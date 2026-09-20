// SPDX-License-Identifier: Apache-2.0
// Adapted from Laya and reimplemented for Candle; see NOTICE for source attribution.
//! Inference-only ModernBERT and Laya heads, using original PyTorch tensor names.
use crate::config::{AgentConfig, EncoderConfig};
use crate::linear::{Linear, linear, linear_b};
use crate::prompt::PreparedQuestion;
use anyhow::{Result, ensure};
use candle_core::{D, DType, Device, IndexOp, Tensor};
use candle_nn::{Embedding, LayerNorm, VarBuilder, embedding, layer_norm, layer_norm_no_bias};

fn norm(size: usize, eps: f64, bias: bool, vb: VarBuilder) -> candle_core::Result<LayerNorm> {
    if bias {
        layer_norm(size, eps, vb)
    } else if vb.device().is_metal() {
        // Candle's no-bias LayerNorm takes the decomposed reduction path.
        // A constant zero bias is algebraically identical and enables its fused
        // Metal kernel without adding or changing any checkpoint parameter.
        let weight = vb.get(size, "weight")?;
        let zero = Tensor::zeros(size, vb.dtype(), vb.device())?;
        Ok(LayerNorm::new(weight, zero, eps))
    } else {
        layer_norm_no_bias(size, eps, vb)
    }
}

fn geglu(x: &Tensor) -> candle_core::Result<Tensor> {
    #[cfg(feature = "metal")]
    if x.device().is_metal() && matches!(x.dtype(), DType::F32 | DType::F16) {
        return crate::metal::geglu(x);
    }
    let parts = x.chunk(2, D::Minus1)?;
    parts[0].gelu_erf()? * &parts[1]
}

fn merge_heads(x: &Tensor) -> candle_core::Result<Tensor> {
    #[cfg(feature = "metal")]
    if x.device().is_metal() && matches!(x.dtype(), DType::F32 | DType::F16) {
        return crate::metal::merge_heads(&x.contiguous()?);
    }
    let (b, h, l, d) = x.dims4()?;
    x.transpose(1, 2)?.contiguous()?.reshape((b, l, h * d))
}

fn pack_heads(x: &Tensor, planes: usize, heads: usize) -> candle_core::Result<Tensor> {
    #[cfg(feature = "metal")]
    if x.device().is_metal() && matches!(x.dtype(), DType::F32 | DType::F16) {
        return crate::metal::pack_heads(x, planes, heads);
    }
    let (b, l, d) = x.dims3()?;
    x.reshape((b, l, planes, heads, d / (planes * heads)))?
        .permute((2, 0, 3, 1, 4))
}

struct Rope {
    cos: Tensor,
    sin: Tensor,
}
impl Rope {
    fn new(
        c: &EncoderConfig,
        local: bool,
        max_len: usize,
        dtype: DType,
        dev: &Device,
    ) -> Result<Self> {
        let dim = c.hidden_size / c.num_attention_heads;
        let theta = c.theta(local);
        let inv: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| (1. / theta.powf(i as f64 / dim as f64)) as f32)
            .collect();
        let freqs: Vec<f32> = (0..max_len)
            .flat_map(|p| inv.iter().map(move |v| p as f32 * v))
            .collect();
        let freqs = Tensor::from_vec(freqs, (max_len, dim / 2), dev)?;
        Ok(Self {
            cos: freqs.cos()?.to_dtype(dtype)?,
            sin: freqs.sin()?.to_dtype(dtype)?,
        })
    }
    fn apply(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        candle_nn::rotary_emb::rope(&x.contiguous()?, &self.cos, &self.sin)
    }
}

struct Attention {
    qkv: Linear,
    out: Linear,
    heads: usize,
}

/// Masks depend on input lengths and geometry, not layer weights. Construct
/// them once per forward pass and reuse them across every matching layer.
struct AttentionBlock {
    start: usize,
    len: usize,
    key_start: usize,
    key_len: usize,
    mask: Option<Tensor>,
}
struct AttentionPlan {
    blocks: Vec<AttentionBlock>,
}
impl AttentionPlan {
    fn new(
        lengths: &[usize],
        window: Option<usize>,
        dtype: DType,
        device: &Device,
        fused: bool,
    ) -> Result<Self> {
        let b = lengths.len();
        let l = *lengths.iter().max().unwrap();
        let window = window.filter(|w| *w < l - 1);
        // Metal SDPA already tiles global attention without an LxL temporary.
        // Keep explicit tiling for the unfused path and sliding attention.
        // Small tiles reduce masked work. Batched/medium-length float32
        // workloads amortize dispatch better with the larger tile.
        let narrow_local = l <= 512 || dtype == DType::F16 || (b == 1 && l >= 2048);
        let tile = if fused && window.is_some() && narrow_local {
            l.min(128)
        } else if l <= 512 || (fused && window.is_none()) {
            l
        } else {
            256
        };
        let padded = lengths.iter().any(|len| *len != l);
        let mut blocks = Vec::new();
        for start in (0..l).step_by(tile) {
            let n = tile.min(l - start);
            let (key_start, key_end) = match window {
                Some(w) if lengths.iter().all(|len| start + n <= *len) => {
                    (start.saturating_sub(w), (start + n + w).min(l))
                }
                _ => (0, l),
            };
            let nk = key_end - key_start;
            let mask = if let Some(w) = window {
                let mut values = Vec::with_capacity(b * n * nk);
                for len in lengths {
                    for i in start..start + n {
                        for j in key_start..key_end {
                            values.push(if j >= *len || (i < *len && i.abs_diff(j) > w) {
                                f32::NEG_INFINITY
                            } else {
                                0.
                            });
                        }
                    }
                }
                Some(Tensor::from_vec(values, (b, 1, n, nk), device)?.to_dtype(dtype)?)
            } else if padded {
                let values: Vec<f32> = lengths
                    .iter()
                    .flat_map(|len| {
                        (key_start..key_end)
                            .map(move |j| if j >= *len { f32::NEG_INFINITY } else { 0. })
                    })
                    .collect();
                Some(Tensor::from_vec(values, (b, 1, 1, nk), device)?.to_dtype(dtype)?)
            } else {
                None
            };
            blocks.push(AttentionBlock {
                start,
                len: n,
                key_start,
                key_len: nk,
                mask,
            });
        }
        Ok(Self { blocks })
    }
}

fn supports_sdpa(device: &Device, head_dim: usize) -> bool {
    device.is_metal() && matches!(head_dim, 32 | 64 | 72 | 80 | 96 | 128 | 256)
}
impl Attention {
    fn encoder(c: &EncoderConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            qkv: linear_b(
                c.hidden_size,
                c.hidden_size * 3,
                c.attention_bias,
                vb.pp("Wqkv"),
            )?,
            out: linear_b(c.hidden_size, c.hidden_size, c.attention_bias, vb.pp("Wo"))?,
            heads: c.num_attention_heads,
        })
    }
    fn head(d: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            qkv: Linear::new(
                vb.get((3 * d, d), "in_proj_weight")?,
                Some(vb.get(3 * d, "in_proj_bias")?),
            ),
            out: linear(d, d, vb.pp("out_proj"))?,
            heads: (d / 64).max(1),
        })
    }
    fn forward(&self, x: &Tensor, plan: &AttentionPlan, rope: Option<&Rope>) -> Result<Tensor> {
        let (b, _, d) = x.dims3()?;
        let hd = d / self.heads;
        let projected = x.apply(&self.qkv)?;
        #[cfg(feature = "metal")]
        let fused = if let Some(rope) = rope {
            if x.device().is_metal() && matches!(x.dtype(), DType::F32 | DType::F16) {
                Some(crate::metal::qkv_rope(
                    &projected, &rope.cos, &rope.sin, self.heads,
                )?)
            } else {
                None
            }
        } else {
            None
        };
        #[cfg(not(feature = "metal"))]
        let fused: Option<Tensor> = None;
        let fused_rope = fused.is_some();
        let qkv = if let Some(qkv) = fused {
            qkv
        } else {
            pack_heads(&projected, 3, self.heads)?
        };
        let mut q = qkv.get(0)?.contiguous()?;
        let mut k = qkv.get(1)?.contiguous()?;
        let v = qkv.get(2)?.contiguous()?;
        if let Some(rope) = rope.filter(|_| !fused_rope) {
            q = rope.apply(&q)?;
            k = rope.apply(&k)?;
        }
        let mut blocks = Vec::new();
        for block in &plan.blocks {
            let (start, n, key_start, nk) =
                (block.start, block.len, block.key_start, block.key_len);
            let qb = q.narrow(2, start, n)?;
            let kb = k.narrow(2, key_start, nk)?;
            let vb = v.narrow(2, key_start, nk)?;
            let output = if supports_sdpa(x.device(), hd) && n > 1 {
                let m = block
                    .mask
                    .as_ref()
                    .map(|m| m.broadcast_as((b, self.heads, n, nk)))
                    .transpose()?;
                candle_nn::ops::sdpa(&qb, &kb, &vb, m.as_ref(), false, (hd as f32).powf(-0.5), 1.)?
            } else {
                let (qb, kb, vb) = (qb.contiguous()?, kb.contiguous()?, vb.contiguous()?);
                let att = (qb.matmul(&kb.transpose(2, 3)?)? * (hd as f64).powf(-0.5))?;
                let mut att = att.to_dtype(DType::F32)?;
                if let Some(m) = &block.mask {
                    att = att.broadcast_add(&m.to_dtype(DType::F32)?)?;
                }
                let att = candle_nn::ops::softmax_last_dim(&att)?.to_dtype(x.dtype())?;
                att.matmul(&vb)?
            };
            blocks.push(output);
        }
        let output = if blocks.len() == 1 {
            blocks.remove(0)
        } else {
            Tensor::cat(&blocks, 2)?
        };
        Ok(merge_heads(&output)?.apply(&self.out)?)
    }
}

struct EncoderLayer {
    attn: Attention,
    attn_norm: Option<LayerNorm>,
    mlp_norm: LayerNorm,
    wi: Linear,
    wo: Linear,
    local: bool,
}
impl EncoderLayer {
    fn load(c: &EncoderConfig, i: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            attn: Attention::encoder(c, vb.pp("attn"))?,
            attn_norm: if i == 0 {
                None
            } else {
                Some(norm(
                    c.hidden_size,
                    c.norm_eps,
                    c.norm_bias,
                    vb.pp("attn_norm"),
                )?)
            },
            mlp_norm: norm(c.hidden_size, c.norm_eps, c.norm_bias, vb.pp("mlp_norm"))?,
            wi: linear_b(
                c.hidden_size,
                c.intermediate_size * 2,
                c.mlp_bias,
                vb.pp("mlp.Wi"),
            )?,
            wo: linear_b(
                c.intermediate_size,
                c.hidden_size,
                c.mlp_bias,
                vb.pp("mlp.Wo"),
            )?,
            local: c.local(i),
        })
    }
    fn forward(&self, x: &Tensor, plan: &AttentionPlan, rope: &Rope) -> Result<Tensor> {
        let normalized = match &self.attn_norm {
            Some(n) => x.apply(n)?,
            None => x.clone(),
        };
        let x = (x + self.attn.forward(&normalized, plan, Some(rope))?)?;
        let projected = x.apply(&self.mlp_norm)?.apply(&self.wi)?;
        let mlp = geglu(&projected)?.apply(&self.wo)?;
        Ok((x + mlp)?)
    }
}

struct HeadLayer {
    attn: Attention,
    norm1: LayerNorm,
    norm2: LayerNorm,
    linear1: Linear,
    linear2: Linear,
}
impl HeadLayer {
    fn load(d: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            attn: Attention::head(d, vb.pp("self_attn"))?,
            norm1: layer_norm(d, 1e-5, vb.pp("norm1"))?,
            norm2: layer_norm(d, 1e-5, vb.pp("norm2"))?,
            linear1: linear(d, 4 * d, vb.pp("linear1"))?,
            linear2: linear(4 * d, d, vb.pp("linear2"))?,
        })
    }
    fn forward(&self, x: &Tensor, plan: &AttentionPlan) -> Result<Tensor> {
        let x = (x + self.attn.forward(&x.apply(&self.norm1)?, plan, None)?)?;
        let mlp = x
            .apply(&self.norm2)?
            .apply(&self.linear1)?
            .relu()?
            .apply(&self.linear2)?;
        Ok((x + mlp)?)
    }

    /// The final head layer only needs CLS and option-marker outputs. Every
    /// token still contributes K/V, preserving full bidirectional attention.
    fn forward_selected(
        &self,
        x: &Tensor,
        indices: &Tensor,
        lengths: &[usize],
        count: usize,
    ) -> Result<Tensor> {
        let (b, l, d) = x.dims3()?;
        let heads = self.attn.heads;
        let hd = d / heads;
        let selected = x
            .reshape((b * l, d))?
            .index_select(indices, 0)?
            .reshape((b, count, d))?;
        let normalized = x.apply(&self.norm1)?;
        let selected_norm = normalized
            .reshape((b * l, d))?
            .index_select(indices, 0)?
            .reshape((b, count, d))?;
        let q_weight = self.attn.qkv.weight().narrow(0, 0, d)?;
        let kv_weight = self.attn.qkv.weight().narrow(0, d, 2 * d)?;
        let q_bias = self
            .attn
            .qkv
            .bias()
            .map(|t| t.narrow(0, 0, d))
            .transpose()?;
        let kv_bias = self
            .attn
            .qkv
            .bias()
            .map(|t| t.narrow(0, d, 2 * d))
            .transpose()?;
        let q = selected_norm
            .apply(&Linear::new(q_weight, q_bias))?
            .reshape((b, count, heads, hd))?
            .transpose(1, 2)?
            .contiguous()?
            // A singleton head axis may still carry its pre-transpose stride.
            // Canonicalize it before the batched matmul fallback reads strides.
            .reshape((b * heads, count, hd))?
            .reshape((b, heads, count, hd))?;
        let kv = pack_heads(
            &normalized.apply(&Linear::new(kv_weight, kv_bias))?,
            2,
            heads,
        )?;
        let k = kv.get(0)?.contiguous()?;
        let v = kv.get(1)?.contiguous()?;
        let mask = if lengths.iter().all(|n| *n == l) {
            None
        } else {
            let values: Vec<f32> = lengths
                .iter()
                .flat_map(|n| (0..l).map(move |i| if i >= *n { f32::NEG_INFINITY } else { 0. }))
                .collect();
            Some(Tensor::from_vec(values, (b, 1, 1, l), x.device())?.to_dtype(x.dtype())?)
        };
        let attended = if supports_sdpa(x.device(), hd) && count <= l {
            let mask = mask
                .as_ref()
                .map(|m| m.broadcast_as((b, heads, count, l)))
                .transpose()?;
            candle_nn::ops::sdpa(&q, &k, &v, mask.as_ref(), false, (hd as f32).powf(-0.5), 1.)?
        } else {
            let att =
                (q.matmul(&k.transpose(2, 3)?)? * (hd as f64).powf(-0.5))?.to_dtype(DType::F32)?;
            let att = match mask {
                Some(m) => att.broadcast_add(&m.to_dtype(DType::F32)?)?,
                None => att,
            };
            candle_nn::ops::softmax_last_dim(&att)?
                .to_dtype(x.dtype())?
                .matmul(&v)?
        };
        let x = (selected + merge_heads(&attended)?.apply(&self.attn.out)?)?;
        let mlp = x
            .apply(&self.norm2)?
            .apply(&self.linear1)?
            .relu()?
            .apply(&self.linear2)?;
        Ok((x + mlp)?)
    }
}

pub(crate) struct DecisionModel {
    emb: Embedding,
    emb_norm: LayerNorm,
    layers: Vec<EncoderLayer>,
    final_norm: LayerNorm,
    global_rope: Rope,
    local_rope: Rope,
    local_window: usize,
    encoder_head_dim: usize,
    head: Vec<HeadLayer>,
    type_emb: Embedding,
    scorer_norm: LayerNorm,
    scorer1: Linear,
    scorer2: Linear,
    act1: Linear,
    act2: Linear,
    device: Device,
}

#[derive(Debug, serde::Serialize)]
pub struct RawOutput {
    /// Uncalibrated logits, without padded option slots.
    pub logits: Vec<Vec<f32>>,
    pub action_logits: Vec<Vec<f32>>,
}

impl DecisionModel {
    pub fn load(c: &EncoderConfig, a: &AgentConfig, vb: VarBuilder) -> Result<Self> {
        let d = c.hidden_size;
        let evb = vb.pp("encoder");
        let layers = (0..c.num_hidden_layers)
            .map(|i| EncoderLayer::load(c, i, evb.pp(format!("layers.{i}"))))
            .collect::<Result<_>>()?;
        let head = (0..a.head_layers)
            .map(|i| HeadLayer::load(d, vb.pp(format!("head.layers.{i}"))))
            .collect::<Result<_>>()?;
        Ok(Self {
            emb: embedding(c.vocab_size, d, evb.pp("embeddings.tok_embeddings"))?,
            emb_norm: norm(d, c.norm_eps, c.norm_bias, evb.pp("embeddings.norm"))?,
            final_norm: norm(d, c.norm_eps, c.norm_bias, evb.pp("final_norm"))?,
            layers,
            global_rope: Rope::new(c, false, a.max_len, vb.dtype(), vb.device())?,
            local_rope: Rope::new(c, true, a.max_len, vb.dtype(), vb.device())?,
            local_window: c.local_attention / 2,
            encoder_head_dim: c.hidden_size / c.num_attention_heads,
            head,
            type_emb: embedding(3, d, vb.pp("type_emb"))?,
            // Keep the small final scorer in float32. Low calibration
            // temperatures can amplify a one-ULP half-precision logit change.
            scorer_norm: layer_norm(d, 1e-5, vb.to_dtype(DType::F32).pp("scorer.0"))?,
            scorer1: linear(d, d, vb.to_dtype(DType::F32).pp("scorer.1"))?,
            scorer2: linear(d, 1, vb.to_dtype(DType::F32).pp("scorer.3"))?,
            act1: linear(d + 4, 256, vb.pp("act_head.0"))?,
            act2: linear(256, a.act_costs.len() + 1, vb.pp("act_head.2"))?,
            device: vb.device().clone(),
        })
    }
    pub fn forward(&self, items: &[PreparedQuestion], pad: u32) -> Result<RawOutput> {
        ensure!(!items.is_empty(), "cannot forward an empty batch");
        let b = items.len();
        let l = items.iter().map(|i| i.ids.len()).max().unwrap();
        let k = items.iter().map(|i| i.markers.len()).max().unwrap().max(2);
        let mut ids = vec![pad; b * l];
        let lengths: Vec<usize> = items.iter().map(|i| i.ids.len()).collect();
        for (r, item) in items.iter().enumerate() {
            ids[r * l..r * l + item.ids.len()].copy_from_slice(&item.ids);
        }
        let ids = Tensor::from_vec(ids, (b, l), &self.device)?;
        let mut h = ids.apply(&self.emb)?.apply(&self.emb_norm)?;
        let global = AttentionPlan::new(
            &lengths,
            None,
            h.dtype(),
            &self.device,
            supports_sdpa(&self.device, self.encoder_head_dim),
        )?;
        let local = AttentionPlan::new(
            &lengths,
            Some(self.local_window),
            h.dtype(),
            &self.device,
            supports_sdpa(&self.device, self.encoder_head_dim),
        )?;
        for layer in &self.layers {
            h = layer.forward(
                &h,
                if layer.local { &local } else { &global },
                if layer.local {
                    &self.local_rope
                } else {
                    &self.global_rope
                },
            )?;
        }
        h = h.apply(&self.final_norm)?;
        let qt = Tensor::from_vec(
            items.iter().map(|i| i.qtype).collect::<Vec<_>>(),
            b,
            &self.device,
        )?;
        let type_embedding = qt.apply(&self.type_emb)?;
        #[cfg(feature = "metal")]
        if self.device.is_metal() && matches!(h.dtype(), DType::F32 | DType::F16) {
            h = crate::metal::add_rows(&h, &type_embedding)?;
        } else {
            h = h.broadcast_add(&type_embedding.unsqueeze(1)?)?;
        }
        #[cfg(not(feature = "metal"))]
        {
            h = h.broadcast_add(&type_embedding.unsqueeze(1)?)?;
        }
        let d = h.dim(2)?;
        let head_plan = AttentionPlan::new(
            &lengths,
            None,
            h.dtype(),
            &self.device,
            supports_sdpa(&self.device, d / (d / 64).max(1)),
        )?;
        let marker_indices: Vec<u32> = items
            .iter()
            .enumerate()
            .flat_map(|(r, item)| {
                (0..k).map(move |j| (r * l) as u32 + item.markers.get(j).copied().unwrap_or(0))
            })
            .collect();
        let indices = Tensor::from_vec(marker_indices, b * k, &self.device)?;
        let markers = if let Some((last, layers)) = self.head.split_last() {
            for layer in layers {
                h = layer.forward(&h, &head_plan)?;
            }
            let select: Vec<u32> =
                items
                    .iter()
                    .enumerate()
                    .flat_map(|(r, item)| {
                        std::iter::once((r * l) as u32).chain((0..k).map(move |j| {
                            (r * l) as u32 + item.markers.get(j).copied().unwrap_or(0)
                        }))
                    })
                    .collect();
            let select = Tensor::from_vec(select, b * (k + 1), &self.device)?;
            h = last.forward_selected(&h, &select, &lengths, k + 1)?;
            h.narrow(1, 1, k)?.contiguous()?
        } else {
            h.reshape((b * l, d))?
                .index_select(&indices, 0)?
                .reshape((b, k, d))?
        };
        let scores = markers
            .to_dtype(DType::F32)?
            .apply(&self.scorer_norm)?
            .apply(&self.scorer1)?
            .gelu_erf()?
            .apply(&self.scorer2)?
            .squeeze(2)?
            .to_dtype(DType::F32)?;
        #[cfg(feature = "metal")]
        if self.device.is_metal() && matches!(h.dtype(), DType::F32 | DType::F16) {
            let counts = Tensor::from_vec(
                items
                    .iter()
                    .map(|item| item.markers.len() as u32)
                    .collect::<Vec<_>>(),
                b,
                &self.device,
            )?;
            let merged = crate::metal::scores_features(&scores, &counts)?;
            let features = merged.narrow(1, k, 4)?.contiguous()?.to_dtype(h.dtype())?;
            let pooled = h.i((.., 0, ..))?.contiguous()?;
            let act = Tensor::cat(&[pooled, features], 1)?
                .apply(&self.act1)?
                .gelu_erf()?
                .apply(&self.act2)?
                .to_dtype(DType::F32)?;
            let packed =
                Tensor::cat(&[merged.narrow(1, 0, k)?.contiguous()?, act], 1)?.to_vec2::<f32>()?;
            let mut logits = Vec::with_capacity(b);
            let mut action_logits = Vec::with_capacity(b);
            for (row, item) in packed.into_iter().zip(items) {
                ensure!(row.iter().all(|z| z.is_finite()), "non-finite model logits");
                logits.push(row[..item.markers.len()].to_vec());
                action_logits.push(row[k..].to_vec());
            }
            return Ok(RawOutput {
                logits,
                action_logits,
            });
        }
        let mut logits = scores.to_vec2::<f32>()?;
        let mut features = Vec::with_capacity(b * 4);
        for (row, item) in logits.iter_mut().zip(items) {
            for z in &mut row[item.markers.len()..] {
                *z = -1e4;
            }
            let p = softmax(row)?;
            let mut top = p.clone();
            top.sort_by(|a, b| b.total_cmp(a));
            let n = item.markers.len().max(2) as f32;
            let entropy = -p.iter().map(|p| p * p.max(1e-9).ln()).sum::<f32>() / n.ln();
            features.extend([top[0], top[0] - top[1], entropy, n / 255.]);
            row.truncate(item.markers.len());
        }
        let features = Tensor::from_vec(features, (b, 4), &self.device)?.to_dtype(h.dtype())?;
        let pooled = h.i((.., 0, ..))?.contiguous()?;
        let act = Tensor::cat(&[pooled, features], 1)?
            .apply(&self.act1)?
            .gelu_erf()?
            .apply(&self.act2)?
            .to_dtype(DType::F32)?
            .to_vec2::<f32>()?;
        ensure!(
            act.iter().flatten().all(|x| x.is_finite()),
            "non-finite action logits"
        );
        Ok(RawOutput {
            logits,
            action_logits: act,
        })
    }
}

pub(crate) fn softmax(z: &[f32]) -> Result<Vec<f32>> {
    ensure!(
        !z.is_empty() && z.iter().all(|v| v.is_finite()),
        "non-finite or empty logits"
    );
    let max = z.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let p: Vec<f32> = z.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = p.iter().sum();
    Ok(p.iter().map(|x| x / sum).collect())
}

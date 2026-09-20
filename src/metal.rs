//! Inference-only fusions. All buffers are registered through Candle's encoder
//! parameter API so its dependency fences and allocator remain authoritative.
use candle_core::{
    CpuStorage, CustomOp1, CustomOp3, DType, Layout, MetalDevice, MetalStorage, Result, Shape,
    Tensor, backend::BackendStorage,
};
use candle_metal_kernels::{
    metal::{ComputeCommandEncoder, ComputePipeline},
    set_params,
    utils::{BufferOffset, Output},
};
use objc2_metal::MTLSize;
use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
};

fn pipeline(device: &MetalDevice, name: &'static str) -> Result<ComputePipeline> {
    static CACHE: OnceLock<Mutex<HashMap<(u64, &'static str), ComputePipeline>>> = OnceLock::new();
    let key = (device.metal_device().registry_id(), name);
    let mut cache = CACHE
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    if let Some(p) = cache.get(&key) {
        return Ok(p.clone());
    }
    let library = device
        .metal_device()
        .new_library_with_source(include_str!("fused.metal"), None)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    let function = library
        .get_function(name, None)
        .map_err(candle_core::Error::wrap)?;
    let p = device
        .metal_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    cache.insert(key, p.clone());
    Ok(p)
}

fn gemm_pipeline(
    device: &MetalDevice,
    name: String,
    alignment: Option<u8>,
) -> Result<ComputePipeline> {
    use candle_metal_kernels::metal::{ConstantValues, Library, Value};
    type Libraries = HashMap<u64, Library>;
    type Pipelines = HashMap<(u64, String, Option<u8>), ComputePipeline>;
    static CACHE: OnceLock<Mutex<(Libraries, Pipelines)>> = OnceLock::new();
    let mut cache = CACHE
        .get_or_init(|| Mutex::new((HashMap::new(), HashMap::new())))
        .lock()
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    let id = device.metal_device().registry_id();
    let key = (id, name.clone(), alignment);
    if let Some(p) = cache.1.get(&key) {
        return Ok(p.clone());
    }
    if let std::collections::hash_map::Entry::Vacant(entry) = cache.0.entry(id) {
        let library = device
            .metal_device()
            .new_library_with_source(include_str!("mlx_gemm_v0322.metal"), None)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        entry.insert(library);
    }
    let constants = alignment.map(|bits| {
        ConstantValues::new(vec![
            (10, Value::Bool(false)),
            (100, Value::Bool(false)),
            (110, Value::Bool(false)),
            (200, Value::Bool(bits & 1 != 0)),
            (201, Value::Bool(bits & 2 != 0)),
            (202, Value::Bool(bits & 4 != 0)),
        ])
    });
    let function = cache.0[&id]
        .get_function(&name, constants.as_ref())
        .map_err(candle_core::Error::wrap)?;
    let p = device
        .metal_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    cache.1.insert(key, p.clone());
    Ok(p)
}
fn dispatch(encoder: &ComputeCommandEncoder, p: &ComputePipeline, count: usize) {
    let width = 256.min(p.max_total_threads_per_threadgroup());
    encoder.dispatch_thread_groups(
        MTLSize {
            width: count.div_ceil(width),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width,
            height: 1,
            depth: 1,
        },
    );
}
fn offset<'a>(storage: &'a MetalStorage, layout: &Layout) -> BufferOffset<'a> {
    BufferOffset {
        buffer: storage.buffer(),
        offset_in_bytes: layout.start_offset() * storage.dtype().size_in_bytes(),
    }
}
fn name(dtype: DType, f32_name: &'static str, f16_name: &'static str) -> Result<&'static str> {
    match dtype {
        DType::F32 => Ok(f32_name),
        DType::F16 => Ok(f16_name),
        _ => candle_core::bail!("unsupported fusion dtype {dtype:?}"),
    }
}

struct GeGlu;
impl CustomOp1 for GeGlu {
    fn name(&self) -> &'static str {
        "laya-geglu"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("Metal-only fusion")
    }
    fn metal_fwd(&self, s: &MetalStorage, l: &Layout) -> Result<(MetalStorage, Shape)> {
        if !l.is_contiguous() {
            candle_core::bail!("GeGLU requires contiguous input")
        }
        let mut dims = l.dims().to_vec();
        let width = *dims
            .last()
            .ok_or_else(|| candle_core::Error::Msg("GeGLU requires rank >= 1".into()))?;
        if width % 2 != 0 {
            candle_core::bail!("GeGLU requires an even last dimension")
        }
        *dims.last_mut().unwrap() = width / 2;
        let count = l.shape().elem_count() / 2;
        if l.shape().elem_count() > u32::MAX as usize {
            candle_core::bail!("GeGLU tensor exceeds 32-bit indexing range")
        }
        let dev = s.device();
        let p = pipeline(dev, name(s.dtype(), "geglu_f32", "geglu_f16")?)?;
        let output = dev
            .new_buffer_builder()
            .with_size_for(count, s.dtype())
            .build()?;
        let guard = dev.command_encoder()?;
        let encoder = guard.as_ref();
        encoder.set_compute_pipeline_state(&p);
        set_params!(
            encoder,
            (count, width / 2, &offset(s, l), Output::new(&output))
        );
        encoder.dispatch_thread_groups(
            MTLSize {
                width: (width / 2).div_ceil(1024),
                height: count / (width / 2),
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        Ok((
            MetalStorage::new(output, dev.clone(), count, s.dtype()),
            Shape::from(dims),
        ))
    }
}
pub fn geglu(x: &Tensor) -> Result<Tensor> {
    x.apply_op1_no_bwd(&GeGlu)
}

struct QkvRope {
    heads: usize,
}
impl CustomOp3 for QkvRope {
    fn name(&self) -> &'static str {
        "laya-qkv-rope"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("Metal-only fusion")
    }
    fn metal_fwd(
        &self,
        s: &MetalStorage,
        l: &Layout,
        cos: &MetalStorage,
        cl: &Layout,
        sin: &MetalStorage,
        sl: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let (b, len, packed) = l.shape().dims3()?;
        if !l.is_contiguous()
            || !cl.is_contiguous()
            || !sl.is_contiguous()
            || packed % (3 * self.heads) != 0
        {
            candle_core::bail!("invalid packed QKV layout")
        }
        let hd = packed / (3 * self.heads);
        if !hd.is_multiple_of(2)
            || cl.dims().len() != 2
            || cl.dims()[0] < len
            || cl.dims()[1] != hd / 2
            || cl.dims() != sl.dims()
            || s.dtype() != cos.dtype()
            || s.dtype() != sin.dtype()
        {
            candle_core::bail!("invalid RoPE table")
        }
        let count = b * len * packed;
        if count > u32::MAX as usize {
            candle_core::bail!("QKV tensor exceeds 32-bit indexing range")
        }
        let dev = s.device();
        let p = pipeline(dev, name(s.dtype(), "qkv_rope_f32", "qkv_rope_f16")?)?;
        let output = dev
            .new_buffer_builder()
            .with_size_for(count, s.dtype())
            .build()?;
        let guard = dev.command_encoder()?;
        let encoder = guard.as_ref();
        encoder.set_compute_pipeline_state(&p);
        set_params!(
            encoder,
            (
                count,
                b,
                len,
                self.heads,
                hd,
                &offset(s, l),
                &offset(cos, cl),
                &offset(sin, sl),
                Output::new(&output)
            )
        );
        encoder.dispatch_thread_groups(
            MTLSize {
                width: (len * hd).div_ceil(256),
                height: self.heads,
                depth: 3 * b,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        Ok((
            MetalStorage::new(output, dev.clone(), count, s.dtype()),
            Shape::from((3, b, self.heads, len, hd)),
        ))
    }
}
pub fn qkv_rope(x: &Tensor, cos: &Tensor, sin: &Tensor, heads: usize) -> Result<Tensor> {
    x.apply_op3_no_bwd(cos, sin, &QkvRope { heads })
}

struct MergeHeads;
impl CustomOp1 for MergeHeads {
    fn name(&self) -> &'static str {
        "laya-merge-heads"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("Metal-only layout conversion")
    }
    fn metal_fwd(&self, s: &MetalStorage, l: &Layout) -> Result<(MetalStorage, Shape)> {
        let (b, h, len, d) = l.shape().dims4()?;
        let count = b * h * len * d;
        if !l.is_contiguous() || count > u32::MAX as usize {
            candle_core::bail!("invalid attention output layout")
        }
        let dev = s.device();
        let p = pipeline(dev, name(s.dtype(), "merge_heads_f32", "merge_heads_f16")?)?;
        let output = dev
            .new_buffer_builder()
            .with_size_for(count, s.dtype())
            .build()?;
        let guard = dev.command_encoder()?;
        let encoder = guard.as_ref();
        encoder.set_compute_pipeline_state(&p);
        set_params!(encoder, (h, len, d, &offset(s, l), Output::new(&output)));
        encoder.dispatch_thread_groups(
            MTLSize {
                width: (h * d).div_ceil(1024),
                height: len,
                depth: b,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        Ok((
            MetalStorage::new(output, dev.clone(), count, s.dtype()),
            Shape::from((b, len, h * d)),
        ))
    }
}
pub fn merge_heads(x: &Tensor) -> Result<Tensor> {
    x.apply_op1_no_bwd(&MergeHeads)
}

struct PackHeads {
    planes: usize,
    heads: usize,
}
impl CustomOp1 for PackHeads {
    fn name(&self) -> &'static str {
        "laya-pack-heads"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("Metal-only layout conversion")
    }
    fn metal_fwd(&self, s: &MetalStorage, l: &Layout) -> Result<(MetalStorage, Shape)> {
        let (b, len, packed) = l.shape().dims3()?;
        let count = l.shape().elem_count();
        if !l.is_contiguous()
            || count > u32::MAX as usize
            || !packed.is_multiple_of(self.planes * self.heads)
        {
            candle_core::bail!("invalid packed attention shape")
        }
        let d = packed / (self.planes * self.heads);
        let dev = s.device();
        let p = pipeline(dev, name(s.dtype(), "pack_heads_f32", "pack_heads_f16")?)?;
        let output = dev
            .new_buffer_builder()
            .with_size_for(count, s.dtype())
            .build()?;
        let guard = dev.command_encoder()?;
        let encoder = guard.as_ref();
        encoder.set_compute_pipeline_state(&p);
        set_params!(
            encoder,
            (
                b,
                len,
                self.heads,
                d,
                self.planes,
                &offset(s, l),
                Output::new(&output)
            )
        );
        encoder.dispatch_thread_groups(
            MTLSize {
                width: (len * d).div_ceil(256),
                height: self.heads,
                depth: self.planes * b,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        Ok((
            MetalStorage::new(output, dev.clone(), count, s.dtype()),
            Shape::from((self.planes, b, self.heads, len, d)),
        ))
    }
}
pub fn pack_heads(x: &Tensor, planes: usize, heads: usize) -> Result<Tensor> {
    x.apply_op1_no_bwd(&PackHeads { planes, heads })
}

struct AddRows;
impl candle_core::CustomOp2 for AddRows {
    fn name(&self) -> &'static str {
        "laya-add-rows"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("Metal-only row addition")
    }
    fn metal_fwd(
        &self,
        x: &MetalStorage,
        xl: &Layout,
        bias: &MetalStorage,
        bl: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let cols = *xl
            .dims()
            .last()
            .ok_or_else(|| candle_core::Error::Msg("missing channel dimension".into()))?;
        let count = xl.shape().elem_count();
        let groups = if bl.dims().len() == 1 {
            1
        } else if bl.dims().len() == 2 && xl.dims().len() == 3 && bl.dims()[0] == xl.dims()[0] {
            bl.dims()[0]
        } else {
            candle_core::bail!("invalid bias shape")
        };
        if !xl.is_contiguous()
            || !bl.is_contiguous()
            || count > u32::MAX as usize
            || bl.shape().elem_count() != groups * cols
            || x.dtype() != bias.dtype()
        {
            candle_core::bail!("invalid row addition")
        }
        let rows = count / (groups * cols);
        let dev = x.device();
        let p = pipeline(dev, name(x.dtype(), "add_rows_f32", "add_rows_f16")?)?;
        let output = dev
            .new_buffer_builder()
            .with_size_for(count, x.dtype())
            .build()?;
        let guard = dev.command_encoder()?;
        let encoder = guard.as_ref();
        encoder.set_compute_pipeline_state(&p);
        set_params!(
            encoder,
            (
                rows,
                cols,
                &offset(x, xl),
                &offset(bias, bl),
                Output::new(&output)
            )
        );
        encoder.dispatch_thread_groups(
            MTLSize {
                width: cols.div_ceil(1024),
                height: rows,
                depth: groups,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        Ok((
            MetalStorage::new(output, dev.clone(), count, x.dtype()),
            xl.shape().clone(),
        ))
    }
}
pub fn add_rows(x: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.apply_op2_no_bwd(bias, &AddRows)
}

struct ScoresFeatures;
impl candle_core::CustomOp2 for ScoresFeatures {
    fn name(&self) -> &'static str {
        "laya-scores-features"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("Metal-only scoring fusion")
    }
    fn metal_fwd(
        &self,
        s: &MetalStorage,
        l: &Layout,
        counts: &MetalStorage,
        cl: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let (b, k) = l.shape().dims2()?;
        if !l.is_contiguous()
            || !cl.is_contiguous()
            || s.dtype() != DType::F32
            || counts.dtype() != DType::U32
            || cl.shape().elem_count() != b
        {
            candle_core::bail!("invalid scoring fusion inputs")
        }
        let dev = s.device();
        let p = pipeline(dev, "scores_features")?;
        let output = dev
            .new_buffer_builder()
            .with_size_for(b * (k + 4), DType::F32)
            .build()?;
        let guard = dev.command_encoder()?;
        let encoder = guard.as_ref();
        encoder.set_compute_pipeline_state(&p);
        set_params!(
            encoder,
            (
                b,
                k,
                &offset(s, l),
                &offset(counts, cl),
                Output::new(&output)
            )
        );
        dispatch(encoder, &p, b);
        Ok((
            MetalStorage::new(output, dev.clone(), b * (k + 4), DType::F32),
            Shape::from((b, k + 4)),
        ))
    }
}
pub fn scores_features(scores: &Tensor, counts: &Tensor) -> Result<Tensor> {
    scores.apply_op2_no_bwd(counts, &ScoresFeatures)
}

// Use the pinned MLX-derived SIMD GEMM kernels through Candle storage and
// command scheduling. Split-K intermediates remain float32 for both dtypes.
struct LinearTile {
    tile: (usize, usize, usize, usize, usize),
}
#[repr(C)]
struct GemmParams {
    m: i32,
    n: i32,
    k: i32,
    lda: i32,
    ldb: i32,
    ldd: i32,
    tiles_n: i32,
    tiles_m: i32,
    batch_stride_a: isize,
    batch_stride_b: isize,
    batch_stride_d: isize,
    swizzle_log: i32,
    gemm_k_iterations_aligned: i32,
    batch_ndim: i32,
}
impl candle_metal_kernels::utils::EncoderParam for GemmParams {
    fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self) {
        encoder.set_bytes(position, &data)
    }
}
impl candle_core::CustomOp2 for LinearTile {
    fn name(&self) -> &'static str {
        "laya-linear-tile"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("Metal-only GEMM")
    }
    fn metal_fwd(
        &self,
        x: &MetalStorage,
        xl: &Layout,
        w: &MetalStorage,
        wl: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let (m, k) = xl.shape().dims2()?;
        let (n, wk) = wl.shape().dims2()?;
        if !xl.is_contiguous() || !wl.is_contiguous() || wk != k || x.dtype() != w.dtype() {
            candle_core::bail!("invalid linear inputs")
        }
        let (bm, bn, bk, wm, wn) = self.tile;
        let dtype = name(x.dtype(), "f32", "f16")?;
        let kernel = format!("laya_gemm_{dtype}_{bm}_{bn}");
        let alignment = (m.is_multiple_of(bm) as u8)
            | ((n.is_multiple_of(bn) as u8) << 1)
            | ((k.is_multiple_of(bk) as u8) << 2);
        let dev = x.device();
        let pipeline = gemm_pipeline(dev, kernel, Some(alignment))?;
        let output = dev
            .new_buffer_builder()
            .with_size_for(m * n, x.dtype())
            .build()?;
        let guard = dev.command_encoder()?;
        let encoder = guard.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);
        let (tn, tm) = (n.div_ceil(bn), m.div_ceil(bm));
        let params = GemmParams {
            m: m as i32,
            n: n as i32,
            k: k as i32,
            lda: k as i32,
            ldb: k as i32,
            ldd: n as i32,
            tiles_n: tn as i32,
            tiles_m: tm as i32,
            batch_stride_a: 0,
            batch_stride_b: 0,
            batch_stride_d: (m * n) as isize,
            swizzle_log: 0,
            gemm_k_iterations_aligned: (k / bk) as i32,
            batch_ndim: 1,
        };
        let strides = [0isize, 0];
        set_params!(
            encoder,
            (
                &offset(x, xl),
                &offset(w, wl),
                (),
                Output::new(&output),
                params,
                (),
                1i32,
                &strides[..]
            )
        );
        encoder.dispatch_thread_groups(
            MTLSize {
                width: tn,
                height: tm,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: wn,
                depth: wm,
            },
        );
        Ok((
            MetalStorage::new(output, dev.clone(), m * n, x.dtype()),
            Shape::from((m, n)),
        ))
    }
}

pub fn linear(x: &Tensor, w: &Tensor, tile: &str) -> Result<Tensor> {
    let mut shape = x.dims().to_vec();
    let k = x.dim(candle_core::D::Minus1)?;
    let m = x.elem_count() / k;
    let n = w.dim(0)?;
    *shape.last_mut().unwrap() = n;
    if m == 1 || n == 1 {
        return x.reshape((m, k))?.matmul(&w.t()?)?.reshape(shape);
    }
    if k >= m.max(n) && k >= 128 && m.div_ceil(16) * n.div_ceil(16) <= 2048 {
        return x
            .reshape((m, k))?
            .apply_op2_no_bwd(w, &SplitK)?
            .reshape(shape);
    }
    let tile = match tile {
        "32x32" => (32, 32, 16, 2, 2),
        "64x64" => (64, 64, 16, 2, 2),
        _ => candle_core::bail!("unknown GEMM tile {tile}"),
    };
    x.reshape((m, k))?
        .apply_op2_no_bwd(w, &LinearTile { tile })?
        .reshape(shape)
}

struct SplitK;
impl candle_core::CustomOp2 for SplitK {
    fn name(&self) -> &'static str {
        "laya-split-k"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("Metal-only split-K")
    }
    fn metal_fwd(
        &self,
        x: &MetalStorage,
        xl: &Layout,
        w: &MetalStorage,
        wl: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let (m, k) = xl.shape().dims2()?;
        let (n, wk) = wl.shape().dims2()?;
        if !xl.is_contiguous() || !wl.is_contiguous() || wk != k || x.dtype() != w.dtype() {
            candle_core::bail!("invalid split-K inputs")
        }
        let dtype = name(x.dtype(), "f32", "f16")?;
        let dev = x.device();
        let bm = if m < 40 { 16 } else { 32 };
        let bn = if n < 40 { 16 } else { 32 };
        let parts = ((k / 16) / (m.div_ceil(32) * n.div_ceil(32)))
            .next_power_of_two()
            .clamp(2, 32);
        let iterations = (k / 16) / parts;
        let partition_size = iterations * 16;
        let p = gemm_pipeline(dev, format!("laya_split_{dtype}_{bm}_{bn}"), None)?;
        let partial = dev
            .new_buffer_builder()
            .with_size_for(parts * m * n, DType::F32)
            .build()?;
        let output = dev
            .new_buffer_builder()
            .with_size_for(m * n, x.dtype())
            .build()?;
        let params = [
            m as i32,
            n as i32,
            k as i32,
            k as i32,
            k as i32,
            n as i32,
            n.div_ceil(bn) as i32,
            m.div_ceil(bm) as i32,
            parts as i32,
            (m * n) as i32,
            partition_size as i32,
            0,
            iterations as i32,
        ];
        {
            let guard = dev.command_encoder()?;
            let encoder = guard.as_ref();
            encoder.set_compute_pipeline_state(&p);
            set_params!(
                encoder,
                (
                    &offset(x, xl),
                    &offset(w, wl),
                    Output::new(&partial),
                    &params[..]
                )
            );
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: n.div_ceil(bn),
                    height: m.div_ceil(bm),
                    depth: parts,
                },
                MTLSize {
                    width: 32,
                    height: 2,
                    depth: 2,
                },
            );
        }
        let p = gemm_pipeline(dev, format!("laya_split_sum_{dtype}"), None)?;
        let guard = dev.command_encoder()?;
        let encoder = guard.as_ref();
        encoder.set_compute_pipeline_state(&p);
        set_params!(
            encoder,
            (
                &BufferOffset::zero_offset(&partial),
                Output::new(&output),
                parts as i32,
                (m * n) as i32,
                n as i32
            )
        );
        encoder.dispatch_threads(
            MTLSize {
                width: n,
                height: m,
                depth: 1,
            },
            MTLSize {
                width: 32.min(n),
                height: 4.min(m),
                depth: 1,
            },
        );
        Ok((
            MetalStorage::new(output, dev.clone(), m * n, x.dtype()),
            Shape::from((m, n)),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{D, Device};

    fn assert_close(a: &Tensor, b: &Tensor, tolerance: f32) -> Result<()> {
        let a = a.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let b = b.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let error = a
            .iter()
            .zip(&b)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            error <= tolerance,
            "maximum kernel difference {error} > {tolerance}"
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires a Metal GPU"]
    fn fused_kernels_match_unfused_operations() -> Result<()> {
        let dev = Device::new_metal(0)?;
        for dtype in [DType::F32, DType::F16] {
            let tolerance = if dtype == DType::F32 { 2e-6 } else { 0.004 };
            for width in [1, 11, 37, 127, 2624] {
                let values: Vec<f32> = (0..6 * width * 2)
                    .map(|i| (i % 137) as f32 / 37. - 1.8)
                    .collect();
                let x = Tensor::from_vec(values, (2, 3, 2 * width), &dev)?.to_dtype(dtype)?;
                let parts = x.chunk(2, D::Minus1)?;
                assert_close(&geglu(&x)?, &(parts[0].gelu_erf()? * &parts[1])?, tolerance)?;
            }
            for (b, l, h, d) in [
                (2, 13, 4, 16),
                (1, 129, 2, 64),
                (1, 513, 2, 64),
                (3, 17, 5, 24),
            ] {
                let values: Vec<f32> = (0..b * l * 3 * h * d)
                    .map(|i| (i % 137) as f32 / 37. - 1.8)
                    .collect();
                let x = Tensor::from_vec(values, (b, l, 3 * h * d), &dev)?.to_dtype(dtype)?;
                let positions: Vec<f32> = (0..l * d / 2).map(|i| i as f32 * 0.013).collect();
                let positions = Tensor::from_vec(positions, (l, d / 2), &dev)?;
                let cos = positions.cos()?.to_dtype(dtype)?;
                let sin = positions.sin()?.to_dtype(dtype)?;
                let expected = x.reshape((b, l, 3, h, d))?.permute((2, 0, 3, 1, 4))?;
                let q = candle_nn::rotary_emb::rope(&expected.get(0)?.contiguous()?, &cos, &sin)?;
                let k = candle_nn::rotary_emb::rope(&expected.get(1)?.contiguous()?, &cos, &sin)?;
                let expected = Tensor::stack(&[q, k, expected.get(2)?.contiguous()?], 0)?;
                assert_close(&qkv_rope(&x, &cos, &sin, h)?, &expected, tolerance)?;
                let packed = pack_heads(&x, 3, h)?;
                assert_close(
                    &packed,
                    &x.reshape((b, l, 3, h, d))?
                        .permute((2, 0, 3, 1, 4))?
                        .contiguous()?,
                    0.,
                )?;
                assert_close(
                    &merge_heads(&packed.get(0)?)?,
                    &packed
                        .get(0)?
                        .transpose(1, 2)?
                        .contiguous()?
                        .reshape((b, l, h * d))?,
                    0.,
                )?;
                let two = x.narrow(2, 0, 2 * h * d)?.contiguous()?;
                assert_close(
                    &pack_heads(&two, 2, h)?,
                    &two.reshape((b, l, 2, h, d))?
                        .permute((2, 0, 3, 1, 4))?
                        .contiguous()?,
                    0.,
                )?;
            }
            let x = Tensor::arange(0u32, 66u32, &dev)?
                .to_dtype(dtype)?
                .reshape((2, 3, 11))?;
            let bias = Tensor::arange(0u32, 11u32, &dev)?.to_dtype(dtype)?;
            assert_close(&add_rows(&x, &bias)?, &x.broadcast_add(&bias)?, 0.)?;
            let bias = Tensor::arange(0u32, 22u32, &dev)?
                .to_dtype(dtype)?
                .reshape((2, 11))?;
            assert_close(
                &add_rows(&x, &bias)?,
                &x.broadcast_add(&bias.unsqueeze(1)?)?,
                0.,
            )?;
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires a Metal GPU"]
    fn scores_features_match_cpu_and_gemm_handles_tail_tiles() -> Result<()> {
        let dev = Device::new_metal(0)?;
        let values = vec![
            -20000., 5., -3., 9., -2., 1., 1., 0., 0., 0., -100., -99., -98., 5., 6., 0., -0.1,
            0.1, 4., -3.,
        ];
        let counts = vec![1u32, 2, 3, 5];
        let scores = Tensor::from_vec(values.clone(), (4, 5), &dev)?;
        let result = scores_features(&scores, &Tensor::from_vec(counts.clone(), 4, &dev)?)?
            .to_vec2::<f32>()?;
        for (row, n) in counts.iter().enumerate() {
            let mut logits = values[row * 5..(row + 1) * 5].to_vec();
            logits[*n as usize..].fill(-1e4);
            assert_eq!(result[row][..5], logits);
            let p = crate::model::softmax(&logits).unwrap();
            let mut sorted = p.clone();
            sorted.sort_by(|a, b| b.total_cmp(a));
            let k = (*n).max(2) as f32;
            let entropy = -p.iter().map(|p| p * p.max(1e-9).ln()).sum::<f32>() / k.ln();
            let expected = [sorted[0], sorted[0] - sorted[1], entropy, k / 255.];
            for (a, b) in result[row][5..].iter().zip(expected) {
                assert!((a - b).abs() < 2e-6);
            }
        }
        for dtype in [DType::F32, DType::F16] {
            for (m, k, n) in [
                (3, 17, 11),
                (33, 65, 73),
                (129, 128, 127),
                (257, 256, 192),
                (3, 129, 65),
                (5, 1028, 256),
                (30, 1024, 1024),
                (110, 2624, 1024),
                (512, 2624, 1024),
            ] {
                let x: Vec<f32> = (0..m * k).map(|i| (i % 29) as f32 / 100. - 0.14).collect();
                let w: Vec<f32> = (0..n * k).map(|i| (i % 31) as f32 / 100. - 0.15).collect();
                let x = Tensor::from_vec(x, (m, k), &dev)?.to_dtype(dtype)?;
                let w = Tensor::from_vec(w, (n, k), &dev)?.to_dtype(dtype)?;
                let expected = x.matmul(&w.t()?)?;
                for tile in ["32x32", "64x64"] {
                    assert_close(
                        &linear(&x, &w, tile)?,
                        &expected,
                        if dtype == DType::F32 { 2e-5 } else { 0.001 },
                    )?;
                }
            }
        }
        Ok(())
    }
}

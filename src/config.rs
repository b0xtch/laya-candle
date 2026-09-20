// SPDX-License-Identifier: Apache-2.0
// Adapted from Laya and reimplemented for Candle; see NOTICE for source attribution.
use anyhow::{Result, ensure};
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Clone, Deserialize)]
pub struct AgentConfig {
    pub encoder: String,
    pub head_layers: usize,
    #[serde(default = "default_max_len")]
    pub max_len: usize,
    #[serde(default = "default_head_len")]
    pub head_max_len: usize,
    #[serde(default)]
    pub act_costs: HashMap<String, f64>,
    #[serde(default = "default_temperatures")]
    pub temperature: [f32; 3],
    #[serde(default)]
    pub temperature_by_options: HashMap<String, f32>,
}
fn default_max_len() -> usize {
    512
}
fn default_head_len() -> usize {
    192
}
fn default_temperatures() -> [f32; 3] {
    [1.; 3]
}

impl AgentConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.max_len >= 8, "max_len must be at least 8");
        ensure!(
            self.head_max_len >= 16 && self.head_max_len < self.max_len,
            "head_max_len must be >= 16 and < max_len"
        );
        ensure!(
            self.temperature
                .iter()
                .chain(self.temperature_by_options.values())
                .all(|v| v.is_finite() && *v > 0.),
            "temperatures must be finite and positive"
        );
        Ok(())
    }
    pub fn temperature(&self, kind: &str, qt: usize, k: usize) -> f32 {
        let bucket = match k {
            0..=2 => "2",
            3..=5 => "3-5",
            6..=10 => "6-10",
            _ => "11+",
        };
        self.temperature_by_options
            .get(&format!("{kind}:{bucket}"))
            .copied()
            .unwrap_or(self.temperature[qt])
            .max(1e-3)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct EncoderConfig {
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub max_position_embeddings: usize,
    #[serde(default = "default_eps", alias = "layer_norm_epsilon")]
    pub norm_eps: f64,
    #[serde(default)]
    pub norm_bias: bool,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub mlp_bias: bool,
    pub hidden_activation: String,
    pub local_attention: usize,
    pub global_attn_every_n_layers: usize,
    #[serde(default = "global_theta")]
    pub global_rope_theta: f64,
    #[serde(default = "local_theta")]
    pub local_rope_theta: f64,
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,
    #[serde(default)]
    pub rope_parameters: HashMap<String, RopeConfig>,
}
fn default_eps() -> f64 {
    1e-5
}
fn global_theta() -> f64 {
    160000.
}
fn local_theta() -> f64 {
    10000.
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeConfig {
    pub rope_type: String,
    pub rope_theta: f64,
}
impl EncoderConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.model_type == "modernbert",
            "unsupported encoder: {}",
            self.model_type
        );
        ensure!(
            self.hidden_activation == "gelu",
            "only GELU encoders are supported"
        );
        ensure!(
            self.hidden_size > 0
                && self.num_attention_heads > 0
                && self.num_hidden_layers > 0
                && self.intermediate_size > 0
                && self.vocab_size > 0,
            "encoder dimensions must be positive"
        );
        ensure!(
            self.hidden_size.is_multiple_of(self.num_attention_heads)
                && (self.hidden_size / self.num_attention_heads).is_multiple_of(2),
            "invalid attention dimensions"
        );
        ensure!(
            self.hidden_size
                .is_multiple_of((self.hidden_size / 64).max(1)),
            "invalid decision head dimensions"
        );
        ensure!(
            self.global_attn_every_n_layers > 0,
            "invalid global attention interval"
        );
        ensure!(
            self.norm_eps.is_finite() && self.norm_eps > 0.,
            "invalid layer norm epsilon"
        );
        if let Some(kinds) = &self.layer_types {
            ensure!(
                kinds.len() == self.num_hidden_layers
                    && kinds
                        .iter()
                        .all(|s| s == "full_attention" || s == "sliding_attention"),
                "invalid layer_types"
            );
        }
        for rope in self.rope_parameters.values() {
            ensure!(rope.rope_type == "default", "scaled RoPE is not supported");
            ensure!(
                rope.rope_theta.is_finite() && rope.rope_theta > 0.,
                "invalid RoPE base"
            );
        }
        ensure!(
            [self.global_rope_theta, self.local_rope_theta]
                .iter()
                .all(|x| x.is_finite() && *x > 0.),
            "invalid RoPE base"
        );
        Ok(())
    }
    pub fn local(&self, i: usize) -> bool {
        self.layer_types
            .as_ref()
            .map(|v| v[i] == "sliding_attention")
            .unwrap_or(!i.is_multiple_of(self.global_attn_every_n_layers))
    }
    pub fn theta(&self, local: bool) -> f64 {
        let name = if local {
            "sliding_attention"
        } else {
            "full_attention"
        };
        self.rope_parameters
            .get(name)
            .map(|r| r.rope_theta)
            .unwrap_or(if local {
                self.local_rope_theta
            } else {
                self.global_rope_theta
            })
    }
}

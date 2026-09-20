// SPDX-License-Identifier: Apache-2.0
// Adapted from Laya and reimplemented for Candle; see NOTICE for source attribution.
//! Native Rust inference for Laya's choice, score, and noul decisions.
//!
//! ```no_run
//! use laya_candle::{Agent, LoadOptions};
//! use serde_json::json;
//! let agent = Agent::load("convaiinnovations/laya", LoadOptions::default())?;
//! let questions = serde_json::from_value(json!({"refund": {
//!     "type": "noul", "instructions": "Does the customer request a refund?"
//! }}))?;
//! let result = agent.predict(&json!("Please refund the duplicate charge."), &questions)?;
//! println!("{}", serde_json::to_string_pretty(&result)?);
//! # Ok::<(), anyhow::Error>(())
//! ```
mod checkpoint;
pub mod config;
mod linear;
#[cfg(feature = "metal")]
mod metal;
mod model;
pub mod prompt;

use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device};
use checkpoint::{Checkpoint, WeightAudit};
use config::{AgentConfig, EncoderConfig};
pub use model::RawOutput;
use model::{DecisionModel, softmax};
pub use prompt::{PreparedQuestion, Questions};
use prompt::{PromptTokenizer, Question};
use serde::Serialize;
use serde_json::{Map, Value, json};

/// Checkpoint source, device, precision, and input budgeting settings.
///
/// Defaults use CPU float32, the Hub's `main` revision, batches of 16 questions,
/// and the checkpoint's context budgets. Use a commit SHA for reproducible Hub loads.
pub struct LoadOptions {
    /// Explicit inference device; GPU requests never silently fall back to CPU.
    pub device: Device,
    /// Arithmetic precision. CPU supports only float32.
    pub dtype: DType,
    /// Hub branch, tag, or commit SHA; ignored for local checkpoint directories.
    pub revision: String,
    /// Relative directory within a local checkpoint or Hub repository.
    pub subfolder: Option<String>,
    /// Resolve Hub files from the local cache without making network requests.
    pub offline: bool,
    /// Maximum number of questions evaluated together; must be positive.
    pub batch_size: usize,
    /// Override total sequence budget, bounded by the encoder's context limit.
    pub max_len: Option<usize>,
    /// Override upstream budgeting for question and option descriptions.
    pub head_max_len: Option<usize>,
    /// Return an error if any question leaves too little room for the full state.
    pub reject_state_truncation: bool,
}
impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            device: Device::Cpu,
            dtype: DType::F32,
            revision: "main".into(),
            subfolder: None,
            offline: false,
            batch_size: 16,
            max_len: None,
            head_max_len: None,
            reject_state_truncation: false,
        }
    }
}

#[derive(Debug, Serialize)]
/// Typed answers and token usage, with upstream-compatible JSON formatting.
pub struct Prediction {
    pub model: &'static str,
    pub answers: Map<String, Value>,
    pub usage: Usage,
}
#[derive(Debug, Serialize)]
/// Input tokens summed across questions. This decision model generates no tokens.
pub struct Usage {
    pub input_tokens: usize,
    pub output_tokens: usize,
}

/// Loaded Laya checkpoint for reusable inference across requests.
pub struct Agent {
    config: AgentConfig,
    encoder: EncoderConfig,
    tokenizer: PromptTokenizer,
    model: DecisionModel,
    batch_size: usize,
    reject_state_truncation: bool,
}
impl Agent {
    /// Load original Laya safetensors from a local directory or Hugging Face model ID.
    ///
    /// Returns an error for unsupported configurations, missing or extra weights,
    /// unavailable devices, invalid budgets, or uncached files in offline mode.
    pub fn load(source: &str, options: LoadOptions) -> Result<Self> {
        ensure!(options.batch_size > 0, "batch_size must be positive");
        ensure!(
            matches!(options.dtype, DType::F32 | DType::F16 | DType::BF16),
            "unsupported inference dtype"
        );
        ensure!(
            !options.device.is_cpu() || options.dtype == DType::F32,
            "CPU inference requires float32"
        );
        let cp = Checkpoint::resolve(
            source,
            &options.revision,
            options.subfolder.as_deref(),
            options.offline,
        )?;
        let mut config: AgentConfig = serde_json::from_slice(&std::fs::read(&cp.agent)?)?;
        if let Some(n) = options.max_len {
            config.max_len = n;
        }
        if let Some(n) = options.head_max_len {
            config.head_max_len = n;
        }
        config.validate()?;
        let encoder: EncoderConfig = serde_json::from_slice(&std::fs::read(&cp.encoder)?)?;
        encoder.validate()?;
        ensure!(
            config.max_len <= encoder.max_position_embeddings,
            "max_len exceeds encoder context limit"
        );
        let tokenizer = PromptTokenizer::load(&cp.tokenizer_dir)?;
        let (vb, audit) = WeightAudit::load(&cp.weights, options.dtype, &options.device)?;
        let model =
            DecisionModel::load(&encoder, &config, vb).context("loading Laya architecture")?;
        audit.finish()?;
        Ok(Self {
            config,
            encoder,
            tokenizer,
            model,
            batch_size: options.batch_size,
            reject_state_truncation: options.reject_state_truncation,
        })
    }
    /// Effective agent configuration after applying load-time overrides.
    pub fn config(&self) -> &AgentConfig {
        &self.config
    }
    /// Encoder architecture and context limits read from the checkpoint.
    pub fn encoder_config(&self) -> &EncoderConfig {
        &self.encoder
    }

    /// Inspect exact token IDs and option marker positions for parity testing.
    pub fn prepare(&self, state: &Value, questions: &Questions) -> Result<Vec<PreparedQuestion>> {
        questions
            .iter()
            .map(|(id, q)| {
                let q = Question::parse(q).with_context(|| format!("question {id:?}"))?;
                self.tokenizer
                    .prepare(
                        state,
                        &q,
                        self.config.max_len,
                        self.config.head_max_len,
                        self.reject_state_truncation,
                    )
                    .with_context(|| format!("question {id:?}"))
            })
            .collect()
    }
    /// Run bounded batches of prepared inputs and return uncalibrated logits.
    pub fn forward(&self, items: &[PreparedQuestion]) -> Result<RawOutput> {
        for item in items {
            ensure!(
                !item.ids.is_empty() && item.ids.len() <= self.config.max_len,
                "invalid sequence length"
            );
            ensure!(
                !item.markers.is_empty()
                    && item.markers.iter().all(|p| (*p as usize) < item.ids.len()),
                "invalid marker positions"
            );
            ensure!(
                item.qtype < 3
                    && item
                        .ids
                        .iter()
                        .all(|id| (*id as usize) < self.encoder.vocab_size),
                "invalid question type or token ID"
            );
        }
        let mut output = RawOutput {
            logits: vec![],
            action_logits: vec![],
        };
        for batch in items.chunks(self.batch_size) {
            let raw = self.model.forward(batch, self.tokenizer.pad)?;
            output.logits.extend(raw.logits);
            output.action_logits.extend(raw.action_logits);
        }
        Ok(output)
    }
    /// Prepare, evaluate, and calibrate questions about a text or JSON state.
    ///
    /// Question IDs and choice labels preserve JSON insertion order. An empty
    /// question map returns empty answers without running the model.
    pub fn predict(&self, state: &Value, questions: &Questions) -> Result<Prediction> {
        let items = self.prepare(state, questions)?;
        let raw = self.forward(&items)?;
        self.format(questions, &items, &raw)
    }
    /// Alias matching the upstream API.
    pub fn system_one(&self, state: &Value, questions: &Questions) -> Result<Prediction> {
        self.predict(state, questions)
    }

    /// Calibrate raw outputs and format answers for matching prepared questions.
    ///
    /// Pass the questions, prepared items, and raw outputs from the same request
    /// in the same order. This is primarily exposed for numerical verification.
    pub fn format(
        &self,
        questions: &Questions,
        items: &[PreparedQuestion],
        raw: &RawOutput,
    ) -> Result<Prediction> {
        ensure!(
            questions.len() == items.len()
                && items.len() == raw.logits.len()
                && items.len() == raw.action_logits.len(),
            "result row count mismatch"
        );
        let mut answers = Map::new();
        for (r, (id, value)) in questions.iter().enumerate() {
            let q = Question::parse(value)?;
            let k = q.options.len();
            ensure!(raw.logits[r].len() == k, "option/logit count mismatch");
            let temp = self.config.temperature(&q.kind, q.qtype as usize, k);
            let p = softmax(&raw.logits[r].iter().map(|z| z / temp).collect::<Vec<_>>())?;
            let action =
                json!({"act_probability": round4(softmax(&raw.action_logits[r])?[0] as f64)});
            let confidence = if k < 2 {
                1.
            } else {
                (1. + p.iter().map(|p| p * p.clamp(1e-12, 1.).ln()).sum::<f32>() / (k as f32).ln())
                    .clamp(0., 1.)
            };
            let probabilities: Map<_, _> = q
                .labels
                .iter()
                .zip(&p)
                .map(|(label, p)| (label.clone(), json!(round4(*p as f64))))
                .collect();
            let answer = match q.kind.as_str() {
                "choice" => {
                    let best = p
                        .iter()
                        .enumerate()
                        .fold(0, |best, (i, x)| if *x > p[best] { i } else { best });
                    json!({"type":"choice","choice":q.labels[best],"probabilities":probabilities,"confidence":round4(confidence as f64),"action":action})
                }
                "score" => {
                    let score: f64 = p
                        .iter()
                        .enumerate()
                        .map(|(i, p)| i as f64 * *p as f64)
                        .sum();
                    let legend: Map<_, _> = q
                        .criteria
                        .as_array()
                        .unwrap()
                        .iter()
                        .enumerate()
                        .map(|(i, v)| (i.to_string(), v.clone()))
                        .collect();
                    json!({"type":"score","score":round4(score),"legend":legend,"probabilities":probabilities,"confidence":round4(confidence as f64),"action":action})
                }
                _ => {
                    json!({"type":"noul","noul":round4(p[1] as f64),"confidence":round4((p[1] as f64).max(1.-p[1] as f64)),"action":action})
                }
            };
            answers.insert(id.clone(), answer);
        }
        Ok(Prediction {
            model: "laya-rl-agent",
            answers,
            usage: Usage {
                input_tokens: items.iter().map(|i| i.ids.len()).sum(),
                output_tokens: 0,
            },
        })
    }
}
fn round4(x: f64) -> f64 {
    (x * 10000.).round_ties_even() / 10000.
}

use anyhow::{Context, Result};
use candle_core::{DType, Device};
use clap::{Parser, Subcommand, ValueEnum};
use laya_candle::{Agent, LoadOptions, Questions};
use serde_json::{Value, json};
use std::path::PathBuf;

#[derive(Parser)]
#[command(version, about = "Native Rust inference for Laya typed decisions")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    Predict(Args),
    Inspect(Args),
}
#[derive(Clone, Copy, ValueEnum)]
enum Backend {
    Cpu,
    Metal,
    Cuda,
}
#[derive(Clone, Copy, ValueEnum)]
enum Precision {
    F32,
    F16,
    Bf16,
}
#[derive(clap::Args)]
struct Args {
    #[arg(long, default_value = "convaiinnovations/laya")]
    model: String,
    #[arg(long, default_value = "main")]
    revision: String,
    #[arg(long)]
    subfolder: Option<String>,
    #[arg(long)]
    offline: bool,
    #[arg(long, value_enum, default_value = "cpu")]
    device: Backend,
    #[arg(long, value_enum, default_value = "f32")]
    dtype: Precision,
    #[arg(long, default_value_t = 16)]
    batch_size: usize,
    #[arg(long)]
    max_len: Option<usize>,
    #[arg(long)]
    head_max_len: Option<usize>,
    /// Fail if the state would be shortened to fit any question's context budget.
    #[arg(long)]
    reject_state_truncation: bool,
    #[arg(
        long,
        conflicts_with = "state_file",
        required_unless_present = "state_file"
    )]
    state: Option<String>,
    #[arg(long)]
    state_file: Option<PathBuf>,
    #[arg(long)]
    questions: PathBuf,
}
impl Args {
    fn load(&self) -> Result<(Agent, Value, Questions)> {
        let state = if let Some(path) = &self.state_file {
            let text = std::fs::read_to_string(path)?;
            serde_json::from_str(&text)
                .context("state-file must contain JSON (use --state for plain text)")?
        } else {
            Value::String(self.state.clone().unwrap())
        };
        let questions = serde_json::from_slice(&std::fs::read(&self.questions)?)?;
        let device = match self.device {
            Backend::Cpu => Device::Cpu,
            Backend::Metal => {
                Device::new_metal(0).context("Metal unavailable; build with --features metal")?
            }
            Backend::Cuda => {
                Device::new_cuda(0).context("CUDA unavailable; build with --features cuda")?
            }
        };
        let dtype = match self.dtype {
            Precision::F32 => DType::F32,
            Precision::F16 => DType::F16,
            Precision::Bf16 => DType::BF16,
        };
        let agent = Agent::load(
            &self.model,
            LoadOptions {
                device,
                dtype,
                revision: self.revision.clone(),
                subfolder: self.subfolder.clone(),
                offline: self.offline,
                batch_size: self.batch_size,
                max_len: self.max_len,
                head_max_len: self.head_max_len,
                reject_state_truncation: self.reject_state_truncation,
            },
        )?;
        Ok((agent, state, questions))
    }
}
fn main() -> Result<()> {
    let result = match Cli::parse().command {
        Command::Predict(args) => {
            let (agent, state, questions) = args.load()?;
            serde_json::to_value(agent.predict(&state, &questions)?)?
        }
        Command::Inspect(args) => {
            let (agent, state, questions) = args.load()?;
            let items = agent.prepare(&state, &questions)?;
            let raw = agent.forward(&items)?;
            json!({"items":items,"raw":raw,"prediction":agent.format(&questions,&items,&raw)?})
        }
    };
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

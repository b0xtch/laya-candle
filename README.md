# laya-candle

Native Rust inference for [Laya](https://github.com/NandhaKishorM/laya) using
[Candle](https://github.com/huggingface/candle). Pass text or JSON and typed
questions to receive choices, ordinal scores, or boolean probabilities.
Loads original Laya safetensors directly, with no Python runtime.

## Usage

Use Rust 1.95 (pinned in `rust-toolchain.toml`) and a native C/C++ build toolchain.
Clone the source and run with CPU float32, the default backend:

```sh
git clone https://github.com/b0xtch/laya-candle.git
cd laya-candle
cargo run --release -- predict \
  --model convaiinnovations/laya \
  --state-file examples/state.json --questions examples/questions.json
```

On Apple Silicon:

```sh
cargo run --release --features metal -- predict \
  --model convaiinnovations/laya --device metal \
  --state 'Please refund the duplicate charge.' \
  --questions examples/questions.json
```

Use `--dtype f16` for reduced precision on Metal. The final scorer and calibration
remain float32, but reduced precision can change probabilities and close decisions.
Metal uses fused kernels, tiled attention, and GEMM dispatch tuned for M1 Pro;
other Metal devices use Candle's matrix dispatch. CPU requires float32.
`--features accelerate` enables Apple's CPU BLAS backend.
CUDA (`--features cuda`, `--device cuda`) and bfloat16 are available but have not
been validated on hardware for this project.

Install the CLI locally with `cargo install --path . --locked` (add
`--features metal` for Apple Silicon), then invoke `laya-candle predict ...`.
The library and CLI are experimental; the 0.1 API may change.

The first Hub load downloads the selected checkpoint. Subsequent loads reuse the
Hugging Face cache. `--model /path/to/checkpoint` loads local weights; `--offline`
uses cached files without network access, and `--revision COMMIT` pins a Hub
revision. `--subfolder NAME` supports checkpoints bundled in a repository.

## Context

| Checkpoint | Default context | Maximum context |
| --- | ---: | ---: |
| `convaiinnovations/laya` | 512 | 8,192 |
| `convaiinnovations/laya-multilingual` | 1,024 | 8,192 |
| `convaiinnovations/laya-typed-decisions` | 1,024 | 8,192 |

For 8k inputs, add `--max-len 8192 --head-max-len 512 --batch-size 1`.
The context budget includes state, questions, options, and special tokens.
`--head-max-len` controls upstream prompt budgeting, not extra context.
Checkpoint defaults apply unless overridden; batch size defaults to 16.

State is truncated on the right to match upstream Laya. Add
`--reject-state-truncation` to fail if any question cannot fit the full state.
Upstream instruction and option shortening still applies. Supporting 8k context
does not establish model accuracy at that length.

## Rust API

```rust,no_run
use laya_candle::{Agent, LoadOptions, Questions};
use serde_json::json;

fn main() -> anyhow::Result<()> {
    let agent = Agent::load("convaiinnovations/laya", LoadOptions::default())?;
    let questions: Questions = serde_json::from_value(json!({
        "refund": {
            "type": "noul",
            "instructions": "Does the customer request a refund?"
        }
    }))?;
    let result = agent.predict(&json!("Refund the duplicate charge."), &questions)?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}
```

`choice` accepts a label list or a label-to-description object, `score` accepts an
ordered rubric array, and `noul` returns P(true). Question and label order is
preserved. Outputs retain upstream calibration, confidence, action probabilities,
rounding, and token usage. See `examples/questions.json` for all three types.

## Verification

```sh
cargo test --locked
cargo fmt --check
cargo clippy --locked --all-targets -- -D warnings
# On macOS with the Metal feature:
cargo clippy --locked --all-targets --features metal -- -D warnings
# On a machine with an accessible Metal device:
cargo test --locked --features metal -- --ignored --test-threads=1
```

Tests use a deterministic tiny checkpoint and PyTorch golden outputs without
requiring Python, downloaded models, or network access. They cover tokenization,
local/global attention, padding, batch invariance, calibration, truncation, error
handling, repeated inference, and CLI output. Metal tests additionally compare
fused kernels and GEMM against reference operations, including tail dimensions
and short sequences. See [tests/README.md](tests/README.md) for reference generation
and comparison against original pretrained checkpoints.

`inspect` accepts the same arguments as `predict` and exposes prepared tokens,
marker positions, raw logits, and the formatted result for numerical verification.

See [CONTRIBUTING.md](CONTRIBUTING.md) for development and packaging checks and
[SECURITY.md](SECURITY.md) for vulnerability reporting and checkpoint trust.

## License

Project code is Apache-2.0; the vendored MLX kernels are MIT. The crate declares
`Apache-2.0 AND MIT` because it distributes both. This is an independent port.
Laya is by Convai Innovations and its
contributors; the [MLX port](https://github.com/mizorewww/laya-mlx) informed this
implementation. Vendored MLX Metal kernels retain their MIT license. See
[LICENSE](LICENSE), [LICENSE-MIT](LICENSE-MIT), and [NOTICE](NOTICE) for attribution
and pinned source revisions. Pretrained model weights are downloaded separately
and retain their original licenses; only a synthetic test checkpoint is included.

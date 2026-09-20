# Contributing

Use Rust 1.95.0 from `rust-toolchain.toml`. The default test suite runs on CPU
without Python, network access, or pretrained model downloads:

```sh
cargo fmt --check
cargo test --locked
cargo clippy --locked --all-targets -- -D warnings
RUSTDOCFLAGS='-D warnings' cargo doc --locked --no-deps
```

For Metal changes, use an Apple Silicon machine with GPU access:

```sh
cargo clippy --locked --all-targets --features metal -- -D warnings
cargo test --locked --features metal -- --include-ignored --test-threads=1
```

Keep numerical tolerances, tokenization, calibration, and output semantics intact.
For changes to model math or kernels, compare the affected original checkpoints
and precisions with the upstream reference as described in [tests/README.md](tests/README.md).
Do not replace golden outputs with the implementation's own output.

Performance changes should include reproducible before/after measurements using
the same hardware, weights, dtype, batch size, context, and warmup. Measure completed
outputs, include tokenization and calibration, and report regressions as well as
wins. Keep run output, downloaded weights, environments, and build caches out of
the source tree; `CARGO_TARGET_DIR` can direct builds to a temporary directory.

For dependency changes, run these checks with cargo-audit 0.22.2 and cargo-deny 0.20.2:

```sh
cargo audit
cargo deny --locked --all-features check licenses sources
```

For packaging changes, run `cargo package --locked --allow-dirty` and inspect
`cargo package --list --allow-dirty`. The archive must contain both licenses,
NOTICE, the Metal sources, and test fixtures. Pretrained weights and credentials
must not be included. CUDA requires a separate CUDA-equipped validation machine;
do not describe a compile check as GPU validation.

Open a focused pull request explaining the problem, change, and validation. Include
only minimal synthetic inputs in public issues. Contributions to project code use
Apache-2.0; preserve MIT headers and provenance when changing vendored MLX code.

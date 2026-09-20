# Correctness tests

`cargo test --locked` runs offline against `fixtures/tiny` and
`fixtures/reference.json`. The small safetensors file is a deterministic synthetic
test checkpoint, not a downloaded pretrained model. The reference contains seven
cases covering all decision types, structured inputs, literal mask tokens, empty
and long states, and mixed option counts.

Reference provenance:

- Laya commit `42626c348753fbb17572a813127df2278a1ec527`.
- PyTorch 2.14.0, Transformers 5.0.0, CPU float32, seed 1234.
- Exact prepared tokens and marker positions; raw logit error below `2e-5`;
  formatted numeric output error at most `0.00011`.

`reference.py` preserves the upstream fixture generator and full-checkpoint
comparison checks. It is development tooling only. Install its pinned dependencies
in a separate environment and check out the pinned upstream revision:

```sh
python3 -m venv /tmp/laya-verification-env
/tmp/laya-verification-env/bin/pip install -r tests/requirements-reference.txt
git clone https://github.com/NandhaKishorM/laya /tmp/laya-reference
git -C /tmp/laya-reference checkout 42626c348753fbb17572a813127df2278a1ec527
```

To compare a local original pretrained checkpoint against the upstream reference:

```sh
cargo build --release --features metal
/tmp/laya-verification-env/bin/python tests/reference.py export --upstream /tmp/laya-reference --model /path/to/checkpoint --output /tmp/laya-reference.json
/tmp/laya-verification-env/bin/python tests/reference.py compare --model /path/to/checkpoint --reference /tmp/laya-reference.json --device metal --output /tmp/laya-comparison.json
```

Use `--dtype f16` for reduced-precision comparison. Float32/float16 raw-logit
checks use `atol=0.001/0.05` and `rtol=0.0001/0.01`; maximum probability error
must be below `0.001/0.01`, and score/confidence/action probability error below
`0.001/0.02`. Choice labels and boolean decisions must match, as must tokenization
and usage. These tolerances are unchanged from the original qualification.

`export --max-len 8192 --long-only` generates a full-context case. To regenerate
the synthetic checkpoint and golden data into a new directory:

```sh
/tmp/laya-verification-env/bin/python tests/reference.py tiny --upstream /tmp/laya-reference --model /tmp/laya-tiny-new --output /tmp/laya-tiny-reference.json
```

Generated comparison files and environments belong outside the source tree.

# SPDX-License-Identifier: Apache-2.0
"""Generate PyTorch golden fixtures or compare a real checkpoint to the Rust CLI.

Requires a checkout of NandhaKishorM/laya at UPSTREAM_REVISION and requirements-reference.txt.
No reference dependency is needed to run the Rust library or checked-in fixture tests.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

UPSTREAM_REVISION = "42626c348753fbb17572a813127df2278a1ec527"


def cases():
    q = {
        "department": {"type": "choice", "instructions": "Which department?", "criteria": {"billing": "refunds", "technical": "bugs", "sales": "purchases"}},
        "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "medium", "high"]},
        "refund": {"type": "noul", "instructions": "Is a refund requested?"},
    }
    return [
        ("basic", "Please refund the duplicate charge today.", q),
        ("structured", {"z": "中文 😀", "a": [False, 0, None]}, {
            "choice": {"type": "choice", "instructions": {"task": "choose 😀"}, "criteria": {"z": {"desc": "refund"}, "a": False, "b": 0}},
            "score": {"type": "score", "instructions": "Rate", "criteria": [{"level": "low"}, "high"]},
            "noul": {"type": "noul", "instructions": "Refund?", "criteria": {"false": False, "true": {"reason": "money back"}}},
        }),
        ("mask_literals", "[MASK] <mask> hello [MASK]", q),
        ("empty_state", "", q),
        ("long", "Please refund the duplicate charge today. " * 150, q),
        ("conversation", [{"role": "user", "content": "Please refund"}], q),
        ("mixed_options", "Please refund", {
            "one": {"type": "choice", "instructions": "Choose", "criteria": ["billing"]},
            "twenty": {"type": "choice", "instructions": "Which department?", "criteria": ["billing"] + [f"department_{i}" for i in range(19)]},
            **q,
        }),
    ]


def setup(upstream):
    rev = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if rev != UPSTREAM_REVISION:
        raise ValueError(f"Expected upstream {UPSTREAM_REVISION}, got {rev}")
    sys.path.insert(0, str(upstream.resolve()))
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(1234)


def tiny(path):
    import torch
    from transformers import ModernBertConfig, ModernBertModel, PreTrainedTokenizerFast
    from tokenizers import Tokenizer, models, pre_tokenizers
    from safetensors.torch import save_file
    from laya.common import DecisionModel
    path.mkdir(parents=True, exist_ok=False)
    config = ModernBertConfig(vocab_size=128, hidden_size=64, intermediate_size=96,
        num_hidden_layers=4, num_attention_heads=4, max_position_embeddings=8192,
        local_attention=8, global_attn_every_n_layers=3, pad_token_id=0,
        cls_token_id=1, sep_token_id=2, bos_token_id=1, eos_token_id=2,
        reference_compile=False, attn_implementation="sdpa",
        global_rope_theta=160000., local_rope_theta=10000.)
    encoder = ModernBertModel(config)
    model = DecisionModel(encoder, head_layers=2, n_act=2).eval()
    # Nontrivial norm biases and independent head weights come from the upstream module.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("bias"):
                p.uniform_(-0.02, 0.02)
    config.save_pretrained(path / "encoder")
    words = ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]", "choice", "score", "noul", "question", ":", "?", "Please", "refund", "the", "duplicate", "charge", "today", ".", "billing", "technical", "sales", "refunds", "bugs", "purchases", "low", "medium", "high", "level", "false", "true", "no", "yes", "statement", "holds", "does", "not", "hold", "Which", "department", "How", "urgent", "Is", "a", "requested", "0", "1", "2"]
    words.extend(f"department_{i}" for i in range(19))
    backend = Tokenizer(models.WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]", unk_token="[UNK]")
    tok.save_pretrained(path / "tokenizer")
    cfg = {"encoder": "tiny-modernbert", "head_layers": 2, "max_len": 768, "head_max_len": 128,
        "act_costs": {"escalate": 0.5}, "temperature": [1.6, 1.2, 1.9],
        "temperature_by_options": {"choice:3-5": 1.75, "score:3-5": 1.25}}
    (path / "rl_agent_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, path / "model.safetensors")


def export(model, upstream, output, max_len=None, long_only=False):
    import torch
    from laya import Agent
    from laya.common import build_sequence, collate_items, QTYPES
    reference = Agent(str(model), device="cpu")
    if max_len:
        reference.cfg["max_len"] = max_len
    entries = []
    workloads = cases()
    if long_only:
        workloads = [("full_context", "Please refund the duplicate charge today. " * 2000,
            {"refund": {"type": "noul", "instructions": "Is a refund requested?"}})]
    for name, state, questions in workloads:
        items = []
        for q in questions.values():
            q = reference._to_internal(q)
            ids, markers = build_sequence(reference.tok, state, q, reference.cfg["max_len"], reference.cfg["head_max_len"])
            items.append({"ids": ids, "markers": markers, "qtype": QTYPES[q["t"]]})
        batch = collate_items([items], reference.tok.pad_token_id)
        with torch.inference_mode():
            logits, action = reference.model(**{k: batch[k] for k in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")})
        entries.append({"name": name, "state": state, "questions": questions, "items": items,
            "raw": {"logits": [row[:len(item["markers"])].tolist() for row, item in zip(logits, items)], "action_logits": action.tolist()},
            "prediction": reference.predict(state, questions)})
        print(f"reference: {name}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    import transformers
    output.write_text(json.dumps({"upstream_revision": UPSTREAM_REVISION, "max_len": reference.cfg["max_len"], "torch": torch.__version__, "transformers": transformers.__version__, "cases": entries}, ensure_ascii=False, indent=2) + "\n")


def compare(binary, model, reference, device, dtype, output):
    import tempfile
    import numpy as np
    reports = []
    tolerance = 1e-3 if dtype == "f32" else 0.05
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        golden = json.loads(reference.read_text())
        for case in golden["cases"]:
            (tmp / "state.json").write_text(json.dumps(case["state"], ensure_ascii=False))
            (tmp / "questions.json").write_text(json.dumps(case["questions"], ensure_ascii=False))
            cmd = [str(binary.resolve()), "inspect", "--model", str(model.resolve()), "--device", device, "--dtype", dtype,
                "--state-file", str(tmp / "state.json"), "--questions", str(tmp / "questions.json")]
            if "max_len" in golden:
                cmd.extend(["--max-len", str(golden["max_len"])])
            actual = json.loads(subprocess.check_output(cmd, text=True))
            assert actual["items"] == case["items"], f"tokenization mismatch: {case['name']}"
            errors = {}
            for field in ("logits", "action_logits"):
                errors[field] = max(float(np.max(np.abs(np.array(a)-np.array(b)))) for a,b in zip(actual["raw"][field], case["raw"][field]))
                assert all(np.allclose(a,b,atol=tolerance,rtol=1e-4 if dtype == "f32" else 0.01)
                    for a,b in zip(actual["raw"][field], case["raw"][field])), (case["name"], field, errors[field])
            max_prob = 0.
            max_output = 0.
            for key, expected in case["prediction"]["answers"].items():
                answer = actual["prediction"]["answers"][key]
                if expected["type"] == "choice":
                    assert answer["choice"] == expected["choice"], (case["name"], key)
                elif expected["type"] == "noul":
                    assert (answer["noul"] >= 0.5) == (expected["noul"] >= 0.5), (case["name"], key)
                if "probabilities" in expected:
                    error = max(abs(answer["probabilities"][k]-v) for k,v in expected["probabilities"].items())
                else:
                    error = abs(answer["noul"]-expected["noul"])
                max_prob = max(max_prob, error)
                for field in ("score", "confidence"):
                    if field in expected:
                        max_output = max(max_output, abs(answer[field]-expected[field]))
                max_output = max(max_output, abs(answer["action"]["act_probability"]-expected["action"]["act_probability"]))
            assert max_prob < (0.001 if dtype == "f32" else 0.01), (case["name"], max_prob)
            assert max_output < (0.001 if dtype == "f32" else 0.02), (case["name"], max_output)
            assert actual["prediction"]["usage"] == case["prediction"]["usage"]
            reports.append({"case": case["name"], "questions": len(case["questions"]), "max_abs_error": errors, "max_probability_error": max_prob, "max_score_confidence_action_error": max_output})
            print(f"PASS {case['name']}: {errors}, probability={max_prob:.6g}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"device": device, "dtype": dtype, "model": str(model), "upstream_revision": UPSTREAM_REVISION, "cases": reports}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("tiny", "export"):
        p = sub.add_parser(command)
        p.add_argument("--upstream", type=Path, required=True)
        p.add_argument("--model", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--max-len", type=int)
        p.add_argument("--long-only", action="store_true")
    p = sub.add_parser("compare")
    p.add_argument("--binary", type=Path, default=Path("target/release/laya-candle"))
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="f32")
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "compare":
        compare(args.binary, args.model, args.reference, args.device, args.dtype, args.output)
    else:
        setup(args.upstream)
        if args.command == "tiny":
            tiny(args.model)
        export(args.model, args.upstream, args.output, args.max_len, args.long_only)


if __name__ == "__main__":
    main()

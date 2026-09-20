use candle_core::{DType, Device};
use laya_candle::{Agent, LoadOptions, PreparedQuestion, Questions};
use serde::Deserialize;
use serde_json::{Value, json};
use std::path::PathBuf;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny")
}
fn load(batch_size: usize) -> Agent {
    Agent::load(
        fixture().to_str().unwrap(),
        LoadOptions {
            batch_size,
            ..Default::default()
        },
    )
    .unwrap()
}
#[derive(Deserialize)]
struct Golden {
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    name: String,
    state: Value,
    questions: Questions,
    items: Vec<PreparedQuestion>,
    raw: Raw,
    prediction: Value,
}
#[derive(Deserialize)]
struct Raw {
    logits: Vec<Vec<f32>>,
    action_logits: Vec<Vec<f32>>,
}
fn cases() -> Vec<Case> {
    serde_json::from_str::<Golden>(include_str!("fixtures/reference.json"))
        .unwrap()
        .cases
}

fn close(a: &Value, b: &Value, path: &str) {
    match (a, b) {
        (Value::Number(a), Value::Number(b)) => assert!(
            (a.as_f64().unwrap() - b.as_f64().unwrap()).abs() <= 0.00011,
            "{path}: {a} != {b}"
        ),
        (Value::Object(a), Value::Object(b)) => {
            assert_eq!(
                a.keys().collect::<Vec<_>>(),
                b.keys().collect::<Vec<_>>(),
                "{path}"
            );
            for (k, v) in a {
                close(v, &b[k], &format!("{path}.{k}"));
            }
        }
        _ => assert_eq!(a, b, "{path}"),
    }
}

#[test]
fn pytorch_golden_parity() {
    let agent = load(16);
    for case in cases() {
        let prepared = agent.prepare(&case.state, &case.questions).unwrap();
        assert_eq!(prepared, case.items, "{} tokenization", case.name);
        let raw = agent.forward(&prepared).unwrap();
        for (actual, expected) in [
            (&raw.logits, &case.raw.logits),
            (&raw.action_logits, &case.raw.action_logits),
        ] {
            for (a, b) in actual.iter().flatten().zip(expected.iter().flatten()) {
                assert!((a - b).abs() < 2e-5, "{}: {a} != {b}", case.name);
            }
        }
        let result =
            serde_json::to_value(agent.predict(&case.state, &case.questions).unwrap()).unwrap();
        close(&result, &case.prediction, &case.name);
    }
}

#[test]
fn batch_invariance_and_repeated_inference() {
    let single = load(1);
    let batched = load(16);
    let case = cases().pop().unwrap();
    let a = serde_json::to_value(single.predict(&case.state, &case.questions).unwrap()).unwrap();
    let b = serde_json::to_value(batched.predict(&case.state, &case.questions).unwrap()).unwrap();
    close(&a, &b, "batch size");
    for _ in 0..3 {
        assert_eq!(
            b,
            serde_json::to_value(batched.predict(&case.state, &case.questions).unwrap()).unwrap()
        );
    }
}

#[test]
fn empty_single_option_and_invalid_inputs() {
    let agent = load(2);
    let empty = agent.predict(&json!(""), &Questions::new()).unwrap();
    assert!(empty.answers.is_empty());
    assert_eq!(empty.usage.input_tokens, 0);
    let q = serde_json::from_value(
        json!({"one":{"type":"choice","instructions":"Choose","criteria":["only"]}}),
    )
    .unwrap();
    let result = agent.predict(&json!(""), &q).unwrap();
    assert_eq!(result.answers["one"]["probabilities"]["only"], 1.);
    assert_eq!(result.answers["one"]["confidence"], 1.);
    assert!(
        agent
            .forward(&[PreparedQuestion {
                ids: vec![1],
                markers: vec![2],
                qtype: 0
            }])
            .is_err()
    );
    let too_many = serde_json::from_value(json!({"many":{"type":"choice","instructions":"Choose","criteria":(0..300).map(|i|format!("option {i}")).collect::<Vec<_>>()}})).unwrap();
    assert!(agent.prepare(&json!(""), &too_many).is_err());
}

#[test]
fn explicit_truncation_policy_and_config_errors() {
    let options = LoadOptions {
        reject_state_truncation: true,
        ..Default::default()
    };
    let agent = Agent::load(fixture().to_str().unwrap(), options).unwrap();
    let long = cases().into_iter().find(|c| c.name == "long").unwrap();
    assert!(
        agent
            .prepare(&long.state, &long.questions)
            .unwrap_err()
            .to_string()
            .contains("question")
    );
    let bad = LoadOptions {
        batch_size: 0,
        ..Default::default()
    };
    assert!(Agent::load(fixture().to_str().unwrap(), bad).is_err());
    let bad = LoadOptions {
        max_len: Some(9000),
        ..Default::default()
    };
    assert!(Agent::load(fixture().to_str().unwrap(), bad).is_err());
    let bad = LoadOptions {
        device: Device::Cpu,
        dtype: DType::F16,
        ..Default::default()
    };
    assert!(Agent::load(fixture().to_str().unwrap(), bad).is_err());
}

#[test]
fn cli_json_output() {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_laya-candle"))
        .args([
            "predict",
            "--model",
            fixture().to_str().unwrap(),
            "--state",
            "Please refund",
            "--questions",
        ])
        .arg(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/questions.json"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let result: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(result["usage"]["output_tokens"], 0);
    assert_eq!(result["answers"].as_object().unwrap().len(), 3);
}

fn copy_fixture(root: &std::path::Path) {
    for file in [
        "rl_agent_config.json",
        "encoder/config.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
        "model.safetensors",
    ] {
        let target = root.join(file);
        std::fs::create_dir_all(target.parent().unwrap()).unwrap();
        std::fs::copy(fixture().join(file), target).unwrap();
    }
}

#[test]
fn reject_incompatible_weights_and_rope() {
    let tmp = tempfile::tempdir().unwrap();
    copy_fixture(tmp.path());
    let weights = tmp.path().join("model.safetensors");
    let mut tensors = candle_core::safetensors::load(&weights, &Device::Cpu).unwrap();
    tensors.insert(
        "unexpected.weight".into(),
        candle_core::Tensor::zeros(1, DType::F32, &Device::Cpu).unwrap(),
    );
    candle_core::safetensors::save(&tensors, &weights).unwrap();
    let result = Agent::load(tmp.path().to_str().unwrap(), LoadOptions::default());
    assert!(
        result
            .err()
            .unwrap()
            .to_string()
            .contains("unexpected checkpoint tensors")
    );
    tensors.remove("unexpected.weight");
    tensors.remove("scorer.1.weight");
    candle_core::safetensors::save(&tensors, &weights).unwrap();
    assert!(Agent::load(tmp.path().to_str().unwrap(), LoadOptions::default()).is_err());
    copy_fixture(tmp.path());
    let path = tmp.path().join("encoder/config.json");
    let mut cfg: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    cfg["rope_parameters"]["full_attention"]["rope_type"] = json!("linear");
    std::fs::write(&path, serde_json::to_vec(&cfg).unwrap()).unwrap();
    assert!(
        Agent::load(tmp.path().to_str().unwrap(), LoadOptions::default())
            .err()
            .unwrap()
            .to_string()
            .contains("scaled RoPE")
    );
}

#[test]
fn offline_pinned_snapshot_without_ref_file() {
    let tmp = tempfile::tempdir().unwrap();
    let revision = "0123456789012345678901234567890123456789";
    let snapshot = tmp
        .path()
        .join("hub/models--test--laya/snapshots")
        .join(revision)
        .join("multilingual");
    copy_fixture(&snapshot);
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_laya-candle"))
        .env("HF_HOME", tmp.path())
        .args([
            "predict",
            "--model",
            "test/laya",
            "--revision",
            revision,
            "--subfolder",
            "multilingual",
            "--offline",
            "--state",
            "Please refund",
            "--questions",
        ])
        .arg(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/questions.json"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(serde_json::from_slice::<Value>(&output.stdout).unwrap()["answers"].is_object());
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "requires a Metal GPU"]
fn tiny_prepared_sequences_with_padded_marker_slots() {
    let cpu = load(2);
    let gpu = Agent::load(
        fixture().to_str().unwrap(),
        LoadOptions {
            device: Device::new_metal(0).unwrap(),
            ..Default::default()
        },
    )
    .unwrap();
    let items = vec![
        PreparedQuestion {
            ids: vec![1],
            markers: vec![0],
            qtype: 0,
        },
        PreparedQuestion {
            ids: vec![1, 2],
            markers: vec![0],
            qtype: 1,
        },
    ];
    let expected = cpu.forward(&items).unwrap();
    let actual = gpu.forward(&items).unwrap();
    for (a, b) in actual
        .logits
        .iter()
        .flatten()
        .chain(actual.action_logits.iter().flatten())
        .zip(
            expected
                .logits
                .iter()
                .flatten()
                .chain(expected.action_logits.iter().flatten()),
        )
    {
        assert!((a - b).abs() < 2e-5, "{a} != {b}");
    }
}

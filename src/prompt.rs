// SPDX-License-Identifier: Apache-2.0
// Adapted from Laya and reimplemented for Candle; see NOTICE for source attribution.
use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::path::Path;
use tokenizers::Tokenizer;

/// Question IDs and choice labels retain their JSON insertion order.
pub type Questions = Map<String, Value>;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
/// A single tokenized question, including state and option markers.
pub struct PreparedQuestion {
    /// Complete encoder input sequence, including special tokens.
    pub ids: Vec<u32>,
    /// Positions of option markers in `ids`, in option order.
    pub markers: Vec<u32>,
    /// Upstream type index: choice = 0, score = 1, noul = 2.
    pub qtype: u32,
}

#[derive(Debug, Clone)]
pub(crate) struct Question {
    pub kind: String,
    pub instructions: String,
    pub criteria: Value,
    pub labels: Vec<String>,
    pub options: Vec<String>,
    pub qtype: u32,
}

/// Python json.dumps(..., ensure_ascii=False) separators, with insertion order preserved.
pub fn serialize_state(v: &Value) -> String {
    if let Value::String(s) = v {
        return s.clone();
    }
    python_json(v, false)
}
fn python_json(v: &Value, ascii: bool) -> String {
    match v {
        Value::Array(v) => format!(
            "[{}]",
            v.iter()
                .map(|x| python_json(x, ascii))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Value::Object(v) => format!(
            "{{{}}}",
            v.iter()
                .map(|(k, v)| format!(
                    "{}: {}",
                    python_json(&Value::String(k.clone()), ascii),
                    python_json(v, ascii)
                ))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Value::String(_) if ascii => serde_json::to_string(v)
            .unwrap()
            .chars()
            .map(|c| {
                if (c as u32) < 127 {
                    c.to_string()
                } else {
                    c.encode_utf16(&mut [0; 2])
                        .iter()
                        .map(|u| format!("\\u{u:04x}"))
                        .collect::<String>()
                }
            })
            .collect(),
        Value::Number(n) if n.is_f64() => {
            let text = n.to_string();
            if let Some((mantissa, exponent)) = text.split_once('e') {
                let exp: i32 = exponent.parse().expect("JSON float exponent");
                // Python uses a sign and at least two digits in scientific exponents.
                format!("{mantissa}e{exp:+03}")
            } else {
                text
            }
        }
        _ => serde_json::to_string(v).unwrap(),
    }
}
fn described(v: &Value) -> bool {
    !v.is_null() && v.as_str() != Some("")
}

impl Question {
    pub fn parse(value: &Value) -> Result<Self> {
        let kind = value
            .get("type")
            .and_then(Value::as_str)
            .context("question requires a string type")?
            .to_owned();
        let ins = value
            .get("instructions")
            .context("question requires instructions")?;
        let instructions = ins
            .as_str()
            .map(str::to_owned)
            .unwrap_or_else(|| python_json(ins, true));
        let mut criteria = value.get("criteria").cloned().unwrap_or(Value::Null);
        let (qtype, labels, options): (u32, Vec<String>, Vec<String>) = match kind.as_str() {
            "choice" => {
                if let Some(list) = criteria.as_array() {
                    let mut map = Map::new();
                    for label in list {
                        let label = label.as_str().context("choice labels must be strings")?;
                        ensure!(!map.contains_key(label), "duplicate choice label: {label}");
                        map.insert(label.to_owned(), Value::Null);
                    }
                    criteria = Value::Object(map);
                }
                let map = criteria
                    .as_object()
                    .context("choice criteria must be a label list or object")?;
                let labels = map.keys().cloned().collect();
                let options = map
                    .iter()
                    .map(|(k, v)| {
                        if described(v) {
                            format!("{k}: {}", serialize_state(v))
                        } else {
                            k.clone()
                        }
                    })
                    .collect();
                (0, labels, options)
            }
            "score" => {
                let list = criteria
                    .as_array()
                    .context("score criteria must be an array")?;
                (
                    1,
                    (0..list.len()).map(|i| i.to_string()).collect(),
                    list.iter()
                        .enumerate()
                        .map(|(i, v)| format!("level {i}: {}", serialize_state(v)))
                        .collect(),
                )
            }
            "noul" => {
                ensure!(
                    criteria.is_null() || criteria.is_object(),
                    "noul criteria must be an object"
                );
                let options = [
                    ("false", "no, the statement does not hold"),
                    ("true", "yes, the statement holds"),
                ]
                .iter()
                .map(|(key, default)| {
                    let v = criteria.get(key).unwrap_or(&Value::Null);
                    format!(
                        "{key}: {}",
                        if described(v) {
                            serialize_state(v)
                        } else {
                            default.to_string()
                        }
                    )
                })
                .collect();
                (2, vec!["false".to_owned(), "true".to_owned()], options)
            }
            _ => bail!("unknown question type: {kind}"),
        };
        ensure!(
            !options.is_empty(),
            "question must contain at least one option"
        );
        Ok(Self {
            kind,
            instructions,
            criteria,
            labels,
            options,
            qtype,
        })
    }
}

pub(crate) struct PromptTokenizer {
    backend: Tokenizer,
    cls: u32,
    sep: u32,
    mask: u32,
    mask_text: String,
    pub pad: u32,
}
impl PromptTokenizer {
    pub fn load(dir: &Path) -> Result<Self> {
        let mut backend =
            Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow::anyhow!("{e}"))?;
        backend.with_padding(None);
        backend
            .with_truncation(None)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let config: Value =
            serde_json::from_slice(&std::fs::read(dir.join("tokenizer_config.json"))?)?;
        let token = |key: &str| -> Result<(String, u32)> {
            let v = &config[key];
            let text = v
                .as_str()
                .or_else(|| v.get("content").and_then(Value::as_str))
                .with_context(|| format!("missing {key}"))?;
            Ok((
                text.to_owned(),
                backend
                    .token_to_id(text)
                    .with_context(|| format!("invalid {key}"))?,
            ))
        };
        let (_, cls) = token("cls_token")?;
        let (_, sep) = token("sep_token")?;
        let (mask_text, mask) = token("mask_token")?;
        let (_, pad) = token("pad_token")?;
        Ok(Self {
            backend,
            cls,
            sep,
            mask,
            mask_text,
            pad,
        })
    }
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self
            .backend
            .encode(text.replace(&self.mask_text, " "), false)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .get_ids()
            .to_vec())
    }
    pub fn prepare(
        &self,
        state: &Value,
        q: &Question,
        max_len: usize,
        head_len: usize,
        reject_state_truncation: bool,
    ) -> Result<PreparedQuestion> {
        let mut head = self.encode(&format!("{} question: {}", q.kind, q.instructions))?;
        let mut opts = q
            .options
            .iter()
            .map(|s| -> Result<Vec<u32>> {
                let mut ids = vec![self.mask];
                ids.extend(self.encode(&format!(" {s}"))?.into_iter().take(48));
                Ok(ids)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut budget = head_len as isize - opts.iter().map(Vec::len).sum::<usize>() as isize;
        if budget < 16 {
            let per = ((head_len as isize - 16) / opts.len() as isize).max(4) as usize;
            for opt in &mut opts {
                opt.truncate(per);
            }
            budget = head_len as isize - opts.iter().map(Vec::len).sum::<usize>() as isize;
        }
        head.truncate(budget.max(8) as usize);
        let mut ids = vec![self.cls];
        ids.extend(head);
        ids.push(self.sep);
        let mut markers = Vec::new();
        for opt in opts {
            markers.push(ids.len() as u32);
            ids.extend(opt);
        }
        ids.push(self.sep);
        let room = max_len.saturating_sub(ids.len() + 1);
        let state_ids = self.encode(&serialize_state(state))?;
        ensure!(
            !reject_state_truncation || state_ids.len() <= room,
            "state would be truncated: {} tokens but only {room} fit after the question; shorten the state or increase max_len",
            state_ids.len()
        );
        ids.extend(state_ids.into_iter().take(room));
        ids.push(self.sep);
        ids.truncate(max_len);
        ensure!(
            markers.iter().all(|p| (*p as usize) < max_len),
            "options exceed token budget; increase head_max_len/max_len or use fewer options"
        );
        Ok(PreparedQuestion {
            ids,
            markers,
            qtype: q.qtype,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn formatting_and_order() {
        assert_eq!(
            serialize_state(&json!({"z": "中文", "a": [1, false]})),
            "{\"z\": \"中文\", \"a\": [1, false]}"
        );
        let q = Question::parse(
            &json!({"type":"choice", "instructions": "x", "criteria":{"z":false,"a":0,"b":null}}),
        )
        .unwrap();
        assert_eq!(q.options, ["z: false", "a: 0", "b"]);
        let q = Question::parse(&json!({"type":"noul", "instructions": {"emoji":"😀"}, "criteria":{"true":{"reason":"yes"}}})).unwrap();
        assert_eq!(q.instructions, "{\"emoji\": \"\\ud83d\\ude00\"}");
        assert_eq!(q.options[1], "true: {\"reason\": \"yes\"}");
    }
    #[test]
    fn reject_invalid_questions() {
        for q in [
            json!({"type":"choice","instructions":"x","criteria":[]}),
            json!({"type":"choice","instructions":"x","criteria":["a","a"]}),
            json!({"type":"score","instructions":"x","criteria":{}}),
            json!({"type":"other","instructions":"x"}),
        ] {
            assert!(Question::parse(&q).is_err());
        }
    }

    #[test]
    fn python_numeric_and_ascii_formatting() {
        assert_eq!(
            serialize_state(&json!([1e-7, 1e16, 0.0001, 1.0])),
            "[1e-07, 1e+16, 0.0001, 1.0]"
        );
        assert_eq!(python_json(&json!("\u{7f}"), true), "\"\\u007f\"");
    }
}

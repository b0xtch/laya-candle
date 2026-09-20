use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device, Shape, Tensor, safetensors::BufferedSafetensors};
use candle_nn::{Init, VarBuilder, var_builder::SimpleBackend};
use hf_hub::{Cache, Repo, RepoType, api::sync::ApiBuilder};
use std::{
    collections::HashSet,
    path::{Component, Path, PathBuf},
    sync::{Arc, Mutex},
};

pub(crate) struct Checkpoint {
    pub agent: PathBuf,
    pub encoder: PathBuf,
    pub tokenizer_dir: PathBuf,
    pub weights: PathBuf,
}
impl Checkpoint {
    pub fn resolve(
        source: &str,
        revision: &str,
        subfolder: Option<&str>,
        offline: bool,
    ) -> Result<Self> {
        let sub = Path::new(subfolder.unwrap_or(""));
        ensure!(
            sub.components().all(|c| matches!(c, Component::Normal(_))),
            "subfolder must be a relative path without '..'"
        );
        let path = Path::new(source);
        let local = path.is_dir();
        if !local {
            ensure!(
                !path.is_absolute() && !source.starts_with('.') && !path.exists(),
                "local model directory not found: {source}"
            );
        }
        let repo = Repo::with_revision(source.to_owned(), RepoType::Model, revision.to_owned());
        let api = if local || offline {
            None
        } else {
            let mut builder = ApiBuilder::from_env().with_progress(false);
            if let Ok(token) = std::env::var("HF_TOKEN") {
                builder = builder.with_token(Some(token));
            }
            Some(builder.build()?.repo(repo.clone()))
        };
        let root_cache = Cache::from_env();
        let snapshot = if revision.len() == 40 && revision.bytes().all(|b| b.is_ascii_hexdigit()) {
            Some(
                root_cache
                    .path()
                    .join(repo.folder_name())
                    .join("snapshots")
                    .join(revision),
            )
        } else {
            None
        };
        let cache = root_cache.repo(repo);
        let file = |name: &str| -> Result<PathBuf> {
            let relative = sub.join(name);
            let p = if local {
                path.join(&relative)
            } else if let Some(api) = &api {
                api.get(&relative.to_string_lossy())
                    .with_context(|| format!("downloading {source}/{}", relative.display()))?
            } else {
                cache
                    .get(&relative.to_string_lossy())
                    .or_else(|| {
                        // Python Hub caches need not create refs/<sha> for pinned snapshots.
                        snapshot
                            .as_ref()
                            .map(|p| p.join(&relative))
                            .filter(|p| p.is_file())
                    })
                    .with_context(|| {
                        format!(
                            "not cached in offline mode: {source}/{}",
                            relative.display()
                        )
                    })?
            };
            ensure!(p.is_file(), "missing checkpoint file: {}", p.display());
            Ok(p)
        };
        let agent = file("rl_agent_config.json")?;
        let encoder = file("encoder/config.json")?;
        let tokenizer = file("tokenizer/tokenizer.json")?;
        file("tokenizer/tokenizer_config.json")?;
        let weights = file("model.safetensors")?;
        Ok(Self {
            agent,
            encoder,
            tokenizer_dir: tokenizer.parent().unwrap().to_owned(),
            weights,
        })
    }
}

/// Records consumed names so unsupported or misspelled tensors cannot silently disappear.
struct CheckedWeights {
    inner: BufferedSafetensors,
    consumed: Arc<Mutex<HashSet<String>>>,
}
impl SimpleBackend for CheckedWeights {
    fn get(
        &self,
        shape: Shape,
        name: &str,
        hints: Init,
        dtype: DType,
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        let t = SimpleBackend::get(&self.inner, shape, name, hints, dtype, device)?;
        self.consumed.lock().unwrap().insert(name.to_owned());
        Ok(t)
    }
    fn get_unchecked(
        &self,
        name: &str,
        dtype: DType,
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        let t = SimpleBackend::get_unchecked(&self.inner, name, dtype, device)?;
        self.consumed.lock().unwrap().insert(name.to_owned());
        Ok(t)
    }
    fn contains_tensor(&self, name: &str) -> bool {
        SimpleBackend::contains_tensor(&self.inner, name)
    }
}

pub(crate) struct WeightAudit {
    names: HashSet<String>,
    consumed: Arc<Mutex<HashSet<String>>>,
}
impl WeightAudit {
    pub fn load(path: &Path, dtype: DType, device: &Device) -> Result<(VarBuilder<'static>, Self)> {
        // Buffered loading avoids memory-map safety assumptions about mutable local files.
        let inner = BufferedSafetensors::new(std::fs::read(path)?)?;
        let names = inner.tensors().into_iter().map(|(name, _)| name).collect();
        let consumed = Arc::new(Mutex::new(HashSet::new()));
        let vb = VarBuilder::from_backend(
            Box::new(CheckedWeights {
                inner,
                consumed: consumed.clone(),
            }),
            dtype,
            device.clone(),
        );
        // This saved buffer is not the calibration source; calibration is in the agent JSON.
        let _ = vb.get(3, "temperature")?;
        Ok((vb, Self { names, consumed }))
    }
    pub fn finish(&self) -> Result<()> {
        let consumed = self.consumed.lock().unwrap();
        let mut extra: Vec<_> = self.names.difference(&consumed).cloned().collect();
        extra.sort();
        ensure!(
            extra.is_empty(),
            "unexpected checkpoint tensors: {extra:?}; use original Laya weights, not MLX-converted weights"
        );
        Ok(())
    }
}

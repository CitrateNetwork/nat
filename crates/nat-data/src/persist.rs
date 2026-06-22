//! Corpus persistence (Data Ops §5): write the pipeline's shards + manifest to a
//! versioned on-disk corpus, and read them back. The layout is keyed by the
//! pipeline `config_hash` so a federated node can verify it trained on the corpus
//! the manifest commits to.
//!
//! Layout:
//! ```text
//!   <root>/<config_hash>/manifest.json
//!   <root>/<config_hash>/shard_0000.json ...
//! ```

use crate::manifest::{CorpusManifest, Shard};
use crate::PipelineOutput;
use std::io::{Error, ErrorKind, Result};
use std::path::{Path, PathBuf};

fn json_err(e: serde_json::Error) -> Error {
    Error::new(ErrorKind::InvalidData, e)
}

/// The directory a corpus with this config hash lives in, under `root`.
pub fn corpus_dir(root: &Path, config_hash: &str) -> PathBuf {
    root.join(config_hash)
}

/// Write a pipeline output to `<root>/<config_hash>/`. Returns the corpus dir.
pub fn write_corpus(root: &Path, output: &PipelineOutput) -> Result<PathBuf> {
    let dir = corpus_dir(root, &output.manifest.config_hash);
    std::fs::create_dir_all(&dir)?;
    let manifest = serde_json::to_vec_pretty(&output.manifest).map_err(json_err)?;
    std::fs::write(dir.join("manifest.json"), manifest)?;
    for shard in &output.shards {
        let bytes = serde_json::to_vec(shard).map_err(json_err)?;
        std::fs::write(dir.join(format!("shard_{:04}.json", shard.index)), bytes)?;
    }
    Ok(dir)
}

/// Read the manifest from a corpus directory.
pub fn read_manifest(dir: &Path) -> Result<CorpusManifest> {
    let bytes = std::fs::read(dir.join("manifest.json"))?;
    serde_json::from_slice(&bytes).map_err(json_err)
}

/// Read all shards from a corpus directory, in index order.
pub fn read_shards(dir: &Path) -> Result<Vec<Shard>> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with("shard_") && n.ends_with(".json"))
                .unwrap_or(false)
        })
        .collect();
    paths.sort();
    let mut shards = Vec::with_capacity(paths.len());
    for p in paths {
        let bytes = std::fs::read(&p)?;
        shards.push(serde_json::from_slice(&bytes).map_err(json_err)?);
    }
    Ok(shards)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{run_pipeline, PipelineConfig};

    #[test]
    fn corpus_round_trips_through_disk() {
        let out = run_pipeline(crate::seed::seed_corpus(), &PipelineConfig::default());
        let root = std::env::temp_dir().join("nat_corpus_test");
        let _ = std::fs::remove_dir_all(&root);
        let dir = write_corpus(&root, &out).unwrap();

        let manifest = read_manifest(&dir).unwrap();
        let shards = read_shards(&dir).unwrap();
        assert_eq!(manifest.manifest_hash(), out.manifest.manifest_hash());
        assert_eq!(shards, out.shards);
        let _ = std::fs::remove_dir_all(&root);
    }
}

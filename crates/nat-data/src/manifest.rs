//! SHARD + MANIFEST stages (Data Ops §4 steps 7–8).
//!
//! The manifest is what a federated node trusts: it verifies the manifest hash
//! before training, so it knows it has exactly the right data (Data Ops §5,
//! "grandma-proof"). The hash is deterministic — same shards → same hash.

use crate::{hex, Document};
use nat_types::{ZoneId, Q16};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// A shard: a fixed-size, deterministically-ordered batch of documents.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Shard {
    pub index: u32,
    pub docs: Vec<Document>,
}

/// Per-shard manifest entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardManifest {
    pub shard_index: u32,
    pub doc_count: u32,
    pub token_count: u64,
    /// Zone-tag counts, in canonical `ZoneId` order, nonzero entries only.
    pub zone_tag_distribution: Vec<(ZoneId, u32)>,
    /// Token-weighted mean quality of this shard. Q16 in [0,1].
    pub mean_quality: Q16,
    /// SHA-256 over the shard's document raw-hashes, in shard order. Lineage root.
    pub provenance_root: String,
}

impl ShardManifest {
    /// Compute the manifest for one shard.
    pub fn of(shard: &Shard) -> ShardManifest {
        let doc_count = shard.docs.len() as u32;
        let token_count: u64 = shard.docs.iter().map(|d| d.token_count).sum();

        // Zone-tag distribution in canonical order.
        let mut zone_tag_distribution = Vec::new();
        for z in ZoneId::ALL {
            let c = shard
                .docs
                .iter()
                .filter(|d| d.zone_tags.contains(&z))
                .count() as u32;
            if c > 0 {
                zone_tag_distribution.push((z, c));
            }
        }

        // Token-weighted mean quality.
        let mut weighted = Q16::ZERO;
        for d in &shard.docs {
            let toks = Q16::from_raw((d.token_count as i64) << 16);
            weighted = weighted.add(d.quality.mul(toks));
        }
        let mean_quality = if token_count == 0 {
            Q16::ZERO
        } else {
            weighted.div(Q16::from_raw((token_count as i64) << 16))
        };

        // Provenance root: hash of concatenated raw-hashes in shard order.
        let mut h = Sha256::new();
        for d in &shard.docs {
            h.update(d.provenance.raw_hash.as_bytes());
        }
        let provenance_root = hex(&h.finalize());

        ShardManifest {
            shard_index: shard.index,
            doc_count,
            token_count,
            zone_tag_distribution,
            mean_quality,
            provenance_root,
        }
    }
}

/// The corpus manifest: the trust anchor for a federated training run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorpusManifest {
    pub config_hash: String,
    pub shard_count: u32,
    pub total_docs: u64,
    pub total_tokens: u64,
    /// Token-weighted mean quality across the corpus. This is the `data_quality`
    /// that feeds `nat_train::StepContribution`. Q16 in [0,1].
    pub aggregate_quality: Q16,
    pub shards: Vec<ShardManifest>,
}

impl CorpusManifest {
    /// Deterministic hash of the manifest. A node compares this to the published
    /// value before it trusts its shards. Struct field order is stable and every
    /// collection is an ordered `Vec`, so the bytes are reproducible.
    pub fn manifest_hash(&self) -> String {
        let bytes = serde_json::to_vec(self).expect("manifest always serializes");
        hex(&Sha256::digest(&bytes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Document, Provenance};

    fn doc(id: &str, tokens: u64, quality: f32, tags: Vec<ZoneId>) -> Document {
        Document {
            id: id.into(),
            provenance: Provenance {
                source: "t".into(),
                license: "MIT".into(),
                fetch_date: "2026-06-18".into(),
                raw_hash: format!("hash-{id}"),
            },
            text: "x".into(),
            modality_refs: vec![],
            quality: Q16::from_f32(quality),
            zone_tags: tags,
            token_count: tokens,
        }
    }

    #[test]
    fn token_weighted_mean_quality() {
        // Two docs: 10 tokens @ 0.4, 30 tokens @ 0.8 → weighted mean 0.7.
        let shard = Shard {
            index: 0,
            docs: vec![
                doc("a", 10, 0.4, vec![ZoneId::PF]),
                doc("b", 30, 0.8, vec![ZoneId::PF, ZoneId::HP]),
            ],
        };
        let m = ShardManifest::of(&shard);
        assert!((m.mean_quality.to_f32() - 0.7).abs() < 1e-3);
    }

    #[test]
    fn manifest_hash_is_stable() {
        let shard = Shard {
            index: 0,
            docs: vec![doc("a", 5, 0.5, vec![ZoneId::PF])],
        };
        let m = CorpusManifest {
            config_hash: "cfg".into(),
            shard_count: 1,
            total_docs: 1,
            total_tokens: 5,
            aggregate_quality: Q16::from_f32(0.5),
            shards: vec![ShardManifest::of(&shard)],
        };
        assert_eq!(m.manifest_hash(), m.manifest_hash());
    }
}

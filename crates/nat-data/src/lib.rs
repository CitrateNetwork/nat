//! The NAT data pipeline (Data Ops §4).
//!
//! ```text
//!   INGEST → NORMALIZE → DEDUP → QUALITY_SCORE → ZONE_TAG → TOKENIZE → SHARD → MANIFEST
//! ```
//!
//! Two properties matter most and both are load-bearing for the rest of the system:
//!
//! 1. **The quality score is the economic signal.** The QUALITY_SCORE stage
//!    produces the per-document quality that aggregates into the manifest's
//!    `aggregate_quality` — the `data_quality` term in
//!    `nat_train::StepContribution`, hence in `reward_weight = compute × quality`
//!    (`docs/SETTLEMENT_SEAM.md`). Garbage data scores low and earns low reward.
//!
//! 2. **Determinism is federated trust.** Same raw input + same config →
//!    byte-identical shards and an identical manifest hash, regardless of input
//!    order. A federated node verifies the manifest hash before training (Data
//!    Ops §5, "grandma-proof"). The shard order is seeded by the config hash, not
//!    by input order or map iteration.
//!
//! L0 scope: the heuristics are real but small (rule-based quality + tagging,
//! whitespace tokenization). Model-based filters and a real tokenizer land at L1.
//! Raw is never mutated; dropped data is quarantined with a reason, not deleted.

pub mod jsonl;
pub mod manifest;
pub mod persist;
pub mod quality;
pub mod seed;
pub mod tokenizer;
pub mod zonetag;

use manifest::{CorpusManifest, Shard, ShardManifest};
use nat_types::{ZoneId, Q16};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Licenses permitted into the corpus (Data Ops §3.1: permissive only, reviewed).
/// A document whose license is not on this list is quarantined, not trained on.
pub const ALLOWED_LICENSES: &[&str] = &[
    "CC0-1.0",
    "CC-BY-4.0",
    "CC-BY-SA-4.0",
    "MIT",
    "Apache-2.0",
    "BSD-3-Clause",
    "public-domain",
];

/// A raw input document, as fetched. Its hash is recorded and it is never edited.
///
/// This is the **ingest contract** (the `RawDoc` JSONL format, HERMES-S1 WP-H2):
/// one JSON object per line, fields `id`, `source`, `license`, `fetch_date`,
/// `text`, and optional `modality_refs`. A collector (Hermes) emits this; the
/// pipeline CLI consumes it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawDoc {
    pub id: String,
    pub source: String,
    pub license: String,
    pub fetch_date: String,
    pub text: String,
    /// Optional references to non-text modality artifacts (defaults to empty).
    #[serde(default)]
    pub modality_refs: Vec<String>,
}

/// Provenance recorded at ingest; immutable (Data Ops §4.1).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Provenance {
    pub source: String,
    pub license: String,
    pub fetch_date: String,
    /// SHA-256 of the raw text. Lineage back to the untouched raw.
    pub raw_hash: String,
}

/// A document after it has passed every gate: normalized, scored, tagged, tokenized.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub provenance: Provenance,
    pub text: String,
    pub modality_refs: Vec<String>,
    pub quality: Q16,
    pub zone_tags: Vec<ZoneId>,
    pub token_count: u64,
}

/// Why a document was quarantined (Data Ops §4.1: quarantine over delete).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuarantineReason {
    UnreviewedLicense(String),
    TooShort,
    TooLong,
    LowQuality(Q16),
    ExactDuplicate,
    NearDuplicate,
    PiiDetected(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Quarantined {
    pub doc_id: String,
    pub reason: QuarantineReason,
}

/// Pipeline configuration. Its hash pins the run (reproducibility floor, Research
/// Method §8): same config + same raw → same shards.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PipelineConfig {
    pub seed: u64,
    pub shard_size: usize,
    /// Documents scoring below this are quarantined. Q16 in [0,1].
    pub min_quality: Q16,
    pub min_len: usize,
    pub max_len: usize,
    /// Jaccard similarity (over token shingles) above which a doc is a near-dup. Q16 in [0,1].
    pub near_dup_threshold: Q16,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        PipelineConfig {
            seed: 0xC1742E,
            shard_size: 3,
            min_quality: Q16::from_f32(0.35),
            min_len: 16,
            max_len: 100_000,
            near_dup_threshold: Q16::from_f32(0.8),
        }
    }
}

impl PipelineConfig {
    /// Deterministic hash of the config — pins the run; goes into the manifest.
    pub fn config_hash(&self) -> String {
        let canonical = format!(
            "seed={};shard_size={};min_quality={};min_len={};max_len={};near_dup={}",
            self.seed,
            self.shard_size,
            self.min_quality.raw(),
            self.min_len,
            self.max_len,
            self.near_dup_threshold.raw(),
        );
        hex(&Sha256::digest(canonical.as_bytes()))
    }
}

/// What the pipeline produces: the shards a node trains on, the manifest a node
/// verifies, and the quarantine a reviewer audits.
#[derive(Debug, Clone)]
pub struct PipelineOutput {
    pub shards: Vec<Shard>,
    pub manifest: CorpusManifest,
    pub quarantine: Vec<Quarantined>,
}

/// Run the full pipeline. Deterministic in (raw set, config) — and independent of
/// the *order* of `raw`, because sharding is sorted by a config-seeded key.
pub fn run_pipeline(raw: Vec<RawDoc>, cfg: &PipelineConfig) -> PipelineOutput {
    let mut quarantine: Vec<Quarantined> = Vec::new();
    let mut kept: Vec<Document> = Vec::new();
    // Shingle sets of kept docs, for near-dup detection.
    let mut kept_shingles: Vec<std::collections::BTreeSet<u64>> = Vec::new();
    let mut seen_exact: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();

    // Process in a deterministic order (by id) so dedup "keep first" is stable
    // regardless of how `raw` was ordered.
    let mut raw_sorted = raw;
    raw_sorted.sort_by(|a, b| a.id.cmp(&b.id));

    for rd in raw_sorted {
        // INGEST — record provenance, screen license.
        let raw_hash = hex(&Sha256::digest(rd.text.as_bytes()));
        if !ALLOWED_LICENSES.contains(&rd.license.as_str()) {
            quarantine.push(Quarantined {
                doc_id: rd.id,
                reason: QuarantineReason::UnreviewedLicense(rd.license),
            });
            continue;
        }
        let provenance = Provenance {
            source: rd.source.clone(),
            license: rd.license.clone(),
            fetch_date: rd.fetch_date.clone(),
            raw_hash,
        };

        // NORMALIZE — collapse whitespace, trim. Raw is untouched; this is a new artifact.
        let text = normalize(&rd.text);

        // Length gate.
        let len = text.chars().count();
        if len < cfg.min_len {
            quarantine.push(Quarantined {
                doc_id: rd.id,
                reason: QuarantineReason::TooShort,
            });
            continue;
        }
        if len > cfg.max_len {
            quarantine.push(Quarantined {
                doc_id: rd.id,
                reason: QuarantineReason::TooLong,
            });
            continue;
        }

        // DEDUP — exact, then near-dup (Jaccard over token shingles).
        let exact_key = hex(&Sha256::digest(text.as_bytes()));
        if seen_exact.contains(&exact_key) {
            quarantine.push(Quarantined {
                doc_id: rd.id,
                reason: QuarantineReason::ExactDuplicate,
            });
            continue;
        }
        let shingles = shingle_set(&text);
        if kept_shingles
            .iter()
            .any(|prev| jaccard(prev, &shingles) >= cfg.near_dup_threshold)
        {
            quarantine.push(Quarantined {
                doc_id: rd.id,
                reason: QuarantineReason::NearDuplicate,
            });
            continue;
        }

        // QUALITY_SCORE — heuristic score + PII screen (a gate, not a warning).
        if let Some(hit) = quality::pii_hit(&text) {
            quarantine.push(Quarantined {
                doc_id: rd.id,
                reason: QuarantineReason::PiiDetected(hit),
            });
            continue;
        }
        let q = quality::score(&text);
        if q < cfg.min_quality {
            quarantine.push(Quarantined {
                doc_id: rd.id,
                reason: QuarantineReason::LowQuality(q),
            });
            continue;
        }

        // ZONE_TAG + TOKENIZE.
        let zone_tags = zonetag::tags(&text);
        let token_count = tokenize_count(&text);

        seen_exact.insert(exact_key);
        kept_shingles.push(shingles);
        kept.push(Document {
            id: rd.id,
            provenance,
            text,
            modality_refs: rd.modality_refs,
            quality: q,
            zone_tags,
            token_count,
        });
    }

    // SHARD — deterministic order seeded by the config, independent of input order.
    kept.sort_by_key(|d| shard_key(cfg.seed, &d.id));
    let shards: Vec<Shard> = kept
        .chunks(cfg.shard_size.max(1))
        .enumerate()
        .map(|(i, docs)| Shard {
            index: i as u32,
            docs: docs.to_vec(),
        })
        .collect();

    // MANIFEST.
    let manifest = build_manifest(cfg, &shards);
    PipelineOutput {
        shards,
        manifest,
        quarantine,
    }
}

fn build_manifest(cfg: &PipelineConfig, shards: &[Shard]) -> CorpusManifest {
    let shard_manifests: Vec<ShardManifest> = shards.iter().map(ShardManifest::of).collect();
    let total_docs: u64 = shard_manifests.iter().map(|m| m.doc_count as u64).sum();
    let total_tokens: u64 = shard_manifests.iter().map(|m| m.token_count).sum();

    // Aggregate quality = token-weighted mean over all kept docs. Q16, deterministic.
    let mut weighted = Q16::ZERO;
    for s in shards {
        for d in &s.docs {
            let toks = Q16::from_raw((d.token_count as i64) << 16);
            weighted = weighted.add(d.quality.mul(toks));
        }
    }
    let aggregate_quality = if total_tokens == 0 {
        Q16::ZERO
    } else {
        weighted.div(Q16::from_raw((total_tokens as i64) << 16))
    };

    CorpusManifest {
        config_hash: cfg.config_hash(),
        shard_count: shards.len() as u32,
        total_docs,
        total_tokens,
        aggregate_quality,
        shards: shard_manifests,
    }
}

// ---- normalization, tokenization, dedup helpers ----------------------------

/// Collapse runs of whitespace to single spaces and trim. Deterministic.
pub fn normalize(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Toy whitespace tokenizer (real tokenizer at L1). Token count only.
pub fn tokenize_count(s: &str) -> u64 {
    s.split_whitespace().count() as u64
}

/// 3-gram (word-shingle) set, hashed to u64, for near-dup Jaccard. A BTreeSet so
/// iteration is ordered and the comparison is deterministic.
fn shingle_set(s: &str) -> std::collections::BTreeSet<u64> {
    let words: Vec<&str> = s.split_whitespace().collect();
    let mut set = std::collections::BTreeSet::new();
    if words.len() < 3 {
        // Short docs: shingle on individual words so they still dedup sanely.
        for w in &words {
            set.insert(fnv1a(w.as_bytes()));
        }
        return set;
    }
    for w in words.windows(3) {
        set.insert(fnv1a(w.join(" ").as_bytes()));
    }
    set
}

fn jaccard(a: &std::collections::BTreeSet<u64>, b: &std::collections::BTreeSet<u64>) -> Q16 {
    if a.is_empty() && b.is_empty() {
        return Q16::ONE;
    }
    let inter = a.intersection(b).count();
    let union = a.union(b).count();
    if union == 0 {
        return Q16::ZERO;
    }
    Q16::from_raw((inter as i64) << 16).div(Q16::from_raw((union as i64) << 16))
}

/// Stable per-doc sort key for sharding: hash(seed || id). Order-independent.
fn shard_key(seed: u64, id: &str) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(seed.to_le_bytes());
    h.update(id.as_bytes());
    h.finalize().into()
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for b in bytes {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Lowercase hex of bytes.
pub fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(char::from_digit((b >> 4) as u32, 16).unwrap());
        s.push(char::from_digit((b & 0xf) as u32, 16).unwrap());
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw(id: &str, text: &str) -> RawDoc {
        RawDoc {
            id: id.into(),
            source: "test".into(),
            license: "MIT".into(),
            fetch_date: "2026-06-18".into(),
            text: text.into(),
            modality_refs: vec![],
        }
    }

    #[test]
    fn pipeline_is_order_independent() {
        let cfg = PipelineConfig::default();
        let docs = vec![
            raw(
                "a",
                "she walked along the quiet shore at dawn thinking of home",
            ),
            raw(
                "b",
                "fn main() { let x = vec![1, 2, 3]; println!(\"{}\", x.len()); }",
            ),
            raw(
                "c",
                "compute the sum 12 + 7 * 3 and explain each arithmetic step clearly",
            ),
            raw(
                "d",
                "a memoir of long afternoons and the smell of rain on warm stone",
            ),
        ];
        let mut reversed = docs.clone();
        reversed.reverse();

        let out1 = run_pipeline(docs, &cfg);
        let out2 = run_pipeline(reversed, &cfg);
        // Same content, different input order → identical manifest hash.
        assert_eq!(out1.manifest.manifest_hash(), out2.manifest.manifest_hash());
    }

    #[test]
    fn exact_and_near_duplicates_are_quarantined() {
        let cfg = PipelineConfig::default();
        let base = "the cartographer folded the map twice and set it down by the lantern";
        let docs = vec![
            raw("orig", base),
            raw("exact", base),
            raw("near", &format!("{base} softly")), // high shingle overlap
            raw(
                "diff",
                "an entirely unrelated sentence about distant orbital mechanics today",
            ),
        ];
        let out = run_pipeline(docs, &cfg);
        let reasons: Vec<&QuarantineReason> = out.quarantine.iter().map(|q| &q.reason).collect();
        assert!(reasons.contains(&&QuarantineReason::ExactDuplicate));
        assert!(reasons.contains(&&QuarantineReason::NearDuplicate));
    }

    #[test]
    fn unreviewed_license_is_quarantined_not_trained() {
        let cfg = PipelineConfig::default();
        let mut bad = raw(
            "x",
            "a perfectly fine sentence that is long enough to pass the length gate",
        );
        bad.license = "GPL-3.0".into(); // not on the permissive allow-list
        let out = run_pipeline(vec![bad], &cfg);
        assert_eq!(out.shards.len(), 0);
        assert!(matches!(
            out.quarantine[0].reason,
            QuarantineReason::UnreviewedLicense(_)
        ));
    }

    #[test]
    fn aggregate_quality_is_in_unit_interval() {
        let cfg = PipelineConfig::default();
        let out = run_pipeline(
            vec![raw(
                "a",
                "a clear and reasonably diverse english sentence about cartography and rivers",
            )],
            &cfg,
        );
        let q = out.manifest.aggregate_quality.to_f32();
        assert!(
            (0.0..=1.0).contains(&q),
            "aggregate quality {q} out of range"
        );
    }
}

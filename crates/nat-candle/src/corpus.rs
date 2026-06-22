//! WP-D3 — the corpus data loader: real `nat-data` shards → next-byte training
//! windows (NAT-S2 / DATA-S1).
//!
//! Replaces `train_loop::synthetic_task` with a real-text objective. For each
//! document in the shards, the byte stream is encoded (`nat_data::tokenizer`) and
//! slid over with a fixed context window: the model reads `seq_len` bytes and
//! predicts the next byte. That is a genuine language-modeling signal over the
//! vocab of 256 bytes, and it slots into the existing single-output
//! `NatTrainModel` (out_dim = vocab) with no architecture change — full
//! per-position autoregression is a later architecture WP.

use candle_core::{Device, Result, Tensor};
use nat_data::manifest::Shard;
use nat_data::tokenizer;

/// Build next-byte training windows from shards: returns `(ids, targets)` where
/// `ids` is `(n_windows, seq_len)` u32 byte contexts and `targets` is
/// `(n_windows,)` u32 next bytes. Caps at `max_windows` (deterministic order: by
/// shard, then document, then position).
pub fn next_byte_windows(
    shards: &[Shard],
    seq_len: usize,
    max_windows: usize,
    dev: &Device,
) -> Result<(Tensor, Tensor)> {
    let mut ids: Vec<u32> = Vec::new();
    let mut targets: Vec<u32> = Vec::new();
    let mut n = 0usize;
    'outer: for shard in shards {
        for doc in &shard.docs {
            let toks = tokenizer::encode(&doc.text);
            if toks.len() <= seq_len {
                continue;
            }
            for i in 0..(toks.len() - seq_len) {
                ids.extend_from_slice(&toks[i..i + seq_len]);
                targets.push(toks[i + seq_len]);
                n += 1;
                if n >= max_windows {
                    break 'outer;
                }
            }
        }
    }
    if n == 0 {
        candle_core::bail!("no training windows produced (docs shorter than seq_len {seq_len})");
    }
    let ids = Tensor::from_vec(ids, (n, seq_len), dev)?;
    let targets = Tensor::from_vec(targets, (n,), dev)?;
    Ok((ids, targets))
}

/// Convenience: run the seed corpus through the pipeline and build windows from
/// it — the self-contained real-text path (seed → pipeline → shards → windows).
pub fn seed_windows(seq_len: usize, max_windows: usize, dev: &Device) -> Result<(Tensor, Tensor)> {
    let out = nat_data::run_pipeline(
        nat_data::seed::seed_corpus(),
        &nat_data::PipelineConfig::default(),
    );
    next_byte_windows(&out.shards, seq_len, max_windows, dev)
}

/// Build **sequence** windows for the autoregressive objective (WP-D7): each row is
/// a contiguous `seq_len`-byte sequence (non-overlapping chunks per document). The
/// model predicts every next byte within the sequence, so one sequence yields
/// `seq_len - 1` predictions — far more sample-efficient than the single-output
/// next-byte windows. Returns `ids` of shape `(n, seq_len)`.
pub fn sequence_windows(
    shards: &[Shard],
    seq_len: usize,
    max_seqs: usize,
    dev: &Device,
) -> Result<Tensor> {
    let mut ids: Vec<u32> = Vec::new();
    let mut n = 0usize;
    'outer: for shard in shards {
        for doc in &shard.docs {
            let toks = tokenizer::encode(&doc.text);
            for chunk in toks.chunks(seq_len) {
                if chunk.len() < seq_len {
                    continue; // drop the short tail
                }
                ids.extend_from_slice(chunk);
                n += 1;
                if n >= max_seqs {
                    break 'outer;
                }
            }
        }
    }
    if n == 0 {
        candle_core::bail!("no sequences produced (docs shorter than seq_len {seq_len})");
    }
    Tensor::from_vec(ids, (n, seq_len), dev)
}

/// Sequence windows from a persisted corpus directory (autoregressive path).
pub fn sequences_from_dir(
    dir: &std::path::Path,
    seq_len: usize,
    max_seqs: usize,
    dev: &Device,
) -> Result<Tensor> {
    let shards = nat_data::persist::read_shards(dir).map_err(candle_core::Error::wrap)?;
    sequence_windows(&shards, seq_len, max_seqs, dev)
}

/// Build windows from a persisted corpus directory (as written by the `nat-corpus`
/// CLI): `<root>/<config_hash>/`. This is how training consumes a corpus Hermes
/// produced — `nat-corpus run … --out <root>` then point here at the config dir.
pub fn windows_from_dir(
    dir: &std::path::Path,
    seq_len: usize,
    max_windows: usize,
    dev: &Device,
) -> Result<(Tensor, Tensor)> {
    let shards = nat_data::persist::read_shards(dir).map_err(candle_core::Error::wrap)?;
    next_byte_windows(&shards, seq_len, max_windows, dev)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_are_well_formed_from_the_seed_corpus() {
        let dev = Device::Cpu;
        let (ids, targets) = seed_windows(24, 500, &dev).unwrap();
        let (n, sl) = ids.dims2().unwrap();
        assert_eq!(sl, 24);
        assert_eq!(targets.dims1().unwrap(), n);
        assert!(n > 0 && n <= 500);
        // All ids are valid byte tokens.
        let flat = ids.flatten_all().unwrap().to_vec1::<u32>().unwrap();
        assert!(flat.iter().all(|&t| (t as usize) < tokenizer::BYTE_VOCAB));
    }

    #[test]
    fn windows_load_from_a_persisted_corpus_dir() {
        // The CLI path: pipeline → persist → load windows from disk.
        let dev = Device::Cpu;
        let out = nat_data::run_pipeline(
            nat_data::seed::seed_corpus(),
            &nat_data::PipelineConfig::default(),
        );
        let root = std::env::temp_dir().join("nat_corpus_loader_test");
        let _ = std::fs::remove_dir_all(&root);
        let dir = nat_data::persist::write_corpus(&root, &out).unwrap();
        let (ids, _t) = windows_from_dir(&dir, 24, 300, &dev).unwrap();
        assert!(ids.dims2().unwrap().0 > 0);
        let _ = std::fs::remove_dir_all(&root);
    }
}

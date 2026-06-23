//! WP-D7 — the per-position autoregressive NAT language model (DATA-S1, toward L2).
//!
//! The earlier `NatTrainModel` predicts a *single* next byte from a fixed context
//! (it slices the input across zones and pools to one output). That is a
//! sequence-classification shape and wastes compute — one prediction per window.
//! This is the real LM shape: each zone is a **causal** sequence operator over the
//! *full* sequence, the zones are merged **per position**, and a readout produces
//! logits at **every** position. One sequence of length `L` yields `L-1`
//! next-token predictions, so the corpus is used far more efficiently.
//!
//! The zone structure is preserved: attention zones (HP/PF/CX) are causal
//! self-attention; SSM zones (SM/CB) are the causal decay-matrix recurrence (already
//! lower-triangular, hence causal by construction). The merge is the same
//! soft (activation×score) combine as WP-2, applied independently at each position.

use crate::seed::{name_seed, seeded_linear, seeded_scalar_var, seeded_uniform, SplitMix64};
use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::optim::{AdamW, ParamsAdamW};
use candle_nn::{loss, Linear, Module, Optimizer, VarBuilder, VarMap};
use nat_types::{CoreType, ZoneId};

/// A causal sequence operator: `(b, seq, d) -> (b, seq, d)`, attending only to the
/// present and past.
trait CausalCore {
    fn forward(&self, x: &Tensor) -> Result<Tensor>;
}

/// Causal single-head self-attention (HP/PF/CX).
struct CausalAttn {
    wq: Linear,
    wk: Linear,
    wv: Linear,
    wo: Linear,
    mask: Tensor, // (seq, seq): 0 on/below diagonal, large-negative above
    d: usize,
}

impl CausalCore for CausalAttn {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let q = self.wq.forward(x)?;
        let k = self.wk.forward(x)?;
        let v = self.wv.forward(x)?;
        let scale = 1.0 / (self.d as f64).sqrt();
        let scores = q.matmul(&k.transpose(1, 2)?)?.affine(scale, 0.0)?; // (b, seq, seq)
        let scores = scores.broadcast_add(&self.mask)?; // causal mask
        let attn = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let ctx = attn.matmul(&v)?; // (b, seq, d)
        self.wo.forward(&ctx)
    }
}

/// Causal state-space core (SM/CB): `y = D·(x·Wb)` with a lower-triangular decay
/// matrix — causal by construction, no mask needed.
struct CausalSsm {
    wb: Linear,
    wc: Linear,
    wo: Linear,
    log_a: Tensor,
    tk: Tensor,   // (seq, seq): (t-k) on/below the diagonal, 0 above
    mask: Tensor, // (seq, seq): lower-triangular ones
}

impl CausalCore for CausalSsm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, seq, _d) = x.dims3()?;
        let proj = self.wb.forward(x)?; // (b, seq, d)
        let decay_rate = self
            .log_a
            .exp()?
            .affine(1.0, 1.0)?
            .log()?
            .affine(-1.0, 0.0)?;
        let decay = self.tk.broadcast_mul(&decay_rate)?.exp()?.mul(&self.mask)?; // (seq, seq)
        let decay = decay.unsqueeze(0)?.broadcast_as((b, seq, seq))?;
        let y = decay.matmul(&proj)?; // (b, seq, d)
        self.wo.forward(&self.wc.forward(&y)?)
    }
}

/// Configuration for the autoregressive NAT LM.
#[derive(Debug, Clone)]
pub struct AutoregConfig {
    pub zones: Vec<ZoneId>,
    pub vocab: usize,
    pub seq_len: usize,
    /// Model width (embedding + per-zone hidden).
    pub d: usize,
    pub tau: f64,
    pub seed: u64,
}

impl AutoregConfig {
    /// A 3-zone byte-level autoregressive LM (vocab 256).
    pub fn byte_3zone() -> Self {
        AutoregConfig {
            zones: vec![ZoneId::HP, ZoneId::PF, ZoneId::CX],
            vocab: nat_data::tokenizer::BYTE_VOCAB,
            seq_len: 64,
            d: 48,
            tau: 1.0,
            seed: 2026,
        }
    }
}

/// The per-position autoregressive NAT model.
pub struct AutoregLm {
    varmap: VarMap,
    emb: Tensor, // (vocab, d)
    cores: Vec<Box<dyn CausalCore>>,
    score_heads: Vec<Linear>, // d -> 1, per zone (per-position score)
    readout: Linear,          // d -> vocab
    cfg: AutoregConfig,
    device: Device,
}

impl AutoregLm {
    pub fn new(cfg: &AutoregConfig) -> Result<Self> {
        let dev = crate::device::device();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let d = cfg.d;
        let seq = cfg.seq_len;

        // Embedding table.
        let table = seeded_uniform((cfg.vocab, d), 0.1, name_seed(cfg.seed, "embedding"), &dev)?;
        let var = candle_core::Var::from_tensor(&table)?;
        let emb = var.as_tensor().clone();
        varmap
            .data()
            .lock()
            .unwrap()
            .insert("embedding.weight".to_string(), var);

        // Constant causal masks (built once, shared by the cores).
        let attn_mask = causal_attn_mask(seq, &dev)?;
        let (tk, tri) = ssm_matrices(seq, &dev)?;

        let mut cores: Vec<Box<dyn CausalCore>> = Vec::with_capacity(cfg.zones.len());
        let mut score_heads = Vec::with_capacity(cfg.zones.len());
        for &z in &cfg.zones {
            let p = format!("zone_{}", z.as_str());
            let core: Box<dyn CausalCore> = match z.default_core() {
                CoreType::Attention => Box::new(CausalAttn {
                    wq: seeded_linear(&varmap, &vb, &format!("{p}.wq"), d, d, cfg.seed, &dev)?,
                    wk: seeded_linear(&varmap, &vb, &format!("{p}.wk"), d, d, cfg.seed, &dev)?,
                    wv: seeded_linear(&varmap, &vb, &format!("{p}.wv"), d, d, cfg.seed, &dev)?,
                    wo: seeded_linear(&varmap, &vb, &format!("{p}.wo"), d, d, cfg.seed, &dev)?,
                    mask: attn_mask.clone(),
                    d,
                }),
                CoreType::Ssm => Box::new(CausalSsm {
                    wb: seeded_linear(&varmap, &vb, &format!("{p}.wb"), d, d, cfg.seed, &dev)?,
                    wc: seeded_linear(&varmap, &vb, &format!("{p}.wc"), d, d, cfg.seed, &dev)?,
                    wo: seeded_linear(&varmap, &vb, &format!("{p}.wo"), d, d, cfg.seed, &dev)?,
                    log_a: seeded_scalar_var(&varmap, &format!("{p}.log_a"), 0.0, &dev)?,
                    tk: tk.clone(),
                    mask: tri.clone(),
                }),
                CoreType::None => candle_core::bail!("zone {z:?} has no learned core"),
            };
            cores.push(core);
            score_heads.push(seeded_linear(
                &varmap,
                &vb,
                &format!("score_{}", z.as_str()),
                d,
                1,
                cfg.seed,
                &dev,
            )?);
        }
        let readout = seeded_linear(&varmap, &vb, "readout", d, cfg.vocab, cfg.seed, &dev)?;

        Ok(AutoregLm {
            varmap,
            emb,
            cores,
            score_heads,
            readout,
            cfg: cfg.clone(),
            device: dev,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn backend(&self) -> &'static str {
        crate::device::backend_label()
    }

    pub fn param_count(&self) -> usize {
        self.varmap
            .all_vars()
            .iter()
            .map(|v| v.as_tensor().elem_count())
            .sum()
    }

    /// Export the model to a GGUF file (lossless F32) with NAT metadata
    /// (g3-gguf / WP-1.4). The container round-trips through candle's GGUF reader.
    pub fn export_gguf(&self, path: &std::path::Path) -> Result<()> {
        let tensors: Vec<(String, Tensor)> = self
            .varmap
            .data()
            .lock()
            .unwrap()
            .iter()
            .map(|(n, v)| (n.clone(), v.as_tensor().clone()))
            .collect();
        let zones = self
            .cfg
            .zones
            .iter()
            .map(|z| z.as_str())
            .collect::<Vec<_>>()
            .join(",");
        let md = [
            ("general.architecture", crate::gguf::s("nat-autoreg")),
            ("general.name", crate::gguf::s("nat")),
            ("nat.vocab", crate::gguf::u(self.cfg.vocab)),
            ("nat.embedding_length", crate::gguf::u(self.cfg.d)),
            ("nat.context_length", crate::gguf::u(self.cfg.seq_len)),
            ("nat.zone_count", crate::gguf::u(self.cfg.zones.len())),
            ("nat.zones", crate::gguf::s(&zones)),
        ];
        crate::gguf::export(&tensors, &md, path)
    }

    /// Logits at every position: ids `(b, seq)` → `(b, seq, vocab)`.
    pub fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        let (b, seq) = ids.dims2()?;
        let h = self
            .emb
            .index_select(&ids.flatten_all()?, 0)?
            .reshape((b, seq, self.cfg.d))?;

        // Each zone runs causally over the full sequence; score per position.
        let mut zone_outs = Vec::with_capacity(self.cores.len());
        let mut scores = Vec::with_capacity(self.cores.len());
        for (i, core) in self.cores.iter().enumerate() {
            let zo = core.forward(&h)?; // (b, seq, d)
            scores.push(self.score_heads[i].forward(&zo)?); // (b, seq, 1)
            zone_outs.push(zo);
        }
        let nz = self.cores.len();
        // Per-position soft merge over zones.
        let score_refs: Vec<&Tensor> = scores.iter().collect();
        let scores = Tensor::cat(&score_refs, 2)?; // (b, seq, nz)
        let weights = candle_nn::ops::softmax(&scores.affine(1.0 / self.cfg.tau, 0.0)?, D::Minus1)?;
        let zo_refs: Vec<&Tensor> = zone_outs.iter().collect();
        let stacked = Tensor::stack(&zo_refs, 2)?; // (b, seq, nz, d)
        let w = weights.reshape((b, seq, nz, 1))?;
        let composed = stacked.broadcast_mul(&w)?.sum(2)?; // (b, seq, d)
        self.readout.forward(&composed) // (b, seq, vocab)
    }

    /// Next-token cross-entropy averaged over all positions of `ids` `(b, seq)`.
    pub fn loss_on(&self, ids: &Tensor) -> Result<f32> {
        self.loss_tensor(ids)?.to_scalar::<f32>()
    }

    fn loss_tensor(&self, ids: &Tensor) -> Result<Tensor> {
        let (b, seq) = ids.dims2()?;
        let logits = self.forward(ids)?; // (b, seq, vocab)
        let pred = logits
            .narrow(1, 0, seq - 1)?
            .contiguous()?
            .reshape((b * (seq - 1), self.cfg.vocab))?;
        let tgt = ids
            .narrow(1, 1, seq - 1)?
            .contiguous()?
            .reshape((b * (seq - 1),))?;
        loss::cross_entropy(&pred, &tgt)
    }

    /// Mini-batch SGD over shuffled sequences. Targets are the inputs shifted by one
    /// (next-token), computed inside the loss.
    pub fn train_minibatched(
        &mut self,
        ids: &Tensor,
        epochs: usize,
        batch_size: usize,
        lr: f64,
        shuffle_seed: u64,
    ) -> Result<()> {
        let n = ids.dims2()?.0;
        let bs = batch_size.clamp(1, n.max(1));
        let mut opt = AdamW::new(
            self.varmap.all_vars(),
            ParamsAdamW {
                lr,
                ..Default::default()
            },
        )?;
        let mut perm: Vec<u32> = (0..n as u32).collect();
        for epoch in 0..epochs {
            shuffle(
                &mut perm,
                shuffle_seed ^ (epoch as u64).wrapping_mul(0x9E37_79B9),
            );
            let mut start = 0;
            while start < n {
                let end = (start + bs).min(n);
                let idx =
                    Tensor::from_vec(perm[start..end].to_vec(), (end - start,), &self.device)?;
                let xb = ids.index_select(&idx, 0)?;
                let l = self.loss_tensor(&xb)?;
                opt.backward_step(&l)?;
                start = end;
            }
        }
        Ok(())
    }
}

fn shuffle(perm: &mut [u32], seed: u64) {
    let mut rng = SplitMix64::new(seed);
    for i in (1..perm.len()).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        perm.swap(i, j);
    }
}

/// `(seq, seq)` additive causal mask: 0 on/below the diagonal, large-negative above.
fn causal_attn_mask(seq: usize, dev: &Device) -> Result<Tensor> {
    let mut m = vec![0f32; seq * seq];
    for t in 0..seq {
        for k in (t + 1)..seq {
            m[t * seq + k] = -1e9;
        }
    }
    Tensor::from_vec(m, (seq, seq), dev)
}

/// The SSM constant matrices: `(t-k)` on/below the diagonal, and the lower-triangular
/// ones mask.
fn ssm_matrices(seq: usize, dev: &Device) -> Result<(Tensor, Tensor)> {
    let mut tkv = vec![0f32; seq * seq];
    for t in 0..seq {
        for k in 0..=t {
            tkv[t * seq + k] = (t - k) as f32;
        }
    }
    let tk = Tensor::from_vec(tkv, (seq, seq), dev)?;
    let tri = Tensor::tril2(seq, DType::F32, dev)?;
    Ok((tk, tri))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_shape_and_param_count() {
        let cfg = AutoregConfig {
            seq_len: 16,
            d: 24,
            ..AutoregConfig::byte_3zone()
        };
        let m = AutoregLm::new(&cfg).unwrap();
        let ids = Tensor::from_vec(
            (0..(2 * cfg.seq_len) as u32)
                .map(|i| i % 256)
                .collect::<Vec<_>>(),
            (2, cfg.seq_len),
            m.device(),
        )
        .unwrap();
        let out = m.forward(&ids).unwrap();
        assert_eq!(out.dims3().unwrap(), (2, cfg.seq_len, cfg.vocab));
        assert!(m.param_count() > 0);
    }

    #[test]
    fn next_token_loss_drops_below_uniform() {
        // A short, fast run on real seed text: per-position next-token loss falls
        // below the uniform-byte baseline (ln 256). Proves the autoregressive
        // objective + causal cores train.
        let cfg = AutoregConfig {
            seq_len: 32,
            d: 32,
            ..AutoregConfig::byte_3zone()
        };
        let mut m = AutoregLm::new(&cfg).unwrap();
        let out = nat_data::run_pipeline(
            nat_data::seed::seed_corpus(),
            &nat_data::PipelineConfig::default(),
        );
        let ids =
            crate::corpus::sequence_windows(&out.shards, cfg.seq_len, 200, m.device()).unwrap();
        let before = m.loss_on(&ids).unwrap();
        m.train_minibatched(&ids, 5, 32, 0.003, 7).unwrap();
        let after = m.loss_on(&ids).unwrap();
        assert!(after < before, "did not learn: {before} -> {after}");
        assert!(
            after < (cfg.vocab as f32).ln(),
            "no better than uniform: {after}"
        );
    }

    #[test]
    fn gguf_export_round_trips() {
        // g3-gguf: export to GGUF, read back via candle's GGUF reader; every weight
        // is present and its values match (lossless F32).
        let cfg = AutoregConfig {
            seq_len: 16,
            d: 24,
            ..AutoregConfig::byte_3zone()
        };
        let m = AutoregLm::new(&cfg).unwrap();
        let path = std::env::temp_dir().join("nat_autoreg.gguf");
        m.export_gguf(&path).unwrap();

        let names = crate::gguf::tensor_names(&path).unwrap();
        let want: std::collections::BTreeSet<String> =
            m.varmap.data().lock().unwrap().keys().cloned().collect();
        let got: std::collections::BTreeSet<String> = names.into_iter().collect();
        assert_eq!(got, want, "GGUF tensor set != model varmap");

        // A sampled weight round-trips exactly.
        let orig = m
            .varmap
            .data()
            .lock()
            .unwrap()
            .get("embedding.weight")
            .unwrap()
            .as_tensor()
            .to_device(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let back = crate::gguf::read_tensor(&path, "embedding.weight")
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(orig, back, "embedding did not round-trip");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn five_zone_autoreg_constructs_and_runs() {
        // The 5-zone config exercises the causal SSM cores (SM/CB) too.
        let cfg = AutoregConfig {
            zones: ZoneId::LEARNED.to_vec(),
            seq_len: 16,
            d: 24,
            ..AutoregConfig::byte_3zone()
        };
        let m = AutoregLm::new(&cfg).unwrap();
        let ids =
            Tensor::from_vec(vec![1u32; 2 * cfg.seq_len], (2, cfg.seq_len), m.device()).unwrap();
        assert_eq!(m.forward(&ids).unwrap().dims3().unwrap().2, cfg.vocab);
    }
}

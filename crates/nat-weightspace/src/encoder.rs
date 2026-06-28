//! WP-B1 — the GMN encoder (research artifact, gated, may fail without blocking Layer A).
//!
//! An **untrained** Graph-Metanetwork encoder: fixed seeded projections, directed
//! message passing with **sum** aggregation (permutation-equivariant), and a **mean**
//! readout (permutation-invariant). Untrained GMN embeddings are a legitimate research
//! baseline — the publishable claim here is structural, not learned: the encoder is
//! invariant to neuron relabeling (a symmetry a flatten-and-MLP encoder violates) and its
//! latent *diagnoses* a planted weight-space property on **both** NAT and transformer
//! graphs with one encoder. That cross-architecture, permutation-invariant readout is
//! exactly the property a weight-conditioned LoRA generator (WS-3) needs.

use crate::{EdgeKind, WeightGraph, EDGE_FEAT_DIM, FEAT_DIM};

/// A fixed (seeded, untrained) GMN encoder.
#[derive(Debug, Clone)]
pub struct GmnEncoder {
    latent: usize,
    rounds: usize,
    proj_node: Vec<Vec<f32>>, // [latent][FEAT_DIM]
    proj_edge: Vec<Vec<f32>>, // [latent][EDGE_FEAT_DIM]
    proj_self: Vec<Vec<f32>>, // [latent][latent]
}

impl GmnEncoder {
    pub fn new(latent: usize, rounds: usize, seed: u64) -> Self {
        let mut rng = crate::rng::SeededRng::new(seed);
        let mat = |rows: usize, cols: usize, rng: &mut crate::rng::SeededRng| {
            (0..rows)
                .map(|_| {
                    (0..cols)
                        .map(|_| rng.next_f32() * 0.5)
                        .collect::<Vec<f32>>()
                })
                .collect::<Vec<_>>()
        };
        let proj_node = mat(latent, FEAT_DIM, &mut rng);
        let proj_edge = mat(latent, EDGE_FEAT_DIM, &mut rng);
        let proj_self = mat(latent, latent, &mut rng);
        GmnEncoder {
            latent,
            rounds,
            proj_node,
            proj_edge,
            proj_self,
        }
    }

    pub fn latent_dim(&self) -> usize {
        self.latent
    }

    /// Encode a weight-graph into a permutation-invariant latent vector.
    pub fn encode(&self, g: &WeightGraph) -> Vec<f32> {
        let n = g.nodes.len();
        // initial node states h = tanh(proj_node · feats)
        let mut h: Vec<Vec<f32>> = g
            .nodes
            .iter()
            .map(|node| tanh_vec(matvec(&self.proj_node, &node.feats)))
            .collect();

        for _ in 0..self.rounds {
            let mut msg = vec![vec![0.0f32; self.latent]; n];
            for e in &g.edges {
                // edge embedding ⊙ source state → message into dst (sum aggregation)
                let ef = edge_feat(e.kind, e.weight);
                let e_emb = matvec(&self.proj_edge, &ef);
                let src_h = &h[e.src];
                let dst_msg = &mut msg[e.dst];
                for k in 0..self.latent {
                    dst_msg[k] += e_emb[k] * src_h[k];
                }
            }
            // node update: h = tanh(h + proj_self · msg)
            for i in 0..n {
                let upd = matvec(&self.proj_self, &msg[i]);
                for k in 0..self.latent {
                    h[i][k] = (h[i][k] + upd[k]).tanh();
                }
            }
        }

        // permutation-invariant mean readout
        let mut out = vec![0.0f32; self.latent];
        for hi in &h {
            for k in 0..self.latent {
                out[k] += hi[k];
            }
        }
        let inv = 1.0 / n.max(1) as f32;
        for v in &mut out {
            *v *= inv;
        }
        out
    }
}

/// A flatten-and-MLP baseline encoder: concatenate node features in **node-index order**
/// and project. This is the standard weight-space encoder the GMN improves on — it is
/// *not* permutation-invariant, which the benchmark demonstrates.
pub fn flatten_baseline_encode(g: &WeightGraph, latent: usize, seed: u64) -> Vec<f32> {
    let flat: Vec<f32> = g.nodes.iter().flat_map(|node| node.feats.clone()).collect();
    let mut rng = crate::rng::SeededRng::new(seed);
    // fixed projection from a fixed max width; rows are truncated/padded deterministically.
    (0..latent)
        .map(|_| {
            let mut acc = 0.0f32;
            for &x in &flat {
                acc += x * rng.next_f32();
            }
            acc / flat.len().max(1) as f32
        })
        .collect()
}

/// Relabel every node by `perm` (`perm[old] = new`) and remap edges accordingly. The
/// result is the *same network* under a neuron permutation — a GMN encoding must not move.
pub fn permute_nodes(g: &WeightGraph, perm: &[usize]) -> WeightGraph {
    assert_eq!(perm.len(), g.nodes.len());
    let mut nodes = g.nodes.clone();
    for (old, node) in g.nodes.iter().enumerate() {
        nodes[perm[old]] = node.clone();
    }
    let edges = g
        .edges
        .iter()
        .map(|e| crate::GraphEdge {
            src: perm[e.src],
            dst: perm[e.dst],
            kind: e.kind,
            weight: e.weight,
        })
        .collect();
    WeightGraph {
        arch: g.arch,
        nodes,
        edges,
    }
}

// ---- small numeric helpers (research/f32 layer) ---------------------------

fn edge_feat(kind: EdgeKind, weight: f32) -> Vec<f32> {
    let mut f = vec![0.0f32; EDGE_FEAT_DIM];
    f[kind.slot()] = 1.0;
    f[EDGE_FEAT_DIM - 1] = weight;
    f
}

fn matvec(m: &[Vec<f32>], x: &[f32]) -> Vec<f32> {
    m.iter()
        .map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum())
        .collect()
}

fn tanh_vec(mut v: Vec<f32>) -> Vec<f32> {
    for x in &mut v {
        *x = x.tanh();
    }
    v
}

/// L2 distance between two equal-length vectors.
pub fn l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

/// Pearson correlation coefficient.
pub fn pearson(xs: &[f32], ys: &[f32]) -> f32 {
    let n = xs.len() as f32;
    let mx = xs.iter().sum::<f32>() / n;
    let my = ys.iter().sum::<f32>() / n;
    let mut sxy = 0.0;
    let mut sxx = 0.0;
    let mut syy = 0.0;
    for (&x, &y) in xs.iter().zip(ys) {
        sxy += (x - mx) * (y - my);
        sxx += (x - mx) * (x - mx);
        syy += (y - my) * (y - my);
    }
    if sxx <= 0.0 || syy <= 0.0 {
        return 0.0;
    }
    sxy / (sxx.sqrt() * syy.sqrt())
}

/// Is `ys` strictly monotonic (increasing or decreasing) across the sequence?
pub fn is_strictly_monotonic(ys: &[f32]) -> bool {
    if ys.len() < 2 {
        return true;
    }
    let inc = ys.windows(2).all(|w| w[1] > w[0]);
    let dec = ys.windows(2).all(|w| w[1] < w[0]);
    inc || dec
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{nat_checkpoint, seeded_linear, transformer_checkpoint};
    use crate::{lower_nat, lower_transformer, NatCheckpoint, TransformerCheckpoint, ZoneWeights};

    fn identity_perm(n: usize) -> Vec<usize> {
        (0..n).collect()
    }

    fn reverse_perm(n: usize) -> Vec<usize> {
        (0..n).rev().collect()
    }

    #[test]
    fn encoder_is_deterministic() {
        let enc = GmnEncoder::new(16, 3, 99);
        let g = lower_nat(&nat_checkpoint(5));
        assert_eq!(enc.encode(&g), enc.encode(&g));
    }

    #[test]
    fn gmn_is_permutation_invariant_where_flatten_baseline_is_not() {
        let enc = GmnEncoder::new(16, 3, 99);
        for g in [
            lower_nat(&nat_checkpoint(5)),
            lower_transformer(&transformer_checkpoint(5, 2, 4)),
        ] {
            let perm = reverse_perm(g.nodes.len());
            let permuted = permute_nodes(&g, &perm);

            let base = enc.encode(&g);
            let gmn_perm_dist = l2(&base, &enc.encode(&permuted));

            let flat_base = flatten_baseline_encode(&g, 16, 7);
            let flat_perm_dist = l2(&flat_base, &flatten_baseline_encode(&permuted, 16, 7));

            // GMN: invariant (float-noise tolerance). Baseline: provably moves.
            assert!(gmn_perm_dist < 1e-4, "GMN not invariant: {gmn_perm_dist}");
            assert!(
                flat_perm_dist > 1e-2,
                "baseline should be permutation-sensitive: {flat_perm_dist}"
            );
            assert!(
                flat_perm_dist > gmn_perm_dist * 100.0,
                "GMN must be far more invariant than flatten baseline"
            );
        }
        // sanity: the identity permutation is exactly invariant in both
        let g = lower_nat(&nat_checkpoint(5));
        assert!(
            l2(
                &enc.encode(&g),
                &enc.encode(&permute_nodes(&g, &identity_perm(g.nodes.len())))
            ) < 1e-6
        );
    }

    // --- diagnosis: one encoder reads a planted property on BOTH architectures ---

    fn scale_nat(seed: u64, s: f32) -> NatCheckpoint {
        let mut ck = nat_checkpoint(seed);
        for (_, zw) in &mut ck.zone_weights {
            match zw {
                ZoneWeights::Attention { wq, wk, wv, wo } => {
                    for m in [wq, wk, wv, wo] {
                        scale_linear(m, s);
                    }
                }
                ZoneWeights::Ssm { wb, wc, wo, .. } => {
                    for m in [wb, wc, wo] {
                        scale_linear(m, s);
                    }
                }
            }
        }
        ck
    }

    fn scale_xf(seed: u64, blocks: usize, d: usize, s: f32) -> TransformerCheckpoint {
        let mut ck = transformer_checkpoint(seed, blocks, d);
        for b in &mut ck.blocks {
            for m in [
                &mut b.wq,
                &mut b.wk,
                &mut b.wv,
                &mut b.wo,
                &mut b.ffn_up,
                &mut b.ffn_down,
            ] {
                scale_linear(m, s);
            }
        }
        ck
    }

    fn scale_linear(m: &mut crate::Linear, s: f32) {
        for row in &mut m.w {
            for w in row {
                *w *= s;
            }
        }
    }

    /// A fixed 1-D readout direction (seeded), shared across architectures.
    fn probe_feature(enc: &GmnEncoder, dir: &[f32], g: &WeightGraph) -> f32 {
        enc.encode(g).iter().zip(dir).map(|(a, b)| a * b).sum()
    }

    #[test]
    fn one_encoder_diagnoses_a_planted_property_cross_arch() {
        let enc = GmnEncoder::new(16, 3, 99);
        // a fixed probe direction, identical for NAT and transformer (cross-arch claim)
        let dir = seeded_linear(1, enc.latent_dim(), 0xD17).w.remove(0);

        let scales: Vec<f32> = (0..11).map(|i| 0.4 + 0.1 * i as f32).collect(); // 0.4..1.4

        let nat_feats: Vec<f32> = scales
            .iter()
            .map(|&s| probe_feature(&enc, &dir, &lower_nat(&scale_nat(5, s))))
            .collect();
        let xf_feats: Vec<f32> = scales
            .iter()
            .map(|&s| probe_feature(&enc, &dir, &lower_transformer(&scale_xf(5, 2, 4, s))))
            .collect();

        // the planted scalar is recovered monotonically on BOTH architectures…
        assert!(
            is_strictly_monotonic(&nat_feats),
            "NAT diagnosis not monotone: {nat_feats:?}"
        );
        assert!(
            is_strictly_monotonic(&xf_feats),
            "transformer diagnosis not monotone: {xf_feats:?}"
        );
        // …and with high linear correlation (the publishable cross-arch measurement).
        let nat_r = pearson(&scales, &nat_feats).abs();
        let xf_r = pearson(&scales, &xf_feats).abs();
        assert!(nat_r > 0.95, "NAT corr too low: {nat_r}");
        assert!(xf_r > 0.95, "transformer corr too low: {xf_r}");
    }
}

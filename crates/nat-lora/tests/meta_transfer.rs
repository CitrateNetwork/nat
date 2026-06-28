//! WP-G3 + WP-G4 — meta-training, held-out transfer, and the ablation control.
//!
//! The frontier-bet-#3 claim, mechanically checked: a generator meta-trained on a family
//! of peers (each carrying a low-rank capability `ΔW* = Σ φ_k D_k` planted in its weights)
//! generates, for **held-out** peers, a LoRA that installs the peer's capability into a
//! base student — and a generator fed a **shuffled** latent does not. The capability is
//! "decide like the peer": the adapter raises the student's agreement with the peer's
//! decisions, McNemar-significantly (WP-G4 reuses `nat-distill::promotion_gate`).
//!
//! Each peer's weights literally contain `φ` (its `ffn_down` is perturbed by `Σ φ_k P_k`),
//! so the GMN latent is a function of `φ`; the generator must recover it from weight-space
//! alone. Run on transformer-graph peers (primary) and NAT-graph peers (cross-arch).

// Matrix-accumulation loops (readouts, perturbations) read clearer with explicit indices.
#![allow(clippy::needless_range_loop)]

use nat_distill::promotion_gate;
use nat_lora::{condition, decisions, LoraGenerator, SkillAtom, ZoneId};
use nat_sidecar::Sidecar;
use nat_types::CoreType;
use nat_weightspace::encoder::GmnEncoder;
use nat_weightspace::rng::SeededRng;
use nat_weightspace::{
    lower_nat, lower_transformer, Linear, NatCheckpoint, TransformerBlock, TransformerCheckpoint,
    WeightGraph, ZoneWeights,
};

const D_IN: usize = 6; // hidden width
const D_OUT: usize = 4; // classes
const K: usize = 3; // skill atoms
const LATENT: usize = 16;
const N_PROBE: usize = 24;

// ---- fixed world (seeded once) --------------------------------------------

struct World {
    h: Vec<Vec<f32>>,           // probe hiddens [N_PROBE][D_IN]
    w0: Vec<Vec<f32>>,          // base readout [D_OUT][D_IN]
    atoms: Vec<SkillAtom>,      // K skill atoms (u⊗v) — the decision dictionary
    base_ffn: Vec<Vec<f32>>,    // peer ffn_down base [D_OUT][D_IN]
    pert: Vec<Vec<Vec<f32>>>,   // K perturbation matrices [D_OUT][D_IN] encoding φ in weights
    fixed: TransformerBlockTmpl,
}

struct TransformerBlockTmpl {
    wq: Linear,
    wk: Linear,
    wv: Linear,
    wo: Linear,
    ffn_up: Linear,
}

fn mat(rng: &mut SeededRng, r: usize, c: usize, scale: f32) -> Vec<Vec<f32>> {
    (0..r).map(|_| (0..c).map(|_| rng.next_f32() * scale).collect()).collect()
}

fn build_world() -> World {
    let mut rng = SeededRng::new(0xA70_5A17);
    let h = mat(&mut rng, N_PROBE, D_IN, 1.0);
    let w0 = mat(&mut rng, D_OUT, D_IN, 0.25); // weak base readout
    let atoms: Vec<SkillAtom> = (0..K)
        .map(|_| SkillAtom { u: mat(&mut rng, 1, D_OUT, 1.0).remove(0), v: mat(&mut rng, 1, D_IN, 1.0).remove(0) })
        .collect();
    let base_ffn = mat(&mut rng, D_OUT, D_IN, 1.0);
    let pert: Vec<Vec<Vec<f32>>> = (0..K).map(|_| mat(&mut rng, D_OUT, D_IN, 1.0)).collect();
    let fixed = TransformerBlockTmpl {
        wq: Linear::new(mat(&mut rng, D_IN, D_IN, 1.0), vec![0.0; D_IN]),
        wk: Linear::new(mat(&mut rng, D_IN, D_IN, 1.0), vec![0.0; D_IN]),
        wv: Linear::new(mat(&mut rng, D_IN, D_IN, 1.0), vec![0.0; D_IN]),
        wo: Linear::new(mat(&mut rng, D_IN, D_IN, 1.0), vec![0.0; D_IN]),
        ffn_up: Linear::new(mat(&mut rng, 2 * D_IN, D_IN, 1.0), vec![0.0; 2 * D_IN]),
    };
    World { h, w0, atoms, base_ffn, pert, fixed }
}

// T_p = W0 + Σ φ_k (u_k ⊗ v_k) — the peer's readout (its decisions).
fn peer_readout(w: &World, phi: &[f32]) -> Vec<Vec<f32>> {
    let mut t = w.w0.clone();
    for k in 0..K {
        for o in 0..D_OUT {
            for i in 0..D_IN {
                t[o][i] += phi[k] * w.atoms[k].u[o] * w.atoms[k].v[i];
            }
        }
    }
    t
}

// the peer's ffn_down = base + Σ φ_k P_k — this is where φ enters the *weight graph*.
fn peer_ffn_down(w: &World, phi: &[f32]) -> Vec<Vec<f32>> {
    let mut f = w.base_ffn.clone();
    for k in 0..K {
        for o in 0..D_OUT {
            for i in 0..D_IN {
                f[o][i] += phi[k] * w.pert[k][o][i];
            }
        }
    }
    f
}

fn peer_transformer_graph(w: &World, phi: &[f32]) -> WeightGraph {
    let block = TransformerBlock {
        wq: w.fixed.wq.clone(),
        wk: w.fixed.wk.clone(),
        wv: w.fixed.wv.clone(),
        wo: w.fixed.wo.clone(),
        ffn_up: w.fixed.ffn_up.clone(),
        ffn_down: Linear::new(peer_ffn_down(w, phi), vec![0.0; D_OUT]),
    };
    lower_transformer(&TransformerCheckpoint { d_model: D_IN, blocks: vec![block] })
}

fn peer_nat_graph(w: &World, phi: &[f32]) -> WeightGraph {
    // a NAT checkpoint where EVERY learned zone's readout (wo) carries φ — so the GMN
    // latent depends pervasively on φ and the shuffled-latent control genuinely degrades.
    let sidecar = Sidecar::default_l0();
    let ro = peer_ffn_down(w, phi); // [D_OUT][D_IN]
    let mut zone_weights = Vec::new();
    for zd in &sidecar.zones {
        let zw = match zd.core {
            CoreType::Attention => ZoneWeights::Attention {
                wq: w.fixed.wq.clone(),
                wk: w.fixed.wk.clone(),
                wv: w.fixed.wv.clone(),
                wo: Linear::new(ro.clone(), vec![0.0; D_OUT]),
            },
            CoreType::Ssm => ZoneWeights::Ssm {
                wb: w.fixed.wq.clone(),
                wc: w.fixed.wk.clone(),
                wo: Linear::new(ro.clone(), vec![0.0; D_OUT]),
                log_a: 0.5,
            },
            CoreType::None => continue,
        };
        zone_weights.push((zd.id, zw));
    }
    lower_nat(&NatCheckpoint { sidecar, zone_weights })
}

fn sample_phis(seed: u64, n: usize) -> Vec<Vec<f32>> {
    let mut rng = SeededRng::new(seed);
    (0..n)
        .map(|_| (0..K).map(|_| 0.2 + 0.8 * (rng.next_f32() * 0.5 + 0.5)).collect()) // [0.2,1.0]
        .collect()
}

fn agreement(preds: &[usize], labels: &[usize]) -> f32 {
    let ok = preds.iter().zip(labels).filter(|(a, b)| a == b).count();
    ok as f32 / labels.len().max(1) as f32
}

fn shuffle(v: &[f32]) -> Vec<f32> {
    // deterministic derangement: reverse, then rotate by 1 — destroys the z→φ signal.
    let mut s: Vec<f32> = v.iter().rev().cloned().collect();
    s.rotate_left(1);
    s
}

/// Run the full pipeline for one peer-graph builder; returns (held-out mean agreements):
/// (base, lora, shuffled) plus the pooled preds for the McNemar gate.
struct Result {
    base: f32,
    lora: f32,
    shuffled: f32,
    pooled_base: Vec<usize>,
    pooled_lora: Vec<usize>,
    pooled_labels: Vec<usize>,
}

fn run(world: &World, build: &dyn Fn(&World, &[f32]) -> WeightGraph) -> Result {
    let enc = GmnEncoder::new(LATENT, 3, 0xE0C);
    let train_phis = sample_phis(0x1111, 80);
    let test_phis = sample_phis(0x2222, 24);

    // meta-train: peer latents → target gains (= φ).
    let train_latents: Vec<Vec<f32>> =
        train_phis.iter().map(|p| condition(&enc, &build(world, p))).collect();
    let mut gen = LoraGenerator::new(ZoneId::PF, world.atoms.clone(), LATENT);
    gen.fit(&train_latents, &train_phis, 1e-2);

    let (mut sb, mut sl, mut ss) = (0.0f32, 0.0f32, 0.0f32);
    let (mut pb, mut pl, mut py) = (Vec::new(), Vec::new(), Vec::new());

    for phi in &test_phis {
        let graph = build(world, phi);
        let z = condition(&enc, &graph);

        let t_p = peer_readout(world, phi);
        let labels = decisions(&t_p, &world.h); // peer's decisions

        let base_pred = decisions(&world.w0, &world.h);
        let lora = gen.generate(&z);
        let lora_pred = decisions(&lora.apply_matrix(&world.w0), &world.h);
        let shuf = gen.generate(&shuffle(&z));
        let shuf_pred = decisions(&shuf.apply_matrix(&world.w0), &world.h);

        sb += agreement(&base_pred, &labels);
        sl += agreement(&lora_pred, &labels);
        ss += agreement(&shuf_pred, &labels);
        pb.extend(base_pred);
        pl.extend(lora_pred);
        py.extend(labels);
    }
    let n = test_phis.len() as f32;
    Result { base: sb / n, lora: sl / n, shuffled: ss / n, pooled_base: pb, pooled_lora: pl, pooled_labels: py }
}

#[test]
fn weight_conditioned_generation_transfers_to_held_out_peers_transformer() {
    let world = build_world();
    let r = run(&world, &peer_transformer_graph);
    eprintln!("[transformer] base={:.3} lora={:.3} shuffled={:.3}", r.base, r.lora, r.shuffled);

    // the generated LoRA installs the peer's capability on held-out peers…
    assert!(r.lora > r.base + 0.25, "no transfer: base={} lora={}", r.base, r.lora);
    // …and meaningfully beats the shuffled-latent control (the conditioning matters).
    assert!(r.lora > r.shuffled + 0.20, "ablation failed: lora={} shuffled={}", r.lora, r.shuffled);

    // WP-G4 — the McNemar promotion gate accepts it (pooled over held-out peers).
    let d = promotion_gate(&r.pooled_base, &r.pooled_lora, &r.pooled_labels, r.base, r.lora, 0.05);
    assert!(d.promote, "promotion gate should accept: {d:?}");
    assert!(d.p_value < 0.05 && d.improved && d.no_regression);
}

#[test]
fn weight_conditioned_generation_transfers_on_nat_graph_peers() {
    // cross-arch: the same pipeline on NAT-graph peers (g-g3 "both archs").
    let world = build_world();
    let r = run(&world, &peer_nat_graph);
    eprintln!("[nat] base={:.3} lora={:.3} shuffled={:.3}", r.base, r.lora, r.shuffled);
    assert!(r.lora > r.base + 0.25, "no NAT transfer: base={} lora={}", r.base, r.lora);
    assert!(r.lora > r.shuffled + 0.15, "NAT ablation failed: lora={} shuffled={}", r.lora, r.shuffled);
}

#[test]
fn shuffled_control_does_not_promote() {
    // the ablation as a hard gate: a generator fed shuffled latents must NOT pass the
    // promotion gate against the base (no real capability was transferred).
    let world = build_world();
    let r = run(&world, &peer_transformer_graph);
    // build pooled shuffled preds by re-running the shuffled path
    let enc = GmnEncoder::new(LATENT, 3, 0xE0C);
    let train_phis = sample_phis(0x1111, 80);
    let train_latents: Vec<Vec<f32>> =
        train_phis.iter().map(|p| condition(&enc, &peer_transformer_graph(&world, p))).collect();
    let mut gen = LoraGenerator::new(ZoneId::PF, world.atoms.clone(), LATENT);
    gen.fit(&train_latents, &train_phis, 1e-2);

    let (mut pb, mut ps, mut py) = (Vec::new(), Vec::new(), Vec::new());
    for phi in &sample_phis(0x2222, 24) {
        let z = condition(&enc, &peer_transformer_graph(&world, phi));
        let labels = decisions(&peer_readout(&world, phi), &world.h);
        let base_pred = decisions(&world.w0, &world.h);
        let shuf = gen.generate(&shuffle(&z));
        let shuf_pred = decisions(&shuf.apply_matrix(&world.w0), &world.h);
        pb.extend(base_pred);
        ps.extend(shuf_pred);
        py.extend(labels);
    }
    let base_acc = agreement(&pb, &py);
    let shuf_acc = agreement(&ps, &py);
    let d = promotion_gate(&pb, &ps, &py, base_acc, shuf_acc, 0.05);
    assert!(!d.promote, "shuffled control must NOT promote: {d:?}");
    let _ = r;
}

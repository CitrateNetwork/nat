//! H-02 (WP-3): a *trained* router differentiates prompt classes better than the
//! hand-wired L0 router, scored by the **same** `separation_ratio` metric.
//!
//! Honest scope: this is the in-sample read (the router is trained and scored on
//! the same labeled battery), which is what the L0 baseline is also measured on.
//! The real H-02 verdict is over held-out batteries at full L1 scale; this proves
//! the trained gate *can* differentiate, and beats the hand-wired baseline on the
//! shared metric.

use candle_core::Tensor;
use nat_candle::router::{LearnedRouter, FEATURE_DIM};
use nat_core::featurize::class_signals;
use nat_core::NatModel;
use nat_eval::battery::PromptBattery;
use nat_eval::{evaluate_routing, separation_ratio, LEARNED_DIM};
use nat_sidecar::Sidecar;

#[test]
fn trained_router_beats_l0_baseline_on_h02() {
    let battery = PromptBattery::default_l0();
    let sidecar = Sidecar::default_l0();
    let mut router = LearnedRouter::new(&sidecar, 16, 7).unwrap();
    let dev = router.device().clone();

    // Build (features, labels) over the whole labeled battery.
    let mut feats: Vec<f32> = Vec::new();
    let mut labels: Vec<u32> = Vec::new();
    for (ci, class) in battery.classes.iter().enumerate() {
        for p in &class.prompts {
            let s = class_signals(p);
            feats.extend_from_slice(&[s.math, s.narrative, s.code, s.sensory]);
            labels.push(ci as u32);
        }
    }
    let n = labels.len();
    let features = Tensor::from_vec(feats, (n, FEATURE_DIM), &dev).unwrap();
    let labels = Tensor::from_vec(labels, (n,), &dev).unwrap();

    // Train the gate to make its activations class-discriminative.
    router
        .train_to_classify(&features, &labels, battery.classes.len(), 400, 0.05)
        .unwrap();

    // Score the trained router's activations with the shared metric.
    let trained: Vec<Vec<[f32; LEARNED_DIM]>> = battery
        .classes
        .iter()
        .map(|c| {
            c.prompts
                .iter()
                .map(|p| {
                    let v = router.activation_vec(p).unwrap();
                    let mut a = [0f32; LEARNED_DIM];
                    a.copy_from_slice(&v);
                    a
                })
                .collect()
        })
        .collect();
    let trained_ratio = separation_ratio(&trained);

    // The L0 hand-wired baseline, same battery, same metric.
    let baseline = evaluate_routing(&NatModel::l0(), &battery).separation_ratio;

    eprintln!("H-02 separation ratio: trained={trained_ratio:.3} vs L0 baseline={baseline:.3}");
    assert!(
        trained_ratio > baseline,
        "trained router did not beat L0 baseline: trained={trained_ratio:.3} baseline={baseline:.3}"
    );
}

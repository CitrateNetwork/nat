//! H-02 **held-out**: train the learned router on a subset of an extended battery,
//! then score routing-differentiation on prompts it never saw — versus the L0
//! hand-wired baseline on the *same* held-out prompts. This closes the in-sample
//! caveat of `h02_trained_router`: the trained router must generalize, not memorize.

use candle_core::Tensor;
use nat_candle::router::{LearnedRouter, FEATURE_DIM};
use nat_core::featurize::class_signals;
use nat_core::NatModel;
use nat_eval::battery::{PromptBattery, PromptClass};
use nat_eval::{evaluate_routing, separation_ratio, LEARNED_DIM};
use nat_sidecar::Sidecar;

fn to_arr(v: Vec<f32>) -> [f32; LEARNED_DIM] {
    let mut a = [0f32; LEARNED_DIM];
    a.copy_from_slice(&v);
    a
}

#[test]
fn trained_router_beats_l0_baseline_held_out() {
    let battery = PromptBattery::default_l0_extended();
    let n_train = 6; // per class; the remaining 4 are held out

    // Split each class into train / held-out.
    let mut train: Vec<PromptClass> = Vec::new();
    let mut held: Vec<PromptClass> = Vec::new();
    for c in &battery.classes {
        train.push(PromptClass {
            label: c.label.clone(),
            prompts: c.prompts[..n_train].to_vec(),
        });
        held.push(PromptClass {
            label: c.label.clone(),
            prompts: c.prompts[n_train..].to_vec(),
        });
    }
    let held_battery = PromptBattery {
        classes: held.clone(),
    };

    // Train the router on the TRAIN prompts only.
    let sidecar = Sidecar::default_l0();
    let mut router = LearnedRouter::new(&sidecar, 16, 7).unwrap();
    let dev = router.device().clone();
    let (mut feats, mut labels) = (Vec::new(), Vec::new());
    for (ci, c) in train.iter().enumerate() {
        for p in &c.prompts {
            let s = class_signals(p);
            feats.extend_from_slice(&[s.math, s.narrative, s.code, s.sensory]);
            labels.push(ci as u32);
        }
    }
    let nrows = labels.len();
    let features = Tensor::from_vec(feats, (nrows, FEATURE_DIM), &dev).unwrap();
    let labels = Tensor::from_vec(labels, (nrows,), &dev).unwrap();
    router
        .train_to_classify(&features, &labels, train.len(), 400, 0.05)
        .unwrap();

    // Score the trained router on the HELD-OUT prompts.
    let trained: Vec<Vec<[f32; LEARNED_DIM]>> = held
        .iter()
        .map(|c| {
            c.prompts
                .iter()
                .map(|p| to_arr(router.activation_vec(p).unwrap()))
                .collect()
        })
        .collect();
    let trained_ratio = separation_ratio(&trained);

    // The L0 baseline on the SAME held-out prompts.
    let baseline = evaluate_routing(&NatModel::l0(), &held_battery).separation_ratio;

    eprintln!("H-02 held-out: trained={trained_ratio:.3} vs L0 baseline={baseline:.3}");
    assert!(
        trained_ratio > baseline,
        "trained router did not beat L0 on held-out: trained={trained_ratio:.3} baseline={baseline:.3}"
    );
}

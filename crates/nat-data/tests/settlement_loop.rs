//! End-to-end: the data pipeline's quality score is the `data_quality` term in a
//! participant's reward weight. This closes the economic loop at L0 scale —
//! pipeline → manifest.aggregate_quality → StepContribution.data_quality →
//! reward_weight = compute × quality (docs/SETTLEMENT_SEAM.md).

use nat_data::{run_pipeline, PipelineConfig, RawDoc};
use nat_train::StepContribution;
use nat_types::Q16;

fn raw(id: &str, text: &str) -> RawDoc {
    RawDoc {
        id: id.into(),
        source: "test".into(),
        license: "Apache-2.0".into(),
        fetch_date: "2026-06-18".into(),
        text: text.into(),
        modality_refs: vec![],
    }
}

#[test]
fn pipeline_quality_drives_reward_weight() {
    let cfg = PipelineConfig::default();
    let out = run_pipeline(
        vec![
            raw("a", "a clear english paragraph about rivers, maps, and the slow craft of cartography"),
            raw("b", "compute 12 + 7 * 3 step by step and explain each arithmetic operation in plain words"),
            raw("c", "she folded the worn map by the lantern and listened to the rain on warm stone"),
        ],
        &cfg,
    );

    // The manifest carries a corpus-level data-quality score in [0,1].
    let dq = out.manifest.aggregate_quality;
    assert!(dq.to_f32() > 0.0 && dq.to_f32() <= 1.0);

    // A node that contributed, say, 4.0 units of metered compute on this corpus.
    let contribution = StepContribution {
        compute_metered: Q16::from_f32(4.0),
        data_quality: dq, // <-- straight from the pipeline manifest
        tokens: out.manifest.total_tokens,
        provenance_hash: out.manifest.manifest_hash(),
    };

    // reward_weight = compute × quality, deterministically on the Q16.16 path.
    let expected = Q16::from_f32(4.0).mul(dq);
    assert_eq!(contribution.reward_weight(), expected);
}

#[test]
fn zero_quality_corpus_yields_zero_reward() {
    // If everything is quarantined (no kept tokens), aggregate quality is 0, so a
    // node earns zero reward weight no matter how much compute it burned.
    let cfg = PipelineConfig::default();
    let mut junk = raw("j", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"); // degenerate → low quality
    junk.license = "GPL-3.0".into(); // also unreviewed → quarantined at ingest
    let out = run_pipeline(vec![junk], &cfg);

    assert_eq!(out.manifest.total_tokens, 0);
    assert_eq!(out.manifest.aggregate_quality, Q16::ZERO);

    let contribution = StepContribution {
        compute_metered: Q16::from_f32(1000.0),
        data_quality: out.manifest.aggregate_quality,
        tokens: 0,
        provenance_hash: out.manifest.manifest_hash(),
    };
    assert_eq!(contribution.reward_weight(), Q16::ZERO);
}

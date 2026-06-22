//! The NAT eval harness (Data Ops §7). Built fully at L1; the metrics that are
//! *measurable at L0* live here now so the harness exists before it is needed.
//!
//! The bet-deciding metric, capability-per-parameter (H-01), cannot be measured
//! at L0 (nothing is trained). What *is* measurable is the structural plumbing:
//! routing differentiation (H-02) and provenance faithfulness (H-03a).
//!
//! ## Routing differentiation (H-02), stated as a real metric
//!
//! For a labeled battery, embed each prompt as its learned-zone activation vector
//! (5-dim, excluding the always-on non-learned `MX`). Then:
//!
//! - **between-class** = mean pairwise distance between class centroids,
//! - **within-class**  = mean distance of prompts to their own class centroid,
//! - **separation ratio** = between / within.
//!
//! A ratio > 1 means classes are farther apart than they are wide — the router
//! differentiates by class. This is a Fisher/silhouette-flavored separation
//! score. At L0 the router is hand-wired, so a positive result *previews* H-02;
//! the real test runs the same battery against a trained router at L1.

pub mod battery;

use battery::PromptBattery;
use nat_core::NatModel;
use nat_provenance::verify_decision_faithful;
use nat_types::ZoneId;

/// The default H-02 significance threshold: classes must be at least as separated
/// as they are wide. Tunable; the real per-task threshold is set at L1.
pub const DEFAULT_SEPARATION_THRESHOLD: f32 = 1.0;

/// Dimensionality of a routing vector: the five learned zones (SM, CB, HP, PF,
/// CX), excluding the always-on MX harness.
pub const LEARNED_DIM: usize = 5;

/// One class's routing summary.
#[derive(Debug, Clone)]
pub struct ClassRouting {
    pub label: String,
    pub centroid: [f32; LEARNED_DIM],
    /// The single most-activated learned zone for this class.
    pub dominant_zone: ZoneId,
    /// Mean distance of this class's prompts to its own centroid (its width).
    pub spread: f32,
}

/// The routing-differentiation report for a battery.
#[derive(Debug, Clone)]
pub struct RoutingReport {
    pub classes: Vec<ClassRouting>,
    pub between_class: f32,
    pub within_class: f32,
    pub separation_ratio: f32,
}

impl RoutingReport {
    /// Does the router differentiate by class at the given threshold (H-02)?
    pub fn differentiates(&self, threshold: f32) -> bool {
        self.separation_ratio >= threshold
    }

    /// Do all classes have distinct dominant zones? A stronger, categorical
    /// signal than the ratio alone (useful when classes are few).
    pub fn dominant_zones_distinct(&self) -> bool {
        let mut seen = Vec::new();
        for c in &self.classes {
            if seen.contains(&c.dominant_zone) {
                return false;
            }
            seen.push(c.dominant_zone);
        }
        true
    }

    /// A short human-readable summary, in the eval-harness voice.
    pub fn summary(&self) -> String {
        let mut s = format!(
            "routing differentiation: ratio={:.2} (between={:.3}, within={:.3}) -> {}\n",
            self.separation_ratio,
            self.between_class,
            self.within_class,
            if self.differentiates(DEFAULT_SEPARATION_THRESHOLD) {
                "DIFFERENTIATES"
            } else {
                "does not differentiate"
            },
        );
        for c in &self.classes {
            s.push_str(&format!(
                "  {:<10} dominant={:?} spread={:.3}\n",
                c.label, c.dominant_zone, c.spread
            ));
        }
        s
    }
}

/// The learned-zone activation vector for one prompt (5-dim, canonical order).
fn activation_vec(model: &NatModel, prompt: &str) -> [f32; LEARNED_DIM] {
    let act = model.forward(prompt, None).trace.router.zone_activation;
    let mut v = [0.0f32; LEARNED_DIM];
    for (i, z) in ZoneId::LEARNED.iter().enumerate() {
        v[i] = act
            .iter()
            .find(|(id, _)| id == z)
            .map(|(_, q)| q.to_f32())
            .unwrap_or(0.0);
    }
    v
}

fn distance(a: &[f32; LEARNED_DIM], b: &[f32; LEARNED_DIM]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

fn centroid(vecs: &[[f32; LEARNED_DIM]]) -> [f32; LEARNED_DIM] {
    let mut c = [0.0f32; LEARNED_DIM];
    if vecs.is_empty() {
        return c;
    }
    for v in vecs {
        for i in 0..LEARNED_DIM {
            c[i] += v[i];
        }
    }
    for x in c.iter_mut() {
        *x /= vecs.len() as f32;
    }
    c
}

/// The separation ratio (between-class / within-class) over per-class sets of
/// routing vectors. Public so any activation source — the L0 model *or a trained
/// L1 router* — is scored by the **same** metric; that identical-yardstick
/// comparison is the H-02 test (does a trained router differentiate better?).
pub fn separation_ratio(class_vectors: &[Vec<[f32; LEARNED_DIM]>]) -> f32 {
    let centroids: Vec<[f32; LEARNED_DIM]> = class_vectors.iter().map(|v| centroid(v)).collect();

    let within_terms: Vec<f32> = class_vectors
        .iter()
        .zip(centroids.iter())
        .map(|(vecs, c)| {
            if vecs.is_empty() {
                0.0
            } else {
                vecs.iter().map(|v| distance(v, c)).sum::<f32>() / vecs.len() as f32
            }
        })
        .collect();

    let mut pair_dists = Vec::new();
    for i in 0..centroids.len() {
        for j in (i + 1)..centroids.len() {
            pair_dists.push(distance(&centroids[i], &centroids[j]));
        }
    }
    let between = if pair_dists.is_empty() {
        0.0
    } else {
        pair_dists.iter().sum::<f32>() / pair_dists.len() as f32
    };
    let within = if within_terms.is_empty() {
        0.0
    } else {
        within_terms.iter().sum::<f32>() / within_terms.len() as f32
    };
    between / (within + 1e-6)
}

/// Evaluate routing differentiation over a labeled battery (H-02). Deterministic:
/// the model is deterministic and the arithmetic is fixed-order.
pub fn evaluate_routing(model: &NatModel, battery: &PromptBattery) -> RoutingReport {
    // Per-class vectors and centroids.
    let mut classes: Vec<ClassRouting> = Vec::new();
    let mut centroids: Vec<[f32; LEARNED_DIM]> = Vec::new();
    let mut within_terms: Vec<f32> = Vec::new();

    for class in &battery.classes {
        let vecs: Vec<[f32; LEARNED_DIM]> = class
            .prompts
            .iter()
            .map(|p| activation_vec(model, p))
            .collect();
        let c = centroid(&vecs);
        let spread = if vecs.is_empty() {
            0.0
        } else {
            vecs.iter().map(|v| distance(v, &c)).sum::<f32>() / vecs.len() as f32
        };
        // Dominant learned zone = argmax of the centroid.
        let dom_idx = c
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        within_terms.push(spread);
        centroids.push(c);
        classes.push(ClassRouting {
            label: class.label.clone(),
            centroid: c,
            dominant_zone: ZoneId::LEARNED[dom_idx],
            spread,
        });
    }

    // Between-class = mean pairwise centroid distance.
    let mut pair_dists = Vec::new();
    for i in 0..centroids.len() {
        for j in (i + 1)..centroids.len() {
            pair_dists.push(distance(&centroids[i], &centroids[j]));
        }
    }
    let between_class = if pair_dists.is_empty() {
        0.0
    } else {
        pair_dists.iter().sum::<f32>() / pair_dists.len() as f32
    };

    // Within-class = mean class spread.
    let within_class = if within_terms.is_empty() {
        0.0
    } else {
        within_terms.iter().sum::<f32>() / within_terms.len() as f32
    };

    let separation_ratio = between_class / (within_class + 1e-6);

    RoutingReport {
        classes,
        between_class,
        within_class,
        separation_ratio,
    }
}

/// Pairwise routing divergence between two prompts (L1 distance over the 6-dim
/// activation, kept for quick spot checks).
pub fn routing_divergence(model: &NatModel, prompt_a: &str, prompt_b: &str) -> f32 {
    let a = model.forward(prompt_a, None).trace.router.zone_activation;
    let b = model.forward(prompt_b, None).trace.router.zone_activation;
    ZoneId::ALL
        .iter()
        .map(|z| {
            let av = a.iter().find(|(id, _)| id == z).unwrap().1.to_f32();
            let bv = b.iter().find(|(id, _)| id == z).unwrap().1.to_f32();
            (av - bv).abs()
        })
        .sum()
}

/// Provenance faithfulness (H-03a, decision-faithful sense): every pass's trace
/// recomputes its own merge decision from its recorded scores.
pub fn faithfulness_holds(model: &NatModel, prompts: &[&str]) -> bool {
    prompts
        .iter()
        .all(|p| verify_decision_faithful(&model.forward(p, None).trace))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_battery_differentiates_by_class() {
        let m = NatModel::l0();
        let report = evaluate_routing(&m, &PromptBattery::default_l0());
        // Even with a hand-wired L0 router, the four classes separate.
        assert!(
            report.differentiates(DEFAULT_SEPARATION_THRESHOLD),
            "{}",
            report.summary()
        );
        assert!(report.between_class > report.within_class);
    }

    #[test]
    fn evaluation_is_deterministic() {
        let m = NatModel::l0();
        let b = PromptBattery::default_l0();
        let r1 = evaluate_routing(&m, &b);
        let r2 = evaluate_routing(&m, &b);
        assert_eq!(r1.separation_ratio.to_bits(), r2.separation_ratio.to_bits());
    }

    #[test]
    fn code_class_lights_codec_more_than_narrative_does() {
        // The honest H-02 signal at L0: differentiation is *relative*. PF is the
        // always-on reasoning floor, so it tends to be the argmax for every class
        // under a hand-wired router; what differentiates classes is the *other*
        // axes. Here: the code class activates the Codec zone (CX, index 4) more
        // than the narrative class does. A trained L1 router sharpens this.
        let m = NatModel::l0();
        let report = evaluate_routing(&m, &PromptBattery::default_l0());
        let cx = ZoneId::LEARNED
            .iter()
            .position(|z| *z == ZoneId::CX)
            .unwrap();
        let code = report.classes.iter().find(|c| c.label == "code").unwrap();
        let narrative = report
            .classes
            .iter()
            .find(|c| c.label == "narrative")
            .unwrap();
        assert!(
            code.centroid[cx] > narrative.centroid[cx],
            "code CX={} narrative CX={}\n{}",
            code.centroid[cx],
            narrative.centroid[cx],
            report.summary()
        );
    }

    #[test]
    fn faithfulness_holds_over_battery() {
        let m = NatModel::l0();
        let b = PromptBattery::default_l0();
        let prompts: Vec<&str> = b
            .classes
            .iter()
            .flat_map(|c| c.prompts.iter().map(|s| s.as_str()))
            .collect();
        assert!(faithfulness_holds(&m, &prompts));
    }
}

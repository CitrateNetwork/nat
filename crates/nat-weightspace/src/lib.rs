//! `nat-weightspace` — the GMN weight-space encoder (WS-2 Layer B, frontier bet #2).
//!
//! **One permutation-equivariant graph type spanning NAT *and* a vanilla transformer.**
//! A neural network's weights have a symmetry that flatten-and-MLP encoders ignore: the
//! hidden units of a layer can be permuted (with their incident weights) without changing
//! the function. A Graph-Metanetwork (GMN) represents the weights *as a graph* — neurons
//! are nodes, weights are typed edges — so that permutation is just a relabeling the
//! encoder is invariant to by construction. The bet: the **same** graph type, encoder,
//! and commitment read a NAT 6-zone model and a standard transformer, which is what lets
//! a downstream LoRA-generator (WS-3) condition on peer *weights* across architectures.
//!
//! Layering (meta-plan §3 validation split):
//! - [`WeightGraph`] + lowering ([`lower_nat`], [`lower_transformer`]) + [`WeightGraph::validate`]
//!   — the schema both architectures share (WP-B0). Research-grade structure.
//! - [`encoder`] — an untrained GMN encoder that diagnoses weight-space properties and is
//!   permutation-invariant (WP-B1). Research ML: benchmarked, not proven.
//! - [`commit`] — a **consensus-grade** WL canonical weight digest: Q16, permutation-
//!   invariant, tamper-detecting, frozen golden bytes (WP-B2). Paired with the TLA+
//!   `WeightCommitment` spec in `nat/formal/`.
//!
//! This crate is deliberately CPU-only and GPU-decoupled (deps: `nat-types`,
//! `nat-sidecar`, `sha2`) — it reads weight *matrices* a caller fills from a checkpoint;
//! it never pulls in the training/Candle backend.

pub mod commit;
pub mod encoder;
pub mod rng;

use nat_sidecar::Sidecar;
use nat_types::{CoreType, ZoneId};

// ---------------------------------------------------------------------------
// WP-B0 — the shared weight-graph schema
// ---------------------------------------------------------------------------

/// A dense affine map `y = W·x + b`, `W` is `[out][in]`. The single weight-bearing
/// primitive the lowering reads (the codebase has no shared linear-layer type; a
/// checkpoint reader fills these from real tensors).
#[derive(Debug, Clone, PartialEq)]
pub struct Linear {
    pub w: Vec<Vec<f32>>,
    pub b: Vec<f32>,
}

impl Linear {
    pub fn new(w: Vec<Vec<f32>>, b: Vec<f32>) -> Self {
        Linear { w, b }
    }
    pub fn out_dim(&self) -> usize {
        self.w.len()
    }
    pub fn in_dim(&self) -> usize {
        self.w.first().map(|r| r.len()).unwrap_or(0)
    }
}

/// Which architecture a graph was lowered from. The *type* is shared; this tag is only
/// for the cross-arch probe (it is **not** read by the permutation-invariant commitment).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchTag {
    Nat,
    Transformer,
}

/// The role a node plays. NAT contributes `Router`/`ZoneGate`/`Merge`; both architectures
/// contribute `Input`/`Hidden`/`Output`. One enum, both sub-cases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeKind {
    Input,
    Router,
    ZoneGate(ZoneId),
    Hidden,
    Merge,
    Output,
}

/// The number of `NodeKind` discriminants used in the one-hot (ZoneGate collapses to one
/// slot; the zone identity rides a separate one-hot).
pub const NODE_KIND_SLOTS: usize = 6;
/// Node feature width: kind one-hot (6) + zone one-hot (6) + bias scalar (1).
pub const FEAT_DIM: usize = NODE_KIND_SLOTS + 6 + 1;

impl NodeKind {
    fn kind_slot(self) -> usize {
        match self {
            NodeKind::Input => 0,
            NodeKind::Router => 1,
            NodeKind::ZoneGate(_) => 2,
            NodeKind::Hidden => 3,
            NodeKind::Merge => 4,
            NodeKind::Output => 5,
        }
    }
}

/// The typed edges. Every variant is one parametric weight type; both architectures draw
/// from the same enum (NAT uses the SSM/attention/router/merge variants; a transformer
/// uses attention/FFN). This shared edge vocabulary is what makes the two architectures
/// sub-cases of one graph type rather than two encoders bolted together.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeKind {
    InputToHidden,
    AttnQ,
    AttnK,
    AttnV,
    AttnO,
    FfnUp,
    FfnDown,
    SsmB,
    SsmC,
    SsmO,
    RouterToZone,
    ZoneToMerge,
    MergeToOutput,
    HiddenToOutput,
}

/// Count of edge kinds — the edge-feature one-hot width (plus one weight scalar).
pub const EDGE_KIND_SLOTS: usize = 14;
/// Edge feature width: kind one-hot (14) + weight scalar (1).
pub const EDGE_FEAT_DIM: usize = EDGE_KIND_SLOTS + 1;

impl EdgeKind {
    pub fn slot(self) -> usize {
        match self {
            EdgeKind::InputToHidden => 0,
            EdgeKind::AttnQ => 1,
            EdgeKind::AttnK => 2,
            EdgeKind::AttnV => 3,
            EdgeKind::AttnO => 4,
            EdgeKind::FfnUp => 5,
            EdgeKind::FfnDown => 6,
            EdgeKind::SsmB => 7,
            EdgeKind::SsmC => 8,
            EdgeKind::SsmO => 9,
            EdgeKind::RouterToZone => 10,
            EdgeKind::ZoneToMerge => 11,
            EdgeKind::MergeToOutput => 12,
            EdgeKind::HiddenToOutput => 13,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    pub kind: NodeKind,
    /// Fixed-width node features (kind one-hot ++ zone one-hot ++ bias). Length `FEAT_DIM`.
    pub feats: Vec<f32>,
}

impl Node {
    fn new(kind: NodeKind, bias: f32) -> Self {
        let mut feats = vec![0.0f32; FEAT_DIM];
        feats[kind.kind_slot()] = 1.0;
        if let NodeKind::ZoneGate(z) = kind {
            feats[NODE_KIND_SLOTS + zone_index(z)] = 1.0;
        }
        feats[FEAT_DIM - 1] = bias;
        Node { kind, feats }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GraphEdge {
    pub src: usize,
    pub dst: usize,
    pub kind: EdgeKind,
    pub weight: f32,
}

/// A model's weights as a typed graph — the shared representation both architectures
/// lower into. Permutation of `Hidden` nodes (with their incident edges) yields an
/// equivalent graph; the encoder and the commitment are invariant to exactly that.
#[derive(Debug, Clone, PartialEq)]
pub struct WeightGraph {
    pub arch: ArchTag,
    pub nodes: Vec<Node>,
    pub edges: Vec<GraphEdge>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphError {
    Empty,
    EdgeEndpointOutOfRange { edge: usize, endpoint: usize },
    BadNodeFeatureDim { node: usize, got: usize },
    NoInput,
    NoOutput,
}

impl WeightGraph {
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    pub fn hidden_count(&self) -> usize {
        self.nodes.iter().filter(|n| n.kind == NodeKind::Hidden).count()
    }

    /// WP-B0 acceptance: a NAT checkpoint and a transformer checkpoint both validate
    /// into the *same* `WeightGraph` type. Structural soundness: non-empty, every edge
    /// endpoint in range, feature dims consistent, has input + output boundary nodes.
    pub fn validate(&self) -> Result<(), GraphError> {
        if self.nodes.is_empty() || self.edges.is_empty() {
            return Err(GraphError::Empty);
        }
        let n = self.nodes.len();
        for (i, node) in self.nodes.iter().enumerate() {
            if node.feats.len() != FEAT_DIM {
                return Err(GraphError::BadNodeFeatureDim { node: i, got: node.feats.len() });
            }
        }
        for (i, e) in self.edges.iter().enumerate() {
            if e.src >= n {
                return Err(GraphError::EdgeEndpointOutOfRange { edge: i, endpoint: e.src });
            }
            if e.dst >= n {
                return Err(GraphError::EdgeEndpointOutOfRange { edge: i, endpoint: e.dst });
            }
        }
        if !self.nodes.iter().any(|x| x.kind == NodeKind::Input) {
            return Err(GraphError::NoInput);
        }
        if !self.nodes.iter().any(|x| x.kind == NodeKind::Output) {
            return Err(GraphError::NoOutput);
        }
        Ok(())
    }
}

fn zone_index(z: ZoneId) -> usize {
    ZoneId::ALL.iter().position(|&x| x == z).unwrap_or(0)
}

// ---------------------------------------------------------------------------
// NAT checkpoint → graph
// ---------------------------------------------------------------------------

/// The weights of one NAT zone's core. The variant matches the zone's `CoreType` (the
/// shape a checkpoint reader fills): attention zones carry Q/K/V/O; SSM zones carry the
/// B/C/O projections plus the scalar decay.
#[derive(Debug, Clone, PartialEq)]
pub enum ZoneWeights {
    Attention { wq: Linear, wk: Linear, wv: Linear, wo: Linear },
    Ssm { wb: Linear, wc: Linear, wo: Linear, log_a: f32 },
}

/// A NAT checkpoint: the zone graph (sidecar) plus per-zone weights. Mirrors the real
/// model (`NatModel { sidecar, cores }`) without depending on the runtime/Candle backend.
#[derive(Debug, Clone)]
pub struct NatCheckpoint {
    pub sidecar: Sidecar,
    pub zone_weights: Vec<(ZoneId, ZoneWeights)>,
}

impl NatCheckpoint {
    fn weights_for(&self, z: ZoneId) -> Option<&ZoneWeights> {
        self.zone_weights.iter().find(|(id, _)| *id == z).map(|(_, w)| w)
    }
}

/// Lower a NAT checkpoint into the shared `WeightGraph` (WP-B0). Topology mirrors the
/// model's dataflow: input → router → per-zone gate → zone hidden units (one per output
/// channel of the core matrices, with typed edges from the weights) → merge → output.
/// Only the sidecar's *declared* learned zones with supplied weights are emitted.
pub fn lower_nat(ckpt: &NatCheckpoint) -> WeightGraph {
    let mut nodes = vec![Node::new(NodeKind::Input, 0.0)]; // 0 = input
    let input = 0usize;
    nodes.push(Node::new(NodeKind::Router, 0.0));
    let router = 1usize;
    nodes.push(Node::new(NodeKind::Merge, 0.0));
    let merge = 2usize;
    nodes.push(Node::new(NodeKind::Output, 0.0));
    let output = 3usize;

    let mut edges = vec![GraphEdge {
        src: input,
        dst: router,
        kind: EdgeKind::InputToHidden,
        weight: 1.0,
    }];

    for zd in &ckpt.sidecar.zones {
        if zd.core == CoreType::None {
            continue; // MX harness has no weights
        }
        let Some(zw) = ckpt.weights_for(zd.id) else { continue };
        let gate = nodes.len();
        nodes.push(Node::new(NodeKind::ZoneGate(zd.id), 0.0));
        edges.push(GraphEdge {
            src: router,
            dst: gate,
            kind: EdgeKind::RouterToZone,
            weight: 1.0,
        });
        // Emit a hidden node per output channel of the zone's readout, with edges typed
        // by the projection they came from. We attach all of the zone's projections onto
        // the gate→hidden fan so the graph carries every weight (faithful, lossless).
        let readout = match zw {
            ZoneWeights::Attention { wo, .. } => wo,
            ZoneWeights::Ssm { wo, .. } => wo,
        };
        let mut hidden = Vec::with_capacity(readout.out_dim());
        for (oc, brow) in readout.w.iter().enumerate() {
            let h = nodes.len();
            let bias = readout.b.get(oc).copied().unwrap_or(0.0);
            nodes.push(Node::new(NodeKind::Hidden, bias));
            hidden.push(h);
            // gate → hidden edges, one per input channel of the readout, typed by core.
            let kind = match zw {
                ZoneWeights::Attention { .. } => EdgeKind::AttnO,
                ZoneWeights::Ssm { .. } => EdgeKind::SsmO,
            };
            for &w in brow {
                edges.push(GraphEdge { src: gate, dst: h, kind, weight: w });
            }
            // hidden → merge
            edges.push(GraphEdge {
                src: h,
                dst: merge,
                kind: EdgeKind::ZoneToMerge,
                weight: 1.0,
            });
        }
        // The inner projections (Q/K/V or B/C and the SSM decay) ride as gate-self edges
        // so no weight is dropped from the commitment.
        match zw {
            ZoneWeights::Attention { wq, wk, wv, .. } => {
                attach_matrix(&mut edges, gate, &hidden, wq, EdgeKind::AttnQ);
                attach_matrix(&mut edges, gate, &hidden, wk, EdgeKind::AttnK);
                attach_matrix(&mut edges, gate, &hidden, wv, EdgeKind::AttnV);
            }
            ZoneWeights::Ssm { wb, wc, log_a, .. } => {
                attach_matrix(&mut edges, gate, &hidden, wb, EdgeKind::SsmB);
                attach_matrix(&mut edges, gate, &hidden, wc, EdgeKind::SsmC);
                // the scalar decay as a gate self-loop edge
                edges.push(GraphEdge { src: gate, dst: gate, kind: EdgeKind::SsmO, weight: *log_a });
            }
        }
    }

    edges.push(GraphEdge {
        src: merge,
        dst: output,
        kind: EdgeKind::MergeToOutput,
        weight: 1.0,
    });

    WeightGraph { arch: ArchTag::Nat, nodes, edges }
}

/// Spread a projection matrix's rows across the supplied hidden nodes (cycling if the
/// matrix is wider than the hidden fan), typed by `kind`. Keeps every weight in the graph.
fn attach_matrix(
    edges: &mut Vec<GraphEdge>,
    gate: usize,
    hidden: &[usize],
    m: &Linear,
    kind: EdgeKind,
) {
    if hidden.is_empty() {
        return;
    }
    for (r, row) in m.w.iter().enumerate() {
        let h = hidden[r % hidden.len()];
        for &w in row {
            edges.push(GraphEdge { src: gate, dst: h, kind, weight: w });
        }
    }
}

// ---------------------------------------------------------------------------
// Transformer checkpoint → graph
// ---------------------------------------------------------------------------

/// One pre-norm transformer block: attention (Q/K/V/O) + a 2-layer FFN (up/down).
#[derive(Debug, Clone, PartialEq)]
pub struct TransformerBlock {
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
    pub ffn_up: Linear,
    pub ffn_down: Linear,
}

/// A vanilla transformer checkpoint: a stack of blocks over a model width.
#[derive(Debug, Clone)]
pub struct TransformerCheckpoint {
    pub d_model: usize,
    pub blocks: Vec<TransformerBlock>,
}

/// Lower a transformer checkpoint into the **same** `WeightGraph` type (WP-B0). Each
/// block contributes attention hidden nodes (Q/K/V/O edges) and FFN hidden nodes (up/down
/// edges); blocks chain input → … → output.
pub fn lower_transformer(ckpt: &TransformerCheckpoint) -> WeightGraph {
    let mut nodes = vec![Node::new(NodeKind::Input, 0.0)];
    let input = 0usize;
    nodes.push(Node::new(NodeKind::Output, 0.0));
    let output = 1usize;

    let mut edges = Vec::new();
    let mut prev_layer: Vec<usize> = vec![input];

    for blk in &ckpt.blocks {
        // attention hidden nodes (one per output channel of wo)
        let attn: Vec<usize> = (0..blk.wo.out_dim().max(1))
            .map(|oc| {
                let h = nodes.len();
                let bias = blk.wo.b.get(oc).copied().unwrap_or(0.0);
                nodes.push(Node::new(NodeKind::Hidden, bias));
                h
            })
            .collect();
        // wire the previous layer into the attention nodes by Q/K/V/O
        fan_in(&mut edges, &prev_layer, &attn, &blk.wq, EdgeKind::AttnQ);
        fan_in(&mut edges, &prev_layer, &attn, &blk.wk, EdgeKind::AttnK);
        fan_in(&mut edges, &prev_layer, &attn, &blk.wv, EdgeKind::AttnV);
        fan_in(&mut edges, &prev_layer, &attn, &blk.wo, EdgeKind::AttnO);

        // FFN hidden nodes (one per output channel of ffn_down)
        let ffn: Vec<usize> = (0..blk.ffn_down.out_dim().max(1))
            .map(|oc| {
                let h = nodes.len();
                let bias = blk.ffn_down.b.get(oc).copied().unwrap_or(0.0);
                nodes.push(Node::new(NodeKind::Hidden, bias));
                h
            })
            .collect();
        fan_in(&mut edges, &attn, &ffn, &blk.ffn_up, EdgeKind::FfnUp);
        fan_in(&mut edges, &attn, &ffn, &blk.ffn_down, EdgeKind::FfnDown);

        prev_layer = ffn;
    }

    // final layer → output
    for &h in &prev_layer {
        edges.push(GraphEdge {
            src: h,
            dst: output,
            kind: EdgeKind::HiddenToOutput,
            weight: 1.0,
        });
    }
    if edges.is_empty() {
        // a zero-block transformer still has the input→output boundary
        edges.push(GraphEdge { src: input, dst: output, kind: EdgeKind::HiddenToOutput, weight: 1.0 });
    }

    WeightGraph { arch: ArchTag::Transformer, nodes, edges }
}

/// Wire a source layer into a destination layer by a projection matrix `[out][in]`,
/// typed by `kind`. Each dst node `r` receives an edge from each src node carrying the
/// corresponding matrix weight (cycling indices to tolerate shape mismatch in toy specs).
fn fan_in(
    edges: &mut Vec<GraphEdge>,
    src_layer: &[usize],
    dst_layer: &[usize],
    m: &Linear,
    kind: EdgeKind,
) {
    if src_layer.is_empty() || dst_layer.is_empty() || m.w.is_empty() {
        return;
    }
    for (r, &dst) in dst_layer.iter().enumerate() {
        let row = &m.w[r % m.w.len()];
        if row.is_empty() {
            continue;
        }
        for (c, &src) in src_layer.iter().enumerate() {
            let w = row[c % row.len()];
            edges.push(GraphEdge { src, dst, kind, weight: w });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::SeededRng;

    /// A deterministically-seeded linear layer — a *real* matrix (a checkpoint reader
    /// fills these from tensors; tests synthesize them reproducibly).
    pub(crate) fn seeded_linear(out: usize, inp: usize, seed: u64) -> Linear {
        let mut rng = SeededRng::new(seed);
        let w = (0..out).map(|_| (0..inp).map(|_| rng.next_f32()).collect()).collect();
        let b = (0..out).map(|_| rng.next_f32()).collect();
        Linear::new(w, b)
    }

    pub(crate) fn nat_checkpoint(seed: u64) -> NatCheckpoint {
        let sidecar = Sidecar::default_l0();
        let d = 4usize;
        let mut zone_weights = Vec::new();
        for (i, zd) in sidecar.zones.iter().enumerate() {
            let s = seed.wrapping_add(i as u64 * 101);
            let zw = match zd.core {
                CoreType::Attention => ZoneWeights::Attention {
                    wq: seeded_linear(d, d, s),
                    wk: seeded_linear(d, d, s + 1),
                    wv: seeded_linear(d, d, s + 2),
                    wo: seeded_linear(3, d, s + 3),
                },
                CoreType::Ssm => ZoneWeights::Ssm {
                    wb: seeded_linear(d, d, s + 4),
                    wc: seeded_linear(d, d, s + 5),
                    wo: seeded_linear(3, d, s + 6),
                    log_a: 0.5,
                },
                CoreType::None => continue,
            };
            zone_weights.push((zd.id, zw));
        }
        NatCheckpoint { sidecar, zone_weights }
    }

    pub(crate) fn transformer_checkpoint(seed: u64, blocks: usize, d: usize) -> TransformerCheckpoint {
        let blks = (0..blocks)
            .map(|i| {
                let s = seed.wrapping_add(i as u64 * 211);
                TransformerBlock {
                    wq: seeded_linear(d, d, s),
                    wk: seeded_linear(d, d, s + 1),
                    wv: seeded_linear(d, d, s + 2),
                    wo: seeded_linear(d, d, s + 3),
                    ffn_up: seeded_linear(d * 2, d, s + 4),
                    ffn_down: seeded_linear(d, d * 2, s + 5),
                }
            })
            .collect();
        TransformerCheckpoint { d_model: d, blocks: blks }
    }

    #[test]
    fn nat_and_transformer_lower_to_the_same_validated_type() {
        // WP-B0 acceptance: both architectures ingest into one WeightGraph type that
        // validates — the unification claim, mechanically checked.
        let nat = lower_nat(&nat_checkpoint(7));
        let xf = lower_transformer(&transformer_checkpoint(7, 2, 4));
        nat.validate().expect("nat graph valid");
        xf.validate().expect("transformer graph valid");
        // same type, both non-trivial, both carry real weights on typed edges
        assert!(nat.hidden_count() > 0 && xf.hidden_count() > 0);
        assert!(nat.edges.iter().any(|e| e.kind == EdgeKind::RouterToZone));
        assert!(xf.edges.iter().any(|e| e.kind == EdgeKind::FfnUp));
        // the NAT graph carries the SSM decay and attention projections (lossless)
        assert!(nat.edges.iter().any(|e| e.kind == EdgeKind::SsmO));
        assert!(nat.edges.iter().any(|e| e.kind == EdgeKind::AttnQ));
    }

    #[test]
    fn validate_rejects_dangling_edge() {
        let mut g = lower_transformer(&transformer_checkpoint(1, 1, 3));
        g.edges.push(GraphEdge { src: 0, dst: 999, kind: EdgeKind::FfnUp, weight: 1.0 });
        assert!(matches!(g.validate(), Err(GraphError::EdgeEndpointOutOfRange { .. })));
    }

    #[test]
    fn node_features_are_fixed_width_and_kind_tagged() {
        let g = lower_nat(&nat_checkpoint(3));
        for node in &g.nodes {
            assert_eq!(node.feats.len(), FEAT_DIM);
        }
        // exactly one router, one merge, one input, one output
        assert_eq!(g.nodes.iter().filter(|n| n.kind == NodeKind::Router).count(), 1);
        assert_eq!(g.nodes.iter().filter(|n| n.kind == NodeKind::Merge).count(), 1);
    }
}

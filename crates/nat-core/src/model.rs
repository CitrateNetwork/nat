//! The NAT model: the zone-partitioned forward pass that emits a provenance
//! trace (Architecture §1, §5–8). This is the L0 Gate-2 deliverable — it wires
//! the whole pass end to end and proves the trace emits and validates.

use crate::cores::{CoreFactory, CoreOutput, ToyCores, D_OUT};
use crate::featurize::{class_signals, embed};
use crate::gather::{arrival_status, gather};
use crate::merge::{merge, output_hash, Gathered};
use crate::router::route;
use nat_mcp::{run as mcp_run, McpInput, McpOutcome};
use nat_provenance::{
    sha256_hex, CodecRecord, EdgeRecord, McpRecord, MergeRecord, RouterRecord, ToolCallRecord,
    Trace, ZoneRecord,
};
use nat_sidecar::Sidecar;
use nat_types::{Verification, ZoneId, ZoneStatus, Q16};

/// Activation below this counts as "not participating" for the `activated` flag
/// and for whether an inter-zone edge carries signal.
const ACTIVATION_EPS: f32 = 0.05;

/// A tool request riding alongside a prompt, to exercise the MX harness path.
#[derive(Debug, Clone)]
pub struct ToolRequest {
    pub tool: String,
    pub preconditions_met: bool,
    pub approved: bool,
}

/// The model output (Architecture: token logits / action signal + the hash).
#[derive(Debug, Clone)]
pub struct Output {
    pub logits: [f32; D_OUT],
    pub output_hash: String,
}

/// Everything one forward pass returns.
#[derive(Debug, Clone)]
pub struct ForwardResult {
    pub output: Output,
    pub trace: Trace,
    pub mcp: McpOutcome,
}

/// The model: a sidecar (zone graph), a pluggable core backend, and the L0
/// simulated per-zone latencies.
pub struct NatModel {
    pub sidecar: Sidecar,
    cores: Box<dyn CoreFactory>,
    latencies: Vec<(ZoneId, u64)>,
}

impl NatModel {
    /// The Gate-2 reference model: default L0 sidecar, **toy cores**, default
    /// latencies (all under the default 100ms deadline, so the happy path is
    /// all-`ok`). Toy cores validate the architecture; a real run uses
    /// [`NatModel::with_cores`] with a non-toy backend.
    pub fn l0() -> Self {
        NatModel {
            sidecar: Sidecar::default_l0(),
            cores: Box::new(ToyCores),
            latencies: default_latencies(),
        }
    }

    pub fn with_sidecar(sidecar: Sidecar) -> Self {
        NatModel {
            sidecar,
            cores: Box::new(ToyCores),
            latencies: default_latencies(),
        }
    }

    /// Build a model with a specific core backend (e.g. the Candle backend from
    /// `nat-candle`). This is how a real run swaps the toy L0 cores for trained ones.
    pub fn with_cores(sidecar: Sidecar, cores: Box<dyn CoreFactory>) -> Self {
        NatModel {
            sidecar,
            cores,
            latencies: default_latencies(),
        }
    }

    /// The core backend identifier (recorded in every trace).
    pub fn backend(&self) -> &str {
        self.cores.backend()
    }

    /// Whether this model is running the L0 toy cores. The L1/DGX path asserts
    /// this is false so a real run can never silently fall back to toys.
    pub fn uses_toy_cores(&self) -> bool {
        self.cores.is_toy()
    }

    /// Override a zone's simulated latency (used to force a straggler timeout).
    pub fn set_latency(&mut self, zone: ZoneId, latency_ms: u64) {
        if let Some(entry) = self.latencies.iter_mut().find(|(z, _)| *z == zone) {
            entry.1 = latency_ms;
        }
    }

    /// Run one forward pass. Deterministic given (prompt, tool, config).
    pub fn forward(&self, prompt: &str, tool: Option<ToolRequest>) -> ForwardResult {
        let input_hash = sha256_hex(prompt.as_bytes());
        let signals = class_signals(prompt);
        let embedding = embed(prompt);
        let router = route(signals, &self.sidecar);

        // 1. Run each learned zone's core, in parallel, over its slice.
        let slices: Vec<(ZoneId, Vec<f32>)> = ZoneId::LEARNED
            .iter()
            .map(|&z| (z, slice_for(&self.sidecar, &embedding, z)))
            .collect();

        let factory = self.cores.as_ref();
        let cores: Vec<(ZoneId, CoreOutput)> = std::thread::scope(|scope| {
            let handles: Vec<_> = slices
                .iter()
                .map(|(z, slice)| {
                    let z = *z;
                    let slice = slice.clone();
                    // The backend (toy or Candle) decides the core; the forward
                    // pass is agnostic to which ran.
                    scope.spawn(move || (z, factory.core_for(z).forward(&slice)))
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        // 2. Async gather: classify arrivals against the deadline.
        let arrivals = gather(&self.latencies, self.sidecar.merge.deadline_ms);

        // 3. Build the gathered set (arrived learned zones only).
        let prune_threshold = Q16::from_f32(self.sidecar.merge.prune_threshold);
        let gathered: Vec<Gathered> = ZoneId::LEARNED
            .iter()
            .filter_map(|&z| {
                let arrived = arrivals.iter().find(|a| a.zone == z)?.arrived;
                if !arrived {
                    return None;
                }
                let core = &cores.iter().find(|(id, _)| *id == z)?.1;
                let score = router.activation_of(z) * core.confidence;
                Some(Gathered {
                    zone: z,
                    score: Q16::from_f32(score),
                    summary: core.summary,
                })
            })
            .collect();

        // 4. Merge: score (done) → prune → re-weight → compose (Q16.16 path).
        let merged = merge(&gathered, prune_threshold);
        let out_hash = output_hash(&merged.composed_q16);

        // 5. Codec verification (CX). L0 never synthesizes a Fail; that is a real
        //    verify outcome reserved for L1+. Pass if CX survived a code-ish prompt.
        // L0 yields Pass only when the Codec zone survived on a code-ish prompt;
        // otherwise Unverified. A real `Fail` is an L1+ verify outcome.
        let cx_survived = merged.decision.survivors.contains(&ZoneId::CX);
        let codec_verification = if cx_survived && signals.code > 0.5 {
            Verification::Pass
        } else {
            Verification::Unverified
        };
        let codec = CodecRecord {
            verification: codec_verification,
            artifact_hash: sha256_hex(format!("cx::{input_hash}").as_bytes()),
        };

        // 6. MX harness: gate any tool use on the codec result and the policy.
        let mcp_input = McpInput {
            tool: tool.as_ref().map(|t| t.tool.clone()),
            preconditions_met: tool.as_ref().map(|t| t.preconditions_met).unwrap_or(true),
            approved: tool.as_ref().map(|t| t.approved).unwrap_or(false),
            codec_verified: codec_verification,
            args_hash: tool
                .as_ref()
                .map(|t| sha256_hex(t.tool.as_bytes()))
                .unwrap_or_default(),
        };
        let mcp_outcome = mcp_run(&mcp_input);

        // 7. Assemble the trace.
        let trace = self.build_trace(
            input_hash,
            &router,
            &cores,
            &arrivals,
            &gathered,
            &merged,
            prune_threshold,
            codec,
            &mcp_outcome,
            out_hash.clone(),
        );

        ForwardResult {
            output: Output {
                logits: merged.composed_f32,
                output_hash: out_hash,
            },
            trace,
            mcp: mcp_outcome,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn build_trace(
        &self,
        input_hash: String,
        router: &crate::router::RouterOutput,
        cores: &[(ZoneId, CoreOutput)],
        arrivals: &[crate::gather::ZoneArrival],
        gathered: &[Gathered],
        merged: &crate::merge::MergeOutput,
        prune_threshold: Q16,
        codec: CodecRecord,
        mcp: &McpOutcome,
        output_hash: String,
    ) -> Trace {
        let router_rec = RouterRecord {
            zone_activation: router
                .zone_activation
                .iter()
                .map(|(z, a)| (*z, Q16::from_f32(*a)))
                .collect(),
            edge_modulation: router
                .edge_modulation
                .iter()
                .map(|(from, to, s)| EdgeRecord {
                    from: *from,
                    to: *to,
                    strength: Q16::from_f32(*s),
                })
                .collect(),
        };

        // Per-zone records in canonical order.
        let zones: Vec<ZoneRecord> = ZoneId::ALL
            .iter()
            .map(|&z| {
                let activated = router.activation_of(z) > ACTIVATION_EPS;
                if z == ZoneId::MX {
                    // Non-learned harness: no core, no gather, always reached.
                    return ZoneRecord {
                        id: z,
                        core: nat_types::CoreType::None,
                        activated: true,
                        confidence: Q16::ZERO,
                        latency_ms: 0,
                        status: ZoneStatus::Ok,
                    };
                }
                let confidence = cores
                    .iter()
                    .find(|(id, _)| *id == z)
                    .map(|(_, c)| Q16::from_f32(c.confidence))
                    .unwrap_or(Q16::ZERO);
                let arrival = arrivals.iter().find(|a| a.zone == z);
                let latency_ms = arrival.map(|a| a.latency_ms).unwrap_or(0);
                let status = match arrival {
                    Some(a) if !a.arrived => ZoneStatus::TimedOut,
                    _ if merged.decision.survivors.contains(&z) => ZoneStatus::Ok,
                    _ if gathered.iter().any(|g| g.zone == z) => ZoneStatus::Pruned,
                    _ => arrival.map(arrival_status).unwrap_or(ZoneStatus::Ok),
                };
                ZoneRecord {
                    id: z,
                    core: z.default_core(),
                    activated,
                    confidence,
                    latency_ms,
                    status,
                }
            })
            .collect();

        // Inter-zone flows: only declared edges whose both endpoints activated.
        let inter_zone_flows: Vec<EdgeRecord> = router
            .edge_modulation
            .iter()
            .filter(|(from, to, _)| {
                router.activation_of(*from) > ACTIVATION_EPS
                    && router.activation_of(*to) > ACTIVATION_EPS
            })
            .map(|(from, to, s)| EdgeRecord {
                from: *from,
                to: *to,
                strength: Q16::from_f32(*s),
            })
            .collect();

        let merge_rec = MergeRecord {
            scores: gathered.iter().map(|g| (g.zone, g.score)).collect(),
            prune_threshold,
            survivors: merged.decision.survivors.clone(),
            weights: merged.decision.weights.clone(),
        };

        let mcp_rec = McpRecord {
            state_transitions: mcp
                .transitions
                .iter()
                .map(|s| s.as_str().to_string())
                .collect(),
            tool_calls: mcp
                .tool_call
                .iter()
                .map(|(tool, args_hash, result_status)| ToolCallRecord {
                    tool: tool.clone(),
                    args_hash: args_hash.clone(),
                    result_status: result_status.clone(),
                })
                .collect(),
            refusal: mcp.refusal.clone(),
        };

        Trace {
            input_hash,
            backend: self.cores.backend().to_string(),
            router: router_rec,
            zones,
            inter_zone_flows,
            merge: merge_rec,
            codec,
            mcp: mcp_rec,
            output_hash,
        }
    }
}

/// Default L0 latencies. All under the default 100ms deadline so the happy path
/// gathers every zone; the async-gather test raises PF past the deadline.
fn default_latencies() -> Vec<(ZoneId, u64)> {
    vec![
        (ZoneId::SM, 10),
        (ZoneId::CB, 15),
        (ZoneId::HP, 30),
        (ZoneId::PF, 80),
        (ZoneId::CX, 60),
    ]
}

fn slice_for(sidecar: &Sidecar, embedding: &[f32], zone: ZoneId) -> Vec<f32> {
    let decl = sidecar
        .zones
        .iter()
        .find(|z| z.id == zone)
        .expect("zone declared");
    let start = decl.slice_offset as usize;
    let end = (start + decl.slice_width as usize).min(embedding.len());
    embedding[start..end].to_vec()
}

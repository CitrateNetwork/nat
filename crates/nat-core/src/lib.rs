//! `nat-core` — the zone-partitioned forward pass.
//!
//! Pipeline (Architecture §5–8):
//!
//! ```text
//!   prompt
//!     → featurize (class signals + hidden embedding)
//!     → router    (zone activation + edge modulation over the FIXED topology)
//!     → zones     (parallel cores: SSM for SM/CB, attention for HP/PF/CX)
//!     → gather    (deadline discipline; stragglers → timed_out)
//!     → merge     (score → prune → re-weight → compose, on the Q16.16 path)
//!     → MX        (the non-learned harness gates any tool use)
//!     → (Output, Trace)
//! ```
//!
//! The whole thing emits a [`nat_provenance::Trace`] every pass — that trace is
//! the product (the wedge against opacity), not a debug aside.

pub mod cores;
pub mod featurize;
pub mod gather;
pub mod merge;
pub mod model;
pub mod router;

pub use model::{ForwardResult, NatModel, Output, ToolRequest};

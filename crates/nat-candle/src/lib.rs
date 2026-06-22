//! `nat-candle` — Candle-backed zone cores (CPU), the L1 training stack behind
//! the L0 toy cores (ADR-0009).
//!
//! Two things, both runnable on CPU now and GPU-ready by swapping the device:
//!
//! - [`cores`] — [`CandleSsmCore`] and [`CandleAttentionCore`] implement the same
//!   [`nat_core::cores::ZoneCore`] trait the toy cores do, so they drop in behind
//!   the trait with no change above. The math is real Candle tensor ops (matmul,
//!   softmax), which is the same op graph a GPU run uses.
//! - [`train`] — a tiny trainable head proving forward + autodiff backward +
//!   AdamW all work on Candle, the smoke test that de-risks the L1 training stack.
//!
//! Kept a separate crate so the default workspace build (the fast L0 path) does
//! not pull Candle unless this crate is built.

pub mod cores;
pub mod device;
pub mod factory;
pub mod merge_train;
pub mod seed;
pub mod train;
pub mod trainable;

pub use cores::{CandleAttentionCore, CandleSsmCore};
pub use factory::{candle_model, candle_model_l0, CandleCores};
pub use train::{train_tiny_zone_head, TrainReport};

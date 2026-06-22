//! Compute-device selection — the single source of truth the cores, the trainer,
//! and the backend label all read.
//!
//! Default builds run on CPU. Building with the `cuda` feature (the DGX / L1 path,
//! §5.1 of `docs/DGX_HANDOFF.md`) moves the *same op graph* to the GPU by swapping
//! the device here — nothing in the cores changes, because the math is already
//! real Candle tensor ops.
//!
//! The label is **honest by construction**: it is derived from the device that was
//! actually built, never asserted independently. So `trace.backend` records
//! `"candle-cuda"` only when a CUDA device truly came up, and falls back to
//! `"candle-cpu"` if the `cuda` feature is on but no GPU/driver is present — the
//! trace can never claim a GPU run that did not happen (the §4 "record reality"
//! guarantee, extended from toy-vs-real to cpu-vs-cuda).

use candle_core::Device;

/// The Candle device this build runs on.
///
/// With the `cuda` feature: the first CUDA device if one is available, else CPU
/// (fail-honest — see the module note on the label). Without it: always CPU.
#[cfg(feature = "cuda")]
pub fn device() -> Device {
    // `cuda_if_available` returns CPU if the feature compiled but no GPU/driver is
    // present, so the label below stays truthful instead of claiming a GPU run.
    Device::cuda_if_available(0).unwrap_or(Device::Cpu)
}

/// The Candle device this build runs on (CPU; the `cuda` feature is off).
#[cfg(not(feature = "cuda"))]
pub fn device() -> Device {
    Device::Cpu
}

/// The backend label recorded in the provenance trace, derived from the device
/// that actually came up: `"candle-cuda"` for a live GPU, `"candle-cpu"` otherwise.
pub fn backend_label() -> &'static str {
    if device().is_cuda() {
        "candle-cuda"
    } else {
        "candle-cpu"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn label_matches_the_actual_device() {
        // The invariant: the label is never independent of the device. Whatever
        // `device()` resolves to, the label reflects it — so the trace cannot lie.
        let on_gpu = device().is_cuda();
        assert_eq!(
            backend_label(),
            if on_gpu { "candle-cuda" } else { "candle-cpu" }
        );
        assert!(backend_label().starts_with("candle-"));
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn cpu_build_is_never_cuda() {
        assert!(!device().is_cuda());
        assert_eq!(backend_label(), "candle-cpu");
    }
}

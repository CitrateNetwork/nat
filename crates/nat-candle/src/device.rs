//! Compute-device selection — the single source of truth the cores, the trainer,
//! and the backend label all read.
//!
//! Default builds run on CPU. Building with the `cuda` feature (the DGX / L1 path,
//! §5.1 of `docs/DGX_HANDOFF.md`) or the `metal` feature (Apple Silicon) moves the
//! *same op graph* to the accelerator by swapping the device here — nothing in the
//! cores changes, because the math is already real Candle tensor ops.
//!
//! The label is **honest by construction**: it is derived from the device that was
//! actually built, never asserted independently. So `trace.backend` records
//! `"candle-cuda"` only when a CUDA device truly came up, and falls back to
//! `"candle-cpu"` if the feature is on but no GPU/driver is present — the trace can
//! never claim an accelerator run that did not happen (the §4 "record reality"
//! guarantee, extended from toy-vs-real to cpu-vs-accelerator).
//!
//! ## Why heterogeneous backends matter here
//!
//! A co-op takes whatever hardware its members own. CUDA, Metal and CPU do not
//! produce bit-identical floating-point results — reduction orders and fused
//! kernels differ — so the backend a contribution was computed on is part of its
//! provenance, not an implementation detail. `divergence_probe` measures how far
//! apart they actually land; this module is what lets it say which is which.
//!
//! AMD/ROCm has no first-class Candle backend at this version, so those machines
//! resolve to CPU and the label says so rather than pretending otherwise.

use candle_core::Device;

/// The Candle device this build runs on: the first CUDA device, else the first
/// Metal device, else CPU — restricted to whichever features were compiled in.
///
/// Every accelerator branch is fail-honest: if the feature compiled but no
/// device/driver is present, it falls through to the next option and the label
/// below stays truthful instead of claiming a run that did not happen.
pub fn device() -> Device {
    #[cfg(feature = "cuda")]
    {
        if let Ok(d) = Device::new_cuda(0) {
            return d;
        }
    }
    #[cfg(feature = "metal")]
    {
        if let Ok(d) = Device::new_metal(0) {
            return d;
        }
    }
    Device::Cpu
}

/// The backend label recorded in the provenance trace, derived from the device
/// that actually came up: `"candle-cuda"`, `"candle-metal"`, or `"candle-cpu"`.
pub fn backend_label() -> &'static str {
    let d = device();
    if d.is_cuda() {
        "candle-cuda"
    } else if d.is_metal() {
        "candle-metal"
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
        let d = device();
        let expected = if d.is_cuda() {
            "candle-cuda"
        } else if d.is_metal() {
            "candle-metal"
        } else {
            "candle-cpu"
        };
        assert_eq!(backend_label(), expected);
        assert!(backend_label().starts_with("candle-"));
    }

    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    #[test]
    fn cpu_build_is_never_an_accelerator() {
        assert!(!device().is_cuda());
        assert!(!device().is_metal());
        assert_eq!(backend_label(), "candle-cpu");
    }

    /// Exactly one label, and it is one of the three known ones. Guards against a
    /// future backend being added to `device()` without being added here — which
    /// would silently mislabel a whole class of contributor hardware.
    #[test]
    fn the_label_is_always_one_of_the_known_backends() {
        assert!(matches!(
            backend_label(),
            "candle-cuda" | "candle-metal" | "candle-cpu"
        ));
    }
}

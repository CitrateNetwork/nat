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

thread_local! {
    /// The device for *this thread*, resolved once per thread.
    ///
    /// The scope here — one per thread, not one per call and not one per process — is
    /// forced from both sides by two independent Metal constraints. Apple Silicon is
    /// the only platform that exhibits either, which is why this survived until the
    /// first Mac run.
    ///
    /// ## Why not one per call (the old behaviour)
    ///
    /// Candle's notion of "the same device" is per-*instantiation*, not per-*GPU*.
    /// `MetalDevice` carries an `id` minted from a global counter, and candle's own
    /// comment on that field says the registryID "is not sufficient as it identifies
    /// the GPU rather than the device itself". `Device::same_device` compares that
    /// counter, so two calls to `Device::new_metal(0)` yield devices candle considers
    /// *different* despite naming one physical GPU, and any op across them fails:
    ///
    /// ```text
    /// device mismatch in index-select,
    ///   lhs: Metal { gpu_id: 4294968237 }, rhs: Metal { gpu_id: 4294968237 }
    /// ```
    ///
    /// The two ids in that message are *identical*, because the comparison uses the
    /// internal counter while `DeviceLocation` prints `registry_id()`. The error can
    /// never show you the field it actually compared — hence this comment rather than
    /// a bare one-line fix. On CPU the whole class is invisible (`Device::Cpu` always
    /// equals itself), so this cost 12 of 46 `--features metal` tests and every
    /// accelerated example, while CI stayed green.
    ///
    /// ## Why not one per process either
    ///
    /// The obvious repair — a `OnceLock` process-wide singleton — trades this bug for
    /// a worse one. The zone cores run concurrently, and candle 0.8.4's Metal backend
    /// cannot take concurrent encoding against one device: sharing it across threads
    /// aborts the process inside Metal itself, which no Rust-level error handling can
    /// catch.
    ///
    /// ```text
    /// failed assertion `A command encoder is already encoding to this command buffer'
    /// failed assertion _status < MTLCommandBufferStatusCommitted   (SIGABRT)
    /// ```
    ///
    /// One device per thread satisfies both: identity is stable everywhere a tensor
    /// and its operands are actually built, and each thread encodes to its own command
    /// buffer.
    ///
    /// ## The constraint this imposes
    ///
    /// Tensors must not be *combined* across threads on Metal — each thread's tensors
    /// live on that thread's device. That is not a limitation this module invents; it
    /// is candle 0.8.4's Metal backend, which the process-wide experiment above shows
    /// cannot be shared regardless. The zone cores already respect it (they exchange
    /// gathered results, not live device tensors), which is why the full suite passes.
    ///
    /// CUDA and CPU are unaffected in behaviour and strictly better off in cost: this
    /// creates *fewer* devices than the per-call version it replaces.
    static DEVICE_TL: Device = resolve_device();
}

/// The Candle device this build runs on: the first CUDA device, else the first
/// Metal device, else CPU — restricted to whichever features were compiled in.
///
/// Every accelerator branch is fail-honest: if the feature compiled but no
/// device/driver is present, it falls through to the next option and the label
/// below stays truthful instead of claiming a run that did not happen.
///
/// The returned `Device` is a clone of this thread's device, so tensors built
/// from separate calls land somewhere candle agrees is one device. Cloning is
/// cheap (the heavy state is behind `Arc`) and, critically, preserves the identity
/// field — a clone is `same_device` as its original, a fresh construction is not.
pub fn device() -> Device {
    DEVICE_TL.with(|d| d.clone())
}

/// The one-time resolution. Separated from `device()` so the ordering — CUDA,
/// then Metal, then CPU — stays readable, and so the fallthrough is exercised
/// exactly once per process rather than on every tensor allocation.
fn resolve_device() -> Device {
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

    /// Two calls to `device()` must give devices candle agrees are the same one.
    ///
    /// This is the invariant the first Apple Silicon run broke. `Device::new_metal`
    /// mints a fresh identity per call, so a per-call `device()` returns devices
    /// that fail `same_device` despite naming one physical GPU — and every binary
    /// op between tensors from different calls dies with a "device mismatch" whose
    /// two printed ids are identical (the message prints the registry id; the
    /// comparison uses the internal counter).
    ///
    /// It passes trivially on CPU, which is exactly why it has to exist: the CPU
    /// build cannot fail it, so without it the accelerator builds go unguarded.
    #[test]
    fn device_is_the_same_device_across_calls() {
        assert!(
            device().same_device(&device()),
            "device() returned two devices candle considers different — \
             tensors from separate calls cannot be combined"
        );
    }

    /// The same invariant one level up: a tensor made on one `device()` handle must
    /// be usable with a tensor made on another. This is the operation that actually
    /// failed (`index-select`, the embedding lookup in the first forward pass), so
    /// it is asserted directly rather than only through `same_device`.
    #[test]
    fn tensors_from_separate_device_calls_can_be_combined() {
        use candle_core::Tensor;
        let a = Tensor::from_vec(vec![1.0f32, 2.0], (2,), &device()).unwrap();
        let b = Tensor::from_vec(vec![3.0f32, 4.0], (2,), &device()).unwrap();
        let sum = (&a + &b).unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(sum, vec![4.0, 6.0]);
    }

    /// Concurrent threads must each be able to do device work.
    ///
    /// The zone cores run in parallel, and candle 0.8.4's Metal backend aborts the
    /// *process* if two threads encode to one device's command buffer. So this
    /// guards the repair that a process-wide `OnceLock` singleton would silently
    /// undo — the natural "obvious" refactor of this module.
    ///
    /// A regression here does not fail politely: it SIGABRTs inside Metal, which is
    /// exactly why it needs a test rather than a comment. Assertions cannot catch
    /// an abort; the test's job is to *reach* the abort under CI rather than
    /// leaving a contributor to discover it in a training run.
    #[test]
    fn concurrent_threads_can_each_use_the_device() {
        use candle_core::Tensor;
        let handles: Vec<_> = (0..4)
            .map(|i| {
                std::thread::spawn(move || {
                    let d = device();
                    let t = Tensor::from_vec(vec![i as f32; 8], (8,), &d).unwrap();
                    (&t * 2.0).unwrap().to_vec1::<f32>().unwrap()
                })
            })
            .collect();
        for (i, h) in handles.into_iter().enumerate() {
            assert_eq!(h.join().unwrap(), vec![i as f32 * 2.0; 8]);
        }
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

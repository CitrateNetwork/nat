//! The training-stack smoke test (de-risks critique #6).
//!
//! A tiny linear "zone head" is trained on Candle to fit a fixed synthetic
//! linear target. If forward + backward + the optimizer all work here on CPU,
//! the same machinery (VarMap, autodiff, AdamW) scales to real zones on a GPU
//! device at L1 — the point of choosing a real tensor framework now rather than
//! discovering its rough edges during the expensive run.

use candle_core::{DType, Device, Tensor};
use candle_nn::optim::{AdamW, ParamsAdamW};
use candle_nn::{linear, loss, Module, Optimizer, VarBuilder, VarMap};

#[derive(Debug, Clone, Copy)]
pub struct TrainReport {
    pub initial_loss: f32,
    pub final_loss: f32,
    pub steps: usize,
}

impl TrainReport {
    /// Did training reduce the loss at all?
    pub fn converged(&self) -> bool {
        self.final_loss < self.initial_loss
    }
}

/// Train a tiny linear head to fit a fixed synthetic linear target, returning the
/// before/after loss. Deterministic inputs and target; the head's initial weights
/// come from Candle's initializer (so the absolute losses vary run to run, but the
/// reduction is reliable — that is what we assert).
pub fn train_tiny_zone_head(
    in_dim: usize,
    out_dim: usize,
    steps: usize,
) -> candle_core::Result<TrainReport> {
    let dev = Device::Cpu;
    let n = 64usize;

    // Deterministic synthetic inputs.
    let xv: Vec<f32> = (0..n * in_dim)
        .map(|i| ((i % 13) as f32 - 6.0) / 6.0)
        .collect();
    let x = Tensor::from_vec(xv, (n, in_dim), &dev)?;

    // The fixed "true" affine map the head must learn: target = x·W_true + b_true.
    let wv: Vec<f32> = (0..in_dim * out_dim)
        .map(|i| ((i % 5) as f32 - 2.0) / 4.0)
        .collect();
    let w_true = Tensor::from_vec(wv, (in_dim, out_dim), &dev)?;
    let bv: Vec<f32> = (0..out_dim).map(|j| (j as f32) * 0.1 - 0.2).collect();
    let b_true = Tensor::from_vec(bv, (1, out_dim), &dev)?;
    let target = x.matmul(&w_true)?.broadcast_add(&b_true)?;

    // The trainable head.
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let head = linear(in_dim, out_dim, vb.pp("head"))?;

    let mut opt = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr: 0.05,
            ..Default::default()
        },
    )?;

    let initial_loss = loss::mse(&head.forward(&x)?, &target)?.to_scalar::<f32>()?;
    let mut final_loss = initial_loss;
    for _ in 0..steps {
        let pred = head.forward(&x)?;
        let l = loss::mse(&pred, &target)?;
        opt.backward_step(&l)?; // backward + parameter update
        final_loss = l.to_scalar::<f32>()?;
    }

    Ok(TrainReport {
        initial_loss,
        final_loss,
        steps,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn training_reduces_loss_toward_zero() {
        // A linear head fitting a linear target should drive the loss near zero —
        // proving forward, autodiff backward, and the AdamW step all work.
        let r = train_tiny_zone_head(8, 4, 400).unwrap();
        assert!(r.converged(), "loss did not decrease: {r:?}");
        assert!(
            r.final_loss < r.initial_loss * 0.2,
            "loss barely moved: {r:?}"
        );
        assert!(r.final_loss < 0.05, "did not converge near zero: {r:?}");
    }
}

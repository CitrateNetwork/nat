//! The two arms of the H-01 ablation, as small trainable Candle models.
//!
//! - [`PartitionedArm`] — the NAT-shaped arm: the first layer is split into
//!   `n_zones` independent blocks (each zone projects the input on its own), then
//!   a head merges the concatenated zone outputs. This is the structural analog of
//!   zone partitioning.
//! - [`DenseArm`] — the baseline: a fully-connected trunk of the same depth.
//!
//! At L1/DGX these scale up (more zones, real widths, real data) and the
//! partitioned arm becomes the full `NatModel` with Candle cores. The harness
//! does not change — only the models under it do.

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::optim::{AdamW, ParamsAdamW};
use candle_nn::{linear, loss, Linear, Module, Optimizer, VarBuilder, VarMap};

/// A fixed synthetic regression task both arms train on (same data, same target).
pub struct TrainData {
    pub x: Tensor,
    pub y: Tensor,
}

/// Deterministic synthetic data: `y = sin-free smooth linear-ish map of x`, so a
/// small MLP can fit it and the comparison reflects structure, not luck.
pub fn synthetic_data(n: usize, in_dim: usize, out_dim: usize, dev: &Device) -> Result<TrainData> {
    let xv: Vec<f32> = (0..n * in_dim)
        .map(|i| ((i % 17) as f32 - 8.0) / 8.0)
        .collect();
    let x = Tensor::from_vec(xv, (n, in_dim), dev)?;
    // Fixed affine-plus-mild-nonlinear target.
    let wv: Vec<f32> = (0..in_dim * out_dim)
        .map(|i| ((i % 7) as f32 - 3.0) / 5.0)
        .collect();
    let w = Tensor::from_vec(wv, (in_dim, out_dim), dev)?;
    let y = x.matmul(&w)?.tanh()?; // bounded, learnable
    Ok(TrainData { x, y })
}

/// What an ablation arm must provide: a parameter count, a name, and a train step.
pub trait AblationArm {
    fn param_count(&self) -> usize;
    fn name(&self) -> &str;
    /// Train on the data for `steps`, returning `(initial_loss, final_loss)`.
    fn train(&mut self, data: &TrainData, steps: usize, lr: f64) -> Result<(f32, f32)>;
}

fn count_params(varmap: &VarMap) -> usize {
    varmap
        .all_vars()
        .iter()
        .map(|v| v.as_tensor().elem_count())
        .sum()
}

/// Analytic parameter count for the partitioned arm — used to size it to match
/// the dense arm before any tensors are built (ADR-0005 equal-params).
pub fn partitioned_params(
    in_dim: usize,
    n_zones: usize,
    zone_hidden: usize,
    out_dim: usize,
) -> usize {
    let merged = n_zones * zone_hidden;
    n_zones * (in_dim * zone_hidden + zone_hidden) + (merged * out_dim + out_dim)
}

/// Analytic parameter count for the dense arm.
pub fn dense_params(in_dim: usize, hidden: usize, out_dim: usize) -> usize {
    in_dim * hidden + hidden + hidden * out_dim + out_dim
}

/// The NAT-shaped arm: `n_zones` independent input projections → concat → head.
pub struct PartitionedArm {
    varmap: VarMap,
    zones: Vec<Linear>,
    head: Linear,
    params: usize,
}

impl PartitionedArm {
    pub fn new(
        in_dim: usize,
        n_zones: usize,
        zone_hidden: usize,
        out_dim: usize,
        dev: &Device,
    ) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, dev);
        let mut zones = Vec::with_capacity(n_zones);
        for i in 0..n_zones {
            zones.push(linear(in_dim, zone_hidden, vb.pp(format!("zone{i}")))?);
        }
        let head = linear(n_zones * zone_hidden, out_dim, vb.pp("head"))?;
        let params = count_params(&varmap);
        Ok(PartitionedArm {
            varmap,
            zones,
            head,
            params,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let zone_outs: Vec<Tensor> = self
            .zones
            .iter()
            .map(|z| z.forward(x)?.relu())
            .collect::<Result<_>>()?;
        let merged = Tensor::cat(&zone_outs, 1)?; // (n, n_zones*zone_hidden)
        self.head.forward(&merged)
    }
}

impl AblationArm for PartitionedArm {
    fn param_count(&self) -> usize {
        self.params
    }
    fn name(&self) -> &str {
        "partitioned"
    }
    fn train(&mut self, data: &TrainData, steps: usize, lr: f64) -> Result<(f32, f32)> {
        let mut opt = AdamW::new(
            self.varmap.all_vars(),
            ParamsAdamW {
                lr,
                ..Default::default()
            },
        )?;
        let initial = loss::mse(&self.forward(&data.x)?, &data.y)?.to_scalar::<f32>()?;
        let mut final_loss = initial;
        for _ in 0..steps {
            let l = loss::mse(&self.forward(&data.x)?, &data.y)?;
            opt.backward_step(&l)?;
            final_loss = l.to_scalar::<f32>()?;
        }
        Ok((initial, final_loss))
    }
}

/// The dense baseline arm: a fully-connected two-layer trunk.
pub struct DenseArm {
    varmap: VarMap,
    l1: Linear,
    l2: Linear,
    params: usize,
}

impl DenseArm {
    pub fn new(in_dim: usize, hidden: usize, out_dim: usize, dev: &Device) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, dev);
        let l1 = linear(in_dim, hidden, vb.pp("l1"))?;
        let l2 = linear(hidden, out_dim, vb.pp("l2"))?;
        let params = count_params(&varmap);
        Ok(DenseArm {
            varmap,
            l1,
            l2,
            params,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.l2.forward(&self.l1.forward(x)?.relu()?)
    }
}

impl AblationArm for DenseArm {
    fn param_count(&self) -> usize {
        self.params
    }
    fn name(&self) -> &str {
        "dense"
    }
    fn train(&mut self, data: &TrainData, steps: usize, lr: f64) -> Result<(f32, f32)> {
        let mut opt = AdamW::new(
            self.varmap.all_vars(),
            ParamsAdamW {
                lr,
                ..Default::default()
            },
        )?;
        let initial = loss::mse(&self.forward(&data.x)?, &data.y)?.to_scalar::<f32>()?;
        let mut final_loss = initial;
        for _ in 0..steps {
            let l = loss::mse(&self.forward(&data.x)?, &data.y)?;
            opt.backward_step(&l)?;
            final_loss = l.to_scalar::<f32>()?;
        }
        Ok((initial, final_loss))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analytic_param_counts_match_the_built_models() {
        let dev = Device::Cpu;
        let dense = DenseArm::new(8, 16, 4, &dev).unwrap();
        assert_eq!(dense.param_count(), dense_params(8, 16, 4));
        let part = PartitionedArm::new(8, 5, 4, 4, &dev).unwrap();
        assert_eq!(part.param_count(), partitioned_params(8, 5, 4, 4));
    }

    #[test]
    fn both_arms_train_and_reduce_loss() {
        let dev = Device::Cpu;
        let data = synthetic_data(48, 8, 4, &dev).unwrap();
        let mut dense = DenseArm::new(8, 16, 4, &dev).unwrap();
        let (di, df) = dense.train(&data, 150, 0.05).unwrap();
        assert!(df < di, "dense did not learn: {di} -> {df}");
        let mut part = PartitionedArm::new(8, 5, 4, 4, &dev).unwrap();
        let (pi, pf) = part.train(&data, 150, 0.05).unwrap();
        assert!(pf < pi, "partitioned did not learn: {pi} -> {pf}");
    }
}

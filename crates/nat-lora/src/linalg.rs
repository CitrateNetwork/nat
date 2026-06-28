//! Minimal dense linear algebra for the generator's meta-training (ridge regression) —
//! small systems only (latent-dim square solves). Research/f32 layer; never on the
//! committed path.

/// `m · x` for `m` `[r][c]` and `x` `[c]`.
pub fn matvec(m: &[Vec<f32>], x: &[f32]) -> Vec<f32> {
    m.iter().map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum()).collect()
}

/// Transpose an `[r][c]` matrix to `[c][r]`.
pub fn transpose(m: &[Vec<f32>]) -> Vec<Vec<f32>> {
    if m.is_empty() {
        return Vec::new();
    }
    let (r, c) = (m.len(), m[0].len());
    (0..c).map(|j| (0..r).map(|i| m[i][j]).collect()).collect()
}

/// `a · b` for `a` `[n][k]`, `b` `[k][p]` → `[n][p]`.
pub fn matmul(a: &[Vec<f32>], b: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let k = b.len();
    let p = if k > 0 { b[0].len() } else { 0 };
    a.iter()
        .map(|arow| {
            (0..p)
                .map(|j| (0..k).map(|t| arow[t] * b[t][j]).sum())
                .collect()
        })
        .collect()
}

/// Invert a square matrix by Gauss-Jordan with partial pivoting. Returns `None` if
/// singular. `n` is small (the augmented latent dimension).
pub fn invert(m: &[Vec<f32>]) -> Option<Vec<Vec<f32>>> {
    let n = m.len();
    let mut a: Vec<Vec<f64>> = m.iter().map(|r| r.iter().map(|&x| x as f64).collect()).collect();
    let mut inv: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect();

    for col in 0..n {
        // partial pivot
        let mut piv = col;
        let mut best = a[col][col].abs();
        for r in (col + 1)..n {
            if a[r][col].abs() > best {
                best = a[r][col].abs();
                piv = r;
            }
        }
        if best < 1e-12 {
            return None;
        }
        a.swap(col, piv);
        inv.swap(col, piv);

        let d = a[col][col];
        for j in 0..n {
            a[col][j] /= d;
            inv[col][j] /= d;
        }
        for r in 0..n {
            if r == col {
                continue;
            }
            let f = a[r][col];
            if f == 0.0 {
                continue;
            }
            for j in 0..n {
                a[r][j] -= f * a[col][j];
                inv[r][j] -= f * inv[col][j];
            }
        }
    }
    Some(inv.iter().map(|r| r.iter().map(|&x| x as f32).collect()).collect())
}

/// Ridge regression. Given inputs `z` `[n][p]` (already including any bias column) and
/// targets `y` `[n][k]`, solve `min_M ||Z M - Y||² + λ||M||²` and return `M` as `[k][p]`
/// (so `ŷ = M · z`). Closed-form `M = (ZᵀZ + λI)⁻¹ Zᵀ Y`, transposed.
pub fn ridge_fit(z: &[Vec<f32>], y: &[Vec<f32>], lambda: f32) -> Vec<Vec<f32>> {
    let zt = transpose(z);
    let mut ztz = matmul(&zt, z); // [p][p]
    for (i, row) in ztz.iter_mut().enumerate() {
        row[i] += lambda;
    }
    let zty = matmul(&zt, y); // [p][k]
    let inv = invert(&ztz).expect("ridge normal matrix is invertible with λ>0");
    let m_pk = matmul(&inv, &zty); // [p][k]
    transpose(&m_pk) // [k][p]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invert_roundtrips() {
        let m = vec![vec![4.0, 7.0], vec![2.0, 6.0]];
        let inv = invert(&m).expect("invertible");
        let prod = matmul(&m, &inv);
        assert!((prod[0][0] - 1.0).abs() < 1e-4 && (prod[1][1] - 1.0).abs() < 1e-4);
        assert!(prod[0][1].abs() < 1e-4 && prod[1][0].abs() < 1e-4);
    }

    #[test]
    fn ridge_recovers_a_linear_map() {
        // y = 3 x0 - 2 x1 ; fit with a bias column should recover [≈3, ≈-2, ≈0].
        let z = vec![
            vec![1.0, 0.0, 1.0],
            vec![0.0, 1.0, 1.0],
            vec![1.0, 1.0, 1.0],
            vec![2.0, -1.0, 1.0],
        ];
        let y: Vec<Vec<f32>> = z.iter().map(|r| vec![3.0 * r[0] - 2.0 * r[1]]).collect();
        let m = ridge_fit(&z, &y, 1e-6);
        assert!((m[0][0] - 3.0).abs() < 1e-2, "{:?}", m);
        assert!((m[0][1] + 2.0).abs() < 1e-2);
        assert!(m[0][2].abs() < 1e-2);
    }
}

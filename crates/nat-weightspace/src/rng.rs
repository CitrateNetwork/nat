//! A tiny deterministic PRNG (splitmix64) — for seeded fixed projections in the encoder
//! and for synthesizing reproducible weight matrices in tests. Never on the committed
//! path (that path is Q16-exact), so float reproducibility within a process is all that
//! is required.

/// splitmix64 — a fast, well-distributed, fully deterministic generator.
#[derive(Debug, Clone)]
pub struct SeededRng {
    state: u64,
}

impl SeededRng {
    pub fn new(seed: u64) -> Self {
        // avoid the all-zero fixed point
        SeededRng { state: seed ^ 0x9E37_79B9_7F4A_7C15 }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[-1, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        let u = self.next_u64() >> 11; // 53 significant bits
        let unit = u as f64 / (1u64 << 53) as f64; // [0,1)
        (unit * 2.0 - 1.0) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_in_range() {
        let mut a = SeededRng::new(42);
        let mut b = SeededRng::new(42);
        for _ in 0..1000 {
            let x = a.next_f32();
            assert_eq!(x, b.next_f32());
            assert!((-1.0..1.0).contains(&x));
        }
        // a different seed yields a different stream
        let mut c = SeededRng::new(43);
        assert_ne!(SeededRng::new(42).next_u64(), c.next_u64());
    }
}

/// Tiny xorshift32 PRNG to avoid external dependencies.
///
/// Not cryptographically secure; intended only for initializing weights.
#[derive(Debug, Clone)]
pub struct XorShift32 {
    state: u32,
}

impl XorShift32 {
    pub fn new(seed: u32) -> Self {
        let seed = if seed == 0 { 0x6d2b_79f5 } else { seed };
        Self { state: seed }
    }

    pub fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.state = x;
        x
    }

    /// Uniform float in [0,1).
    pub fn next_f32(&mut self) -> f32 {
        // 24-bit mantissa
        let v = self.next_u32() >> 8;
        (v as f32) / ((1u32 << 24) as f32)
    }

    /// Uniform float in [low, high).
    pub fn gen_range_f32(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * self.next_f32()
    }
}

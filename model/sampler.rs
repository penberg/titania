/// Picks the next token from a model's logits: from the `top_k` most likely
/// tokens, then the smallest set of those whose probability reaches `top_p`,
/// at `temperature`. A temperature of zero always picks the most likely token.
pub struct Sampler {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    rng: Rng,
}

impl Sampler {
    pub fn new(temperature: f32, top_k: usize, top_p: f32, seed: u64) -> Self {
        Self {
            temperature,
            top_k,
            top_p,
            rng: Rng(seed.max(1)),
        }
    }

    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        let mut candidates: Vec<(u32, f32)> = logits
            .iter()
            .enumerate()
            .map(|(token, &logit)| (token as u32, logit))
            .collect();
        let k = self.top_k.clamp(1, candidates.len());
        candidates.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1));
        candidates.truncate(k);
        candidates.sort_by(|a, b| b.1.total_cmp(&a.1));
        if self.temperature == 0.0 {
            return candidates[0].0;
        }

        let max = candidates[0].1;
        let mut probs: Vec<f32> = candidates
            .iter()
            .map(|&(_, logit)| ((logit - max) / self.temperature).exp())
            .collect();
        let sum: f32 = probs.iter().sum();
        let mut cumulative = 0.0;
        let mut n = 0;
        for p in probs.iter_mut() {
            *p /= sum;
            if cumulative < self.top_p {
                cumulative += *p;
                n += 1;
            }
        }

        let mut r = self.rng.next_f32() * cumulative;
        for (&(token, _), &p) in candidates.iter().zip(&probs[..n]) {
            r -= p;
            if r <= 0.0 {
                return token;
            }
        }
        candidates[n - 1].0
    }
}

/// xorshift64*: a small, fast pseudorandom number generator.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// Uniform in `[0, 1)`.
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

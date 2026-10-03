//! Next-code selection, in the order Hugging Face `generate` applies it for
//! XTTS: repetition penalty over every id seen so far (the prompt ids
//! included), then — when sampling — temperature, top-k, top-p, and a draw
//! from the softmax. Without sampling it is the arg-max after the penalty.

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

#[derive(Debug, Clone)]
pub struct SamplingOptions {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
    /// `false`: greedy (arg-max after the repetition penalty).
    pub do_sample: bool,
    pub seed: Option<u64>,
    /// End the sequence as soon as the stop code's probability (softmax of
    /// the penalized logits) reaches this value, even if it is not drawn;
    /// 0 disables. XTTS often keeps babbling after the last word when the
    /// stop code is likely but not sampled.
    pub stop_prob: f32,
}

impl Default for SamplingOptions {
    fn default() -> Self {
        // XTTS-v2's config.json.
        Self {
            temperature: 0.75,
            top_k: 50,
            top_p: 0.85,
            repetition_penalty: 5.0,
            do_sample: true,
            seed: None,
            stop_prob: 0.0,
        }
    }
}

pub struct Sampler {
    opts: SamplingOptions,
    stop: Option<u32>,
    seen: Vec<bool>,
    rng: StdRng,
}

impl Sampler {
    /// `prompt_ids`: the ids `generate` sees before the first new code.
    pub fn new(opts: SamplingOptions, vocab: usize, prompt_ids: &[u32]) -> Self {
        let mut seen = vec![false; vocab];
        for &id in prompt_ids {
            if (id as usize) < vocab {
                seen[id as usize] = true;
            }
        }
        let rng = match opts.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_os_rng(),
        };
        Self {
            opts,
            stop: None,
            seen,
            rng,
        }
    }

    /// The code [`SamplingOptions::stop_prob`] watches.
    pub fn with_stop(mut self, stop: u32) -> Self {
        self.stop = Some(stop);
        self
    }

    pub fn next(&mut self, logits: &[f32]) -> u32 {
        let mut scores = logits.to_vec();
        let p = self.opts.repetition_penalty;
        if p != 1.0 {
            for (s, &seen) in scores.iter_mut().zip(&self.seen) {
                if seen {
                    *s = if *s < 0.0 { *s * p } else { *s / p };
                }
            }
        }
        if let Some(stop) = self.stop.filter(|_| self.opts.stop_prob > 0.0) {
            let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = scores.iter().map(|s| (s - max).exp()).sum();
            let p_stop = (scores[stop as usize] - max).exp() / sum;
            tracing::trace!("p(stop) = {p_stop:.4}");
            if p_stop >= self.opts.stop_prob {
                return stop;
            }
        }
        let id = if self.opts.do_sample {
            self.sample(&mut scores)
        } else {
            argmax(&scores)
        };
        self.seen[id as usize] = true;
        id
    }

    fn sample(&mut self, scores: &mut [f32]) -> u32 {
        let t = self.opts.temperature.max(1e-5);
        for s in scores.iter_mut() {
            *s /= t;
        }
        let mut order: Vec<usize> = (0..scores.len()).collect();
        order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
        let k = if self.opts.top_k == 0 {
            order.len()
        } else {
            self.opts.top_k.min(order.len())
        };
        order.truncate(k);
        // Softmax over the kept candidates, then the smallest prefix whose
        // mass reaches top_p (HF drops tokens whose ascending cumulative
        // mass is <= 1 - top_p, which keeps the same set).
        let max = scores[order[0]];
        let mut probs: Vec<f32> = order.iter().map(|&i| (scores[i] - max).exp()).collect();
        let sum: f32 = probs.iter().sum();
        probs.iter_mut().for_each(|p| *p /= sum);
        if self.opts.top_p < 1.0 {
            let mut keep = probs.len();
            let mut tail = 0.0f32;
            // Walk from the least likely: drop while the dropped mass stays
            // <= 1 - top_p.
            for i in (1..probs.len()).rev() {
                if tail + probs[i] <= 1.0 - self.opts.top_p {
                    tail += probs[i];
                    keep = i;
                } else {
                    break;
                }
            }
            order.truncate(keep);
            probs.truncate(keep);
            let sum: f32 = probs.iter().sum();
            probs.iter_mut().for_each(|p| *p /= sum);
        }
        let r: f32 = self.rng.random();
        let mut acc = 0.0;
        for (i, p) in probs.iter().enumerate() {
            acc += p;
            if r < acc {
                return order[i] as u32;
            }
        }
        order[probs.len() - 1] as u32
    }
}

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0;
    for (i, x) in v.iter().enumerate() {
        if *x > v[best] {
            best = i;
        }
    }
    best as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// covers: REQ-SMP-001
    #[test]
    fn penalty_and_greedy() {
        let opts = SamplingOptions {
            do_sample: false,
            repetition_penalty: 2.0,
            ..Default::default()
        };
        let mut s = Sampler::new(opts, 4, &[1]);
        // id 1 is penalized: 3.0 / 2 = 1.5 < 2.0.
        assert_eq!(s.next(&[0.0, 3.0, 2.0, -1.0]), 2);
        // Now 2 is seen too: 1.5 vs 1.0.
        assert_eq!(s.next(&[0.0, 3.0, 2.0, -1.0]), 1);
    }

    /// covers: REQ-SMP-001
    #[test]
    fn top_k_one_is_greedy() {
        let opts = SamplingOptions {
            top_k: 1,
            repetition_penalty: 1.0,
            seed: Some(1),
            ..Default::default()
        };
        let mut s = Sampler::new(opts, 3, &[]);
        for _ in 0..10 {
            assert_eq!(s.next(&[0.1, 5.0, 0.2]), 1);
        }
    }
}

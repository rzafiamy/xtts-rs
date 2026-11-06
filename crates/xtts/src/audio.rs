//! Audio front end of voice cloning: WAV reading, torchaudio's sinc
//! resampler and mel spectrograms, written to match PyTorch numerically
//! (including the places where torchaudio rounds through F32).
//!
//! Spec: `torchaudio.functional.resample`, `torch.stft`,
//! `torchaudio.functional.melscale_fbanks`.

use anyhow::{Context, Result, bail};
use std::f64::consts::PI;
use std::path::Path;

/// Mono samples in [-1, 1] and their rate. Channels are averaged, as Coqui's
/// `load_audio` does. PCM is scaled like libsndfile (`x / 2^(bits-1)`).
pub fn read_wav(path: &Path) -> Result<(Vec<f32>, u32)> {
    let mut r = hound::WavReader::open(path).with_context(|| format!("opening {path:?}"))?;
    let spec = r.spec();
    let ch = spec.channels as usize;
    let raw: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => r.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = (1u64 << (spec.bits_per_sample - 1)) as f64;
            r.samples::<i32>()
                .map(|s| s.map(|v| (v as f64 / scale) as f32))
                .collect::<Result<_, _>>()?
        }
    };
    if raw.is_empty() {
        bail!("{path:?}: empty audio");
    }
    if ch == 1 {
        return Ok((raw, spec.sample_rate));
    }
    let mono = raw
        .chunks_exact(ch)
        .map(|f| f.iter().sum::<f32>() / ch as f32)
        .collect();
    Ok((mono, spec.sample_rate))
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// `torchaudio.functional.resample` with its defaults (`sinc_interp_hann`,
/// `lowpass_filter_width=6`, `rolloff=0.99`).
pub fn resample(x: &[f32], orig: u32, new: u32) -> Vec<f32> {
    if orig == new {
        return x.to_vec();
    }
    const LPW: f64 = 6.0;
    let g = gcd(orig, new);
    let (orig, new) = ((orig / g) as usize, (new / g) as usize);
    let base = orig.min(new) as f64 * 0.99;
    let width = (LPW * orig as f64 / base).ceil() as usize;
    let klen = 2 * width + orig;
    let scale = base / orig as f64;

    // Kernels: F64, but torchaudio's `arange(0, -new, -1) / new` is an
    // integer tensor divided in the default dtype, F32.
    let mut kernels = vec![0f32; new * klen];
    for j in 0..new {
        let off = (-(j as f32) / new as f32) as f64;
        for i in 0..klen {
            let idx = (i as f64 - width as f64) / orig as f64;
            let t = ((off + idx) * base).clamp(-LPW, LPW);
            let window = (t * PI / LPW / 2.0).cos().powi(2);
            let t = t * PI;
            let s = if t == 0.0 { 1.0 } else { t.sin() / t };
            kernels[j * klen + i] = (s * (window * scale)) as f32;
        }
    }

    // Conv1d with stride `orig` over `x` padded by (width, width + orig).
    let n = x.len() as isize;
    let target = ((new as f64 * x.len() as f64 / orig as f64) as f32).ceil() as usize;
    let blocks = (x.len() + 2 * width + orig - klen) / orig + 1;
    let mut out = Vec::with_capacity(blocks * new);
    for b in 0..blocks {
        let start = (b * orig) as isize - width as isize;
        let first = (-start).clamp(0, klen as isize) as usize;
        let last = (n - start).clamp(0, klen as isize) as usize;
        for j in 0..new {
            let k = &kernels[j * klen..(j + 1) * klen];
            let mut acc = 0f64;
            for i in first..last {
                acc += k[i] as f64 * x[(start + i as isize) as usize] as f64;
            }
            out.push(acc as f32);
        }
    }
    out.truncate(target);
    out
}

/// Iterative radix-2 FFT of a fixed size.
struct Fft {
    n: usize,
    rev: Vec<u32>,
    tw: Vec<(f64, f64)>,
}

impl Fft {
    fn new(n: usize) -> Self {
        assert!(n.is_power_of_two());
        let bits = n.trailing_zeros();
        let rev = (0..n as u32)
            .map(|i| i.reverse_bits() >> (32 - bits))
            .collect();
        let tw = (0..n / 2)
            .map(|k| {
                let a = -2.0 * PI * k as f64 / n as f64;
                (a.cos(), a.sin())
            })
            .collect();
        Self { n, rev, tw }
    }

    /// In place, on separate real and imaginary arrays.
    fn run(&self, re: &mut [f64], im: &mut [f64]) {
        let n = self.n;
        for i in 0..n {
            let j = self.rev[i] as usize;
            if i < j {
                re.swap(i, j);
                im.swap(i, j);
            }
        }
        let mut len = 2;
        while len <= n {
            let step = n / len;
            for start in (0..n).step_by(len) {
                for k in 0..len / 2 {
                    let (wr, wi) = self.tw[k * step];
                    let (a, b) = (start + k, start + k + len / 2);
                    let tr = re[b] * wr - im[b] * wi;
                    let ti = re[b] * wi + im[b] * wr;
                    re[b] = re[a] - tr;
                    im[b] = im[a] - ti;
                    re[a] += tr;
                    im[a] += ti;
                }
            }
            len *= 2;
        }
    }
}

/// `torch.hann_window(n)` (periodic), rounded through F32 like PyTorch.
pub fn hann_window(n: usize) -> Vec<f32> {
    let c = (2.0 * PI / n as f64) as f32;
    (0..n).map(|i| 0.5 - 0.5 * (i as f32 * c).cos()).collect()
}

/// Power spectrogram `[frames][n_fft/2 + 1]` as `torch.stft(center=True,
/// pad_mode="reflect", onesided=True)` then `abs().pow(2)`. A window shorter
/// than `n_fft` is centered in it.
pub fn power_spectrogram(
    x: &[f32],
    n_fft: usize,
    hop: usize,
    window: &[f32],
) -> Result<Vec<Vec<f32>>> {
    let pad = n_fft / 2;
    if x.len() <= pad {
        bail!("audio too short: {} samples, need more than {pad}", x.len());
    }
    let n = x.len();
    let at = |i: isize| -> f32 {
        // Reflection without repeating the edge sample.
        let i = i - pad as isize;
        if i < 0 {
            x[(-i) as usize]
        } else if (i as usize) < n {
            x[i as usize]
        } else {
            x[2 * n - 2 - i as usize]
        }
    };
    let left = (n_fft - window.len()) / 2;
    let fft = Fft::new(n_fft);
    let frames = 1 + n / hop;
    let mut out = Vec::with_capacity(frames);
    let (mut re, mut im) = (vec![0f64; n_fft], vec![0f64; n_fft]);
    for f in 0..frames {
        let start = (f * hop) as isize;
        re.fill(0.0);
        im.fill(0.0);
        for (i, w) in window.iter().enumerate() {
            re[left + i] = (at(start + (left + i) as isize) * w) as f64;
        }
        fft.run(&mut re, &mut im);
        out.push(
            (0..=n_fft / 2)
                .map(|k| (re[k] * re[k] + im[k] * im[k]) as f32)
                .collect(),
        );
    }
    Ok(out)
}

/// `torch.linspace` on F32.
fn linspace(start: f32, end: f32, steps: usize) -> Vec<f32> {
    let step = (end - start) / (steps - 1) as f32;
    (0..steps)
        .map(|i| {
            if i < steps / 2 {
                start + step * i as f32
            } else {
                end - step * (steps - i - 1) as f32
            }
        })
        .collect()
}

/// `torchaudio.functional.melscale_fbanks` with the HTK scale; result is
/// `[n_freqs][n_mels]`, in F32 like PyTorch's.
pub fn mel_filterbank(
    n_freqs: usize,
    f_min: f64,
    f_max: f64,
    n_mels: usize,
    sample_rate: u32,
    slaney_norm: bool,
) -> Vec<Vec<f32>> {
    let hz_to_mel = |f: f64| 2595.0 * (1.0 + f / 700.0).log10();
    let freqs = linspace(0.0, (sample_rate / 2) as f32, n_freqs);
    let mels = linspace(hz_to_mel(f_min) as f32, hz_to_mel(f_max) as f32, n_mels + 2);
    let pts: Vec<f32> = mels
        .iter()
        .map(|m| 700.0 * (10f32.powf(m / 2595.0) - 1.0))
        .collect();
    let mut fb = vec![vec![0f32; n_mels]; n_freqs];
    for (fi, f) in freqs.iter().enumerate() {
        for m in 0..n_mels {
            let down = (-(pts[m] - f)) / (pts[m + 1] - pts[m]);
            let up = (pts[m + 2] - f) / (pts[m + 2] - pts[m + 1]);
            let mut v = down.min(up).max(0.0);
            if slaney_norm {
                v *= 2.0 / (pts[m + 2] - pts[m]);
            }
            fb[fi][m] = v;
        }
    }
    fb
}

/// Spectrogram times filterbank: `[frames][n_mels]`.
pub fn apply_filterbank(spec: &[Vec<f32>], fb: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let n_mels = fb[0].len();
    // Only the nonzero span of each filter counts.
    let spans: Vec<(usize, usize)> = (0..n_mels)
        .map(|m| {
            let lo = fb.iter().position(|r| r[m] != 0.0).unwrap_or(0);
            let hi = fb.iter().rposition(|r| r[m] != 0.0).map_or(0, |i| i + 1);
            (lo, hi)
        })
        .collect();
    spec.iter()
        .map(|frame| {
            spans
                .iter()
                .enumerate()
                .map(|(m, &(lo, hi))| {
                    (lo..hi)
                        .map(|k| frame[k] as f64 * fb[k][m] as f64)
                        .sum::<f64>() as f32
                })
                .collect()
        })
        .collect()
}

/// The speaker encoder's pre-emphasis: `y[n] = x[n] - c x[n-1]`, the signal
/// reflected by one sample on the left.
pub fn pre_emphasis(x: &[f32], c: f32) -> Vec<f32> {
    let mut y = Vec::with_capacity(x.len());
    let prev0 = x.get(1).copied().unwrap_or(0.0);
    y.push(x[0] - c * prev0);
    for i in 1..x.len() {
        y.push(x[i] - c * x[i - 1]);
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_matches_dft() {
        let n = 16;
        let x: Vec<f64> = (0..n).map(|i| ((i * i) as f64 * 0.37).sin()).collect();
        let mut re = x.clone();
        let mut im = vec![0.0; n];
        Fft::new(n).run(&mut re, &mut im);
        for k in 0..n {
            let (mut r, mut s) = (0.0, 0.0);
            for (i, v) in x.iter().enumerate() {
                let a = -2.0 * PI * (k * i) as f64 / n as f64;
                r += v * a.cos();
                s += v * a.sin();
            }
            assert!((r - re[k]).abs() < 1e-9 && (s - im[k]).abs() < 1e-9);
        }
    }

    #[test]
    fn resample_lengths() {
        let x = vec![0.5f32; 24000];
        assert_eq!(resample(&x, 24000, 22050).len(), 22050);
        assert_eq!(resample(&x, 22050, 16000).len(), 17415);
    }
}

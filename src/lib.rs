//! # Spectral Estimation
//!
//! Power spectral density estimation methods including periodogram,
//! Welch's method, Bartlett's method, and autoregressive (Yule-Walker) estimation.

pub mod periodogram;
pub mod welch;
pub mod ar;
pub mod bartlett;
pub mod window;

pub use periodogram::periodogram;
pub use welch::welch_psd;
pub use ar::yule_walker;
pub use bartlett::bartlett_psd;

/// Compute the Discrete Fourier Transform of a real signal.
/// Returns complex (real, imag) pairs for N/2+1 frequency bins.
pub fn dft(signal: &[f64]) -> Vec<(f64, f64)> {
    let n = signal.len();
    let n_bins = n / 2 + 1;
    let mut result = Vec::with_capacity(n_bins);

    for k in 0..n_bins {
        let mut re = 0.0;
        let mut im = 0.0;
        for (t, &x) in signal.iter().enumerate() {
            let angle = -2.0 * std::f64::consts::PI * k as f64 * t as f64 / n as f64;
            re += x * angle.cos();
            im += x * angle.sin();
        }
        result.push((re, im));
    }

    result
}

/// Compute power spectrum from DFT output.
pub fn power_spectrum(dft_output: &[(f64, f64)], n: usize) -> Vec<f64> {
    dft_output.iter().map(|(re, im)| {
        (re * re + im * im) / n as f64
    }).collect()
}

/// Convert power to dB.
pub fn power_to_db(power: f64) -> f64 {
    10.0 * power.max(1e-20).log10()
}

/// Generate a test signal: sine wave at given frequency.
pub fn sine_wave(freq: f64, n: usize, sample_rate: f64) -> Vec<f64> {
    (0..n).map(|t| {
        (2.0 * std::f64::consts::PI * freq * t as f64 / sample_rate).sin()
    }).collect()
}

/// Generate white noise.
pub fn white_noise(n: usize, amplitude: f64, seed: u64) -> Vec<f64> {
    let mut state = seed;
    let mut result = Vec::with_capacity(n);
    for _ in 0..n {
        // Simple LCG random number generator
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let x = ((state >> 33) as i64 as f64) / (1i64 << 31) as f64;
        result.push(x * amplitude);
    }
    result
}

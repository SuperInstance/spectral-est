//! Window functions for spectral estimation.

/// Apply a Hann window to a signal.
pub fn hann_window(signal: &[f64]) -> Vec<f64> {
    let n = signal.len();
    signal.iter().enumerate().map(|(i, &x)| {
        let w = 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos());
        x * w
    }).collect()
}

/// Apply a Hamming window to a signal.
pub fn hamming_window(signal: &[f64]) -> Vec<f64> {
    let n = signal.len();
    signal.iter().enumerate().map(|(i, &x)| {
        let w = 0.54 - 0.46 * (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos();
        x * w
    }).collect()
}

/// Apply a Bartlett (triangular) window to a signal.
pub fn bartlett_window(signal: &[f64]) -> Vec<f64> {
    let n = signal.len();
    signal.iter().enumerate().map(|(i, &x)| {
        let w = if i <= n / 2 {
            2.0 * i as f64 / (n - 1) as f64
        } else {
            2.0 - 2.0 * i as f64 / (n - 1) as f64
        };
        x * w
    }).collect()
}

/// Compute the coherent gain of a window (average of window values).
pub fn coherent_gain(window: &[f64]) -> f64 {
    window.iter().sum::<f64>() / window.len() as f64
}

/// Generate a Hann window of given length.
pub fn generate_hann(n: usize) -> Vec<f64> {
    (0..n).map(|i| {
        0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos())
    }).collect()
}

/// Generate a Hamming window of given length.
pub fn generate_hamming(n: usize) -> Vec<f64> {
    (0..n).map(|i| {
        0.54 - 0.46 * (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos()
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hann_endpoints() {
        let w = generate_hann(101);
        assert!(w[0].abs() < 1e-14);
        assert!(w[100].abs() < 1e-14);
    }

    #[test]
    fn test_hann_peak() {
        let w = generate_hann(101);
        assert!((w[50] - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_hamming_symmetry() {
        let w = generate_hamming(51);
        for i in 0..25 {
            assert!((w[i] - w[50 - i]).abs() < 1e-14);
        }
    }

    #[test]
    fn test_bartlett_peak() {
        let w: Vec<f64> = (0..51).map(|i| {
            if i <= 25 { 2.0 * i as f64 / 50.0 } else { 2.0 - 2.0 * i as f64 / 50.0 }
        }).collect();
        assert!((w[25] - 1.0).abs() < 1e-14);
        assert!((w[0]).abs() < 1e-14);
    }

    #[test]
    fn test_coherent_gain_hann() {
        let w = generate_hann(1001);
        let gain = coherent_gain(&w);
        assert!((gain - 0.5).abs() < 0.01, "Hann coherent gain should be ~0.5, got {}", gain);
    }
}

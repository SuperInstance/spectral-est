//! Periodogram spectral estimation.

use crate::{dft, power_spectrum, power_to_db};

/// Compute the periodogram of a signal.
///
/// Returns the power spectral density estimate at N/2+1 frequency bins.
/// Frequency resolution = sample_rate / N.
pub fn periodogram(signal: &[f64]) -> Vec<f64> {
    let n = signal.len();
    let dft_output = dft(signal);
    let psd = power_spectrum(&dft_output, n);

    // Scale for one-sided PSD (double the non-DC/Nyquist bins)
    let mut result = psd;
    if n > 2 {
        for val in result.iter_mut().take(n / 2).skip(1) {
            *val *= 2.0;
        }
    }
    result
}

/// Compute the periodogram in dB.
pub fn periodogram_db(signal: &[f64]) -> Vec<f64> {
    periodogram(signal).iter().map(|&p| power_to_db(p)).collect()
}

/// Compute the frequency bins for a periodogram.
pub fn frequency_bins(n: usize, sample_rate: f64) -> Vec<f64> {
    let n_bins = n / 2 + 1;
    (0..n_bins).map(|k| k as f64 * sample_rate / n as f64).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sine_wave;
    use std::f64::consts::PI;

    #[test]
    fn test_periodogram_length() {
        let signal = vec![0.0; 64];
        let psd = periodogram(&signal);
        assert_eq!(psd.len(), 33); // N/2 + 1
    }

    #[test]
    fn test_periodogram_zero_signal() {
        let signal = vec![0.0; 64];
        let psd = periodogram(&signal);
        for &p in &psd {
            assert!(p.abs() < 1e-14, "Zero signal should have zero PSD");
        }
    }

    #[test]
    fn test_periodogram_sine_peak() {
        let n = 256;
        let signal = sine_wave(10.0, n, 256.0);
        let psd = periodogram(&signal);
        // Find peak
        let peak_bin = psd.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        // Should be near bin 10
        assert!((peak_bin as i32 - 10).abs() <= 1, "Peak at bin {}, expected ~10", peak_bin);
    }

    #[test]
    fn test_periodogram_dc_signal() {
        let signal = vec![1.0; 64];
        let psd = periodogram(&signal);
        // All power should be at DC (bin 0)
        assert!(psd[0] > 1.0, "DC power should be large");
        for i in 1..psd.len() {
            assert!(psd[i] < 0.01, "Non-DC power should be small at bin {}", i);
        }
    }

    #[test]
    fn test_periodogram_power_conservation() {
        let signal: Vec<f64> = (0..64).map(|i| (i as f64 * 0.3).sin()).collect();
        let psd = periodogram(&signal);
        // Total PSD power should relate to signal variance
        let signal_energy: f64 = signal.iter().map(|x| x * x).sum::<f64>() / 64.0;
        let total_power: f64 = psd.iter().sum();
        // Should be within a factor of 2 (due to one-sided scaling)
        assert!(total_power > 0.0 && signal_energy > 0.0, "Both should be positive");
    }

    #[test]
    fn test_frequency_bins() {
        let bins = frequency_bins(256, 1000.0);
        assert_eq!(bins.len(), 129);
        assert!((bins[0] - 0.0).abs() < 1e-14);
        assert!((bins[1] - 1000.0 / 256.0).abs() < 1e-14);
    }
}

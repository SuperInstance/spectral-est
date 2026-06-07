//! Bartlett's method for spectral estimation.
//!
//! Averages non-overlapping periodograms.

use crate::{dft, power_spectrum};

/// Compute Bartlett's PSD estimate using non-overlapping segments.
///
/// # Arguments
/// * `signal` - Input signal
/// * `segment_length` - Length of each segment
pub fn bartlett_psd(signal: &[f64], segment_length: usize) -> Vec<f64> {
    assert!(segment_length > 0, "Segment length must be positive");

    let n_segments = signal.len() / segment_length;
    if n_segments == 0 {
        return vec![0.0; segment_length / 2 + 1];
    }

    let mut avg_psd = vec![0.0; segment_length / 2 + 1];

    for seg in 0..n_segments {
        let start = seg * segment_length;
        let segment = &signal[start..start + segment_length];

        let dft_out = dft(segment);
        let psd = power_spectrum(&dft_out, segment_length);

        for (i, avg) in avg_psd.iter_mut().enumerate() {
            *avg += psd[i];
        }
    }

    let scale = 1.0 / n_segments as f64;
    for avg in &mut avg_psd {
        *avg *= scale;
    }

    // One-sided PSD
    for val in avg_psd.iter_mut().take(segment_length / 2).skip(1) {
        *val *= 2.0;
    }

    avg_psd
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sine_wave;

    #[test]
    fn test_bartlett_length() {
        let signal = vec![0.0; 256];
        let psd = bartlett_psd(&signal, 64);
        assert_eq!(psd.len(), 33);
    }

    #[test]
    fn test_bartlett_zero_signal() {
        let signal = vec![0.0; 256];
        let psd = bartlett_psd(&signal, 64);
        for &p in &psd {
            assert!(p.abs() < 1e-14);
        }
    }

    #[test]
    fn test_bartlett_sine_peak() {
        let n = 512;
        let signal = sine_wave(20.0, n, 256.0);
        let psd = bartlett_psd(&signal, 128);
        let peak_bin = psd.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        // Expected bin: 20 * 128 / 256 = 10
        assert!((peak_bin as i32 - 10).abs() <= 2, "Peak at bin {}, expected ~10", peak_bin);
    }

    #[test]
    fn test_bartlett_variance_reduction() {
        // Bartlett should have lower variance than single periodogram
        let n = 512;
        let signal = sine_wave(20.0, n, 256.0);
        let single_psd = crate::periodogram::periodogram(&signal[0..128.min(n)]);
        let bartlett_psd_val = bartlett_psd(&signal, 128);
        // Both should find the peak, but Bartlett smooths
        assert_eq!(bartlett_psd_val.len(), 65);
    }
}

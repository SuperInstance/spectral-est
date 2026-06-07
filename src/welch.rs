//! Welch's method for power spectral density estimation.
//!
//! Averages modified periodograms from overlapping segments.

use crate::{dft, power_spectrum};
use crate::window::hann_window;

/// Compute Welch's PSD estimate.
///
/// # Arguments
/// * `signal` - Input signal
/// * `segment_length` - Length of each segment (should be power of 2)
/// * `overlap` - Number of overlapping samples between segments
pub fn welch_psd(signal: &[f64], segment_length: usize, overlap: usize) -> Vec<f64> {
    assert!(segment_length > 0, "Segment length must be positive");
    assert!(overlap < segment_length, "Overlap must be less than segment length");

    let hop = segment_length - overlap;
    let n_segments = if signal.len() >= segment_length {
        (signal.len() - overlap) / hop
    } else {
        1
    };

    if n_segments == 0 {
        return vec![0.0; segment_length / 2 + 1];
    }

    let mut avg_psd = vec![0.0; segment_length / 2 + 1];

    for seg in 0..n_segments {
        let start = seg * hop;
        if start + segment_length > signal.len() {
            break;
        }

        let segment = &signal[start..start + segment_length];
        let windowed = hann_window(segment);

        // Normalize by window power
        let win_power: f64 = windowed.iter().map(|x| x * x).sum();

        let dft_out = dft(&windowed);
        let psd = power_spectrum(&dft_out, segment_length);

        for (i, avg) in avg_psd.iter_mut().enumerate() {
            *avg += if win_power > 1e-30 { psd[i] / win_power * segment_length as f64 } else { 0.0 };
        }
    }

    // Average over segments
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
    fn test_welch_length() {
        let signal = vec![0.0; 256];
        let psd = welch_psd(&signal, 64, 32);
        assert_eq!(psd.len(), 33); // 64/2 + 1
    }

    #[test]
    fn test_welch_zero_signal() {
        let signal = vec![0.0; 256];
        let psd = welch_psd(&signal, 64, 32);
        for &p in &psd {
            assert!(p.abs() < 1e-14);
        }
    }

    #[test]
    fn test_welch_sine_peak() {
        let n = 1024;
        let signal = sine_wave(50.0, n, 1000.0);
        let psd = welch_psd(&signal, 256, 128);
        // Find peak
        let peak_bin = psd.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        // Expected bin: 50 * 256 / 1000 = 12.8
        let expected_bin = (50.0_f64 * 256.0 / 1000.0).round() as usize;
        assert!((peak_bin as i32 - expected_bin as i32).abs() <= 2,
            "Peak at bin {}, expected ~{}", peak_bin, expected_bin);
    }

    #[test]
    fn test_welch_no_overlap() {
        let signal = sine_wave(10.0, 256, 256.0);
        let psd = welch_psd(&signal, 64, 0);
        assert_eq!(psd.len(), 33);
    }

    #[test]
    fn test_welch_full_overlap() {
        let signal = sine_wave(10.0, 128, 256.0);
        let psd = welch_psd(&signal, 64, 63);
        assert_eq!(psd.len(), 33);
    }
}

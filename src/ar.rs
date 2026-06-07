//! Autoregressive (AR) spectral estimation using Yule-Walker method.

/// Solve a Toeplitz system using Levinson-Durbin recursion.
/// Given autocorrelation r[0..p+1], returns AR coefficients a[1..p] and prediction error.
pub fn levinson_durbin(autocorr: &[f64]) -> (Vec<f64>, f64) {
    let p = autocorr.len() - 1;
    if p == 0 {
        return (vec![], autocorr[0]);
    }

    let mut a = vec![0.0; p];
    let mut error = autocorr[0];

    for m in 1..=p {
        let mut reflection = 0.0;
        for k in 1..m {
            reflection += a[k - 1] * autocorr[m - k];
        }
        reflection = (autocorr[m] - reflection) / error;

        // Update coefficients
        let mut new_a = vec![0.0; m];
        new_a[m - 1] = reflection;
        for k in 1..m {
            new_a[k - 1] = a[k - 1] - reflection * a[m - 1 - k];
        }

        for (i, &val) in new_a.iter().enumerate() {
            a[i] = val;
        }

        error *= 1.0 - reflection * reflection;
    }

    (a, error)
}

/// Estimate AR model parameters using the Yule-Walker method.
///
/// # Arguments
/// * `signal` - Input signal
/// * `order` - AR model order
///
/// Returns AR coefficients (a[1], a[2], ..., a[p]) and prediction error variance.
pub fn yule_walker(signal: &[f64], order: usize) -> (Vec<f64>, f64) {
    let autocorr = autocorrelation(signal, order);
    levinson_durbin(&autocorr)
}

/// Compute biased autocorrelation estimate.
pub fn autocorrelation(signal: &[f64], max_lag: usize) -> Vec<f64> {
    let n = signal.len();
    let max_lag = max_lag.min(n - 1);
    let mut r = vec![0.0; max_lag + 1];

    for lag in 0..=max_lag {
        let mut sum = 0.0;
        for i in 0..(n - lag) {
            sum += signal[i] * signal[i + lag];
        }
        r[lag] = sum / n as f64;
    }

    r
}

/// Compute AR spectral estimate from AR coefficients.
///
/// Returns power spectral density at `n_points` frequency bins (0 to π).
pub fn ar_spectrum(ar_coeffs: &[f64], error_var: f64, n_points: usize) -> Vec<f64> {
    (0..n_points).map(|i| {
        let omega = std::f64::consts::PI * i as f64 / (n_points - 1) as f64;
        let mut denom_re = 1.0;
        let mut denom_im = 0.0;
        for (k, &a) in ar_coeffs.iter().enumerate() {
            let angle = omega * (k + 1) as f64;
            denom_re += a * angle.cos();
            denom_im -= a * angle.sin();
        }
        let denom_mag_sq = denom_re * denom_re + denom_im * denom_im;
        error_var / denom_mag_sq
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_autocorrelation_at_zero_lag() {
        let signal = [1.0, 2.0, 3.0, 4.0];
        let r = autocorrelation(&signal, 3);
        // r[0] should be the average power
        let expected = (1.0 + 4.0 + 9.0 + 16.0) / 4.0;
        assert!((r[0] - expected).abs() < 1e-14);
    }

    #[test]
    fn test_autocorrelation_length() {
        let signal = [1.0, 2.0, 3.0];
        let r = autocorrelation(&signal, 2); // max_lag capped at n-1=2
        assert_eq!(r.len(), 3);
    }

    #[test]
    fn test_levinson_durbin_order_1() {
        let r = [1.0, 0.5];
        let (a, err) = levinson_durbin(&r);
        assert_eq!(a.len(), 1);
        // Levinson returns a[0] = reflection = 0.5
        assert!((a[0] - 0.5).abs() < 1e-14, "a[0]={}", a[0]);
        assert!((err - 0.75).abs() < 1e-14, "err={}", err);
    }

    #[test]
    fn test_yule_walker_white_noise() {
        // Use actual-like noise
        let signal = [0.3, -0.7, 0.5, -0.2, 0.8, -0.4, 0.1, -0.6, 0.9, -0.3, 0.4, -0.8, 0.6, -0.1, 0.2, -0.5];
        let (coeffs, err) = yule_walker(&signal, 2);
        assert_eq!(coeffs.len(), 2);
        assert!(err > 0.0, "Error variance should be positive");
    }

    #[test]
    fn test_yule_walker_order() {
        let signal = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let (coeffs, _) = yule_walker(&signal, 3);
        assert_eq!(coeffs.len(), 3);
    }

    #[test]
    fn test_ar_spectrum_length() {
        let coeffs = vec![0.5, -0.3];
        let spectrum = ar_spectrum(&coeffs, 1.0, 128);
        assert_eq!(spectrum.len(), 128);
    }

    #[test]
    fn test_ar_spectrum_positive() {
        let coeffs = vec![0.5, -0.3];
        let spectrum = ar_spectrum(&coeffs, 1.0, 64);
        for &s in &spectrum {
            assert!(s > 0.0, "AR spectrum should be positive");
        }
    }

    #[test]
    fn test_levinson_durbin_stability() {
        // Reflection coefficients should be < 1 for stable AR process
        let r = [1.0, 0.8, 0.5, 0.3, 0.1];
        let (_, err) = levinson_durbin(&r);
        assert!(err > 0.0, "Prediction error should be positive for stable process");
    }
}

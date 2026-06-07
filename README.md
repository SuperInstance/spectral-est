# spectral-est

Power spectral density estimation in pure Rust.

## Features

- **Periodogram** — Classic non-parametric spectral estimation
- **Welch's method** — Averaged overlapping windowed periodograms
- **Bartlett's method** — Averaged non-overlapping periodograms
- **AR (Yule-Walker)** — Autoregressive parametric spectral estimation
- **Window functions** — Hann, Hamming, Bartlett windows

## Modules

| Module | Description |
|--------|-------------|
| `periodogram` | Periodogram spectral estimation |
| `welch` | Welch's method |
| `ar` | Autoregressive / Yule-Walker |
| `bartlett` | Bartlett's method |
| `window` | Window functions |

## Quick Start

```rust
use spectral_est::{periodogram, welch_psd, sine_wave};

let signal = sine_wave(50.0, 1024, 1000.0);
let psd = periodogram(&signal);
let welch = welch_psd(&signal, 256, 128);
```

## License

MIT OR Apache-2.0

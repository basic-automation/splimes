<p align="center">
  <img src="https://raw.githubusercontent.com/basic-automation/splimes/main/assets/logo.svg" alt="splimes" width="420">
</p>

<p align="center">
  <a href="https://crates.io/crates/splimes"><img src="https://img.shields.io/crates/v/splimes.svg" alt="crates.io"></a>
  <a href="https://docs.rs/splimes"><img src="https://img.shields.io/docsrs/splimes" alt="docs.rs"></a>
  <a href="https://github.com/basic-automation/splimes/actions/workflows/ci.yml"><img src="https://github.com/basic-automation/splimes/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/MSRV-1.95-blue.svg" alt="MSRV 1.95">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</p>

Spline interpolation over **irregularly sampled** time series.

`splimes` reconstructs a continuous signal from points that didn't arrive on a clean
grid, and resamples it onto the regular grid you ask for: every second, minute, day,
and so on.

- **Methods:** linear, quadratic, cubic and polynomial splines. If there are too few
  points for the method you asked for, it steps down to one that fits: cubic →
  quadratic → linear.
- **Backends:** SIMD and `rayon`-parallel CPU, or GPU via [`wgpu`](https://wgpu.rs)
  (Vulkan / Metal / DX12). Selection is automatic. Without a usable GPU the engine
  runs on the CPU, and small workloads stay on the CPU because dispatch overhead
  would dominate.
- **Values are [`BigDecimal`](https://docs.rs/bigdecimal):** precision is declared,
  not quietly lost. GPUs without f64 support use an f32 path.

## Quick start

```toml
[dependencies]
splimes = "0.1"
bigdecimal = "0.4"
chrono = "0.4"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
anyhow = "1"
```

```rust
use bigdecimal::BigDecimal;
use chrono::{TimeZone, Utc};
use splimes::{Point, Resolution, Spline, auto_interpolate};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let at = |secs: i64| Utc.timestamp_opt(1_700_000_000 + secs, 0).unwrap();

    // Four readings, unevenly spaced.
    let mut readings = vec![
        Point::new(at(0), BigDecimal::from(10)),
        Point::new(at(7), BigDecimal::from(14)),
        Point::new(at(19), BigDecimal::from(11)),
        Point::new(at(30), BigDecimal::from(20)),
    ];

    // Resample onto a one-second grid with a cubic spline.
    let series = auto_interpolate(&mut readings, at(0), at(30), Resolution::Seconds, Spline::Cubic).await?;

    for point in series.iter().take(3) {
        println!("{} {}", point.timestamp, point.value);
    }
    Ok(())
}
```

`auto_interpolate` picks the backend for you. `cpu_interpolate`,
`parallel_interpolate` and `gpu_interpolate` force a specific backend, and
`prewarm_gpu` moves the GPU's one-time start-up cost (about a second) out of your
first query.

## Features

| Feature | Default | Effect |
|---------|---------|--------|
| `gpu-eager-init` | off | Initialise the GPU at process start (via `ctor`) instead of on first use. Trades start-up time for a warm first query. Leave it off if you want to call `prewarm_gpu_with_config`. |

## Status

**Pre-1.0.** The API may change between minor versions until 1.0; see
[`ROADMAP.md`](ROADMAP.md) for what 1.0 requires and [`CHANGELOG.md`](CHANGELOG.md)
for what changed. The minimum supported Rust version is **1.95**, and raising it
counts as a breaking change after 1.0.

`splimes` is the interpolation engine behind [WeftDB](https://github.com/basic-automation/weftdb),
but it has no dependency on the database and works on its own.

## Contributing

`cargo test` runs anywhere. Without a GPU, the tests that compare CPU and GPU output
skip their GPU half; set `SPLIMES_REQUIRE_GPU=1` to make a missing GPU a failure instead
(CI does, on a software Vulkan driver). Formatting uses nightly rustfmt:
`cargo +nightly fmt`.

## License

MIT; see [LICENSE](LICENSE).

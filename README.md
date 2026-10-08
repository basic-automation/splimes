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

Resample **irregularly sampled** time series onto a regular grid.

You have points that arrived whenever they arrived; you want a value every second,
minute or day. `splimes` fits a local spline through the points, evaluates it on the
grid, and labels every output point as an observation, an interpolation or an
extrapolation.

- **Methods:** linear, quadratic, cubic and polynomial (degree 1 to 8). Each is
  precisely defined, inside the data and outside it, and every backend computes the
  same thing.
- **Backends:** one CPU thread, rayon's thread pool, or the GPU via
  [`wgpu`](https://wgpu.rs) (Vulkan, Metal, DX12), picked per call by size or chosen
  explicitly. A GPU failure never panics: automatic selection reruns on the CPU and
  tells you why.
- **Values:** `BigDecimal` or `f64`. Computation is in `f64` (or `f32` on the GPU, if
  you ask), against a published, tested error bound. Points that land on an input
  return that input exactly, every digit intact.
- **Provenance:** each output point is `Raw`, `Interpolated` or `Extrapolated`.
- **Honest reporting:** the result says which method and backend actually ran. If
  there are too few points for the method you asked for, it steps down (cubic →
  quadratic → linear) and says so, or refuses if you'd rather it didn't.

## Quick start

```toml
[dependencies]
splimes = "1"
bigdecimal = "0.4"
chrono = "0.4"
```

```rust
use bigdecimal::BigDecimal;
use chrono::{TimeZone, Utc};
use splimes::{Point, PointKind, Resolution, Spline};

fn main() -> Result<(), splimes::Error> {
    let at = |secs: i64| Utc.timestamp_opt(1_700_000_000 + secs, 0).unwrap();

    // Four readings, unevenly spaced.
    let readings = [
        Point::new(at(0), BigDecimal::from(10)),
        Point::new(at(7), BigDecimal::from(14)),
        Point::new(at(19), BigDecimal::from(11)),
        Point::new(at(30), BigDecimal::from(20)),
    ];

    // A value every second, with a cubic spline, a little past the last reading.
    let series = splimes::interpolate(&readings, at(0), at(33), Resolution::Seconds, Spline::Cubic)?;

    for (timestamp, value, kind) in series.iter().take(3) {
        println!("{timestamp} {value} {kind}");
    }
    assert_eq!(series.kinds()[7], PointKind::Raw);
    assert_eq!(series.kinds()[32], PointKind::Extrapolated);
    Ok(())
}
```

For more control, configure an `Interpolator`:

```rust
use chrono::{TimeDelta, TimeZone, Utc};
use splimes::{Backend, Interpolator, Resolution, Spline};

let t0 = Utc.timestamp_opt(1_700_000_000, 0).unwrap();
let timestamps = [t0, t0 + TimeDelta::seconds(4), t0 + TimeDelta::seconds(9)];
let values = [1.0, 3.5, 2.0];

let series = Interpolator::new(Spline::Quadratic, Resolution::Milliseconds)
    .backend(Backend::Parallel) // or Auto (the default), Cpu, Gpu
    .exact(true)                // error instead of stepping down to a simpler method
    .run_f64(&timestamps, &values, t0, t0 + TimeDelta::seconds(9))?; // f64 in, f64 out

assert_eq!(series.len(), 9_001);
assert_eq!(series.spline(), Spline::Quadratic);
# Ok::<(), splimes::Error>(())
```

From async code, `spawn` (or `spawn_f64`) runs the work on rayon's pool and returns a
future that any executor can await, with no feature flag:
`interpolator.spawn(points, start, end).await?`. With the `tokio` feature, `run_async`
uses tokio's blocking pool instead.

## Accuracy

Every backend computes from the same normalised inputs, with every time difference
taken exactly, and `tests/contract.rs` checks each one against an exact 60-digit
reference. Each result is as accurate as perturbing every input value by **10⁻¹³** of
the value range allows in `f64` (CPU or GPU), or **10⁻⁵** in GPU `f32` — for every
method, inside the data and out, however long the series. Where the problem is
ill-conditioned (very irregular spacing, high degree, far extrapolation) the error grows
exactly as much as the method amplifies such a perturbation, and no more. The
[crate documentation](https://docs.rs/splimes/latest/splimes/#numerical-contract)
has the precise statement. It is part of the semver promise.

## Performance

On a 16-core Ryzen 9 7950X3D with an RTX 4070 Ti SUPER, a cubic resample of 4,096
irregular points onto a 16.7-million-point grid takes about **100 ms** on rayon's pool,
against 670 ms on one thread, and a million points about 7–12 ms. Building the output
(timestamps, values, provenance) costs the same whichever backend computed the values,
and it dominates, so the GPU gains little: here it at best ties rayon in `f64` and wins
modestly in `f32` at a million points and more. `Backend::Auto` therefore uses the GPU
only once `splimes::calibrate()` has started it and measured that it pays off on your
machine. [BENCHMARKS.md](BENCHMARKS.md) has the numbers and how to reproduce them.

## Features

| Feature | Default | Effect |
|---------|---------|--------|
| `gpu` | yes | The wgpu backend. Without it, `Backend::Gpu` returns `Error::GpuUnavailable`, and the build drops wgpu entirely. |
| `serde` | yes | `Serialize`/`Deserialize` for `Point`, `PointKind`, `Resolution` and `Spline`. |
| `tokio` | no | `Interpolator::run_async` and `run_f64_async`. |

## Stability

`splimes` 1.x follows [semantic versioning](https://semver.org). Covered: the public
API, what each method computes (beyond rounding within the published bound), and the
serde formats. The minimum supported Rust version is **1.95**; raising it is a
minor-version change, announced in the [changelog](CHANGELOG.md), and only to a
version at least six months old. Upgrading from 0.1? See [MIGRATING.md](MIGRATING.md).

`splimes` is the interpolation engine behind [WeftDB](https://github.com/basic-automation/weftdb),
but it has no dependency on the database and works on its own.

## Contributing

`cargo test` runs anywhere. Without a GPU, the GPU tests skip themselves; set
`SPLIMES_REQUIRE_GPU=1` to make a missing GPU a failure instead, or
`SPLIMES_REQUIRE_GPU_F64=1` to require one with `f64` support too (CI sets that, on a
software Vulkan driver). Formatting uses nightly rustfmt: `cargo +nightly fmt`.

## License

MIT; see [LICENSE](LICENSE).

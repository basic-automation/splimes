# Migrating from 0.1 to 1.0

1.0 replaces the 0.1 entry points with one builder, makes everything synchronous,
returns a typed error, and reports provenance and the method and backend actually used.
What each method computes is now defined once and computed identically on every
backend, so a few results change too; they are listed at the end.

## Entry points

| 0.1 | 1.0 |
|-----|-----|
| `auto_interpolate(&mut points, start, end, resolution, spline).await?` | `interpolate(&points, start, end, resolution, spline)?` |
| `cpu_interpolate(&mut points, start, end, resolution, spline).await?` | `Interpolator::new(spline, resolution).backend(Backend::Cpu).run(&points, start, end)?` |
| `parallel_interpolate(&mut points, &start, &end, spline, resolution).await?` | `….backend(Backend::Parallel).run(…)?` |
| `gpu_interpolate(&mut points, start, end, resolution, spline).await?` | `….backend(Backend::Gpu).run(…)?` |
| result: `Vec<Point>` | result: `Interpolation<BigDecimal>`; call `.into_points()` for a `Vec<Point>` |

From async code, run the call off the executor. With the `tokio` feature:

```rust,ignore
// 0.1
let series = splimes::auto_interpolate(&mut points, start, end, resolution, spline).await?;

// 1.0
let series = Interpolator::new(spline, resolution).run_async(points, start, end).await?.into_points();
```

`run_async` takes the points by value (the blocking task has to own them). Without the
feature, wrap `run` in your runtime's `spawn_blocking`.

## Provenance

Code that rebuilt raw / interpolated / extrapolated labels from the inputs — WeftDB's
`classify` in `weft-server/src/interpolate.rs` does exactly this — can read them from
the result instead:

```rust,ignore
let result = Interpolator::new(spline, resolution).run(&points, start, end)?;
for (timestamp, value, kind) in result.iter() {
    // kind: PointKind::Raw | Interpolated | Extrapolated, serialised as "raw" | …
}
```

The definitions match: `Raw` when the grid instant equals an input timestamp,
`Extrapolated` outside `[first, last]`, `Interpolated` otherwise. A `Raw` value is the
input's own `BigDecimal`, every digit intact.

If you widen `f64` to `BigDecimal` only to call splimes and narrow it again afterwards,
use `run_f64` and skip both conversions.

If the time range or resolution comes from a request, cap the grid with
`Interpolator::max_points(n)`: 0.1 would try to allocate whatever was asked for.

## Errors

Every function returns `splimes::Result<T>`, with `splimes::Error` in place of
`anyhow::Error`. It implements `std::error::Error`, so `?` into `anyhow` still works.
Match on variants instead of strings:

| 0.1 | 1.0 |
|-----|-----|
| `InsufficientMeasurementsError` | `NoPoints` |
| `InvalidTimeRangeError` | `InvalidTimeRange { start, end }`, now only when `start > end` |
| `InvalidDegreeError(d)` | `InvalidDegree { degree, max }` |
| `DecimalConversionError` | `ValueOutOfRange { timestamp }` (input), `NonFiniteResult { timestamp }` (output) |
| `InsufficientPointsError` | `InsufficientPoints { spline, required, available }`, only with `exact(true)` |
| GPU failures (often a panic) | `GpuUnavailable(reason)`, `Gpu(reason)` |
| `ConversionError` from `FromStr` | `Parse { kind, input }` |
| `TimeError`, `InvalidTimestampError` | none: time arithmetic is exact `i128` nanoseconds and can't fail |
| `IOError` | none: nothing is spilled to disk any more |

The enum is `#[non_exhaustive]`; keep a `_` arm.

## Removed items

| 0.1 | 1.0 |
|-----|-----|
| `pub mod helpers`, `should_use_gpu`, `InterpolationStrategy` | `Backend::Auto` with `AutoThresholds`, or `calibrate()` |
| `estimate_output_points`, `generate_target_times`, `TargetTimesIterator` | the grid is `start + k·resolution.step()` for `k` in `0..=(end − start) / step`; `Interpolation::timestamps()` has it |
| `apply_fast_path` | gone: methods are never silently downgraded |
| `BASE_BATCH_SIZE`, `POINT_SIZE` | gone |
| `SECONDS_IN_*`, `DAYS_IN_*` | `Resolution::step()` / `step_nanos()` |
| `Resolution::to_step()` | `Resolution::step()` |
| `Resolution::difference(&a, &b)` | `(a - b).num_nanoseconds()` and divide by `step_nanos()`, or chrono directly |
| `Resolution::round(&t)` | gone; it returned `t` unchanged for every resolution |
| `Resolution::to_base`, `to_step_base` | gone |
| `Spline::number_of_points_required()` | `Spline::min_points()` |
| `Spline::pre_check(…)` | `Spline::validate()`, plus `Error::InvalidTimeRange` from the call |
| `prewarm_gpu() -> Result<()>` | `prewarm_gpu() -> Result<GpuInfo>` |
| `prewarm_gpu_with_config`, `GpuConfig` presets | unchanged names, still `const`; `GpuConfig` now has `max_pool_bytes`, `chunk_points`, `low_power` |
| `gpu_config_applied()`, `effective_gpu_config()` | `gpu_config()`; `configure_gpu()` errors if it's too late |
| `gpu_buffer_pool_stats()`, `BufferPoolStats` | `gpu_pool_stats() -> Option<GpuPoolStats>` |
| feature `gpu-eager-init` | call `prewarm_gpu()` at startup |

## Types that didn't change

`Point`, `Resolution` and `Spline` keep their variants, fields and **serde formats**, so
persisted configurations still deserialise: `"Seconds"`, `"Cubic"`,
`{"Polynomial":[3,1.5]}`. `Display` for `Spline` prints the same text as 0.1, and
`FromStr` still parses it — but it now validates, so a persisted string with degree 0,
a degree above 8, or a negative or non-finite bounds factor, which 0.1 accepted, is an
`Error::InvalidDegree` / `InvalidBoundsFactor`. (Serde doesn't validate; the error then
comes from the interpolation call.) `Spline::degree` and `Spline::bounds_factor` keep
their `&self` receivers.

`Resolution` and `Spline` are now `#[non_exhaustive]`: an exhaustive `match` on them
needs a `_` arm.

## Results that change

Each change makes the backends agree with each other and with the documentation in
`Spline`. Expect differences in these cases only:

- **Large inputs keep their method.** 0.1 silently replaced cubic with quadratic from
  2,500 input points and with linear from 5,000 (quadratic with linear from 5,000).
- **`Polynomial(1 | 2 | 3, b)` stays a polynomial.** 0.1's `auto_interpolate` turned
  these into `Linear`, `Quadratic` and `Cubic`, dropping the bounds factor and, for
  degree 3, holding flat outside the data instead of extending.
- **Coarse resolutions over fine data.** 0.1's CPU paths measured time in whole
  resolution units, so with `Hours` output, knots minutes apart coincided. Time is now
  exact to the nanosecond on every path.
- **`Polynomial` with too few points** stays a polynomial (degree `n − 1`, same bounds
  factor) instead of becoming `Cubic`, `Quadratic` or `Linear`, so it keeps extending
  and clamping outside the data rather than, for example, holding flat as `Cubic` does.
- **Polynomial degrees above 8** are an error instead of being capped at 8 silently.
- **Duplicate timestamps** keep the last value on every method; 0.1 varied by method.
- **`start == end`** returns one point instead of an error.
- **`BigDecimal` output** is the shortest decimal that round-trips to the computed
  value (`0.1`), where 0.1 sometimes rounded to 10 places and sometimes returned the
  full binary expansion.
- **GPU polynomials.** 0.1's GPU path evaluated `Polynomial(d, b)` at degree `d + 1`,
  and its `f64` kernel ignored any bounds factor other than 1.0, so GPU results (which
  `auto_interpolate` used for 100–999 and 50,000+ grid points) differed from CPU ones.
  Every backend now computes degree `d` and applies `b`.
- **GPU `f32`** is used only when asked for with `gpu_precision(Precision::F32)`.
  0.1 used it whenever the adapter lacked `f64`. With the default (`F64`) on such an
  adapter, `Auto` stays on the CPU and `Backend::Gpu` returns `GpuUnavailable`.
- **`Auto` doesn't use the GPU by default, and never starts it.** On the hardware in
  BENCHMARKS.md it only matched the rayon backend. Run `calibrate()` once at startup to
  start the GPU and let `Auto` use it where it measures faster, or call `prewarm_gpu()`
  and set the thresholds yourself.
- **Leap seconds.** A `23:59:60.x` timestamp is the instant `00:00:00.x` of the next
  second (the POSIX scale), so it collapses with an input at that instant (last wins),
  and a grid starting in one starts at the folded instant.

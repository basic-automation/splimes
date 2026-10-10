# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
From 1.0, what each interpolation method computes (beyond rounding within the
published error bound) is covered by semver, as is the public API. Raising the minimum
supported Rust version is a minor-version change.

## [Unreleased]

### Fixed

- **A GPU device lost mid-call is an error, never stale values.** splimes checked for
  device loss only when a call started. wgpu 30 can report a result buffer's mapping as
  successful although the device was lost before the mapping was processed
  ([gfx-rs/wgpu#10301](https://github.com/gfx-rs/wgpu/pull/10301), fixed after 30.0.1),
  and the buffer then holds what it held before: for a pooled buffer, an earlier call's
  results. splimes now also checks after every read-back, so a call during which the
  device is lost returns `Error::GpuUnavailable` (`Backend::Auto` reruns it on the CPU
  and reports why).

## [1.1.0] - 2026-10-08

### Added

- `Interpolator::spawn` and `Interpolator::spawn_f64`: start an interpolation on
  rayon's pool and get an `InterpolationFuture` that any executor can await (tokio,
  async-std, smol, `futures`), with no feature flag and no new dependency. A panic in
  the work becomes `Error::Task` instead of aborting. Dropping the future cancels work
  that hasn't started. A process that exits while the GPU driver is busy on one of
  rayon's threads can crash in the driver, so: with `Backend::Gpu`, the first `spawn`
  opens the device on the calling thread (splimes still never opens it on a thread of
  its own); `Backend::Auto` uses the GPU only if it had started when `spawn` was called;
  and dropping a future whose GPU work has started waits for it, except on a rayon
  worker. Called from a rayon worker, `spawn` runs the interpolation there, so blocking
  on its future can't deadlock the pool.
- `Error::Task` no longer requires the `tokio` feature: `InterpolationFuture` reports
  through it too.

### Changed

- **Faster input preparation.** Every backend except `Cpu` now converts and sorts inputs
  of 16 Ki points or more on rayon's pool, and each value is converted to `f64` once
  instead of twice. A million shuffled `BigDecimal` inputs prepare in 43 ms instead of
  368 ms on `Parallel`, and in 187 ms instead of 371 ms on `Cpu`, which stays on the
  calling thread ([BENCHMARKS.md](BENCHMARKS.md#preparing-input)). Results are unchanged.
- **Faster CPU kernel.** When the knots and the grid lie within about 146 years of the
  first knot (nearly every series), the CPU backends take their exact time differences
  in `i64` rather than `i128`: 27–39% less time single-threaded, across methods, and
  less on `Parallel`, where the gain is within run-to-run variance
  ([BENCHMARKS.md](BENCHMARKS.md#methods)). Results are bit-identical, and longer spans
  keep the `i128` path.
- **Cheaper output timestamps.** Grid timestamps are built from integer POSIX seconds on
  a calendar date computed once per day, instead of one chrono `checked_add_signed` per
  point (about 10 ns each): 26% less time for a single-threaded linear interpolation of a
  million points. The timestamps are the same.

## [1.0.0] - 2026-10-05

A rewrite of the engine around one precisely defined kernel that every backend
computes. It is a breaking release: [MIGRATING.md](MIGRATING.md) maps the 0.1 API to
its replacement and lists the results that change.

### Added

- `Interpolator`, a `Copy` builder: method, resolution, `backend(Backend)`,
  `gpu_precision(Precision)` and `exact(bool)`, run with `run(&[Point], start, end)`
  or `run_f64(&[DateTime<Utc>], &[f64], start, end)`.
- `Interpolation<V>`: the series as timestamp, value and `PointKind` columns, plus a
  report of the method actually used (`spline()`, `requested_spline()`), the backend
  (`backend()`), the precision (`precision()`) and why the GPU was abandoned, if it
  was (`gpu_fallback()`).
- **Provenance:** every output point is `PointKind::Raw` (it coincides with an input,
  and returns that input's value exactly), `Interpolated` or `Extrapolated`.
- **Typed `f64` input and output** with `Interpolator::run_f64`, skipping `BigDecimal`
  conversion entirely.
- **Numerical contract:** every result is within `bound · range · Λ(t)` of the exact
  value (Λ the window's Lebesgue function), with `bound` 1e-13 in `f64` and 1e-5 in GPU
  `f32`, for every method, inside the data and out. Enforced by `tests/contract.rs`
  against an exact 60-digit reference that computes Λ, on randomised and stress series.
- `Interpolation::points_recomputed_in_f64`: how many points the GPU couldn't compute
  reliably (`f32` overflow, or `f32` windows mixing dense and sparse knots) and were
  recomputed in `f64` on the CPU.
- `splimes::Error`, a `#[non_exhaustive]` enum, and `splimes::Result`.
- `Backend::Auto` falls back to the CPU when the GPU fails mid-call, logs why, and
  reports it. It never starts the GPU itself: it uses it once the program has (with
  `prewarm_gpu`, `calibrate` or a `Backend::Gpu` call), so no driver ever initialises on
  a thread the process can't wait for at exit.
- `calibrate()`, `auto_thresholds()` and `set_auto_thresholds()`: measure this
  machine's CPU and GPU crossovers, in both GPU precisions, instead of trusting fixed
  thresholds. `AutoThresholds` has a GPU threshold per precision.
- `Precision::F32` for the GPU. Times are 96-bit integer nanoseconds on the GPU too, so
  long series keep their resolution in single precision, and compilers that reassociate
  floating point (Metal's fast math) can't undo it. Points where the `f32` kernel overflows but
  `f64` wouldn't are recomputed in `f64` on the CPU.
- `Interpolator::max_points`, to refuse oversized grids from untrusted input before
  allocating.
- `gpu_info()`, `gpu_config()`, `gpu_pool_stats()`, `configure_gpu()`.
- Features: `gpu` (default; turn it off to drop wgpu), `serde` (default), and
  `tokio` (`Interpolator::run_async`, `run_f64_async`).
- `Resolution::ALL`, `Resolution::step_nanos`, `Resolution::as_str`,
  `Spline::min_points`, `Spline::validate`, `Spline::fallback_for`,
  `PointKind::as_str`, `MAX_POLYNOMIAL_DEGREE`.
- `BENCHMARKS.md`, with hardware, versions and reproduction steps.

### Changed

- **All entry points are synchronous.** The 0.1 `async fn`s did all their work
  without yielding, blocking the executor; use `run_async` (feature `tokio`) or your
  runtime's `spawn_blocking` from async code. tokio is no longer a dependency.
- **One kernel, every backend.** Quadratic, cubic and polynomial windows, and
  extrapolation, are defined once (see `Spline`) and computed identically on CPU and
  GPU; `Cpu` and `Parallel` are bit-identical.
- **Time is exact, and never truncated to the resolution.** 0.1's CPU paths measured
  knot spacing in whole resolution units, so hourly output from sub-hour data saw
  coincident knots. All paths now take every time difference exactly, in integer
  nanoseconds (96-bit integer arithmetic on the GPU), so accuracy doesn't degrade with the
  length of the series.
- **Time is on the POSIX scale.** A leap second (`23:59:60.x`, which chrono can
  represent) is the same instant as `00:00:00.x` the next second, for inputs, `start`
  and `end`; output timestamps never contain one. 0.1 mixed chrono's subtraction (which
  counts leap seconds) with its addition (which doesn't).
- **No silent method downgrades.** 0.1 replaced cubic with quadratic from 2,500
  points and with linear from 5,000, and capped polynomial degree at 8, without
  saying so. Now the method you ask for runs, `Polynomial` degrees above 8 are an
  error, and stepping down for too few points is reported (or refused with
  `exact(true)`).
- **Polynomial falls back within its family:** `Polynomial(d, b)` with `n ≤ d` points
  runs `Polynomial(n − 1, b)`, keeping its extrapolation and bounds, instead of
  becoming `Cubic`.
- Input is taken by shared reference and no longer sorted in place.
- Duplicate timestamps keep the last value, on every method. Every input value must be
  finite, including ones a later duplicate replaces.
- `GpuConfig` presets are `const fn` as before, but describe the new buffer pool:
  `high_performance()` keeps 1 GiB + 1 MiB, enough for both of a call's 16 Mi-point
  `f64` buffer sets. A GPU call holds at most two chunks' buffers at a time.
- `start == end` is a one-point grid instead of an error.
- `BigDecimal` results are the shortest decimal that round-trips to the computed
  `f64`, not its full binary expansion or a 10-place rounding.
- `Resolution` and `Spline` are `#[non_exhaustive]`; `Spline` parsing is
  case-insensitive for the unit variants and validates parameters.
- The GPU device is opened with `pollster` on the calling thread instead of a
  private tokio runtime on a helper thread.

### Fixed

- `Debug` for `Point` wrote the value's plain expansion, so a value like
  `1e-10000000000` (a 15-byte JSON string) allocated gigabytes or aborted. It now
  prints in space proportional to the digits.

- GPU validation, out-of-memory and internal errors panicked (wgpu's default with no
  handler). They are captured in error scopes and returned as `Error::Gpu`; a lost
  device is detected and reported as `Error::GpuUnavailable`.
- The f32 GPU kernels computed in raw nanoseconds, so cubic Lagrange products
  overflowed `f32` for knot spacings of a couple of hours and beyond, and the f32
  linear kernel clamped every result to ±10⁶.
- The f64 GPU path measured target times from the first *unsorted* input point, so
  unsorted input came out shifted in time.
- The GPU cubic and polynomial kernels added a chunk offset to their thread index
  while binding a per-chunk buffer, so grids larger than one chunk lost points in every
  chunk after the first.
- Non-finite results were silently returned as zero; they are now
  `Error::NonFiniteResult`.
- `BigDecimal` inputs near `f64::MAX` failed conversion.
- Grids too large to allocate aborted the process; they are now
  `Error::OutputTooLarge`.
- Spilling large results to temporary files (and back through a lossy text format)
  is gone, along with the `tempfile` and `sysinfo` dependencies.

### Removed

- `auto_interpolate`, `cpu_interpolate`, `parallel_interpolate`, `gpu_interpolate`:
  use `interpolate` or an `Interpolator` with a `Backend`.
- `pub mod helpers` and its contents (`should_use_gpu`, `InterpolationStrategy`,
  `estimate_output_points`, `generate_target_times`, `TargetTimesIterator`, batch
  state), `apply_fast_path`, `BASE_BATCH_SIZE`, `POINT_SIZE`, and the
  `SECONDS_IN_*` / `DAYS_IN_*` constants.
- `Resolution::to_step`, `to_base`, `to_step_base`, `difference` and `round`
  (`round` was the identity for every resolution); use `step` and `step_nanos`.
- `Spline::number_of_points_required` (now `min_points`) and `Spline::pre_check`.
- `effective_gpu_config`, `gpu_config_applied`, `gpu_buffer_pool_stats`,
  `BufferPoolStats` and the `GpuConfig` fields `buffer_pool`, `num_staging_buffers`
  and the never-implemented `max_command_batch_size`.
- The `gpu-eager-init` feature (a `ctor` running GPU start-up before `main`); call
  `prewarm_gpu()` at startup instead.
- Dependencies: `anyhow`, `tokio` (now optional), `sysinfo`, `tempfile`, `wide`,
  `ctor`.

## [0.1.0] - 2026-10-05

First release as a standalone crate. splimes was developed inside the
[WeftDB](https://github.com/basic-automation/weftdb) workspace, and this repository keeps
that history.

### Added

- Linear, quadratic, cubic and polynomial spline interpolation over irregularly sampled
  `BigDecimal` series, resampled onto a regular grid from nanoseconds to years.
- Automatic backend selection (`auto_interpolate`) between SIMD CPU, `rayon`-parallel
  CPU and GPU (`wgpu`: Vulkan, Metal, DX12), with explicit `cpu_interpolate`,
  `parallel_interpolate` and `gpu_interpolate`.
- An f32 GPU path for adapters without f64 support.
- `prewarm_gpu`, `prewarm_gpu_with_config` (`GpuConfig` presets) and
  `gpu_buffer_pool_stats`.
- The `gpu-eager-init` feature.

### Fixed

- The f32 quadratic shader didn't parse, so quadratic GPU interpolation panicked on
  adapters without f64 support (Apple, most integrated GPUs, Windows WARP).
- Each interpolation call built a full `sysinfo::System` (every process, disk and
  network interface) just to read memory figures: about 445 ms per call, now about
  38 µs.

### Changed

- `Point::random` is now test-only, and `fake` is no longer a runtime dependency.

[Unreleased]: https://github.com/basic-automation/splimes/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/basic-automation/splimes/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/basic-automation/splimes/compare/v0.1.0...v1.0.0
[0.1.0]: https://github.com/basic-automation/splimes/releases/tag/v0.1.0

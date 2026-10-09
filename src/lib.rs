//! Resample irregularly sampled time series onto a regular grid.
//!
//! You have points that arrived whenever they arrived; you want a value every second,
//! minute or day. splimes fits a local spline through the points and evaluates it on the
//! grid, on the CPU (one thread or rayon's pool) or the GPU (wgpu), and tells you, for
//! every output point, whether it is an observation, an interpolation or an extrapolation.
//!
//! ```
//! use bigdecimal::BigDecimal;
//! use chrono::{TimeZone, Utc};
//! use splimes::{Point, PointKind, Resolution, Spline};
//!
//! let at = |secs: i64| Utc.timestamp_opt(1_700_000_000 + secs, 0).unwrap();
//! let readings = [
//!     Point::new(at(0), BigDecimal::from(10)),
//!     Point::new(at(7), BigDecimal::from(14)),
//!     Point::new(at(19), BigDecimal::from(11)),
//!     Point::new(at(30), BigDecimal::from(20)),
//! ];
//!
//! let series = splimes::interpolate(&readings, at(0), at(35), Resolution::Seconds, Spline::Cubic)?;
//!
//! assert_eq!(series.len(), 36);
//! assert_eq!(series.values()[7], BigDecimal::from(14)); // an input, returned exactly
//! assert_eq!(series.kinds()[7], PointKind::Raw);
//! assert_eq!(series.kinds()[8], PointKind::Interpolated);
//! assert_eq!(series.kinds()[33], PointKind::Extrapolated);
//! # Ok::<(), splimes::Error>(())
//! ```
//!
//! For more control — the backend, GPU precision, refusing to fall back to a simpler
//! method — use an [`Interpolator`]. For `f64` data, [`Interpolator::run_f64`] skips the
//! `BigDecimal` conversions.
//!
//! # The grid
//!
//! The output grid is `start, start + step, start + 2·step, …`, up to and including the
//! last point not after `end`, where `step` is the [`Resolution`]. It is anchored at
//! `start`, not aligned to the epoch. `start == end` gives a one-point grid; `start > end`
//! is an error.
//!
//! # Methods
//!
//! [`Spline`] documents what each method computes, inside the data and outside it, and
//! how it steps down when there are too few points. The method actually used is reported
//! by [`Interpolation::spline`].
//!
//! # Input
//!
//! Points may arrive in any order. Points at the same instant collapse to the last one
//! given. Values must be finite in `f64` (`BigDecimal`s beyond ±1.8 × 10³⁰⁸ are rejected,
//! not saturated).
//!
//! Time is on the POSIX scale, exact to the nanosecond. chrono can represent a leap
//! second (`23:59:60.x`); splimes treats it as the same instant as `00:00:00.x` the next
//! second, for inputs, `start` and `end` alike, and never returns one.
//!
//! # Numerical contract
//!
//! Every backend computes the same formula from the same normalised inputs, taking every
//! time difference exactly. `BigDecimal` inputs are first rounded to the nearest `f64`;
//! the contract is about what happens after that. Write `exact` for the method evaluated
//! in exact arithmetic on those rounded inputs, `range` for the spread of the input values
//! (`max − min`), and `ε = 2⁻⁵²`. Then every value `v` satisfies
//!
//! `|v − exact| ≤ bound · range · Λ(t) + ε · |exact|`
//!
//! with `bound` = **1 × 10⁻¹³** in `f64` (`Cpu`, `Parallel`, `Gpu`) and **1 × 10⁻⁵** in
//! `f32` (`Gpu` only), for every method, inside and outside the data. `ε · |exact|` is the
//! rounding any `f64` of that magnitude carries.
//!
//! `Λ(t) = Σⱼ |Lⱼ(t)|` is the Lebesgue function of the window used at `t`: the sum of the
//! magnitudes of its Lagrange basis polynomials, which is how much the method amplifies
//! any perturbation of the input values. In other words, a result is as accurate as
//! perturbing every input value by `bound · range` allows. `Λ` is exactly 1 for linear
//! interpolation inside the data, close to 1 for evenly spread knots, and grows where the
//! problem itself is ill-conditioned: with very irregular spacing (a burst of dense samples
//! beside sparse ones, geometric gaps), with high degree, and outside the data, roughly
//! like `(D / h)^degree` at a distance `D` from the edge window of spacing `h`.
//! Extrapolating a high-degree polynomial far is numerically as well as statistically
//! fragile, and the bound says by how much. `Cubic` holds the first or last input value
//! outside the data, exactly.
//!
//! Beyond the bound:
//!
//! - `Cpu` and `Parallel` results are bit-identical, and `Gpu` `f64` results usually are too.
//! - Points that coincide with an input ([`PointKind::Raw`]) are that input's value,
//!   exactly, on every backend and precision.
//! - `BigDecimal` results are the shortest decimal that round-trips to the computed `f64`.
//! - Non-finite results are errors, never values: an extrapolation that overflows `f64`
//!   returns [`Error::NonFiniteResult`].
//! - On the GPU in `f32`, points it can't compute reliably — values that overflow, and
//!   windows whose knot gaps differ from the mean spacing by more than 1,024× — are
//!   recomputed in `f64` on the CPU and counted by
//!   [`Interpolation::points_recomputed_in_f64`].
//!
//! `tests/contract.rs` checks every backend and precision against an exact 60-digit
//! reference that also computes `Λ`: 258 series (256 randomised, with knot spacings from
//! microseconds to 30 days, plus a dense-burst and a geometric-gap series), each on five
//! grids — across both edges, the middle, the densest stretch, and 100 spacings out — for
//! six methods, plus a two-million-knot series, far-out bounded extrapolation and `f32`
//! windows at the 1,024× gap limit. It fails if a bound is exceeded. `tests/properties.rs`
//! checks the same bound, and the input and provenance rules above, on 1,024 generated
//! series of the input the docs promise to accept: unsorted, with duplicate instants,
//! single points and constant values.
//!
//! # Backends
//!
//! [`Backend::Auto`] (the default) picks by grid size; see [`AutoThresholds`]. It never
//! starts the GPU: it uses it only once the program has, with [`prewarm_gpu`],
//! [`calibrate`] or an explicit [`Backend::Gpu`] call, and only above the GPU thresholds,
//! which by default are never (see `BENCHMARKS.md`). If the GPU fails mid-call, `Auto`
//! reruns on the CPU and says so in [`Interpolation::gpu_fallback`]; an explicit
//! [`Backend::Gpu`] returns the error. splimes never opens the device on a thread of its
//! own: a process exiting while a driver initialises on another thread can crash.
//!
//! The GPU is the high-performance adapter wgpu finds (or the low-power one, with
//! [`GpuConfig::low_power`]). wgpu's environment variables choose another, as they do
//! in any wgpu program: `WGPU_BACKEND` limits the graphics APIs tried (for example
//! `vulkan`, `dx12`, `metal` or `gl`), and `WGPU_ADAPTER_NAME` picks the first adapter
//! whose name contains it, ignoring case. A name that matches no adapter makes the GPU
//! unavailable, with the adapters found in the error; splimes never falls back to another.
//! [`gpu_info`] reports the adapter and API in use.
//!
//! All entry points are synchronous and CPU- or GPU-bound. From async code, use
//! [`Interpolator::spawn`], which runs the work on rayon's pool and returns a future any
//! executor can await (its docs say when it blocks, and what dropping it does, with the
//! GPU), the `tokio` feature's `Interpolator::run_async` (tokio's blocking pool), or your
//! runtime's equivalent of `spawn_blocking`.
//!
//! # Features
//!
//! | Feature | Default | Effect |
//! |---------|---------|--------|
//! | `gpu` | yes | The wgpu backend. Without it, `Backend::Gpu` returns [`Error::GpuUnavailable`]. |
//! | `serde` | yes | `Serialize`/`Deserialize` for [`Point`], [`PointKind`], [`Resolution`] and [`Spline`]. |
//! | `tokio` | no | `Interpolator::run_async` and `Interpolator::run_f64_async`. |
//!
//! # Stability
//!
//! splimes follows semantic versioning. The public API, the numerical contract above and
//! the method definitions on [`Spline`] are covered: a change to what a method computes,
//! beyond rounding within the stated bounds, is a breaking change. The minimum supported
//! Rust version is 1.95; raising it is a minor-version change, announced in the changelog.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![forbid(unsafe_code)]
#![warn(missing_docs, clippy::pedantic, clippy::nursery)]
#![allow(clippy::module_name_repetitions, clippy::cast_precision_loss)]
// `mul_add` is a slow libm call on targets without hardware FMA (including baseline
// x86-64), and fusing would make the CPU round differently from the WGSL kernels.
#![allow(clippy::suboptimal_flops)]
// Proving `Send` through wgpu's nested types exceeds the default limit of 128.
#![recursion_limit = "256"]

pub use auto::{AutoThresholds, auto_thresholds, set_auto_thresholds};
pub use calibrate::{Calibration, CalibrationSample, calibrate};
pub use error::{Error, Result};
pub use gpu::{GpuConfig, GpuInfo, GpuPoolStats, configure_gpu, gpu_config, gpu_info, gpu_pool_stats, prewarm_gpu, prewarm_gpu_with_config};
pub use interpolation::{Backend, Interpolation, Interpolator, Precision, interpolate};
pub use point::{Point, PointKind};
pub use resolution::Resolution;
pub use spawn::InterpolationFuture;
pub use spline::{MAX_POLYNOMIAL_DEGREE, Spline};
pub use value::Value;

#[cfg(feature = "tokio")]
mod asynk;
mod auto;
mod calibrate;
mod error;
mod gpu;
mod interpolation;
mod kernel;
mod point;
mod prepare;
mod resolution;
mod spawn;
mod spline;
mod time;
mod value;

// Compile and run the README's examples as doctests, so they can't drift from the API.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;

use chrono::{DateTime, Utc};

use crate::Spline;

/// The result type returned by every fallible splimes function.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Everything that can go wrong in splimes.
///
/// The enum is `#[non_exhaustive]`: match the variants you handle and keep a `_` arm, so
/// a new failure mode in a minor release isn't a breaking change.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
	/// No input points were supplied.
	#[error("no input points to interpolate")]
	NoPoints,

	/// `start` is after `end`.
	#[error("invalid time range: start {start} is after end {end}")]
	InvalidTimeRange {
		/// The requested start of the output grid.
		start: DateTime<Utc>,
		/// The requested end of the output grid.
		end: DateTime<Utc>,
	},

	/// A polynomial degree of 0 or above [`MAX_POLYNOMIAL_DEGREE`](crate::MAX_POLYNOMIAL_DEGREE).
	#[error("invalid polynomial degree {degree}: must be between 1 and {max}")]
	InvalidDegree {
		/// The degree that was asked for.
		degree: usize,
		/// The largest supported degree.
		max: usize,
	},

	/// The timestamp and value columns passed to
	/// [`Interpolator::run_f64`](crate::Interpolator::run_f64) differ in length.
	#[error("{timestamps} timestamps but {values} values")]
	LengthMismatch {
		/// Length of the timestamp column.
		timestamps: usize,
		/// Length of the value column.
		values: usize,
	},

	/// A polynomial `bounds_factor` that is negative, NaN or infinite.
	#[error("invalid bounds factor {0}: must be finite and not negative")]
	InvalidBoundsFactor(f64),

	/// An input value has no finite `f64` representation (its magnitude is above about
	/// 1.8 × 10³⁰⁸), or an `f64` input is NaN or infinite.
	#[error("input value at {timestamp} is not representable as a finite f64")]
	ValueOutOfRange {
		/// Timestamp of the offending input point.
		timestamp: DateTime<Utc>,
	},

	/// The interpolated value at this timestamp is beyond `f64`'s range: extrapolating far
	/// outside the data with a high-degree method, or values near `f64`'s limits. It names
	/// the first such grid point, on every backend.
	#[error("interpolated value at {timestamp} is beyond f64's range")]
	NonFiniteResult {
		/// Timestamp of the output point that overflowed.
		timestamp: DateTime<Utc>,
	},

	/// The output grid has more points than [`Interpolator::max_points`](crate::Interpolator::max_points)
	/// allows or than can be allocated, for example a nanosecond grid spanning years.
	#[error("output grid of {points} points is too large")]
	OutputTooLarge {
		/// The number of grid points the request describes.
		points: u128,
	},

	/// Too few distinct timestamps for the requested method, and fallback was disabled
	/// with [`Interpolator::exact`](crate::Interpolator::exact).
	#[error("{spline} needs at least {required} distinct points, got {available}")]
	InsufficientPoints {
		/// The method that was asked for.
		spline: Spline,
		/// Distinct timestamps the method needs.
		required: usize,
		/// Distinct timestamps supplied.
		available: usize,
	},

	/// No usable GPU: the `gpu` feature is off, no adapter was found, the device was
	/// lost, or the adapter can't run the requested precision.
	#[error("GPU unavailable: {0}")]
	GpuUnavailable(String),

	/// The GPU reported an error (validation, out of memory, or an internal error) while
	/// running an interpolation. [`Backend::Auto`](crate::Backend::Auto) catches this and
	/// reruns on the CPU.
	#[error("GPU error: {0}")]
	Gpu(String),

	/// [`configure_gpu`](crate::configure_gpu) was called after the GPU had already
	/// started, or a second time.
	#[error("GPU already configured: {0}")]
	GpuAlreadyConfigured(&'static str),

	/// A string didn't name a [`Resolution`](crate::Resolution) or
	/// [`Spline`](crate::Spline).
	#[error("cannot parse {kind} from {input:?}")]
	Parse {
		/// What was being parsed: `"resolution"` or `"spline"`.
		kind: &'static str,
		/// The input that failed to parse.
		input: String,
	},

	/// The blocking task running an async interpolation panicked or was cancelled.
	#[cfg(feature = "tokio")]
	#[cfg_attr(docsrs, doc(cfg(feature = "tokio")))]
	#[error("interpolation task failed: {0}")]
	Task(String),
}

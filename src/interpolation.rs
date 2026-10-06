use std::fmt;

use bigdecimal::BigDecimal;
use chrono::{DateTime, Utc};
use rayon::prelude::*;

use crate::{
	Error, Point, PointKind, Resolution, Result, Spline, kernel::{self, Method}, prepare::Knots, time::Grid, value::Value
};

/// Where an interpolation runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum Backend {
	/// Choose per call by grid size (the default). Small grids run on [`Cpu`](Self::Cpu),
	/// larger ones on [`Parallel`](Self::Parallel), and grids above the GPU threshold on
	/// [`Gpu`](Self::Gpu) if the program has started the GPU (`Auto` never does) and it can
	/// run the requested [`Precision`]. A GPU failure falls back to `Parallel` and is
	/// reported by [`Interpolation::gpu_fallback`]. See [`AutoThresholds`](crate::AutoThresholds).
	#[default]
	Auto,
	/// The calling thread only.
	Cpu,
	/// rayon's global thread pool.
	Parallel,
	/// The GPU, via wgpu. Fails with [`Error::GpuUnavailable`] or [`Error::Gpu`] instead of
	/// falling back.
	Gpu,
}

impl fmt::Display for Backend {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.write_str(match self {
			Self::Auto => "auto",
			Self::Cpu => "cpu",
			Self::Parallel => "parallel",
			Self::Gpu => "gpu",
		})
	}
}

/// Floating-point precision of the kernel.
///
/// The CPU backends always compute in `f64`. The GPU computes in `f64` when the adapter
/// supports it (`SHADER_F64`, which most discrete desktop GPUs have and Apple and most
/// integrated GPUs lack). `f32` must be asked for: it is faster on GPUs with slow `f64`,
/// and the only option on the rest, at the cost documented under
/// "Numerical contract" in the crate docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum Precision {
	/// IEEE 754 double precision (the default).
	#[default]
	F64,
	/// IEEE 754 single precision, on the GPU only. CPU backends ignore it.
	F32,
}

impl fmt::Display for Precision {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.write_str(match self {
			Self::F64 => "f64",
			Self::F32 => "f32",
		})
	}
}

/// A configured interpolation: method, grid resolution and where to run.
///
/// Cheap to build and `Copy`, so keep one around or make one per call.
///
/// ```
/// use bigdecimal::BigDecimal;
/// use chrono::{TimeZone, Utc};
/// use splimes::{Backend, Interpolator, Point, PointKind, Resolution, Spline};
///
/// let at = |s| Utc.timestamp_opt(s, 0).unwrap();
/// let points = [Point::new(at(0), BigDecimal::from(0)), Point::new(at(10), BigDecimal::from(10))];
///
/// let series = Interpolator::new(Spline::Linear, Resolution::Seconds).backend(Backend::Cpu).run(&points, at(0), at(12))?;
///
/// assert_eq!(series.len(), 13);
/// assert_eq!(series.values()[5], BigDecimal::from(5));
/// assert_eq!(series.kinds()[0], PointKind::Raw);
/// assert_eq!(series.kinds()[5], PointKind::Interpolated);
/// assert_eq!(series.kinds()[12], PointKind::Extrapolated);
/// # Ok::<(), splimes::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Interpolator {
	spline: Spline,
	resolution: Resolution,
	backend: Backend,
	precision: Precision,
	exact: bool,
	max_points: usize,
}

impl Interpolator {
	/// An interpolator for `spline` onto a `resolution` grid, on [`Backend::Auto`] in
	/// [`Precision::F64`], stepping the method down when there are too few points.
	#[must_use]
	pub const fn new(spline: Spline, resolution: Resolution) -> Self {
		Self { spline, resolution, backend: Backend::Auto, precision: Precision::F64, exact: false, max_points: usize::MAX }
	}

	/// Where to run.
	#[must_use]
	pub const fn backend(mut self, backend: Backend) -> Self {
		self.backend = backend;
		self
	}

	/// GPU precision. The default, `F64`, keeps [`Backend::Auto`] off GPUs without `f64`
	/// support and makes [`Backend::Gpu`] fail on them.
	#[must_use]
	pub const fn gpu_precision(mut self, precision: Precision) -> Self {
		self.precision = precision;
		self
	}

	/// With `true`, too few distinct points for the method is an
	/// [`Error::InsufficientPoints`] instead of a step down to a simpler method.
	#[must_use]
	pub const fn exact(mut self, exact: bool) -> Self {
		self.exact = exact;
		self
	}

	/// Refuse grids of more than `points` points with [`Error::OutputTooLarge`], before
	/// allocating anything. Unlimited by default.
	///
	/// Set this when `start`, `end` or the resolution come from untrusted input: a
	/// nanosecond grid over a day is 86 trillion points, and a grid that fits in address
	/// space can still exhaust memory. Each output point costs about 40 bytes with `f64`
	/// values and more with `BigDecimal`.
	#[must_use]
	pub const fn max_points(mut self, points: usize) -> Self {
		self.max_points = points;
		self
	}

	/// The configured method.
	#[must_use]
	pub const fn spline(&self) -> Spline {
		self.spline
	}

	/// The configured grid resolution.
	#[must_use]
	pub const fn resolution(&self) -> Resolution {
		self.resolution
	}

	/// Resamples `points` onto the grid `start, start + step, …, ≤ end`.
	///
	/// `points` may be in any order. Points sharing a timestamp collapse to the last one
	/// in slice order.
	///
	/// # Errors
	///
	/// - [`Error::InvalidDegree`] / [`Error::InvalidBoundsFactor`]: bad `Polynomial` parameters.
	/// - [`Error::InvalidTimeRange`]: `start > end`.
	/// - [`Error::OutputTooLarge`]: the grid exceeds [`max_points`](Self::max_points) or
	///   can't be allocated.
	/// - [`Error::NoPoints`]: `points` is empty.
	/// - [`Error::ValueOutOfRange`]: a value is beyond `f64`'s range.
	/// - [`Error::InsufficientPoints`]: too few points, with [`exact`](Self::exact) set.
	/// - [`Error::NonFiniteResult`]: extrapolation overflowed.
	/// - [`Error::GpuUnavailable`] / [`Error::Gpu`]: on [`Backend::Gpu`] only.
	pub fn run(&self, points: &[Point], start: DateTime<Utc>, end: DateTime<Utc>) -> Result<Interpolation<BigDecimal>> {
		self.run_generic(points.iter().map(|p| p.timestamp), points.iter().map(|p| &p.value), start, end)
	}

	/// [`run`](Self::run) for `f64` samples given as parallel columns, skipping the
	/// `BigDecimal` conversions on the way in and out.
	///
	/// # Errors
	///
	/// As [`run`](Self::run), plus [`Error::LengthMismatch`] if the columns differ in
	/// length; [`Error::ValueOutOfRange`] covers NaN and infinite inputs.
	pub fn run_f64(&self, timestamps: &[DateTime<Utc>], values: &[f64], start: DateTime<Utc>, end: DateTime<Utc>) -> Result<Interpolation<f64>> {
		if timestamps.len() != values.len() {
			return Err(Error::LengthMismatch { timestamps: timestamps.len(), values: values.len() });
		}
		self.run_generic(timestamps.iter().copied(), values.iter(), start, end)
	}

	fn run_generic<'a, V: Value>(&self, timestamps: impl ExactSizeIterator<Item = DateTime<Utc>>, values: impl ExactSizeIterator<Item = &'a V>, start: DateTime<Utc>, end: DateTime<Utc>) -> Result<Interpolation<V>> {
		self.spline.validate()?;
		let grid = Grid::new(start, end, self.resolution)?;
		if grid.len > self.max_points {
			return Err(Error::OutputTooLarge { points: grid.len as u128 });
		}
		let knots = Knots::new(timestamps, values)?;
		let spline = self.spline.fallback_for(knots.len());
		if self.exact && knots.len() < self.spline.min_points() {
			return Err(Error::InsufficientPoints { spline: self.spline, required: self.spline.min_points(), available: knots.len() });
		}
		let method = Method::new(spline, &knots);

		let parallel = self.backend != Backend::Cpu;
		let mut normalised = filled(grid.len, 0.0, parallel)?;
		let run = self.compute(&knots, &grid, &method, &mut normalised)?;
		assemble(&knots, &grid, &method, &normalised, run, spline, self.spline)
	}

	fn compute<V: Value>(&self, knots: &Knots<'_, V>, grid: &Grid, method: &Method, out: &mut [f64]) -> Result<Run> {
		let cpu = |backend: Backend, out: &mut [f64], gpu_fallback: Option<Error>| {
			if backend == Backend::Cpu {
				kernel::eval_serial(knots, grid, method, out);
			} else {
				kernel::eval_parallel(knots, grid, method, out);
			}
			Run { backend, precision: Precision::F64, gpu_fallback, recomputed: 0 }
		};
		let gpu = |out: &mut [f64]| {
			crate::gpu::eval(knots, grid, method, self.precision, out)?;
			let unsafe_windows = if self.precision == Precision::F32 { kernel::f32_unsafe_windows(knots, method) } else { None };
			let recomputed = kernel::repair(knots, grid, method, out, unsafe_windows.as_deref());
			if recomputed > 0 {
				log::debug!("splimes: recomputed {recomputed} point(s) in f64 on the CPU that the GPU couldn't compute reliably");
			}
			Ok::<_, Error>(Run { backend: Backend::Gpu, precision: self.precision, gpu_fallback: None, recomputed })
		};
		match self.backend {
			Backend::Cpu | Backend::Parallel => Ok(cpu(self.backend, out, None)),
			Backend::Gpu => gpu(out),
			Backend::Auto => match crate::auto::choose(grid.len, self.precision) {
				Backend::Gpu => match gpu(out) {
					Ok(run) => Ok(run),
					Err(e) => {
						log::warn!("splimes: GPU interpolation failed, rerunning on the CPU: {e}");
						Ok(cpu(Backend::Parallel, out, Some(e)))
					}
				},
				backend => Ok(cpu(backend, out, None)),
			},
		}
	}
}

/// How a computation actually ran.
struct Run {
	backend: Backend,
	precision: Precision,
	gpu_fallback: Option<Error>,
	recomputed: usize,
}

/// Grid points per rayon task when assembling output.
const ASSEMBLE_CHUNK: usize = 16 * 1024;

/// A `len`-element vector of `value`, or [`Error::OutputTooLarge`] if it can't be
/// allocated. With `parallel`, rayon fills large ones, so first-touch page faults — most
/// of the cost of a fresh multi-hundred-megabyte buffer — are spread over every core.
fn filled<T: Clone + Send + Sync>(len: usize, value: T, parallel: bool) -> Result<Vec<T>> {
	let mut v = Vec::new();
	v.try_reserve_exact(len).map_err(|_| Error::OutputTooLarge { points: len as u128 })?;
	if parallel && len >= 1 << 16 {
		v.par_extend(rayon::iter::repeat_n(value, len));
	} else {
		v.resize(len, value);
	}
	Ok(v)
}

/// One chunk of the three output columns, as zipped chunk iterators yield them.
type Columns<'c, V> = ((&'c mut [DateTime<Utc>], &'c mut [PointKind]), &'c mut [V]);

/// Builds the output columns from the normalised kernel results: timestamps, provenance
/// by a merge walk against the knots, and values (the caller's own for raw points).
fn assemble<V: Value>(knots: &Knots<'_, V>, grid: &Grid, method: &Method, normalised: &[f64], run: Run, spline: Spline, requested: Spline) -> Result<Interpolation<V>> {
	let parallel = run.backend != Backend::Cpu;
	let mut timestamps = filled(grid.len, grid.start, parallel)?;
	let mut kinds = filled(grid.len, PointKind::Interpolated, parallel)?;
	let mut values = filled(grid.len, V::placeholder(), parallel)?;
	// Fills one chunk; on a non-finite value, returns the index of the first one in it.
	let chunk = |c: usize, ((ts, ks), vs): Columns<'_, V>| -> Result<(), usize> {
		let first = c * ASSEMBLE_CHUNK;
		let ys = &normalised[first..first + vs.len()];
		let offsets = &knots.offsets;
		let (first_knot, last_knot) = (knots.originals[0], knots.originals[offsets.len() - 1]);
		let last = offsets[offsets.len() - 1];
		let mut cursor = None;
		for (i, (((kind, value), &y), offset)) in ks.iter_mut().zip(vs.iter_mut()).zip(ys).zip(grid.offsets(knots.t0, first)).enumerate() {
			let mut c = cursor.unwrap_or_else(|| offsets.partition_point(|&o| o < offset));
			while c < offsets.len() && offsets[c] < offset {
				c += 1;
			}
			cursor = Some(c);
			if c < offsets.len() && offsets[c] == offset {
				*kind = PointKind::Raw;
				*value = knots.originals[c].clone();
			} else if offset < 0 || offset > last {
				*kind = PointKind::Extrapolated;
				// `Cubic` holds the edge values: return them exactly, not their round trip
				// through the normalised kernel.
				*value = match (method.hold, offset < 0) {
					(true, true) => first_knot.clone(),
					(true, false) => last_knot.clone(),
					(false, _) => denormalise(knots, y).ok_or(first + i)?,
				};
			} else {
				*value = denormalise(knots, y).ok_or(first + i)?;
			}
		}
		for (t, time) in ts.iter_mut().zip(grid.timestamps(first, vs.len())) {
			*t = time;
		}
		Ok(())
	};

	// The first failing point in grid order, whichever chunk finishes first, so every
	// backend reports the same timestamp.
	let failure = if parallel { timestamps.par_chunks_mut(ASSEMBLE_CHUNK).zip(kinds.par_chunks_mut(ASSEMBLE_CHUNK)).zip(values.par_chunks_mut(ASSEMBLE_CHUNK)).enumerate().filter_map(|(c, cols)| chunk(c, cols).err()).min() } else { timestamps.chunks_mut(ASSEMBLE_CHUNK).zip(kinds.chunks_mut(ASSEMBLE_CHUNK)).zip(values.chunks_mut(ASSEMBLE_CHUNK)).enumerate().find_map(|(c, cols)| chunk(c, cols).err()) };
	if let Some(k) = failure {
		return Err(Error::NonFiniteResult { timestamp: grid.at(k) });
	}
	Ok(Interpolation { timestamps, values, kinds, spline, requested, backend: run.backend, precision: run.precision, gpu_fallback: run.gpu_fallback, recomputed: run.recomputed })
}

/// A normalised kernel result back in the caller's units, if it is finite there.
fn denormalise<V: Value>(knots: &Knots<'_, V>, y: f64) -> Option<V> {
	let v = y * knots.scale + knots.centre;
	// Near f64::MAX the product can overflow although the sum doesn't; a fused multiply-add
	// rounds once, at the end. (It's slow without hardware FMA, so only as a fallback.) A
	// value within the knots' own range is finite however `centre` and `scale` rounded.
	let v = if v.is_finite() {
		v
	} else if y.abs() <= 1.0 {
		y.mul_add(knots.scale, knots.centre).clamp(knots.min, knots.max)
	} else {
		y.mul_add(knots.scale, knots.centre)
	};
	V::from_finite_f64(v)
}

/// A resampled series and a report of how it was produced.
///
/// The series is stored as three parallel columns — timestamps, values and
/// [`PointKind`]s — in grid order.
#[derive(Debug, Clone, PartialEq)]
pub struct Interpolation<V> {
	timestamps: Vec<DateTime<Utc>>,
	values: Vec<V>,
	kinds: Vec<PointKind>,
	spline: Spline,
	requested: Spline,
	backend: Backend,
	precision: Precision,
	gpu_fallback: Option<Error>,
	recomputed: usize,
}

impl<V> Interpolation<V> {
	/// The number of grid points.
	#[must_use]
	pub const fn len(&self) -> usize {
		self.timestamps.len()
	}

	/// Whether the grid is empty. It never is: a grid always contains its start.
	#[must_use]
	pub const fn is_empty(&self) -> bool {
		self.timestamps.is_empty()
	}

	/// Grid timestamps, ascending.
	#[must_use]
	pub fn timestamps(&self) -> &[DateTime<Utc>] {
		&self.timestamps
	}

	/// Values, one per grid timestamp.
	#[must_use]
	pub fn values(&self) -> &[V] {
		&self.values
	}

	/// Provenance, one per grid timestamp.
	#[must_use]
	pub fn kinds(&self) -> &[PointKind] {
		&self.kinds
	}

	/// The method actually used. It differs from [`requested_spline`](Self::requested_spline)
	/// when there were too few distinct points (see [`Spline`]).
	#[must_use]
	pub const fn spline(&self) -> Spline {
		self.spline
	}

	/// The method that was asked for.
	#[must_use]
	pub const fn requested_spline(&self) -> Spline {
		self.requested
	}

	/// The backend that produced the values: `Cpu`, `Parallel` or `Gpu`, never `Auto`.
	#[must_use]
	pub const fn backend(&self) -> Backend {
		self.backend
	}

	/// The precision the kernel ran in.
	#[must_use]
	pub const fn precision(&self) -> Precision {
		self.precision
	}

	/// How many points the GPU couldn't compute reliably, so they were recomputed in `f64`
	/// on the CPU: values that overflowed, and in [`Precision::F32`], every point in a
	/// window whose knot gaps differ from the series' mean spacing by more than a factor
	/// of 1,024 (dense bursts beside sparse stretches), where single precision underflows.
	/// Always 0 on the CPU backends, and normally 0 on the GPU.
	#[must_use]
	pub const fn points_recomputed_in_f64(&self) -> usize {
		self.recomputed
	}

	/// Why [`Backend::Auto`] abandoned the GPU for this call, if it tried and failed.
	#[must_use]
	pub const fn gpu_fallback(&self) -> Option<&Error> {
		self.gpu_fallback.as_ref()
	}

	/// `(timestamp, value, kind)` for each grid point.
	pub fn iter(&self) -> impl ExactSizeIterator<Item = (DateTime<Utc>, &V, PointKind)> + '_ {
		self.timestamps.iter().zip(&self.values).zip(&self.kinds).map(|((&t, v), &k)| (t, v, k))
	}

	/// The three columns, consuming the result.
	#[must_use]
	pub fn into_parts(self) -> (Vec<DateTime<Utc>>, Vec<V>, Vec<PointKind>) {
		(self.timestamps, self.values, self.kinds)
	}
}

impl Interpolation<BigDecimal> {
	/// The series as [`Point`]s, dropping the provenance.
	#[must_use]
	pub fn into_points(self) -> Vec<Point> {
		self.timestamps.into_iter().zip(self.values).map(|(timestamp, value)| Point { timestamp, value }).collect()
	}
}

/// Resamples `points` onto a `resolution` grid from `start` to `end` with `spline`, on
/// [`Backend::Auto`].
///
/// Shorthand for `Interpolator::new(spline, resolution).run(points, start, end)`; see
/// [`Interpolator::run`] for the errors.
///
/// # Errors
///
/// See [`Interpolator::run`].
pub fn interpolate(points: &[Point], start: DateTime<Utc>, end: DateTime<Utc>, resolution: Resolution, spline: Spline) -> Result<Interpolation<BigDecimal>> {
	Interpolator::new(spline, resolution).run(points, start, end)
}

#[cfg(test)]
mod tests {
	use chrono::TimeZone;

	use super::*;

	#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
	fn at(secs: i64) -> DateTime<Utc> {
		Utc.timestamp_opt(secs, 0).single().expect("valid")
	}

	#[cfg(feature = "gpu")]
	#[test]
	fn auto_falls_back_to_the_cpu_when_the_gpu_fails() {
		crate::auto::FORCE_GPU.set(true);
		crate::gpu::run::tests::FAIL.set(true);
		let points = [Point::new(at(0), BigDecimal::from(0)), Point::new(at(10), BigDecimal::from(10))];
		let auto = Interpolator::new(Spline::Linear, Resolution::Seconds).run(&points, at(0), at(10));
		let explicit = Interpolator::new(Spline::Linear, Resolution::Seconds).backend(Backend::Gpu).run(&points, at(0), at(10));
		crate::auto::FORCE_GPU.set(false);
		crate::gpu::run::tests::FAIL.set(false);

		let auto = auto.expect("Auto recovers on the CPU");
		assert_eq!(auto.backend(), Backend::Parallel);
		assert!(matches!(auto.gpu_fallback(), Some(Error::Gpu(_) | Error::GpuUnavailable(_))), "{:?}", auto.gpu_fallback());
		assert_eq!(auto.values()[4], BigDecimal::from(4));
		assert!(explicit.is_err(), "an explicit GPU request reports the failure instead");
	}
}

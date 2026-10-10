use std::{
	fmt, sync::atomic::{AtomicUsize, Ordering}
};

use bigdecimal::BigDecimal;
use chrono::{DateTime, Utc};
use rayon::prelude::*;

use crate::{
	Error, Point, PointKind, Resolution, Result, Spline, kernel::{self, Method, Nanos}, prepare::Knots, time::{Clock, Grid}, value::Value
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
	///
	/// Whatever the grid size, inputs of 16 Ki points or more are prepared on rayon's pool;
	/// only [`Cpu`](Self::Cpu) keeps the whole call on the calling thread.
	#[default]
	Auto,
	/// The calling thread only, including preparing the input.
	Cpu,
	/// rayon's global thread pool. Every backend but `Cpu` also prepares large inputs
	/// (16 Ki points or more) there.
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
	pub(crate) backend: Backend,
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
		self.run_points(points, start, end, true)
	}

	/// [`run`](Self::run); `auto_gpu` false bars [`Backend::Auto`] from the GPU, as for a
	/// job spawned before the GPU started.
	pub(crate) fn run_points(&self, points: &[Point], start: DateTime<Utc>, end: DateTime<Utc>, auto_gpu: bool) -> Result<Interpolation<BigDecimal>> {
		self.run_generic(points.len(), |i| points[i].timestamp, |i| &points[i].value, (start, end), auto_gpu)
	}

	/// [`run`](Self::run) for `f64` samples given as parallel columns, skipping the
	/// `BigDecimal` conversions on the way in and out.
	///
	/// # Errors
	///
	/// As [`run`](Self::run), plus [`Error::LengthMismatch`] if the columns differ in
	/// length; [`Error::ValueOutOfRange`] covers NaN and infinite inputs.
	pub fn run_f64(&self, timestamps: &[DateTime<Utc>], values: &[f64], start: DateTime<Utc>, end: DateTime<Utc>) -> Result<Interpolation<f64>> {
		self.run_columns(timestamps, values, start, end, true)
	}

	/// [`run_f64`](Self::run_f64), with `auto_gpu` as for [`run_points`](Self::run_points).
	pub(crate) fn run_columns(&self, timestamps: &[DateTime<Utc>], values: &[f64], start: DateTime<Utc>, end: DateTime<Utc>, auto_gpu: bool) -> Result<Interpolation<f64>> {
		if timestamps.len() != values.len() {
			return Err(Error::LengthMismatch { timestamps: timestamps.len(), values: values.len() });
		}
		self.run_generic(timestamps.len(), |i| timestamps[i], |i| &values[i], (start, end), auto_gpu)
	}

	/// The `len` samples `(timestamp(i), value(i))`, resampled onto `start..=end`.
	fn run_generic<'a, V: Value>(&self, len: usize, timestamp: impl Fn(usize) -> DateTime<Utc> + Sync, value: impl Fn(usize) -> &'a V + Sync, (start, end): (DateTime<Utc>, DateTime<Utc>), auto_gpu: bool) -> Result<Interpolation<V>> {
		self.spline.validate()?;
		let grid = Grid::new(start, end, self.resolution)?;
		if grid.len > self.max_points {
			return Err(Error::OutputTooLarge { points: grid.len as u128 });
		}
		let knots = Knots::new(len, timestamp, value, self.backend != Backend::Cpu)?;
		let spline = self.spline.fallback_for(knots.len());
		if self.exact && knots.len() < self.spline.min_points() {
			return Err(Error::InsufficientPoints { spline: self.spline, required: self.spline.min_points(), available: knots.len() });
		}
		let method = Method::new(spline, &knots);

		let parallel = self.backend != Backend::Cpu;
		let mut normalised = filled(grid.len, 0.0, parallel)?;
		let run = self.compute(&knots, &grid, &method, &mut normalised, auto_gpu)?;
		assemble(&knots, &grid, &method, &normalised, run, spline, self.spline)
	}

	fn compute<V: Value>(&self, knots: &Knots<'_, V>, grid: &Grid, method: &Method, out: &mut [f64], auto_gpu: bool) -> Result<Run> {
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
			Backend::Auto => match crate::auto::choose(grid.len, self.precision, auto_gpu) {
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

/// Builds the output columns from the normalised kernel results: timestamps, provenance
/// by a merge walk against the knots, and values (the caller's own for raw points).
///
/// Each column is written once, straight into its allocation: in parallel by rayon's
/// indexed collects, whose jobs each take a run of consecutive points, so the cursors in
/// [`Clock`] and [`Walk`] advance by addition and seek only at the start of a run.
fn assemble<V: Value>(knots: &Knots<'_, V>, grid: &Grid, method: &Method, normalised: &[f64], run: Run, spline: Spline, requested: Spline) -> Result<Interpolation<V>> {
	fn reserved<T>(len: usize) -> Result<Vec<T>> {
		let mut v = Vec::new();
		v.try_reserve_exact(len).map_err(|_| Error::OutputTooLarge { points: len as u128 })?;
		Ok(v)
	}
	let len = grid.len;
	let mut columns = Columns { timestamps: reserved(len)?, kinds: reserved(len)?, values: reserved(len)? };
	let parallel = run.backend != Backend::Cpu;
	// The provenance walk compares offsets in `i64` when they all fit, as the kernel does.
	let failure = match kernel::narrow_offsets(knots, grid) {
		Some(offsets) => columns.fill(offsets, knots, grid, method, normalised, parallel),
		None => columns.fill(&knots.offsets, knots, grid, method, normalised, parallel),
	};
	if let Some(k) = failure {
		return Err(Error::NonFiniteResult { timestamp: grid.at(k) });
	}
	let Columns { timestamps, kinds, values } = columns;
	Ok(Interpolation { timestamps, values, kinds, spline, requested, backend: run.backend, precision: run.precision, gpu_fallback: run.gpu_fallback, recomputed: run.recomputed })
}

/// The three output columns, allocated but empty until [`fill`](Self::fill).
struct Columns<V> {
	timestamps: Vec<DateTime<Utc>>,
	kinds: Vec<PointKind>,
	values: Vec<V>,
}

impl<V: Value> Columns<V> {
	/// Fills every column, `offsets` being the knots' offsets from the first knot. Returns the
	/// first point in grid order whose value isn't finite, whichever job meets it first, so
	/// every backend reports the same timestamp.
	fn fill<N: Nanos>(&mut self, offsets: &[N], knots: &Knots<'_, V>, grid: &Grid, method: &Method, normalised: &[f64], parallel: bool) -> Option<usize> {
		let step = N::saturating_from(i128::from(grid.step_nanos));
		let len = grid.len;
		if !parallel {
			// One pass, stopping at the first non-finite value.
			let (mut clock, mut walk) = (Clock::new(grid), Walk::new(step));
			for (k, &y) in normalised.iter().enumerate() {
				let (kind, value) = walk.point(offsets, knots, grid, method, k, y);
				let Some(value) = value else {
					return Some(k);
				};
				self.timestamps.push(clock.at(k));
				self.kinds.push(kind);
				self.values.push(value);
			}
			return None;
		}
		let failure = AtomicUsize::new(usize::MAX);
		let point = |walk: &mut Walk<N>, k: usize| -> (PointKind, V) {
			let (kind, value) = walk.point(offsets, knots, grid, method, k, normalised[k]);
			let value = value.unwrap_or_else(|| {
				failure.fetch_min(k, Ordering::Relaxed);
				V::placeholder()
			});
			(kind, value)
		};
		(0..len).into_par_iter().with_min_len(ASSEMBLE_CHUNK).map_init(|| Clock::new(grid), Clock::at).collect_into_vec(&mut self.timestamps);
		(0..len).into_par_iter().with_min_len(ASSEMBLE_CHUNK).map_init(|| Walk::new(step), point).unzip_into_vecs(&mut self.kinds, &mut self.values);
		Some(failure.into_inner()).filter(|&k| k != usize::MAX)
	}
}

/// The provenance walk: where grid point `k` falls among the knots. Ascending indices
/// advance by addition and a forward scan; any other index seeks with a binary search.
struct Walk<N> {
	/// The index `offset` and `cursor` describe.
	next: usize,
	/// Nanoseconds from the first knot to the last grid point visited.
	offset: N,
	/// The grid step in nanoseconds.
	step: N,
	/// The first knot at or after `offset`.
	cursor: usize,
}

impl<N: Nanos> Walk<N> {
	const fn new(step: N) -> Self {
		Self { next: usize::MAX, offset: step, step, cursor: 0 }
	}

	/// Point `k`'s provenance and value, from its normalised kernel result `y`; `None` if the
	/// value isn't finite in the caller's units. `offsets` are the knots' offsets from the
	/// first knot, so `offsets[0]` is zero.
	fn point<V: Value>(&mut self, offsets: &[N], knots: &Knots<'_, V>, grid: &Grid, method: &Method, k: usize, y: f64) -> (PointKind, Option<V>) {
		if k == self.next {
			self.offset = self.offset + self.step;
		} else {
			self.offset = N::saturating_from(grid.offset_nanos(knots.t0, k));
			self.cursor = offsets.partition_point(|&o| o < self.offset);
		}
		self.next = k.saturating_add(1);
		let offset = self.offset;
		while self.cursor < offsets.len() && offsets[self.cursor] < offset {
			self.cursor += 1;
		}
		let c = self.cursor;
		if c < offsets.len() && offsets[c] == offset {
			return (PointKind::Raw, Some(knots.originals[c].clone()));
		}
		let last = offsets.len() - 1;
		let before = offset < offsets[0];
		if before || offset > offsets[last] {
			// `Cubic` holds the edge values: return them exactly, not their round trip
			// through the normalised kernel.
			let value = match (method.hold, before) {
				(true, true) => Some(knots.originals[0].clone()),
				(true, false) => Some(knots.originals[last].clone()),
				(false, _) => denormalise(knots, y),
			};
			return (PointKind::Extrapolated, value);
		}
		(PointKind::Interpolated, denormalise(knots, y))
	}
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

	/// The backend that computed the values: `Cpu`, `Parallel` or `Gpu`, never `Auto`. Under
	/// [`Backend::Auto`], a large input may still have been prepared on rayon's pool.
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

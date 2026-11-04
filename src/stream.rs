//! Streamed output: the grid delivered a chunk at a time, in order, each as soon as it is
//! computed, so a caller can serialise or send the first points while later ones are still
//! being computed (on the GPU, the next chunk computes while the callback runs).

use std::{
	ops::ControlFlow, panic::{AssertUnwindSafe, catch_unwind, resume_unwind}
};

use chrono::{DateTime, Utc};
use rayon::prelude::*;

use crate::{
	Backend, Error, Interpolator, PointKind, Precision, Result, Spline, TickUnit, gpu::ChunkSink, interpolation::{ASSEMBLE_CHUNK, Prepared, Walk}, kernel::{self, Method, Nanos}, prepare::Knots, time::{Clock, Grid}, value::Value
};

/// One chunk of a streamed interpolation: grid points `first()..first() + len()`, as
/// parallel columns. `T` is the time representation: `DateTime<Utc>` from
/// [`Interpolator::stream_f64`], `i64` epoch ticks from [`Interpolator::stream_epoch`].
///
/// The slices borrow buffers splimes reuses for the next chunk; copy out what you keep.
#[derive(Debug, Clone, Copy)]
pub struct Chunk<'a, T> {
	first: usize,
	times: &'a [T],
	values: &'a [f64],
	kinds: &'a [PointKind],
}

impl<'a, T: Copy> Chunk<'a, T> {
	/// The grid index of this chunk's first point.
	#[must_use]
	pub const fn first(&self) -> usize {
		self.first
	}

	/// The number of points in the chunk. Never zero.
	#[must_use]
	pub const fn len(&self) -> usize {
		self.values.len()
	}

	/// Always `false`: splimes never delivers an empty chunk.
	#[must_use]
	pub const fn is_empty(&self) -> bool {
		self.values.is_empty()
	}

	/// The grid times, ascending.
	#[must_use]
	pub const fn times(&self) -> &'a [T] {
		self.times
	}

	/// The values, one per time.
	#[must_use]
	pub const fn values(&self) -> &'a [f64] {
		self.values
	}

	/// Provenance, one per time.
	#[must_use]
	pub const fn kinds(&self) -> &'a [PointKind] {
		self.kinds
	}

	/// `(time, value, kind)` for each point.
	pub fn iter(&self) -> impl ExactSizeIterator<Item = (T, f64, PointKind)> + 'a {
		self.times.iter().zip(self.values).zip(self.kinds).map(|((&t, &v), &k)| (t, v, k))
	}
}

/// How a streamed interpolation ran: the report [`Interpolation`](crate::Interpolation)
/// carries, plus how much was delivered.
#[derive(Debug, Clone, PartialEq)]
pub struct StreamSummary {
	len: usize,
	delivered: usize,
	chunks: usize,
	spline: Spline,
	requested: Spline,
	backend: Backend,
	precision: Precision,
	gpu_fallback: Option<Error>,
	recomputed: usize,
}

impl StreamSummary {
	/// The number of points in the grid.
	#[must_use]
	pub const fn len(&self) -> usize {
		self.len
	}

	/// Whether the grid is empty. It never is: a grid always contains its start.
	#[must_use]
	pub const fn is_empty(&self) -> bool {
		self.len == 0
	}

	/// Points delivered to the callback: [`len`](Self::len) unless it stopped the run.
	#[must_use]
	pub const fn delivered(&self) -> usize {
		self.delivered
	}

	/// Whether every grid point was delivered.
	#[must_use]
	pub const fn is_complete(&self) -> bool {
		self.delivered == self.len
	}

	/// Chunks delivered.
	#[must_use]
	pub const fn chunks(&self) -> usize {
		self.chunks
	}

	/// The method actually used, as [`Interpolation::spline`](crate::Interpolation::spline).
	#[must_use]
	pub const fn spline(&self) -> Spline {
		self.spline
	}

	/// The method that was asked for.
	#[must_use]
	pub const fn requested_spline(&self) -> Spline {
		self.requested
	}

	/// The backend that computed the values: `Cpu`, `Parallel` or `Gpu`, never `Auto`. After
	/// a GPU fallback it is `Parallel`, although chunks delivered before the failure came
	/// from the GPU.
	#[must_use]
	pub const fn backend(&self) -> Backend {
		self.backend
	}

	/// The precision the kernel ran in (`F64` after a GPU fallback).
	#[must_use]
	pub const fn precision(&self) -> Precision {
		self.precision
	}

	/// As [`Interpolation::gpu_fallback`](crate::Interpolation::gpu_fallback).
	#[must_use]
	pub const fn gpu_fallback(&self) -> Option<&Error> {
		self.gpu_fallback.as_ref()
	}

	/// As [`Interpolation::points_recomputed_in_f64`](crate::Interpolation::points_recomputed_in_f64),
	/// over the chunks computed on the GPU.
	#[must_use]
	pub const fn points_recomputed_in_f64(&self) -> usize {
		self.recomputed
	}
}

impl Interpolator {
	/// [`run_f64`](Self::run_f64), delivered a chunk at a time: `sink` receives grid points
	/// `0..k`, then `k..2k`, …, in order, each chunk as soon as it is computed, so output
	/// can be serialised while the rest of the grid is still being computed. Chunks hold
	/// at most [`stream_chunk_points`](Self::stream_chunk_points) points.
	///
	/// Every time, value and kind is bit-identical to what `run_f64` returns for the same
	/// request on the same backend and precision. Return `ControlFlow::Break(())` from `sink`
	/// to stop early (say, the client went away): nothing more is computed beyond what is
	/// already in flight, and the summary says how much was delivered.
	///
	/// On the GPU, the next chunk computes while `sink` runs. On the CPU backends each
	/// chunk is computed, then delivered.
	///
	/// ```
	/// use std::ops::ControlFlow;
	///
	/// use chrono::{TimeZone, Utc};
	/// use splimes::{Backend, Interpolator, Resolution, Spline};
	///
	/// let at = |s| Utc.timestamp_opt(s, 0).unwrap();
	/// let (ts, vs) = ([at(0), at(10)], [0.0, 10.0]);
	/// let mut seen = Vec::new();
	/// let summary = Interpolator::new(Spline::Linear, Resolution::Seconds).backend(Backend::Cpu).stream_chunk_points(4).stream_f64(&ts, &vs, at(0), at(10), |chunk| {
	/// 	seen.extend_from_slice(chunk.values());
	/// 	ControlFlow::Continue(())
	/// })?;
	/// assert_eq!((summary.len(), summary.chunks()), (11, 3));
	/// assert_eq!(seen, (0..=10).map(f64::from).collect::<Vec<_>>());
	/// # Ok::<(), splimes::Error>(())
	/// ```
	///
	/// # Errors
	///
	/// As [`run_f64`](Self::run_f64), with one difference: chunks before a failing point
	/// may already have been delivered when [`Error::NonFiniteResult`] is returned. It
	/// names the same point `run_f64` names, and no chunk containing or after it is
	/// delivered. Under [`Backend::Auto`], a GPU failure part-way through continues on
	/// the CPU from the first chunk not yet delivered, and is reported by
	/// [`StreamSummary::gpu_fallback`].
	///
	/// # Panics
	///
	/// A panic in `sink` stops the run and is resumed once the GPU is idle.
	pub fn stream_f64(&self, timestamps: &[DateTime<Utc>], values: &[f64], start: DateTime<Utc>, end: DateTime<Utc>, sink: impl FnMut(Chunk<'_, DateTime<Utc>>) -> ControlFlow<()>) -> Result<StreamSummary> {
		if timestamps.len() != values.len() {
			return Err(Error::LengthMismatch { timestamps: timestamps.len(), values: values.len() });
		}
		let prepared = self.prepare(|| Grid::new(start, end, self.resolution()), || Knots::new(timestamps.len(), |i| timestamps[i], |i| &values[i], self.backend_choice() != Backend::Cpu))?;
		self.stream(prepared, &DateTimes, sink)
	}

	/// [`stream_f64`](Self::stream_f64) for epoch ticks in and out: the input as in
	/// [`run_epoch`](Self::run_epoch), and each chunk's times as ticks of the same `unit`,
	/// so no `DateTime` is built for any input or output point. Values and kinds are
	/// bit-identical to [`run_f64`](Self::run_f64) on the same instants.
	///
	/// ```
	/// use std::ops::ControlFlow;
	///
	/// use splimes::{Interpolator, Resolution, Spline, TickUnit};
	///
	/// // Millisecond ticks, a one-second grid.
	/// let (ticks, values) = ([1_000, 4_000], [1.0, 4.0]);
	/// let mut out = Vec::new();
	/// Interpolator::new(Spline::Linear, Resolution::Seconds).stream_epoch(&ticks, TickUnit::MILLISECONDS, &values, 0, 5_000, |chunk| {
	/// 	out.extend(chunk.iter().map(|(t, v, _)| (t, v)));
	/// 	ControlFlow::Continue(())
	/// })?;
	/// assert_eq!(out, [(0, 0.0), (1_000, 1.0), (2_000, 2.0), (3_000, 3.0), (4_000, 4.0), (5_000, 5.0)]);
	/// # Ok::<(), splimes::Error>(())
	/// ```
	///
	/// # Errors
	///
	/// As [`stream_f64`](Self::stream_f64) and [`run_epoch`](Self::run_epoch), plus
	/// [`Error::GridNotInTicks`] if the resolution is not a whole number of ticks (a
	/// millisecond grid in second ticks).
	///
	/// # Panics
	///
	/// As [`stream_f64`](Self::stream_f64).
	pub fn stream_epoch(&self, ticks: &[i64], unit: TickUnit, values: &[f64], start: i64, end: i64, sink: impl FnMut(Chunk<'_, i64>) -> ControlFlow<()>) -> Result<StreamSummary> {
		if ticks.len() != values.len() {
			return Err(Error::LengthMismatch { timestamps: ticks.len(), values: values.len() });
		}
		let resolution = self.resolution();
		let mut steps = (0, 0);
		let prepared = self.prepare(
			|| {
				let grid = unit.grid(start, end, resolution)?;
				steps = unit.grid_ticks(&grid, resolution)?;
				Ok(grid)
			},
			|| unit.knots(ticks, values, self.backend_choice() != Backend::Cpu),
		)?;
		self.stream(prepared, &Ticks { first: steps.0, step: steps.1 }, sink)
	}

	/// Runs a prepared request, delivering it chunk by chunk.
	fn stream<TT: Times>(&self, Prepared { grid, knots, spline, method }: Prepared<'_, f64>, times: &TT, sink: impl FnMut(Chunk<'_, TT::T>) -> ControlFlow<()>) -> Result<StreamSummary> {
		let chunk = self.stream_chunk().min(grid.len);
		let mut normalised = Vec::new();
		normalised.try_reserve_exact(chunk).map_err(|_| Error::OutputTooLarge { points: chunk as u128 })?;
		let mut emitter = Emitter { knots: &knots, grid: &grid, method: &method, times, narrow: kernel::narrow_offsets(&knots, &grid), parallel: self.backend_choice() != Backend::Cpu, repair: None, normalised, out_times: Vec::new(), kinds: Vec::new(), values: Vec::new(), sink, delivered: 0, chunks: 0, recomputed: 0, failure: None, panic: None };

		let precision = self.precision_choice();
		let gpu = |emitter: &mut Emitter<'_, '_, TT, _>| {
			emitter.repair = Some(if precision == Precision::F32 { kernel::f32_unsafe_windows(&knots, &method) } else { None });
			let result = crate::gpu::eval_chunks(&knots, &grid, &method, precision, chunk, emitter);
			emitter.repair = None;
			result
		};
		let (backend, precision, gpu_fallback) = match self.backend_choice() {
			backend @ (Backend::Cpu | Backend::Parallel) => {
				emitter.cpu(backend, chunk);
				(backend, Precision::F64, None)
			}
			Backend::Gpu => {
				let result = gpu(&mut emitter);
				emitter.settle()?;
				result?;
				(Backend::Gpu, precision, None)
			}
			Backend::Auto => match crate::auto::choose(grid.len, precision, true) {
				Backend::Gpu => {
					let result = gpu(&mut emitter);
					emitter.settle()?;
					match result {
						Ok(()) => (Backend::Gpu, precision, None),
						Err(e) => {
							log::warn!("splimes: GPU interpolation failed after {} of {} points, continuing on the CPU: {e}", emitter.delivered, grid.len);
							emitter.cpu(Backend::Parallel, chunk);
							(Backend::Parallel, Precision::F64, Some(e))
						}
					}
				}
				backend => {
					emitter.cpu(backend, chunk);
					(backend, Precision::F64, None)
				}
			},
		};
		emitter.settle()?;
		Ok(StreamSummary { len: grid.len, delivered: emitter.delivered, chunks: emitter.chunks, spline, requested: self.requested(), backend, precision, gpu_fallback, recomputed: emitter.recomputed })
	}
}

/// How a stream writes grid times.
pub(crate) trait Times: Sync {
	type T: Copy + Send;

	/// Grid points `first..first + len` into `out`, replacing its contents.
	fn fill(&self, grid: &Grid, first: usize, len: usize, parallel: bool, out: &mut Vec<Self::T>);
}

/// Grid times as `DateTime`s, exactly as [`Interpolation::timestamps`](crate::Interpolation::timestamps).
pub(crate) struct DateTimes;

impl Times for DateTimes {
	type T = DateTime<Utc>;

	fn fill(&self, grid: &Grid, first: usize, len: usize, parallel: bool, out: &mut Vec<Self::T>) {
		if parallel {
			(first..first + len).into_par_iter().with_min_len(ASSEMBLE_CHUNK).map_init(|| Clock::new(grid), Clock::at).collect_into_vec(out);
		} else {
			out.clear();
			let mut clock = Clock::new(grid);
			out.extend((first..first + len).map(|k| clock.at(k)));
		}
	}
}

/// Grid times as epoch ticks: point `k` is `first + k × step`.
pub(crate) struct Ticks {
	first: i64,
	step: i64,
}

impl Times for Ticks {
	type T = i64;

	fn fill(&self, _: &Grid, first: usize, len: usize, _: bool, out: &mut Vec<i64>) {
		out.clear();
		// Every grid point is an in-range tick, so neither the index nor the tick overflows.
		let mut t = self.first + i64::try_from(first).unwrap_or(i64::MAX).saturating_mul(self.step);
		out.extend((0..len).map(|_| {
			let tick = t;
			t = t.saturating_add(self.step);
			tick
		}));
	}
}

/// Turns normalised chunks into output chunks and hands them to the caller.
struct Emitter<'s, 'k, TT: Times, F> {
	knots: &'s Knots<'k, f64>,
	grid: &'s Grid,
	method: &'s Method,
	times: &'s TT,
	/// The knots' offsets as `i64`, when the kernel's fast path applies.
	narrow: Option<&'s [i64]>,
	parallel: bool,
	/// For chunks from the GPU: what to recompute in `f64` (see `kernel::repair`).
	repair: Option<Option<Vec<bool>>>,
	normalised: Vec<f64>,
	out_times: Vec<TT::T>,
	kinds: Vec<PointKind>,
	values: Vec<f64>,
	sink: F,
	delivered: usize,
	chunks: usize,
	recomputed: usize,
	/// Set when a point isn't finite: the run stops and returns it.
	failure: Option<Error>,
	/// Set when the caller's callback panicked: resumed once the GPU is idle.
	panic: Option<Box<dyn std::any::Any + Send>>,
}

impl<TT: Times, F: FnMut(Chunk<'_, TT::T>) -> ControlFlow<()>> Emitter<'_, '_, TT, F> {
	/// Computes and delivers the rest of the grid on the CPU, from the first point not yet
	/// delivered.
	fn cpu(&mut self, backend: Backend, chunk: usize) {
		// Assembled as `run_f64` assembles a CPU run: on rayon's pool unless on `Cpu`.
		self.parallel = backend != Backend::Cpu;
		let mut first = self.delivered;
		while first < self.grid.len {
			let len = chunk.min(self.grid.len - first);
			let (knots, grid, method) = (self.knots, self.grid, self.method);
			let dest = self.dest(first, len);
			if backend == Backend::Cpu {
				kernel::eval_serial(knots, grid, method, first, dest);
			} else {
				kernel::eval_parallel(knots, grid, method, first, dest);
			}
			if self.filled(first, len).is_break() {
				return;
			}
			first += len;
		}
	}

	/// The error or panic that stopped the run, if any.
	fn settle(&mut self) -> Result<()> {
		if let Some(panic) = self.panic.take() {
			resume_unwind(panic);
		}
		self.failure.take().map_or(Ok(()), Err)
	}

	/// Fills kinds and values for points `first..first + normalised.len()`; returns the
	/// first point whose value isn't finite.
	fn kinds_and_values<N: Nanos>(&mut self, offsets: &[N], first: usize, len: usize) -> Option<usize> {
		let (knots, grid, method, normalised) = (self.knots, self.grid, self.method, &self.normalised[..len]);
		let step = N::saturating_from(i128::from(grid.step_nanos));
		if !self.parallel {
			self.kinds.clear();
			self.values.clear();
			let mut walk = Walk::new(step);
			for (i, &y) in normalised.iter().enumerate() {
				let (kind, value) = walk.point(offsets, knots, grid, method, first + i, y);
				let Some(value) = value else {
					return Some(first + i);
				};
				self.kinds.push(kind);
				self.values.push(value);
			}
			return None;
		}
		let failure = std::sync::atomic::AtomicUsize::new(usize::MAX);
		let point = |walk: &mut Walk<N>, k: usize| -> (PointKind, f64) {
			let (kind, value) = walk.point(offsets, knots, grid, method, k, normalised[k - first]);
			let value = value.unwrap_or_else(|| {
				failure.fetch_min(k, std::sync::atomic::Ordering::Relaxed);
				f64::placeholder()
			});
			(kind, value)
		};
		(first..first + len).into_par_iter().with_min_len(ASSEMBLE_CHUNK).map_init(|| Walk::new(step), point).unzip_into_vecs(&mut self.kinds, &mut self.values);
		Some(failure.into_inner()).filter(|&k| k != usize::MAX)
	}

	/// Delivers points `first..first + len`, whose normalised values are in the buffer
	/// [`dest`](ChunkSink::dest) returned.
	fn emit(&mut self, first: usize, len: usize) -> ControlFlow<()> {
		let (knots, grid, method) = (self.knots, self.grid, self.method);
		if let Some(unsafe_windows) = &self.repair {
			let recomputed = kernel::repair(knots, grid, method, first, &mut self.normalised[..len], unsafe_windows.as_deref());
			if recomputed > 0 {
				log::debug!("splimes: recomputed {recomputed} point(s) in f64 on the CPU that the GPU couldn't compute reliably");
			}
			self.recomputed += recomputed;
		}
		// The provenance walk compares offsets in `i64` when they all fit, as the kernel does.
		let failure = match self.narrow {
			Some(offsets) => self.kinds_and_values(offsets, first, len),
			None => self.kinds_and_values(&knots.offsets, first, len),
		};
		if let Some(k) = failure {
			self.failure = Some(Error::NonFiniteResult { timestamp: grid.at(k) });
			return ControlFlow::Break(());
		}
		self.times.fill(grid, first, len, self.parallel, &mut self.out_times);
		let chunk = Chunk { first, times: &self.out_times, values: &self.values, kinds: &self.kinds };
		let sink = &mut self.sink;
		match catch_unwind(AssertUnwindSafe(|| sink(chunk))) {
			Ok(flow) => {
				self.delivered += len;
				self.chunks += 1;
				flow
			}
			Err(panic) => {
				self.panic = Some(panic);
				ControlFlow::Break(())
			}
		}
	}
}

impl<TT: Times, F: FnMut(Chunk<'_, TT::T>) -> ControlFlow<()>> ChunkSink for Emitter<'_, '_, TT, F> {
	fn dest(&mut self, _: usize, len: usize) -> &mut [f64] {
		// Within the capacity reserved for one chunk: never reallocates.
		self.normalised.resize(len, 0.0);
		&mut self.normalised[..len]
	}

	fn filled(&mut self, first: usize, len: usize) -> ControlFlow<()> {
		self.emit(first, len)
	}
}

#[cfg(test)]
mod tests {
	use chrono::TimeDelta;

	use super::*;
	use crate::Resolution;

	/// Under `Auto`, a GPU failure part-way through continues on the CPU from the first
	/// chunk not yet delivered: every point is delivered once, in order, and the report
	/// says what happened. An explicit `Gpu` request reports the failure instead.
	#[cfg(feature = "gpu")]
	#[test]
	fn auto_continues_on_the_cpu_after_a_gpu_failure() {
		if crate::prewarm_gpu().is_err() {
			eprintln!("auto_continues_on_the_cpu_after_a_gpu_failure: skipped, no GPU");
			return;
		}
		let at = |s: i64| DateTime::<Utc>::UNIX_EPOCH + TimeDelta::seconds(s);
		let ts: Vec<_> = (0..40).map(|i| at(i * 7 + i % 3)).collect();
		let vs: Vec<f64> = (0..40).map(|i| f64::from(i).sin()).collect();
		let (start, end) = (at(-20), at(300));
		let parallel = Interpolator::new(Spline::Cubic, Resolution::Seconds).backend(Backend::Parallel).run_f64(&ts, &vs, start, end).expect("runs");
		for precision in [Precision::F64, Precision::F32] {
			if precision == Precision::F64 && !crate::gpu_info().is_some_and(|i| i.supports_f64) {
				continue;
			}
			let interpolator = Interpolator::new(Spline::Cubic, Resolution::Seconds).gpu_precision(precision).stream_chunk_points(50);
			let (mut first, mut values) = (Vec::new(), Vec::new());
			crate::auto::FORCE_GPU.set(true);
			crate::gpu::run::tests::FAIL_AFTER_CHUNKS.set(Some(2));
			let summary = interpolator.stream_f64(&ts, &vs, start, end, |c| {
				first.push(c.first());
				values.extend_from_slice(c.values());
				ControlFlow::Continue(())
			});
			crate::gpu::run::tests::FAIL_AFTER_CHUNKS.set(Some(2));
			let explicit = interpolator.backend(Backend::Gpu).stream_f64(&ts, &vs, start, end, |_| ControlFlow::Continue(()));
			crate::auto::FORCE_GPU.set(false);
			crate::gpu::run::tests::FAIL_AFTER_CHUNKS.set(None);

			let summary = summary.expect("Auto recovers on the CPU");
			assert_eq!(first, (0..parallel.len()).step_by(50).collect::<Vec<_>>(), "{precision}: each point once, in order");
			assert!(summary.is_complete());
			assert_eq!((summary.backend(), summary.precision()), (Backend::Parallel, Precision::F64), "{precision}");
			assert_eq!(summary.gpu_fallback(), Some(&Error::Gpu("injected failure".to_owned())), "{precision}");
			// The CPU's part is exactly the CPU's; the GPU's part is within the contract.
			assert!(values[100..].iter().zip(&parallel.values()[100..]).all(|(a, b)| a.to_bits() == b.to_bits()), "{precision}");
			assert!(values[..100].iter().zip(parallel.values()).all(|(a, b)| (a - b).abs() < 1e-4), "{precision}");
			assert_eq!(explicit, Err(Error::Gpu("injected failure".to_owned())), "{precision}");
		}
	}
}

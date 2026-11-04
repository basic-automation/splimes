//! GPU interpolations that run while the caller does something else, with no thread
//! waiting for them.

use std::{
	future::Future, pin::Pin, task::{Context, Poll}
};

use bigdecimal::BigDecimal;
use chrono::{DateTime, Utc};

use crate::{Error, Interpolation, Interpolator, Point, Result};
#[cfg(feature = "gpu")]
use crate::{interpolation::Backend, prepare::Knots, time::Grid, value::Value};

impl Interpolator {
	/// Starts [`run`](Self::run) and returns at once with a handle to its result, while the
	/// GPU computes. Nothing waits for the GPU in the meantime: no rayon worker and no
	/// thread of the caller's.
	///
	/// When `run` would compute on the GPU ([`Backend::Gpu`](crate::Backend::Gpu), or
	/// [`Backend::Auto`](crate::Backend::Auto) above its GPU threshold with the GPU
	/// started), `submit` prepares the input and submits the first chunks of the grid on
	/// the calling thread, as `run` would, and returns. A single process-wide poller thread
	/// then polls the device without blocking, reads each chunk back as its mapping
	/// completes, submits the next, and wakes the handle when the last is in. The `f32`
	/// repair and the output assembly, which are CPU work, run on the thread that polls or
	/// [`wait`](GpuHandle::wait)s on the handle once the GPU is done (as part of `run`
	/// they run on the caller's thread too). Otherwise `submit` runs `run` there and then
	/// and returns a handle that is already complete.
	///
	/// The result is the one `run` returns, bit for bit, including
	/// [`Interpolation::backend`], [`Interpolation::gpu_fallback`] (an `Auto` call whose GPU
	/// work fails reruns on the CPU when the handle is polled) and
	/// [`Interpolation::recomputed`].
	///
	/// [`GpuHandle`] is a [`Future`] any executor can await, and has a blocking
	/// [`wait`](GpuHandle::wait) for synchronous code. Compare
	/// [`spawn`](Self::spawn), which occupies a rayon worker for the whole call, blocked in
	/// the driver while the GPU computes.
	///
	/// ```
	/// use bigdecimal::BigDecimal;
	/// use chrono::{TimeZone, Utc};
	/// use splimes::{Interpolator, Point, Resolution, Spline};
	///
	/// let at = |s| Utc.timestamp_opt(s, 0).unwrap();
	/// let points = vec![Point::new(at(0), BigDecimal::from(0)), Point::new(at(10), BigDecimal::from(10))];
	/// let handle = Interpolator::new(Spline::Linear, Resolution::Seconds).submit(points, at(0), at(10));
	/// // … parse the next request while the GPU works …
	/// let series = handle.wait()?;
	/// assert_eq!(series.values()[3], BigDecimal::from(3));
	/// # Ok::<(), splimes::Error>(())
	/// ```
	///
	/// # Dropping a handle, and process exit
	///
	/// Dropping a handle before its result is taken submits nothing more of its grid and
	/// discards the results, then blocks until the chunks already on the GPU finish (one or
	/// two dispatches), so that a process exiting after the drop isn't inside the driver.
	/// The poller sleeps outside the driver when no work is in flight. The process mustn't
	/// end while a handle's work is on the GPU and nothing waits for it, as after
	/// `std::process::exit` or `mem::forget`.
	///
	/// # Errors
	///
	/// The handle resolves to [`run`](Self::run)'s errors, plus [`Error::Task`] if it is
	/// polled again after it resolved, and [`Error::Gpu`] if the poller thread can't be
	/// started.
	#[must_use = "dropping the handle discards the interpolation"]
	pub fn submit(self, points: Vec<Point>, start: DateTime<Utc>, end: DateTime<Utc>) -> GpuHandle<BigDecimal> {
		#[cfg(feature = "gpu")]
		if let Some(grid) = self.gpu_grid(start, end) {
			return self.submit_gpu(grid, points);
		}
		GpuHandle::ready(self.run(&points, start, end))
	}

	/// [`submit`](Self::submit) for [`run_f64`](Self::run_f64).
	///
	/// # Errors
	///
	/// As [`submit`](Self::submit), with [`run_f64`](Self::run_f64)'s errors.
	#[must_use = "dropping the handle discards the interpolation"]
	pub fn submit_f64(self, timestamps: Vec<DateTime<Utc>>, values: Vec<f64>, start: DateTime<Utc>, end: DateTime<Utc>) -> GpuHandle<f64> {
		#[cfg(feature = "gpu")]
		if timestamps.len() == values.len()
			&& let Some(grid) = self.gpu_grid(start, end)
		{
			return self.submit_gpu(grid, Columns(timestamps, values));
		}
		GpuHandle::ready(self.run_f64(&timestamps, &values, start, end))
	}

	/// The grid, if `run` would compute this call on the GPU. Any refusal is left to `run`.
	#[cfg(feature = "gpu")]
	fn gpu_grid(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> Option<Grid> {
		self.grid(start, end).ok().filter(|grid| self.resolved_backend(grid, true) == Backend::Gpu)
	}

	/// The GPU path of [`submit`](Self::submit): `run`'s steps, with the GPU's part handed
	/// to the poller between them.
	#[cfg(feature = "gpu")]
	fn submit_gpu<S: Source>(self, grid: Grid, input: S) -> GpuHandle<S::V> {
		let prepared = (|| {
			let (knots, indices) = Knots::new_indexed(input.len(), |i| input.timestamp(i), |i| input.value(i), self.backend != Backend::Cpu)?;
			let (spline, method) = self.method(&knots)?;
			let out = self.output(&grid)?;
			Ok((knots, indices, spline, method, out))
		})();
		let (knots, indices, spline, method, out) = match prepared {
			Ok(prepared) => prepared,
			Err(e) => return GpuHandle::ready(Err(e)),
		};
		let started = crate::gpu::start(&knots, &grid, &method, self.precision(), out);
		// The knots stop borrowing the input, which moves into the finishing step.
		let knots = knots.detach();
		let finish = move |gpu: Result<()>, mut out: Vec<f64>| {
			let knots = knots.attach(indices.iter().map(|&i| input.value(i)).collect());
			self.finish_gpu(&knots, &grid, &method, spline, &mut out, gpu)
		};
		match started {
			Ok(job) => GpuHandle { state: State::Gpu { job, finish: Box::new(finish) } },
			Err((e, out)) => GpuHandle::ready(finish(Err(e), out)),
		}
	}
}

/// An owned input a handle keeps until its output is assembled.
#[cfg(feature = "gpu")]
trait Source: Send + Sync + 'static {
	type V: Value;
	fn len(&self) -> usize;
	fn timestamp(&self, i: usize) -> DateTime<Utc>;
	fn value(&self, i: usize) -> &Self::V;
}

#[cfg(feature = "gpu")]
impl Source for Vec<Point> {
	type V = BigDecimal;

	fn len(&self) -> usize {
		Self::len(self)
	}

	fn timestamp(&self, i: usize) -> DateTime<Utc> {
		self[i].timestamp
	}

	fn value(&self, i: usize) -> &BigDecimal {
		&self[i].value
	}
}

/// `run_f64`'s columns, of equal length.
#[cfg(feature = "gpu")]
struct Columns(Vec<DateTime<Utc>>, Vec<f64>);

#[cfg(feature = "gpu")]
impl Source for Columns {
	type V = f64;

	fn len(&self) -> usize {
		self.0.len()
	}

	fn timestamp(&self, i: usize) -> DateTime<Utc> {
		self.0[i]
	}

	fn value(&self, i: usize) -> &f64 {
		&self.1[i]
	}
}

/// The finishing step of a handle: the GPU's verdict and normalised results in, the output out.
#[cfg(feature = "gpu")]
type Finish<V> = Box<dyn FnOnce(Result<()>, Vec<f64>) -> Result<Interpolation<V>> + Send>;

/// An interpolation the GPU is computing, from [`Interpolator::submit`] or
/// [`Interpolator::submit_f64`]: a [`Future`] of its result, with a blocking
/// [`wait`](Self::wait).
///
/// It works on any executor: it registers the waker it is polled with, and the GPU poller
/// wakes it when the GPU part finishes. It is `Send` and `Unpin`. Dropping it before the
/// result is taken discards the interpolation; see [`Interpolator::submit`].
pub struct GpuHandle<V> {
	state: State<V>,
}

enum State<V> {
	Ready(Result<Interpolation<V>>),
	#[cfg(feature = "gpu")]
	Gpu {
		job: std::sync::Arc<crate::gpu::pending::Job>,
		finish: Finish<V>,
	},
	Taken,
}

impl<V> std::fmt::Debug for GpuHandle<V> {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		let state = match &self.state {
			State::Ready(_) => "ready",
			#[cfg(feature = "gpu")]
			State::Gpu { job, .. } if job.is_finished() => "gpu finished",
			#[cfg(feature = "gpu")]
			State::Gpu { .. } => "on the gpu",
			State::Taken => "taken",
		};
		f.debug_struct("GpuHandle").field("state", &state).finish()
	}
}

impl<V> GpuHandle<V> {
	const fn ready(result: Result<Interpolation<V>>) -> Self {
		Self { state: State::Ready(result) }
	}

	/// Whether the result can be taken without waiting for the GPU. Never blocks.
	#[must_use]
	pub fn is_finished(&self) -> bool {
		match &self.state {
			#[cfg(feature = "gpu")]
			State::Gpu { job, .. } => job.is_finished(),
			_ => true,
		}
	}

	/// Blocks until the GPU has finished, then returns the result.
	///
	/// Safe on any thread, including rayon's workers: the poller completes GPU work
	/// without rayon.
	///
	/// # Errors
	///
	/// As [`Interpolator::submit`].
	pub fn wait(mut self) -> Result<Interpolation<V>> {
		match std::mem::replace(&mut self.state, State::Taken) {
			State::Ready(result) => result,
			#[cfg(feature = "gpu")]
			State::Gpu { job, finish } => {
				let (verdict, out) = job.wait();
				finish(verdict, out)
			}
			State::Taken => Err(taken()),
		}
	}
}

fn taken() -> Error {
	Error::Task("polled after the interpolation's result was taken".to_owned())
}

// Nothing in a handle is pinned: the GPU work refers to buffers, not to the handle.
impl<V> Unpin for GpuHandle<V> {}

impl<V> Future for GpuHandle<V> {
	type Output = Result<Interpolation<V>>;

	fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
		let this = self.get_mut();
		match std::mem::replace(&mut this.state, State::Taken) {
			State::Ready(result) => Poll::Ready(result),
			#[cfg(feature = "gpu")]
			State::Gpu { job, finish } => match job.poll(cx.waker()) {
				Some((verdict, out)) => Poll::Ready(finish(verdict, out)),
				None => {
					this.state = State::Gpu { job, finish };
					Poll::Pending
				}
			},
			State::Taken => {
				let _ = cx;
				Poll::Ready(Err(taken()))
			}
		}
	}
}

impl<V> Drop for GpuHandle<V> {
	fn drop(&mut self) {
		#[cfg(feature = "gpu")]
		if let State::Gpu { job, .. } = &self.state {
			job.cancel();
		}
	}
}

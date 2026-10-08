//! Interpolations on rayon's pool, awaited from any executor.

use std::{
	future::Future, panic::{AssertUnwindSafe, catch_unwind}, pin::Pin, sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError}, task::{Context, Poll, Waker}
};

use bigdecimal::BigDecimal;
use chrono::{DateTime, Utc};

use crate::{Backend, Error, Interpolation, Interpolator, Point, Result};

impl Interpolator {
	/// Starts [`run`](Self::run) on rayon's thread pool and returns a future of its result,
	/// which any executor can await (tokio, async-std, smol, `futures`), or none: it only
	/// needs to be polled. Takes the points by value because the task must own them.
	///
	/// The work starts at once, not on the first poll. Dropping the future doesn't stop it;
	/// the result is discarded.
	///
	/// The interpolation occupies a rayon worker while it runs (and uses the rest of the
	/// pool as [`Backend::Parallel`](crate::Backend::Parallel) would), so the caller's
	/// executor isn't blocked, with two exceptions:
	///
	/// - With [`Backend::Gpu`](crate::Backend::Gpu), if this is the program's first use of
	///   the GPU, `spawn` opens the device on the calling thread before it returns (a few
	///   hundred milliseconds), because splimes never opens it on a thread of its own.
	///   Call [`prewarm_gpu`](crate::prewarm_gpu) at startup to pay that up front.
	/// - Called from a rayon worker, `spawn` runs the interpolation there and then, as
	///   [`run`](Self::run) would, and returns a future that is already complete. Queued
	///   on the pool instead, it could wait behind the very worker blocked on it.
	///
	/// When the GPU may compute the result ([`Backend::Gpu`](crate::Backend::Gpu), or
	/// [`Backend::Auto`](crate::Backend::Auto) once the GPU is started), dropping the future
	/// before it completes waits for the work to finish. rayon's threads aren't joined at
	/// exit, a process that exits while one of them is in the GPU driver can crash there,
	/// and dropping a future is often the last thing a program does (a runtime shutting
	/// down, `main` returning). For the same reason, don't end the process any other way,
	/// such as `std::process::exit`, while such a future is pending.
	///
	/// With the `tokio` feature, `Interpolator::run_async` uses tokio's blocking pool
	/// instead, which the runtime waits for when it shuts down.
	///
	/// ```
	/// use bigdecimal::BigDecimal;
	/// use chrono::{TimeZone, Utc};
	/// use splimes::{Interpolator, Point, Resolution, Spline};
	///
	/// # fn block_on<F: std::future::Future>(f: F) -> F::Output { pollster_like::block_on(f) }
	/// # mod pollster_like {
	/// #     use std::{future::Future, pin::pin, sync::Arc, task::{Context, Poll, Wake}, thread::{self, Thread}};
	/// #     struct Unpark(Thread);
	/// #     impl Wake for Unpark { fn wake(self: Arc<Self>) { self.0.unpark(); } }
	/// #     pub fn block_on<F: Future>(f: F) -> F::Output {
	/// #         let waker = Arc::new(Unpark(thread::current())).into();
	/// #         let mut cx = Context::from_waker(&waker);
	/// #         let mut f = pin!(f);
	/// #         loop { if let Poll::Ready(v) = f.as_mut().poll(&mut cx) { return v; } thread::park(); }
	/// #     }
	/// # }
	/// let at = |s| Utc.timestamp_opt(s, 0).unwrap();
	/// let points = vec![Point::new(at(0), BigDecimal::from(0)), Point::new(at(10), BigDecimal::from(10))];
	///
	/// // In an async fn, on any executor: `interpolator.spawn(points, start, end).await?`
	/// let series = block_on(Interpolator::new(Spline::Linear, Resolution::Seconds).spawn(points, at(0), at(10)))?;
	/// assert_eq!(series.values()[3], BigDecimal::from(3));
	/// # Ok::<(), splimes::Error>(())
	/// ```
	///
	/// # Errors
	///
	/// The future resolves to [`run`](Self::run)'s errors, plus [`Error::Task`] if the
	/// interpolation panicked, which is a bug, or the future was polled again after it
	/// completed.
	#[must_use = "the interpolation runs regardless, but its result is in the future"]
	pub fn spawn(self, points: Vec<Point>, start: DateTime<Utc>, end: DateTime<Utc>) -> InterpolationFuture<BigDecimal> {
		self.open_gpu_here();
		InterpolationFuture::spawn(self.may_use_gpu(), move || self.run(&points, start, end))
	}

	/// [`spawn`](Self::spawn) for [`run_f64`](Self::run_f64).
	///
	/// # Errors
	///
	/// The future resolves to [`run_f64`](Self::run_f64)'s errors, plus [`Error::Task`] as
	/// for [`spawn`](Self::spawn).
	#[must_use = "the interpolation runs regardless, but its result is in the future"]
	pub fn spawn_f64(self, timestamps: Vec<DateTime<Utc>>, values: Vec<f64>, start: DateTime<Utc>, end: DateTime<Utc>) -> InterpolationFuture<f64> {
		self.open_gpu_here();
		InterpolationFuture::spawn(self.may_use_gpu(), move || self.run_f64(&timestamps, &values, start, end))
	}

	/// With [`Backend::Gpu`](crate::Backend::Gpu), opens the device on the calling thread
	/// if no call has yet, so the job never opens it on a rayon worker: a process exiting
	/// while a driver initialises on a thread nothing joins can crash in the driver. A
	/// failure to open is remembered, and the job reports it as `GpuUnavailable`.
	fn open_gpu_here(&self) {
		if self.backend == Backend::Gpu {
			let _ = crate::prewarm_gpu();
		}
	}

	/// Whether the GPU may compute this call: [`Backend::Gpu`], or [`Backend::Auto`] once
	/// the GPU is started.
	fn may_use_gpu(&self) -> bool {
		self.backend == Backend::Gpu || (self.backend == Backend::Auto && crate::gpu_info().is_some())
	}
}

/// An interpolation running on rayon's pool: a [`Future`] of its result, from
/// [`Interpolator::spawn`] or [`Interpolator::spawn_f64`].
///
/// It works on any executor: it registers the waker it is polled with and wakes it when
/// the interpolation finishes. It is `Send` and `Unpin`. Dropping it before it completes
/// waits for the work if the GPU may be computing it; see [`Interpolator::spawn`].
#[derive(Debug)]
pub struct InterpolationFuture<V> {
	shared: Arc<Shared<V>>,
	/// Whether dropping the future waits for the work to finish: the GPU may be computing it.
	wait_on_drop: bool,
}

#[derive(Debug)]
struct Shared<V> {
	state: Mutex<State<V>>,
	/// Signalled when the work finishes.
	finished: Condvar,
}

#[derive(Debug)]
enum State<V> {
	Running(Option<Waker>),
	Done(Result<Interpolation<V>>),
	Taken,
}

impl<V> Shared<V> {
	const fn new(state: State<V>) -> Self {
		Self { state: Mutex::new(state), finished: Condvar::new() }
	}

	fn lock(&self) -> MutexGuard<'_, State<V>> {
		self.state.lock().unwrap_or_else(PoisonError::into_inner)
	}
}

impl<V: Send + 'static> InterpolationFuture<V> {
	fn spawn(wait_on_drop: bool, work: impl FnOnce() -> Result<Interpolation<V>> + Send + 'static) -> Self {
		// On a rayon worker already, run here: a job queued on the pool could otherwise wait
		// behind the very worker that is blocked on its future, which never finishes.
		if rayon::current_thread_index().is_some() {
			return Self { shared: Arc::new(Shared::new(State::Done(caught(work)))), wait_on_drop: false };
		}
		let shared = Arc::new(Shared::new(State::Running(None)));
		let task = Arc::clone(&shared);
		rayon::spawn(move || {
			let result = caught(work);
			let previous = std::mem::replace(&mut *task.lock(), State::Done(result));
			// Both outside the lock, so the executor can poll straight away.
			task.finished.notify_all();
			if let State::Running(Some(waker)) = previous {
				waker.wake();
			}
		});
		Self { shared, wait_on_drop }
	}
}

impl<V> Drop for InterpolationFuture<V> {
	fn drop(&mut self) {
		// Dropping a future is often the last thing a program does before it exits (a runtime
		// shutting down, `main` returning), and a process that exits while the GPU driver is
		// busy on one of rayon's threads, which nothing joins, can crash in the driver. So if
		// the GPU may be computing the result, wait for the work. Not on a rayon worker,
		// though: the job could be queued behind this very thread.
		if !self.wait_on_drop || rayon::current_thread_index().is_some() {
			return;
		}
		drop(self.shared.finished.wait_while(self.shared.lock(), |state| matches!(state, State::Running(_))).unwrap_or_else(PoisonError::into_inner));
	}
}

/// Runs `work`, turning a panic into the future's error. A panic must not unwind out of a
/// rayon task, which aborts the process. The closure owns everything it touches.
fn caught<V>(work: impl FnOnce() -> Result<Interpolation<V>>) -> Result<Interpolation<V>> {
	catch_unwind(AssertUnwindSafe(work)).unwrap_or_else(|panic| {
		let message = panic.downcast_ref::<&str>().map(|s| (*s).to_owned()).or_else(|| panic.downcast_ref::<String>().cloned()).unwrap_or_default();
		Err(Error::Task(format!("the interpolation panicked: {message}")))
	})
}

impl<V> Future for InterpolationFuture<V> {
	type Output = Result<Interpolation<V>>;

	fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
		let mut state = self.shared.lock();
		match &mut *state {
			State::Running(waker) => {
				// The executor may move the task between polls; keep the latest waker.
				match waker {
					Some(w) if w.will_wake(cx.waker()) => {}
					_ => *waker = Some(cx.waker().clone()),
				}
				Poll::Pending
			}
			State::Done(_) => match std::mem::replace(&mut *state, State::Taken) {
				State::Done(result) => Poll::Ready(result),
				_ => Poll::Ready(Err(Error::Task("the interpolation's result was lost".to_owned()))),
			},
			State::Taken => Poll::Ready(Err(Error::Task("polled after the interpolation's result was taken".to_owned()))),
		}
	}
}

#[cfg(test)]
mod tests {
	use std::{
		sync::atomic::{AtomicUsize, Ordering}, task::Wake, thread::{self, Thread}
	};

	use chrono::TimeZone;

	use super::*;
	use crate::{Resolution, Spline};

	/// Wakes a parked thread, and counts wakes.
	struct Unpark(Thread, AtomicUsize);

	impl Wake for Unpark {
		fn wake(self: Arc<Self>) {
			self.1.fetch_add(1, Ordering::SeqCst);
			self.0.unpark();
		}
	}

	/// The smallest executor there is: poll, park until woken, repeat.
	fn block_on<F: Future + Unpin>(mut f: F) -> (F::Output, usize) {
		let unpark = Arc::new(Unpark(thread::current(), AtomicUsize::new(0)));
		let waker = Waker::from(Arc::clone(&unpark));
		let mut cx = Context::from_waker(&waker);
		loop {
			if let Poll::Ready(v) = Pin::new(&mut f).poll(&mut cx) {
				return (v, unpark.1.load(Ordering::SeqCst));
			}
			thread::park();
		}
	}

	fn at(secs: i64) -> DateTime<Utc> {
		Utc.timestamp_opt(secs, 0).single().expect("valid")
	}

	#[test]
	fn resolves_to_what_run_returns() {
		let interpolator = Interpolator::new(Spline::Cubic, Resolution::Milliseconds);
		let ts: Vec<_> = (0..50).map(|i| at(i * 3)).collect();
		let vs: Vec<f64> = (0..50).map(|i| f64::from(i).sin()).collect();
		let (spawned, _) = block_on(interpolator.spawn_f64(ts.clone(), vs.clone(), at(-5), at(160)));
		assert_eq!(spawned, interpolator.run_f64(&ts, &vs, at(-5), at(160)));
		let points: Vec<Point> = ts.iter().zip(&vs).map(|(&t, &v)| Point::new(t, BigDecimal::try_from(v).expect("finite"))).collect();
		let (spawned, _) = block_on(interpolator.spawn(points.clone(), at(0), at(10)));
		assert_eq!(spawned, interpolator.run(&points, at(0), at(10)));
		// Errors arrive through the future too.
		let (err, _) = block_on(interpolator.spawn(Vec::new(), at(0), at(1)));
		assert_eq!(err, Err(Error::NoPoints));
	}

	#[test]
	fn wakes_the_task_that_is_waiting() {
		// Ten million points: long enough that the first poll finds it still running, so
		// completion has to come through the waker.
		let interpolator = Interpolator::new(Spline::Linear, Resolution::Microseconds).backend(crate::Backend::Cpu);
		let (out, wakes) = block_on(interpolator.spawn_f64(vec![at(0), at(100)], vec![0.0, 100.0], at(0), at(10)));
		assert_eq!(out.expect("runs").len(), 10_000_001);
		assert_eq!(wakes, 1);
	}

	#[test]
	fn a_second_poll_after_completion_is_an_error() {
		let mut f = Interpolator::new(Spline::Linear, Resolution::Seconds).spawn_f64(vec![at(0), at(1)], vec![0.0, 1.0], at(0), at(1));
		let (first, _) = block_on(&mut f);
		assert!(first.is_ok());
		let (again, _) = block_on(&mut f);
		assert!(matches!(again, Err(Error::Task(_))), "{again:?}");
	}

	#[test]
	fn a_panicking_task_becomes_an_error() {
		let f: InterpolationFuture<f64> = InterpolationFuture::spawn(false, || panic!("boom"));
		let (out, _) = block_on(f);
		assert_eq!(out, Err(Error::Task("the interpolation panicked: boom".to_owned())));
	}

	#[test]
	fn dropping_the_future_lets_the_work_finish() {
		let (finished, done) = std::sync::mpsc::channel();
		let f: InterpolationFuture<f64> = InterpolationFuture::spawn(false, move || {
			std::thread::sleep(std::time::Duration::from_millis(50));
			finished.send(()).expect("the test is waiting");
			Err(Error::NoPoints)
		});
		let shared = Arc::downgrade(&f.shared);
		drop(f);
		// rayon still runs the task to the end, and its result goes nowhere...
		done.recv_timeout(std::time::Duration::from_secs(30)).expect("the dropped work finished");
		// ...and once it has, nothing of it is left behind.
		let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
		while shared.upgrade().is_some() {
			assert!(std::time::Instant::now() < deadline, "the task's state was never freed");
			thread::yield_now();
		}
		// A later call is unaffected.
		let (out, _) = block_on(Interpolator::new(Spline::Linear, Resolution::Seconds).spawn_f64(vec![at(0), at(1)], vec![0.0, 1.0], at(0), at(1)));
		assert_eq!(out.expect("runs").len(), 2);
	}

	#[test]
	fn dropping_a_future_the_gpu_may_compute_waits_for_the_work() {
		let finished = Arc::new(std::sync::atomic::AtomicBool::new(false));
		let flag = Arc::clone(&finished);
		let f: InterpolationFuture<f64> = InterpolationFuture::spawn(true, move || {
			std::thread::sleep(std::time::Duration::from_millis(100));
			flag.store(true, Ordering::SeqCst);
			Err(Error::NoPoints)
		});
		drop(f);
		assert!(finished.load(Ordering::SeqCst), "drop returned before the work finished");
		// An explicit GPU interpolation is such a future, whether or not a GPU is present.
		assert!(Interpolator::new(Spline::Linear, Resolution::Seconds).backend(crate::Backend::Gpu).spawn_f64(vec![at(0), at(1)], vec![0.0, 1.0], at(0), at(1)).wait_on_drop);
		assert!(!Interpolator::new(Spline::Linear, Resolution::Seconds).backend(crate::Backend::Parallel).spawn_f64(vec![at(0), at(1)], vec![0.0, 1.0], at(0), at(1)).wait_on_drop);
	}

	#[test]
	fn blocking_on_it_from_a_rayon_worker_does_not_deadlock() {
		// One worker, blocked on the future: had the job been queued on the pool, nothing
		// would be left to run it.
		let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().expect("a pool");
		let (sent, received) = std::sync::mpsc::channel();
		thread::spawn(move || {
			let out = pool.install(|| block_on(Interpolator::new(Spline::Linear, Resolution::Seconds).spawn_f64(vec![at(0), at(4)], vec![0.0, 4.0], at(0), at(4))));
			sent.send(out).expect("the test is waiting");
		});
		let (out, wakes) = received.recv_timeout(std::time::Duration::from_secs(30)).expect("no deadlock");
		assert_eq!(out.expect("runs").len(), 5);
		assert_eq!(wakes, 0, "complete on the first poll");
	}
}

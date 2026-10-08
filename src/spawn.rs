//! Interpolations on rayon's pool, awaited from any executor.

use std::{
	future::Future, panic::{AssertUnwindSafe, catch_unwind}, pin::Pin, sync::{Arc, Mutex, PoisonError}, task::{Context, Poll, Waker}
};

use bigdecimal::BigDecimal;
use chrono::{DateTime, Utc};

use crate::{Error, Interpolation, Interpolator, Point, Result};

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
	/// executor is never blocked. If this is the program's first use of the GPU
	/// ([`Backend::Gpu`](crate::Backend::Gpu)), the device opens on that worker; call
	/// [`prewarm_gpu`](crate::prewarm_gpu) at startup to keep driver initialisation on a
	/// thread you control.
	///
	/// With the `tokio` feature, `Interpolator::run_async` uses tokio's blocking pool
	/// instead.
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
		InterpolationFuture::spawn(move || self.run(&points, start, end))
	}

	/// [`spawn`](Self::spawn) for [`run_f64`](Self::run_f64).
	///
	/// # Errors
	///
	/// The future resolves to [`run_f64`](Self::run_f64)'s errors, plus [`Error::Task`] as
	/// for [`spawn`](Self::spawn).
	#[must_use = "the interpolation runs regardless, but its result is in the future"]
	pub fn spawn_f64(self, timestamps: Vec<DateTime<Utc>>, values: Vec<f64>, start: DateTime<Utc>, end: DateTime<Utc>) -> InterpolationFuture<f64> {
		InterpolationFuture::spawn(move || self.run_f64(&timestamps, &values, start, end))
	}
}

/// An interpolation running on rayon's pool: a [`Future`] of its result, from
/// [`Interpolator::spawn`] or [`Interpolator::spawn_f64`].
///
/// It works on any executor: it registers the waker it is polled with and wakes it when
/// the interpolation finishes. It is `Send` and `Unpin`.
#[derive(Debug)]
pub struct InterpolationFuture<V> {
	shared: Arc<Mutex<Shared<V>>>,
}

#[derive(Debug)]
enum Shared<V> {
	Running(Option<Waker>),
	Done(Result<Interpolation<V>>),
	Taken,
}

impl<V: Send + 'static> InterpolationFuture<V> {
	fn spawn(work: impl FnOnce() -> Result<Interpolation<V>> + Send + 'static) -> Self {
		let shared = Arc::new(Mutex::new(Shared::Running(None)));
		let task = Arc::clone(&shared);
		rayon::spawn(move || {
			// A panic must not unwind out of a rayon task, which aborts the process; it
			// becomes the future's error instead. The closure owns everything it touches.
			let result = catch_unwind(AssertUnwindSafe(work)).unwrap_or_else(|panic| {
				let message = panic.downcast_ref::<&str>().map(|s| (*s).to_owned()).or_else(|| panic.downcast_ref::<String>().cloned()).unwrap_or_default();
				Err(Error::Task(format!("the interpolation panicked: {message}")))
			});
			let previous = std::mem::replace(&mut *task.lock().unwrap_or_else(PoisonError::into_inner), Shared::Done(result));
			// Woken outside the lock, so the executor can poll straight away.
			if let Shared::Running(Some(waker)) = previous {
				waker.wake();
			}
		});
		Self { shared }
	}
}

impl<V> Future for InterpolationFuture<V> {
	type Output = Result<Interpolation<V>>;

	fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
		let mut shared = self.shared.lock().unwrap_or_else(PoisonError::into_inner);
		match &mut *shared {
			Shared::Running(waker) => {
				// The executor may move the task between polls; keep the latest waker.
				match waker {
					Some(w) if w.will_wake(cx.waker()) => {}
					_ => *waker = Some(cx.waker().clone()),
				}
				Poll::Pending
			}
			Shared::Done(_) => match std::mem::replace(&mut *shared, Shared::Taken) {
				Shared::Done(result) => Poll::Ready(result),
				_ => Poll::Ready(Err(Error::Task("the interpolation's result was lost".to_owned()))),
			},
			Shared::Taken => Poll::Ready(Err(Error::Task("polled after the interpolation's result was taken".to_owned()))),
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
		let f: InterpolationFuture<f64> = InterpolationFuture::spawn(|| panic!("boom"));
		let (out, _) = block_on(f);
		assert_eq!(out, Err(Error::Task("the interpolation panicked: boom".to_owned())));
	}

	#[test]
	fn dropping_the_future_lets_the_work_finish() {
		drop(Interpolator::new(Spline::Linear, Resolution::Seconds).spawn_f64(vec![at(0), at(1)], vec![0.0, 1.0], at(0), at(1)));
		// rayon still runs the task, and its result goes nowhere; a later call is unaffected.
		let (out, _) = block_on(Interpolator::new(Spline::Linear, Resolution::Seconds).spawn_f64(vec![at(0), at(1)], vec![0.0, 1.0], at(0), at(1)));
		assert_eq!(out.expect("runs").len(), 2);
	}
}

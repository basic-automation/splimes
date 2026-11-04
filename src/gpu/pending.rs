//! GPU work that finishes without a thread waiting for it.
//!
//! A job's first chunks are submitted on the caller's thread. From then on one
//! process-wide poller thread drives every job: it polls each device without blocking
//! (`PollType::Poll`), which runs the `map_async` callbacks of finished chunks, copies
//! their results out, submits the next chunks, and wakes the job's waiter when the last
//! chunk is in. Between polls that find nothing to do it sleeps, briefly; with no job in
//! flight it waits on a condition variable, outside the driver.
//!
//! The poller never blocks in the driver and never uses rayon, so no thread is parked on
//! the GPU, and waiting for a job can't deadlock on any pool. It only polls with
//! `PollType::Poll`, for which wgpu-core 30.0.1's `Device::maintain` has no wait index and
//! skips the release-mode `assert!` that concurrent `PollType::Wait` polls can trip
//! (gfx-rs/wgpu#9958; see `ROADMAP.md`).

use std::{
	collections::VecDeque, sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, PoisonError, atomic::Ordering}, task::Waker, thread::{self, ThreadId}, time::{Duration, Instant}
};

use wgpu::{MapMode, PollType};

use super::{
	context::{Context, Kernel}, run::{CHUNK_TIMEOUT, IN_FLIGHT, Slot, Stream, acquire, captured, copy_out, precision_bytes, release}
};
use crate::{Error, Precision, Result, kernel::Method, prepare::Knots, time::Grid, value::Value};

/// The poller's first sleep after a poll that found nothing to do; it doubles up to
/// [`MAX_IDLE`] and resets on progress.
const MIN_IDLE: Duration = Duration::from_micros(10);
const MAX_IDLE: Duration = Duration::from_micros(200);

/// One call's GPU work, shared by its handle and the poller.
pub struct Job {
	ctx: &'static Context,
	state: Mutex<State>,
	/// Signalled when the GPU part finishes.
	changed: Condvar,
}

struct State {
	/// The chunks still to submit or read back; `None` once finished.
	active: Option<Active>,
	outcome: Outcome,
	waker: Option<Waker>,
	/// The handle was dropped: submit nothing more, and discard what comes back.
	cancelled: bool,
}

enum Outcome {
	Running,
	/// The GPU's verdict, and the normalised results (complete only if it is `Ok`).
	Done(Result<()>, Vec<f64>),
	Taken,
}

struct Active {
	kernel: &'static Kernel,
	stream: Stream,
	precision: Precision,
	/// Grid points in the call.
	len: usize,
	/// The first grid point not yet submitted.
	next: usize,
	in_flight: VecDeque<Chunk>,
	/// Buffer sets read back and free for the next chunk.
	spare: Vec<Slot>,
	out: Vec<f64>,
}

struct Chunk {
	slot: Slot,
	first: usize,
	len: usize,
	/// Set by the `map_async` callback, on whichever thread polls the device.
	mapped: Arc<OnceLock<std::result::Result<(), String>>>,
	since: Instant,
}

/// Uploads the knots and submits the first chunks on the calling thread, then hands the
/// rest to the poller. `out` is the results buffer, of the grid's length; on failure it
/// comes back with the error, for a CPU rerun.
pub fn start<V: Value>(ctx: &'static Context, knots: &Knots<'_, V>, grid: &Grid, method: &Method, precision: Precision, out: Vec<f64>) -> std::result::Result<Arc<Job>, (Error, Vec<f64>)> {
	#[cfg(test)]
	if super::run::tests::FAIL.get() {
		return Err((Error::Gpu("injected failure".to_owned()), out));
	}
	let pump = match pump() {
		Ok(pump) => pump,
		Err(e) => return Err((e, out)),
	};
	let kernel = match ctx.kernel(precision) {
		Ok(kernel) => kernel,
		Err(e) => return Err((e, out)),
	};
	let active = captured(&ctx.device, || {
		let stream = Stream::new(ctx, knots, grid, method, precision)?;
		let mut active = Active { kernel, stream, precision, len: grid.len, next: 0, in_flight: VecDeque::with_capacity(IN_FLIGHT), spare: Vec::with_capacity(IN_FLIGHT), out: Vec::new() };
		active.fill(ctx, true)?;
		Ok(active)
	});
	let mut active = match active {
		Ok(active) => active,
		Err(e) => return Err((e, out)),
	};
	active.out = out;
	#[cfg(test)]
	match super::run::tests::LOSE_BEFORE_POLL.get() {
		Some(super::run::tests::Loss::Destroyed) => ctx.device.destroy(),
		Some(super::run::tests::Loss::Signalled) => ctx.lost.store(true, Ordering::Release),
		None => {}
	}
	let job = Arc::new(Job { ctx, state: Mutex::new(State { active: Some(active), outcome: Outcome::Running, waker: None, cancelled: false }), changed: Condvar::new() });
	lock(&pump.jobs).push(Arc::clone(&job));
	pump.work.notify_one();
	Ok(job)
}

impl Active {
	/// Submits chunks until [`IN_FLIGHT`] are in flight or the grid is all submitted. With
	/// `parallel`, the grid times are staged on rayon's pool.
	fn fill(&mut self, ctx: &Context, parallel: bool) -> Result<bool> {
		let mut submitted = false;
		while self.in_flight.len() < IN_FLIGHT && self.next < self.len {
			let len = self.stream.chunk.min(self.len - self.next);
			let slot = self.spare.iter().position(|s| s.capacity >= len).map_or_else(|| acquire(ctx, self.precision, len), |i| self.spare.swap_remove(i));
			self.stream.submit(ctx, self.kernel, &slot, self.next, len, parallel)?;
			let mapped = Arc::new(OnceLock::new());
			let flag = Arc::clone(&mapped);
			slot.staging.slice(..len as u64 * precision_bytes(self.precision)).map_async(MapMode::Read, move |r| {
				let _ = flag.set(r.map_err(|e| e.to_string()));
			});
			self.in_flight.push_back(Chunk { slot, first: self.next, len, mapped, since: Instant::now() });
			self.next += len;
			submitted = true;
		}
		Ok(submitted)
	}

	/// Reads back every chunk whose mapping has completed, in order, and submits more.
	/// Returns whether anything happened, and whether the job is finished.
	fn advance(&mut self, ctx: &Context, cancelled: bool) -> Result<(bool, bool)> {
		let lost = || Error::GpuUnavailable("the GPU device was lost during the call".to_owned());
		if ctx.lost.load(Ordering::Acquire) {
			return Err(lost());
		}
		let mut progressed = false;
		while let Some(front) = self.in_flight.front() {
			let Some(mapped) = front.mapped.get() else {
				if front.since.elapsed() > CHUNK_TIMEOUT {
					return Err(Error::Gpu(format!("waiting for the GPU: no result within {CHUNK_TIMEOUT:?}")));
				}
				break;
			};
			mapped.clone().map_err(|e| Error::Gpu(format!("mapping the result buffer: {e}")))?;
			// wgpu 30 can report a mapping as successful although the device was lost before
			// it was processed (gfx-rs/wgpu#10301); see `run::read_back`.
			if ctx.lost.load(Ordering::Acquire) {
				return Err(lost());
			}
			let Some(chunk) = self.in_flight.pop_front() else { break };
			if cancelled {
				chunk.slot.staging.unmap();
			} else {
				copy_out(&chunk.slot, self.precision, &mut self.out[chunk.first..chunk.first + chunk.len], false)?;
			}
			self.spare.push(chunk.slot);
			progressed = true;
		}
		if !cancelled {
			progressed |= captured(&ctx.device, || self.fill(ctx, false))?;
		}
		Ok((progressed, self.in_flight.is_empty() && (cancelled || self.next >= self.len)))
	}
}

impl Job {
	/// One step of the poller's loop for this job, after its device was polled with
	/// `polled`. Returns whether anything happened, and whether the job is finished.
	fn step(&self, polled: Option<&Error>) -> (bool, bool) {
		let mut state = lock(&self.state);
		let cancelled = state.cancelled;
		let Some(active) = state.active.as_mut() else { return (false, true) };
		let result = polled.map_or_else(|| active.advance(self.ctx, cancelled), |e| Err(e.clone()));
		let verdict = match result {
			Ok((progressed, false)) => return (progressed, false),
			Ok((_, true)) => Ok(()),
			Err(e) => Err(e),
		};
		let Some(active) = state.active.take() else { return (true, true) };
		let Active { spare, out, .. } = active;
		// Buffer sets go back to the pool only if nothing went wrong, as in `run::eval`.
		if verdict.is_ok() {
			release(self.ctx, spare);
		}
		state.outcome = Outcome::Done(verdict, out);
		let waker = state.waker.take();
		drop(state);
		self.changed.notify_all();
		if let Some(waker) = waker {
			waker.wake();
		}
		(true, true)
	}

	/// The GPU's verdict and the results, if the GPU part has finished; otherwise registers
	/// `waker` to be woken when it does.
	pub fn poll(&self, waker: &Waker) -> Option<(Result<()>, Vec<f64>)> {
		let mut state = lock(&self.state);
		match state.outcome {
			Outcome::Running => {
				match &state.waker {
					Some(w) if w.will_wake(waker) => {}
					_ => state.waker = Some(waker.clone()),
				}
				None
			}
			_ => Some(take(&mut state)),
		}
	}

	/// Blocks until the GPU part has finished, then returns its verdict and the results.
	pub fn wait(&self) -> (Result<()>, Vec<f64>) {
		if on_poller() {
			return (Err(Error::Task("waited for GPU work on the thread that completes it".to_owned())), Vec::new());
		}
		let mut state = self.changed.wait_while(lock(&self.state), |s| matches!(s.outcome, Outcome::Running)).unwrap_or_else(PoisonError::into_inner);
		take(&mut state)
	}

	/// Whether the GPU part has finished.
	pub fn is_finished(&self) -> bool {
		!matches!(lock(&self.state).outcome, Outcome::Running)
	}

	/// Submits nothing more and discards the results, then waits for the chunks already in
	/// flight, so a process exiting after the drop isn't inside the driver. Doesn't wait on
	/// the poller's own thread, which would never finish them.
	pub fn cancel(&self) {
		let mut state = lock(&self.state);
		state.cancelled = true;
		state.waker = None;
		if on_poller() {
			drop(state);
		} else {
			drop(self.changed.wait_while(state, |s| matches!(s.outcome, Outcome::Running)).unwrap_or_else(PoisonError::into_inner));
		}
	}
}

fn take(state: &mut State) -> (Result<()>, Vec<f64>) {
	match std::mem::replace(&mut state.outcome, Outcome::Taken) {
		Outcome::Done(verdict, out) => (verdict, out),
		_ => (Err(Error::Task("the GPU results were already taken".to_owned())), Vec::new()),
	}
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
	m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The poller: the jobs in flight, and the thread that drives them.
struct Pump {
	jobs: Mutex<Vec<Arc<Job>>>,
	/// Signalled when a job is added.
	work: Condvar,
	thread: OnceLock<ThreadId>,
}

static PUMP: OnceLock<std::result::Result<Arc<Pump>, String>> = OnceLock::new();

/// The poller, started on first use.
fn pump() -> Result<&'static Arc<Pump>> {
	PUMP.get_or_init(|| {
		let pump = Arc::new(Pump { jobs: Mutex::new(Vec::new()), work: Condvar::new(), thread: OnceLock::new() });
		let driver = Arc::clone(&pump);
		let handle = thread::Builder::new().name("splimes-gpu-poll".to_owned()).spawn(move || drive(&driver)).map_err(|e| format!("could not start the GPU poller thread: {e}"))?;
		let _ = pump.thread.set(handle.thread().id());
		Ok(pump)
	})
	.as_ref()
	.map_err(|e| Error::Gpu(e.clone()))
}

/// Whether this is the poller's thread.
fn on_poller() -> bool {
	PUMP.get().and_then(|p| p.as_ref().ok()).and_then(|p| p.thread.get()).is_some_and(|&id| id == thread::current().id())
}

fn drive(pump: &Pump) {
	let mut idle = MIN_IDLE;
	loop {
		let jobs: Vec<Arc<Job>> = {
			let jobs = pump.work.wait_while(lock(&pump.jobs), |jobs| jobs.is_empty()).unwrap_or_else(PoisonError::into_inner);
			jobs.clone()
		};
		// One non-blocking poll per device. It runs the callbacks of finished mappings,
		// which only record their result.
		let mut polled: Vec<(&'static Context, Option<Error>)> = Vec::new();
		for job in &jobs {
			if !polled.iter().any(|(ctx, _)| std::ptr::eq(*ctx, job.ctx)) {
				let error = job.ctx.device.poll(PollType::Poll).err().map(|e| Error::Gpu(format!("polling the GPU: {e}")));
				polled.push((job.ctx, error));
			}
		}
		let mut progressed = false;
		let mut finished = Vec::new();
		for job in &jobs {
			let error = polled.iter().find(|(ctx, _)| std::ptr::eq(*ctx, job.ctx)).and_then(|(_, e)| e.as_ref());
			let (p, done) = job.step(error);
			progressed |= p;
			if done {
				finished.push(Arc::clone(job));
			}
		}
		if !finished.is_empty() {
			lock(&pump.jobs).retain(|j| !finished.iter().any(|f| Arc::ptr_eq(j, f)));
		}
		if progressed {
			idle = MIN_IDLE;
		} else {
			thread::sleep(idle);
			idle = (idle * 2).min(MAX_IDLE);
		}
	}
}

#[cfg(test)]
mod tests {
	use std::{
		task::Waker, thread, time::{Duration, Instant}
	};

	use chrono::{DateTime, Utc};

	use super::start;
	use crate::{
		Error, Precision, Resolution, Spline, gpu::{
			context, run::tests::{LOSE_BEFORE_POLL, Loss}
		}, kernel::Method, prepare::Knots, time::Grid
	};

	/// A device lost while a job's chunks are in flight makes the job an error, never the
	/// staging buffers' previous contents, and it never hangs.
	#[test]
	fn a_device_lost_mid_job_is_an_error() {
		for loss in [Loss::Signalled, Loss::Destroyed] {
			// A device of its own: losing it must not disturb the shared one.
			let Ok(ctx) = context::open() else {
				eprintln!("a_device_lost_mid_job_is_an_error: skipped, no GPU");
				return;
			};
			let ctx: &'static context::Context = Box::leak(Box::new(ctx));
			let at = |s: i64| DateTime::<Utc>::from_timestamp(s, 0).expect("valid");
			let values: Vec<f64> = (0..200).map(f64::from).collect();
			let ts: Vec<_> = (0..).step_by(10).take(values.len()).map(at).collect();
			let knots = Knots::new(values.len(), |i| ts[i], |i| &values[i], false).expect("knots");
			let grid = Grid::new(at(0), ts[ts.len() - 1], Resolution::Seconds).expect("grid");
			let method = Method::new(Spline::Linear, &knots);
			// Lost after the first chunks are submitted, before the poller sees the job.
			LOSE_BEFORE_POLL.set(Some(loss));
			let started = start(ctx, &knots, &grid, &method, Precision::F32, vec![0.0; grid.len]);
			LOSE_BEFORE_POLL.set(None);
			let job = match started {
				Ok(job) => job,
				Err((e, _)) => panic!("{loss:?}: could not start: {e}"),
			};
			let deadline = Instant::now() + Duration::from_secs(30);
			let waker = Waker::noop();
			let verdict = loop {
				if let Some((verdict, _)) = job.poll(waker) {
					break verdict;
				}
				assert!(Instant::now() < deadline, "{loss:?}: the job never finished");
				thread::sleep(Duration::from_millis(1));
			};
			assert!(matches!(verdict, Err(Error::GpuUnavailable(_) | Error::Gpu(_))), "{loss:?}: {verdict:?}");
			// A later job on the lost device fails at once.
			assert!(start(ctx, &knots, &grid, &method, Precision::F32, Vec::new()).is_err());
		}
	}
}

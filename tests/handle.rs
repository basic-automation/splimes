//! `Interpolator::submit` / `submit_f64`: GPU work that no thread waits for.
//!
//! The GPU is a process-wide singleton, so this binary configures it once, with tiny
//! dispatch chunks so modest grids take many chunks, which the poller submits and reads
//! back on its own.

mod common;

use std::{
	future::Future, pin::Pin, sync::{
		Arc, Barrier, Mutex, MutexGuard, OnceLock, PoisonError, atomic::{AtomicUsize, Ordering}, mpsc
	}, task::{Context, Poll, Wake, Waker}, thread::{self, Thread}, time::{Duration, Instant}
};

use bigdecimal::BigDecimal;
use chrono::{DateTime, TimeDelta, Utc};
use common::epoch;
use splimes::{Backend, Error, GpuConfig, GpuInfo, Interpolator, Point, Precision, Resolution, Spline};

const CHUNK: u32 = 1_000;
const PATIENCE: Duration = Duration::from_secs(60);

/// Serialises the tests in this binary: one of them blocks every rayon worker.
static SERIAL: Mutex<()> = Mutex::new(());

fn gpu(test: &str) -> Option<(GpuInfo, MutexGuard<'static, ()>)> {
	static CONFIGURED: OnceLock<()> = OnceLock::new();
	let guard = SERIAL.lock().unwrap_or_else(PoisonError::into_inner);
	CONFIGURED.get_or_init(|| splimes::configure_gpu(GpuConfig::default().with_chunk_points(CHUNK)).expect("first configuration succeeds"));
	common::gpu_or_skip(test).map(|info| (info, guard))
}

fn precisions(info: &GpuInfo) -> Vec<Precision> {
	if info.supports_f64 { vec![Precision::F64, Precision::F32] } else { vec![Precision::F32] }
}

/// `n` irregular knots a second or so apart, from `seed`.
fn series(n: i64, seed: i64) -> (Vec<DateTime<Utc>>, Vec<f64>) {
	let ts = (0..n).map(|i| epoch() + TimeDelta::milliseconds(i * 1_000 + (i * 7_919 + seed * 31) % 700)).collect();
	let vs = (0..n).map(|i| ((i + seed) as f64 * 0.05).sin() * 50.0 + i as f64 * 0.01).collect();
	(ts, vs)
}

/// Wakes a parked thread, and counts wakes.
struct Unpark(Thread, AtomicUsize);

impl Wake for Unpark {
	fn wake(self: Arc<Self>) {
		self.1.fetch_add(1, Ordering::SeqCst);
		self.0.unpark();
	}
}

/// The smallest executor there is: poll, park until woken, repeat. Returns the output and
/// how many times it was woken.
fn block_on<F: Future + Unpin>(mut f: F) -> (F::Output, usize) {
	let unpark = Arc::new(Unpark(thread::current(), AtomicUsize::new(0)));
	let waker = Waker::from(Arc::clone(&unpark));
	let mut cx = Context::from_waker(&waker);
	let deadline = Instant::now() + PATIENCE;
	loop {
		if let Poll::Ready(v) = Pin::new(&mut f).poll(&mut cx) {
			return (v, unpark.1.load(Ordering::SeqCst));
		}
		assert!(Instant::now() < deadline, "never woken");
		thread::park_timeout(PATIENCE);
	}
}

/// Many handles in flight at once, of every size from one chunk to many, both value
/// types, every precision, resolved by `.await` and by `wait` in an order unrelated to
/// submission: each is bit-identical to `run`, including how it ran.
#[test]
fn many_handles_in_flight_match_run() {
	let Some((info, _serial)) = gpu("many_handles_in_flight_match_run") else { return };
	let splines = [Spline::Linear, Spline::Cubic, Spline::Polynomial(5, None), Spline::Polynomial(3, Some(0.5))];
	let mut cases = Vec::new();
	for (k, &spline) in splines.iter().enumerate() {
		for &precision in &precisions(&info) {
			for (j, &(knots, points)) in [(2_i64, 50_i64), (40, 999), (300, 3_500), (1_000, 25_017)].iter().enumerate() {
				let seed = i64::try_from(k * 10 + j).expect("small");
				let (ts, vs) = series(knots, seed);
				// From before the data to after it, at 1 grid point per `step` ms.
				let span = (knots + 2) * 1_000;
				let step = (span / points).max(1);
				let (start, end) = (ts[0] - TimeDelta::milliseconds(1_000), ts[0] - TimeDelta::milliseconds(1_000) + TimeDelta::milliseconds(step * (points - 1)));
				let interpolator = Interpolator::new(spline, Resolution::Milliseconds).backend(Backend::Gpu).gpu_precision(precision);
				cases.push((interpolator, ts, vs, start, end));
			}
		}
	}
	// Every handle submitted before any is resolved.
	let mut f64_handles = Vec::new();
	let mut decimal_handles = Vec::new();
	for (i, (interpolator, ts, vs, start, end)) in cases.iter().enumerate() {
		f64_handles.push((i, interpolator.submit_f64(ts.clone(), vs.clone(), *start, *end)));
		let points: Vec<Point> = ts.iter().zip(vs).map(|(&t, &v)| Point::new(t, BigDecimal::try_from(v).expect("finite"))).collect();
		decimal_handles.push((i, interpolator.submit(points, *start, *end)));
	}
	assert!(f64_handles.len() >= 16, "{} handles", f64_handles.len());
	// Resolved newest first, alternating `.await` and `wait`.
	for (n, ((i, f), (j, d))) in f64_handles.into_iter().rev().zip(decimal_handles.into_iter().rev()).enumerate() {
		let (interpolator, ts, vs, start, end) = &cases[i];
		let expected = interpolator.run_f64(ts, vs, *start, *end);
		assert!(expected.is_ok(), "{expected:?}");
		let got = if n % 2 == 0 { block_on(f).0 } else { f.wait() };
		assert_eq!(got, expected, "case {i}: {interpolator:?}");
		assert_eq!(got.as_ref().map(splimes::Interpolation::backend).ok(), Some(Backend::Gpu));

		let points: Vec<Point> = ts.iter().zip(vs).map(|(&t, &v)| Point::new(t, BigDecimal::try_from(v).expect("finite"))).collect();
		let expected = cases[j].0.run(&points, *start, *end);
		let got = if n % 2 == 0 { d.wait() } else { block_on(d).0 };
		assert_eq!(got, expected, "decimal case {j}");
	}
}

/// A handle that the GPU hasn't finished is completed through its waker.
#[test]
fn a_pending_handle_is_woken() {
	let Some((info, _serial)) = gpu("a_pending_handle_is_woken") else { return };
	let (ts, vs) = series(2_000, 1);
	// 2,000 s of milliseconds: two thousand chunks, far too many to finish before the
	// first poll.
	let (start, end) = (ts[0], ts[1_999]);
	let interpolator = Interpolator::new(Spline::Cubic, Resolution::Milliseconds).backend(Backend::Gpu).gpu_precision(precisions(&info)[0]);
	let handle = interpolator.submit_f64(ts.clone(), vs.clone(), start, end);
	assert!(!handle.is_finished());
	let (got, wakes) = block_on(handle);
	assert!(wakes >= 1, "completed without a wake");
	assert_eq!(got, interpolator.run_f64(&ts, &vs, start, end));
}

/// The `f32` windows the GPU can't compute reliably are recomputed in `f64` when a handle
/// is resolved, exactly as `run` does: the gap-limit fixture of `tests/contract.rs`, just
/// past the 1,024× limit.
#[test]
fn the_f32_repair_runs_for_handles() {
	let Some((_, _serial)) = gpu("the_f32_repair_runs_for_handles") else { return };
	let gaps: Vec<i64> = (0..39).map(|g| if (5..35).contains(&g) { 1_000 } else { 4_763_333 }).collect();
	let mut t = epoch();
	let mut ts = vec![t];
	for gap in &gaps {
		t += TimeDelta::milliseconds(*gap);
		ts.push(t);
	}
	let vs: Vec<f64> = (0..40).map(|i| (f64::from(i) * 0.3).sin()).collect();
	let (start, end) = (ts[5] - TimeDelta::milliseconds(3_600_500), ts[9]);
	for spline in [Spline::Cubic, Spline::Polynomial(5, None)] {
		let interpolator = Interpolator::new(spline, Resolution::Seconds).backend(Backend::Gpu).gpu_precision(Precision::F32);
		let expected = interpolator.run_f64(&ts, &vs, start, end).expect("runs");
		assert!(expected.points_recomputed_in_f64() > 0, "{spline}: the fixture must need the repair");
		let got = interpolator.submit_f64(ts.clone(), vs.clone(), start, end).wait().expect("runs");
		assert_eq!(got, expected, "{spline}");
	}
}

/// Dropping handles while their chunks are on the GPU returns (after the chunks in
/// flight), on any thread, including a rayon worker, and leaves the GPU working.
#[test]
fn dropping_handles_before_completion() {
	let Some((info, _serial)) = gpu("dropping_handles_before_completion") else { return };
	let precision = precisions(&info)[0];
	let (ts, vs) = series(2_000, 2);
	let (start, end) = (ts[0], ts[1_999]);
	let interpolator = Interpolator::new(Spline::Linear, Resolution::Milliseconds).backend(Backend::Gpu).gpu_precision(precision);
	let handles: Vec<_> = (0..16).map(|_| interpolator.submit_f64(ts.clone(), vs.clone(), start, end)).collect();
	assert!(handles.iter().any(|h| !h.is_finished()), "nothing was in flight to drop");
	let (dropped, has_dropped) = mpsc::channel();
	let mut handles = handles.into_iter();
	let on_worker: Vec<_> = handles.by_ref().take(4).collect();
	let rest: Vec<_> = handles.collect();
	thread::spawn(move || {
		drop(rest);
		let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().expect("a pool");
		pool.install(move || drop(on_worker));
		dropped.send(()).expect("the test is waiting");
	});
	has_dropped.recv_timeout(PATIENCE).expect("the drops returned");
	// The GPU still works, and a handle dropped straight after submission is harmless too.
	drop(interpolator.submit_f64(ts.clone(), vs.clone(), start, ts[10]));
	let small_end = ts[20];
	assert_eq!(interpolator.submit_f64(ts.clone(), vs.clone(), start, small_end).wait(), interpolator.run_f64(&ts, &vs, start, small_end));
}

/// The GPU part of a handle completes with every rayon worker blocked: the poller needs no
/// rayon thread to read chunks back and submit the next ones.
#[test]
fn gpu_progress_needs_no_rayon_thread() {
	let Some((info, _serial)) = gpu("gpu_progress_needs_no_rayon_thread") else { return };
	let (ts, vs) = series(200, 3);
	// 50,001 grid points: fifty-one chunks, all but the first two submitted by the poller.
	let (start, end) = (ts[0], ts[0] + TimeDelta::milliseconds(50_000));
	let interpolator = Interpolator::new(Spline::Cubic, Resolution::Milliseconds).backend(Backend::Gpu).gpu_precision(precisions(&info)[0]);
	let expected = interpolator.run_f64(&ts, &vs, start, end);
	let handle = interpolator.submit_f64(ts.clone(), vs.clone(), start, end);
	let workers = rayon::current_num_threads();
	let (busy, release) = (Arc::new(Barrier::new(workers + 1)), Arc::new(Barrier::new(workers + 1)));
	for _ in 0..workers {
		let (busy, release) = (Arc::clone(&busy), Arc::clone(&release));
		rayon::spawn(move || {
			busy.wait();
			release.wait();
		});
	}
	busy.wait();
	let deadline = Instant::now() + PATIENCE;
	while !handle.is_finished() {
		if Instant::now() > deadline {
			release.wait();
			panic!("the GPU part never finished with rayon's workers blocked");
		}
		thread::sleep(Duration::from_millis(1));
	}
	release.wait();
	assert_eq!(handle.wait(), expected);
}

/// The CPU backends and every refusal come through `submit` as `run` returns them.
#[test]
fn cpu_backends_and_refusals_match_run() {
	let (ts, vs) = series(50, 4);
	let (start, end) = (ts[0], ts[49]);
	for backend in [Backend::Cpu, Backend::Parallel] {
		let interpolator = Interpolator::new(Spline::Cubic, Resolution::Seconds).backend(backend);
		let handle = interpolator.submit_f64(ts.clone(), vs.clone(), start, end);
		assert!(handle.is_finished(), "{backend}: a CPU handle is complete at once");
		assert_eq!(handle.wait(), interpolator.run_f64(&ts, &vs, start, end));
	}
	let gpu = Interpolator::new(Spline::Cubic, Resolution::Seconds).backend(Backend::Gpu);
	let refusals: Vec<(Interpolator, Vec<DateTime<Utc>>, Vec<f64>, DateTime<Utc>, DateTime<Utc>)> = vec![
		(gpu, Vec::new(), Vec::new(), start, end),
		(gpu, ts.clone(), vs.clone(), end, start),
		(gpu, ts.clone(), vs[1..].to_vec(), start, end),
		(gpu.max_points(3), ts.clone(), vs.clone(), start, end),
		(Interpolator::new(Spline::Cubic, Resolution::Seconds).backend(Backend::Gpu).exact(true), ts[..2].to_vec(), vs[..2].to_vec(), start, end),
		(
			gpu,
			ts.clone(),
			{
				let mut v = vs.clone();
				v[7] = f64::NAN;
				v
			},
			start,
			end,
		),
		(Interpolator::new(Spline::Polynomial(0, None), Resolution::Seconds).backend(Backend::Gpu), ts.clone(), vs.clone(), start, end),
	];
	for (interpolator, ts, vs, start, end) in refusals {
		let expected = interpolator.run_f64(&ts, &vs, start, end);
		assert!(expected.is_err(), "{expected:?}");
		assert_eq!(interpolator.submit_f64(ts.clone(), vs.clone(), start, end).wait(), expected);
		let (got, _) = block_on(interpolator.submit_f64(ts, vs, start, end));
		assert_eq!(got, expected);
	}
}

/// A handle resolves once; polling it again is an error, not a panic.
#[test]
fn a_second_poll_is_an_error() {
	let (ts, vs) = series(10, 5);
	let mut handle = Interpolator::new(Spline::Linear, Resolution::Seconds).backend(Backend::Cpu).submit_f64(ts.clone(), vs, ts[0], ts[9]);
	assert!(block_on(&mut handle).0.is_ok());
	assert!(matches!(block_on(&mut handle).0, Err(Error::Task(_))));
}

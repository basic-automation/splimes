//! GPU work from `spawn`, its future dropped, then the process exits moments later. It must
//! exit cleanly. rayon's threads aren't joined at exit, and a process that exits while one
//! of them is in the GPU driver can crash there (before 1.1.0's fixes it did in about half
//! such runs on an NVIDIA GPU): `spawn` must open the device on the calling thread, and the
//! drop must cancel the work or wait for it, and an `Auto` job spawned before the GPU
//! started must stay off it.
//!
//! Each run is a child process, so this binary has its own `main` (`harness = false`): the
//! scenario needs the device opened on the main thread, which libtest's per-test threads
//! are not.

mod common;

use std::{
	future::Future, pin::pin, process::Command, sync::{Arc, Barrier}, task::{Context, Poll, Wake}, thread::{self, Thread}, time::Duration
};

use chrono::TimeDelta;
use common::epoch;
use splimes::{AutoThresholds, Backend, Interpolator, Precision, Resolution, Spline};

const NAME: &str = "a_dropped_gpu_spawn_lets_the_process_exit";

/// Set in the children: the scenario, how long to wait before dropping the future (so
/// the work may be queued or running), and how long after it before exiting.
const MODE: &str = "SPLIMES_SPAWN_EXIT_MODE";
const DELAYS: &str = "SPLIMES_SPAWN_EXIT_DELAYS_MS";
/// Set in the children when the adapter has no `f64`, so the work still reaches the GPU.
const F32: &str = "SPLIMES_SPAWN_EXIT_F32";

/// `Backend::Gpu` as the program's first GPU use; `Auto` with the GPU started first; `Auto`
/// spawned before the GPU starts.
const MODES: [&str; 3] = ["gpu", "auto", "auto-late"];
/// Run once, awaited rather than dropped: `Auto` spawned before the GPU starts, with the
/// job held back until the GPU is ready, must not pick the GPU.
const LATE_AWAITED: &str = "auto-late-awaited";
/// (Before the drop, after it) in milliseconds: dropped while queued and exiting at once
/// or later, and dropped while the work is (usually) running.
const DELAYS_MS: [(u64, u64); 7] = [(0, 0), (0, 20), (0, 150), (2, 0), (5, 0), (10, 0), (25, 0)];

fn main() {
	if let Some(mode) = std::env::var_os(MODE) {
		let delays = std::env::var(DELAYS).expect("delays");
		let (before, after) = delays.split_once(',').and_then(|(b, a)| Some((b.parse().ok()?, a.parse().ok()?))).expect("two delays in milliseconds");
		child(mode.to_str().expect("a mode"), Duration::from_millis(before), Duration::from_millis(after));
		return;
	}
	if !libtest_selects(NAME) {
		return;
	}
	let Some(info) = common::gpu_or_skip(NAME) else { return };
	// The children's `Auto` jobs, with the GPU started, really run on it: else the drops in
	// the "auto" mode would have nothing to wait for.
	let (interpolator, ts, vs) = workload(!info.supports_f64);
	let end = *ts.last().expect("knots");
	let out = block_on(interpolator.spawn_f64(ts, vs, epoch(), end)).expect("runs");
	assert_eq!(out.backend(), Backend::Gpu, "{NAME}: Auto didn't use the started GPU");
	let exe = std::env::current_exe().expect("this test's path");
	let mut late = Command::new(&exe);
	late.env(MODE, LATE_AWAITED).env(DELAYS, "0,0");
	if !info.supports_f64 {
		late.env(F32, "1");
	}
	let out = late.output().expect("the child runs");
	assert!(out.status.success(), "{NAME}: {LATE_AWAITED} ended with {}\n{}", out.status, String::from_utf8_lossy(&out.stderr));
	for mode in MODES {
		for (before, after) in DELAYS_MS {
			let mut child = Command::new(&exe);
			child.env(MODE, mode).env(DELAYS, format!("{before},{after}"));
			if !info.supports_f64 {
				child.env(F32, "1");
			}
			let out = child.output().expect("the child runs");
			assert!(out.status.success(), "{NAME}: {mode}, dropped {before} ms after spawning and exiting {after} ms later, ended with {}\n{}", out.status, String::from_utf8_lossy(&out.stderr));
		}
	}
	eprintln!("{NAME}: {} runs exited cleanly", MODES.len() * DELAYS_MS.len() + 1);
}

/// The interpolation every child spawns, with `Auto` thresholds that send it to the GPU:
/// four million grid points, so the GPU work takes a while.
fn workload(f32: bool) -> (Interpolator, Vec<chrono::DateTime<chrono::Utc>>, Vec<f64>) {
	let ts: Vec<_> = (0..4_096).map(|i| epoch() + TimeDelta::seconds(i)).collect();
	let vs: Vec<f64> = (0..4_096).map(|i| f64::from(i).sin()).collect();
	splimes::set_auto_thresholds(AutoThresholds::new(0, 0).with_gpu_f32(0));
	let precision = if f32 { Precision::F32 } else { Precision::F64 };
	(Interpolator::new(Spline::Cubic, Resolution::Milliseconds).gpu_precision(precision), ts, vs)
}

/// Spawns GPU work, drops the future and returns, which ends the process.
fn child(mode: &str, before: Duration, after: Duration) {
	let (interpolator, ts, vs) = workload(std::env::var_os(F32).is_some());
	let end = *ts.last().expect("knots");
	if mode == LATE_AWAITED {
		// Every rayon worker waits at a barrier, so the job can't pick its backend until the
		// GPU is ready.
		let workers = rayon::current_num_threads();
		let release = Arc::new(Barrier::new(workers + 1));
		for _ in 0..workers {
			let release = Arc::clone(&release);
			rayon::spawn(move || drop(release.wait()));
		}
		let future = interpolator.spawn_f64(ts, vs, epoch(), end);
		splimes::prewarm_gpu().expect("the parent found a GPU");
		release.wait();
		let out = block_on(future).expect("runs");
		assert_ne!(out.backend(), Backend::Gpu, "an Auto job spawned before the GPU started picked it");
		return;
	}
	let future = match mode {
		"gpu" => {
			let future = interpolator.backend(Backend::Gpu).spawn_f64(ts, vs, epoch(), end);
			assert!(splimes::gpu_info().is_some(), "spawn returned before opening the device on this thread");
			future
		}
		"auto" => {
			splimes::prewarm_gpu().expect("the parent found a GPU");
			interpolator.spawn_f64(ts, vs, epoch(), end)
		}
		"auto-late" => {
			let future = interpolator.spawn_f64(ts, vs, epoch(), end);
			splimes::prewarm_gpu().expect("the parent found a GPU");
			future
		}
		other => panic!("unknown mode {other}"),
	};
	thread::sleep(before);
	drop(future);
	thread::sleep(after);
}

/// Whether libtest's command line, as cargo and cargo-nextest pass it, selects the test
/// `name`. Answers `--list` itself.
fn libtest_selects(name: &str) -> bool {
	let (mut list, mut exact, mut ignored) = (false, false, false);
	let (mut filters, mut skips) = (Vec::new(), Vec::new());
	let mut args = std::env::args().skip(1);
	while let Some(arg) = args.next() {
		match arg.as_str() {
			"--list" => list = true,
			"--exact" => exact = true,
			"--ignored" => ignored = true,
			"--skip" => skips.extend(args.next()),
			// Options that take a value, which isn't a filter.
			"--test-threads" | "--format" | "--color" | "--logfile" | "--shuffle-seed" | "-Z" => drop(args.next()),
			flag if flag.starts_with('-') => {}
			_ => filters.push(arg),
		}
	}
	let matches = |pattern: &String| if exact { pattern == name } else { name.contains(pattern.as_str()) };
	let selected = !ignored && (filters.is_empty() || filters.iter().any(matches)) && !skips.iter().any(matches);
	if list {
		if selected {
			println!("{name}: test");
		}
		return false;
	}
	selected
}

/// The smallest executor: poll, park until woken, repeat.
fn block_on<F: Future>(future: F) -> F::Output {
	struct Unpark(Thread);
	impl Wake for Unpark {
		fn wake(self: Arc<Self>) {
			self.0.unpark();
		}
	}
	let waker = Arc::new(Unpark(thread::current())).into();
	let mut cx = Context::from_waker(&waker);
	let mut future = pin!(future);
	loop {
		if let Poll::Ready(out) = future.as_mut().poll(&mut cx) {
			return out;
		}
		thread::park();
	}
}

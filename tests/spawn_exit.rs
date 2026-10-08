//! `spawn` as the program's first use of the GPU, with the future dropped and the process
//! exiting moments later. splimes must have opened the device on the calling thread by
//! then: a process that exits while a driver initialises on one of rayon's threads, which
//! nothing joins, can crash in the driver (it did in about half of such runs on an NVIDIA
//! GPU before `spawn` opened the device itself).
//!
//! Each run is a child process, so this binary has its own `main` (`harness = false`): the
//! scenario needs the device opened on the main thread, which libtest's per-test threads
//! are not.

mod common;

use std::{process::Command, thread, time::Duration};

use chrono::TimeDelta;
use common::epoch;
use splimes::{Backend, Interpolator, Resolution, Spline};

/// Set in the children: how long to wait, in milliseconds, before exiting.
const CHILD: &str = "SPLIMES_SPAWN_EXIT_CHILD";

/// Exit delays, from immediately to after the device is usually open.
const DELAYS_MS: [u64; 10] = [0, 0, 5, 10, 20, 30, 50, 100, 150, 250];

fn main() {
	if let Some(delay) = std::env::var_os(CHILD) {
		let delay = delay.to_str().and_then(|d| d.parse().ok()).expect("a delay in milliseconds");
		child(Duration::from_millis(delay));
		return;
	}
	let test = "a_dropped_gpu_spawn_lets_the_process_exit";
	if common::gpu_or_skip(test).is_none() {
		return;
	}
	let exe = std::env::current_exe().expect("this test's path");
	for (run, delay) in DELAYS_MS.into_iter().enumerate() {
		let out = Command::new(&exe).env(CHILD, delay.to_string()).output().expect("the child runs");
		assert!(out.status.success(), "{test}: run {run} (exiting {delay} ms after spawning) ended with {}\n{}", out.status, String::from_utf8_lossy(&out.stderr));
	}
	println!("{test}: {} runs exited cleanly", DELAYS_MS.len());
}

/// Spawns GPU work as the process's first GPU call, drops the future and exits.
fn child(delay: Duration) {
	let ts: Vec<_> = (0..4_096).map(|i| epoch() + TimeDelta::milliseconds(i * 250)).collect();
	let vs: Vec<f64> = (0..4_096).map(|i| f64::from(i).sin()).collect();
	let end = *ts.last().expect("knots");
	drop(Interpolator::new(Spline::Cubic, Resolution::Milliseconds).backend(Backend::Gpu).spawn_f64(ts, vs, epoch(), end));
	thread::sleep(delay);
}

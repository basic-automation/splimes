//! Requests that interleave CPU work with GPU interpolation, two ways: `run` (the caller
//! blocks on each) and `spawn` (a rayon worker blocks on each). A third mode, `submit`
//! (non-blocking GPU handles completed by one poller thread), was measured for 1.3 and
//! dropped: see "Overlapping requests" in BENCHMARKS.md.
//!
//! Each request first does `--parse-us` microseconds of CPU work on the calling thread
//! (standing in for parsing or reading the next request's input), then interpolates a
//! `--knots`-knot series onto a `--points`-point grid on the GPU. `spawn`
//! keeps up to `--depth` requests in flight, resolving the oldest when full.
//!
//! ```text
//! cargo run --release --example handles -- --smoke
//! cargo run --release --example handles -- --requests 512 --points 1048576 --parse-us 2000 --depth 8
//! ```
//!
//! Reported per mode: wall time, requests per second, the caller's time inside splimes
//! calls (`run`, `spawn`) and waiting for results, and the thread each request occupies
//! while the GPU computes: `run` blocks the caller, `spawn` a rayon worker.

use std::{
	collections::VecDeque, future::Future, hint::black_box, pin::Pin, sync::Arc, task::{Context, Poll, Wake, Waker}, thread::{self, Thread}, time::{Duration, Instant}
};

use chrono::{DateTime, TimeDelta, Utc};
use splimes::{Backend, Interpolation, Interpolator, Precision, Resolution, Spline};

struct Args {
	requests: usize,
	points: i64,
	knots: i64,
	parse_us: u64,
	depth: usize,
	precision: Precision,
	modes: Vec<String>,
}

fn args() -> Args {
	let mut a = Args { requests: 256, points: 1 << 20, knots: 4096, parse_us: 2_000, depth: 8, precision: Precision::F64, modes: vec!["run".into(), "spawn".into()] };
	let mut it = std::env::args().skip(1);
	while let Some(flag) = it.next() {
		let mut value = || it.next().unwrap_or_else(|| panic!("{flag} needs a value"));
		match flag.as_str() {
			"--smoke" => {
				a.requests = 16;
				a.points = 1 << 14;
				a.knots = 256;
				a.parse_us = 200;
				a.depth = 4;
			}
			"--requests" => a.requests = value().parse().expect("a count"),
			"--points" => a.points = value().parse().expect("a count"),
			"--knots" => a.knots = value().parse().expect("a count"),
			"--parse-us" => a.parse_us = value().parse().expect("microseconds"),
			"--depth" => a.depth = value().parse::<usize>().expect("a count").max(1),
			"--precision" => a.precision = if value() == "f32" { Precision::F32 } else { Precision::F64 },
			"--modes" => a.modes = value().split(',').map(str::to_owned).collect(),
			other => panic!("unknown flag {other}"),
		}
	}
	a
}

/// `knots` irregular samples over a grid of `points` milliseconds, varied by `seed`.
fn series(knots: i64, points: i64, seed: usize) -> (Vec<DateTime<Utc>>, Vec<f64>, DateTime<Utc>, DateTime<Utc>) {
	let start = DateTime::<Utc>::UNIX_EPOCH;
	let spacing_us = 1000.0 * points as f64 / knots as f64;
	let ts = (0..knots).map(|i| start + TimeDelta::microseconds((spacing_us * (i as f64 + 0.4 * ((i * 7919) % 100) as f64 / 100.0)) as i64)).collect();
	let vs = (0..knots).map(|i| ((i as f64 + seed as f64) * 0.01).sin() * 100.0).collect();
	(ts, vs, start, start + TimeDelta::milliseconds(points - 1))
}

/// `us` microseconds of CPU work on this thread, standing in for parsing a request.
fn parse(us: u64, seed: usize) -> u64 {
	let until = Instant::now() + Duration::from_micros(us);
	let mut h = seed as u64 ^ 0x9e37_79b9_7f4a_7c15;
	while Instant::now() < until {
		for _ in 0..256 {
			h = h.rotate_left(5).wrapping_mul(0x0100_0000_01b3) ^ 0xcbf2_9ce4_8422_2325;
		}
	}
	h
}

struct Unpark(Thread);

impl Wake for Unpark {
	fn wake(self: Arc<Self>) {
		self.0.unpark();
	}
}

fn block_on<F: Future + Unpin>(mut f: F) -> F::Output {
	let waker = Waker::from(Arc::new(Unpark(thread::current())));
	let mut cx = Context::from_waker(&waker);
	loop {
		if let Poll::Ready(v) = Pin::new(&mut f).poll(&mut cx) {
			return v;
		}
		thread::park();
	}
}

#[derive(Default)]
struct Totals {
	wall: Duration,
	/// The caller's time inside `run` or `spawn`.
	calling: Duration,
	/// The caller's time waiting for a result.
	waiting: Duration,
	points: usize,
	checksum: f64,
}

fn tally(totals: &mut Totals, out: &Interpolation<f64>) {
	totals.points += out.len();
	totals.checksum += out.values().iter().step_by(4099).sum::<f64>();
}

fn main() {
	let a = args();
	let info = match splimes::prewarm_gpu() {
		Ok(info) => info,
		Err(e) => {
			eprintln!("no GPU: {e}");
			return;
		}
	};
	let precision = if a.precision == Precision::F64 && !info.supports_f64 { Precision::F32 } else { a.precision };
	println!("# GPU: {} ({}), splimes {}, precision {precision}, rayon threads {}", info.name, info.api, env!("CARGO_PKG_VERSION"), rayon::current_num_threads());
	println!("# {} requests, {} knots onto {} points each, {} us CPU work before each, depth {}", a.requests, a.knots, a.points, a.parse_us, a.depth);
	let interpolator = Interpolator::new(Spline::Cubic, Resolution::Milliseconds).backend(Backend::Gpu).gpu_precision(precision);
	let inputs: Vec<_> = (0..8).map(|s| series(a.knots, a.points, s)).collect();

	// The modes must agree bit for bit.
	let (ts, vs, start, end) = &inputs[0];
	let reference = interpolator.run_f64(ts, vs, *start, *end).expect("runs");
	assert_eq!(block_on(interpolator.spawn_f64(ts.clone(), vs.clone(), *start, *end)).expect("runs"), reference, "spawn differs from run");

	println!("mode,requests,wall_ms,requests_per_s,caller_calling_ms,caller_waiting_ms,threads_blocked_per_in_flight_request,points,checksum");
	for mode in &a.modes {
		let mut t = Totals::default();
		let begin = Instant::now();
		let mut sink = 0_u64;
		match mode.as_str() {
			"run" => {
				for r in 0..a.requests {
					sink ^= parse(a.parse_us, r);
					let (ts, vs, start, end) = &inputs[r % inputs.len()];
					let at = Instant::now();
					let out = interpolator.run_f64(ts, vs, *start, *end).expect("runs");
					t.calling += at.elapsed();
					tally(&mut t, &out);
				}
			}
			"spawn" => {
				let mut queue: VecDeque<splimes::InterpolationFuture<f64>> = VecDeque::new();
				for r in 0..a.requests {
					sink ^= parse(a.parse_us, r);
					if queue.len() >= a.depth {
						let at = Instant::now();
						let out = block_on(queue.pop_front().expect("full")).expect("runs");
						t.waiting += at.elapsed();
						tally(&mut t, &out);
					}
					let (ts, vs, start, end) = inputs[r % inputs.len()].clone();
					let at = Instant::now();
					queue.push_back(interpolator.spawn_f64(ts, vs, start, end));
					t.calling += at.elapsed();
				}
				for f in queue {
					let at = Instant::now();
					let out = block_on(f).expect("runs");
					t.waiting += at.elapsed();
					tally(&mut t, &out);
				}
			}
			other => panic!("unknown mode {other}"),
		}
		t.wall = begin.elapsed();
		black_box(sink);
		let blocked = if mode == "run" { "1 (the caller)" } else { "1 (a rayon worker)" };
		println!("{mode},{},{:.1},{:.1},{:.1},{:.1},{blocked},{},{:.6e}", a.requests, t.wall.as_secs_f64() * 1e3, a.requests as f64 / t.wall.as_secs_f64(), t.calling.as_secs_f64() * 1e3, t.waiting.as_secs_f64() * 1e3, t.points, t.checksum);
	}
}

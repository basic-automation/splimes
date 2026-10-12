//! `Backend::Cpu` runs on the calling thread only: preparing the input, the kernel and
//! assembling the output. This binary occupies every rayon worker while it checks that,
//! so it holds a single test.

use std::{
	sync::{Arc, Barrier, mpsc}, time::Duration
};

use bigdecimal::BigDecimal;
use chrono::{TimeDelta, TimeZone, Utc};
use splimes::{Backend, Interpolator, Point, Resolution, Spline};

#[test]
fn cpu_never_waits_for_rayon() {
	// Every rayon worker blocks until the calls below have finished, so any use of the
	// pool would wait for them.
	let workers = rayon::current_num_threads();
	let (started, release) = (Arc::new(Barrier::new(workers + 1)), Arc::new(Barrier::new(workers + 1)));
	{
		let (started, release) = (Arc::clone(&started), Arc::clone(&release));
		rayon::spawn_broadcast(move |_| {
			started.wait();
			release.wait();
		});
	}
	started.wait();

	// On a thread of its own, so a call that waits for the pool times out instead of
	// hanging the test.
	let (tx, rx) = mpsc::channel();
	std::thread::spawn(move || {
		let t0 = Utc.timestamp_opt(1_700_000_000, 0).single().expect("valid");
		// Above every size at which the other backends go parallel: 64 Ki unsorted inputs
		// (prepared on rayon from 16 Ki) and a grid of a million points.
		let n: i64 = 65_536;
		let ts: Vec<_> = (0..n).rev().map(|i| t0 + TimeDelta::milliseconds(i * 16)).collect();
		let vs: Vec<f64> = (0..n).map(|i| (i as f64 * 0.01).sin()).collect();
		let points: Vec<Point> = ts.iter().zip(&vs).map(|(&t, &v)| Point::new(t, BigDecimal::try_from(v).expect("finite"))).collect();
		let cpu = Interpolator::new(Spline::Cubic, Resolution::Milliseconds).backend(Backend::Cpu);
		let end = t0 + TimeDelta::milliseconds((n - 1) * 16);
		let f64s = cpu.run_f64(&ts, &vs, t0, end).map(|r| (r.len(), r.backend()));
		let decimals = cpu.run(&points, t0, end).map(|r| (r.len(), r.backend()));
		let _ = tx.send((f64s, decimals));
	});
	let result = rx.recv_timeout(Duration::from_secs(120));
	release.wait();

	let (f64s, decimals) = result.expect("Backend::Cpu waited for rayon's pool");
	let expected = Ok((1_048_561, Backend::Cpu));
	assert_eq!(f64s, expected);
	assert_eq!(decimals, expected);
}

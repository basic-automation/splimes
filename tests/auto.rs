//! `Backend::Auto`'s choices. Thresholds are process-wide, so this binary holds a single
//! test that owns them.

mod common;

use chrono::TimeDelta;
use common::epoch;
use splimes::{AutoThresholds, Backend, Interpolator, Precision, Resolution, Spline};

#[test]
fn auto_follows_the_thresholds() {
	assert_eq!(splimes::auto_thresholds(), AutoThresholds::DEFAULT);
	let ts = vec![epoch(), epoch() + TimeDelta::seconds(100)];
	let vs = vec![0.0, 1.0];
	let run = |points: i64, precision| Interpolator::new(Spline::Linear, Resolution::Seconds).gpu_precision(precision).run_f64(&ts, &vs, ts[0], ts[0] + TimeDelta::seconds(points - 1)).expect("runs").backend();

	splimes::set_auto_thresholds(AutoThresholds::new(10, usize::MAX));
	assert_eq!(run(9, Precision::F64), Backend::Cpu);
	assert_eq!(run(10, Precision::F64), Backend::Parallel);

	// GPU-eligible from 20 points, but the GPU hasn't been started: Auto runs on the CPU and
	// does not start it (opening a device on a thread of its own could crash the process
	// at exit).
	splimes::set_auto_thresholds(AutoThresholds::new(10, 20));
	assert_eq!(run(25, Precision::F64), Backend::Parallel);
	assert_eq!(run(25, Precision::F32), Backend::Parallel);
	// Long enough for a background start-up, had one been launched, to have finished.
	std::thread::sleep(std::time::Duration::from_millis(1500));
	assert!(splimes::gpu_info().is_none(), "Auto must not start the GPU, even in the background");

	// Once the caller starts it, eligible calls go to it, each precision by its own
	// threshold.
	if let Some(info) = common::gpu_or_skip("auto_follows_the_thresholds (GPU part)") {
		assert_eq!(run(15, Precision::F64), Backend::Parallel, "below the GPU threshold");
		assert_eq!(run(25, Precision::F64), if info.supports_f64 { Backend::Gpu } else { Backend::Parallel }, "f64: {}", info.supports_f64);
		assert_eq!(run(25, Precision::F32), Backend::Gpu);
		splimes::set_auto_thresholds(AutoThresholds::new(10, usize::MAX).with_gpu_f32(20));
		assert_eq!(run(25, Precision::F64), Backend::Parallel, "f64 GPU threshold is never");
		assert_eq!(run(25, Precision::F32), Backend::Gpu);
	}

	splimes::set_auto_thresholds(AutoThresholds::DEFAULT);
}

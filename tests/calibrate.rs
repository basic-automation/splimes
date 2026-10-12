//! `calibrate()` end to end. It sets the process-wide thresholds and starts the GPU, so
//! this binary holds a single test that owns them.

mod common;

use splimes::{AutoThresholds, Precision};

#[test]
fn calibrate_measures_every_size_and_sets_what_it_returns() {
	assert_eq!(splimes::auto_thresholds(), AutoThresholds::DEFAULT);
	let calibration = splimes::calibrate().expect("the CPU backends never fail here");

	// Every documented size, smallest first, each timed on both CPU backends.
	let sizes: Vec<usize> = calibration.samples.iter().map(|s| s.points).collect();
	assert_eq!(sizes, (0..8).map(|k| 1 << (10 + 2 * k)).collect::<Vec<usize>>());
	assert!(calibration.samples.iter().all(|s| !s.cpu.is_zero() && !s.parallel.is_zero()), "{calibration:?}");

	// What it returns is what Auto now uses, and every threshold is a measured size or never.
	let t = calibration.thresholds;
	assert_eq!(splimes::auto_thresholds(), t);
	for threshold in [t.parallel_min_points, t.gpu_min_points, t.gpu_f32_min_points] {
		assert!(threshold == usize::MAX || sizes.contains(&threshold), "{threshold} in {calibration:?}");
	}

	// The GPU is measured in exactly the precisions it can run, and a precision it can't
	// run is never chosen.
	match common::gpu_or_skip("calibrate (GPU part)") {
		Some(info) => {
			assert!(splimes::gpu_info().is_some(), "calibrate starts the GPU");
			assert!(calibration.samples.iter().all(|s| s.gpu_f32.is_some() && s.gpu_f64.is_some() == info.supports_f64), "{calibration:?}");
			if !info.supports_f64 {
				assert_eq!(t.gpu_min_points_for(Precision::F64), usize::MAX);
			}
		}
		None => {
			assert!(calibration.samples.iter().all(|s| s.gpu_f32.is_none() && s.gpu_f64.is_none()), "{calibration:?}");
			assert_eq!((t.gpu_min_points, t.gpu_f32_min_points), (usize::MAX, usize::MAX));
		}
	}

	splimes::set_auto_thresholds(AutoThresholds::DEFAULT);
}

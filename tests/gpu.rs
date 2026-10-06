//! The GPU backend's lifecycle: configuration, chunked dispatch, buffer reuse and
//! concurrent callers. Accuracy is `contract.rs`'s job.
//!
//! The GPU is a process-wide singleton, so this binary configures it once, with tiny
//! dispatch chunks so modest grids exercise the multi-chunk pipeline.

mod common;

use std::sync::{Mutex, MutexGuard, OnceLock, PoisonError};

use chrono::{DateTime, TimeDelta, Utc};
use common::epoch;
use splimes::{Backend, Error, GpuConfig, GpuInfo, Interpolator, Precision, Resolution, Spline};

const CHUNK: u32 = 1_000;
const POOL_CAP: u64 = 64 << 20;

/// Serialises the tests in this binary: several of them read the shared pool counters.
static SERIAL: Mutex<()> = Mutex::new(());

fn gpu(test: &str) -> Option<(GpuInfo, MutexGuard<'static, ()>)> {
	static CONFIGURED: OnceLock<()> = OnceLock::new();
	let guard = SERIAL.lock().unwrap_or_else(PoisonError::into_inner);
	CONFIGURED.get_or_init(|| splimes::configure_gpu(GpuConfig::default().with_chunk_points(CHUNK).with_max_pool_bytes(POOL_CAP)).expect("first configuration succeeds"));
	common::gpu_or_skip(test).map(|info| (info, guard))
}

fn series(n: i64) -> (Vec<DateTime<Utc>>, Vec<f64>) {
	let ts = (0..n).map(|i| epoch() + TimeDelta::milliseconds(i * 1_000 + (i * 7_919) % 700)).collect();
	let vs = (0..n).map(|i| (i as f64 * 0.05).sin() * 50.0 + i as f64 * 0.01).collect();
	(ts, vs)
}

fn precisions(info: &GpuInfo) -> Vec<Precision> {
	if info.supports_f64 { vec![Precision::F64, Precision::F32] } else { vec![Precision::F32] }
}

#[test]
fn reports_the_adapter() {
	let Some((info, _serial)) = gpu("reports_the_adapter") else { return };
	assert!(!info.name.is_empty());
	assert_eq!(splimes::gpu_info(), Some(info.clone()));
	assert_eq!(splimes::gpu_config().chunk_points, CHUNK);
	println!("GPU: {info:?}");
}

#[test]
fn configuration_after_start_is_refused() {
	let Some((_, _serial)) = gpu("configuration_after_start_is_refused") else { return };
	assert!(matches!(splimes::configure_gpu(GpuConfig::high_performance()), Err(Error::GpuAlreadyConfigured(_))));
	assert!(matches!(splimes::prewarm_gpu_with_config(GpuConfig::minimal()), Err(Error::GpuAlreadyConfigured(_))));
}

#[test]
fn many_chunks_match_the_cpu() {
	let Some((info, _serial)) = gpu("many_chunks_match_the_cpu") else { return };
	let (ts, vs) = series(2_000);
	// 10,007 grid points: ten full chunks and a ragged one, none a multiple of the
	// 256-thread workgroup.
	let (start, end) = (ts[0] - TimeDelta::milliseconds(3_000), ts[0] + TimeDelta::milliseconds(3_000 + 10_006 * 200));
	for precision in precisions(&info) {
		for spline in [Spline::Linear, Spline::Cubic, Spline::Polynomial(5, Some(0.1))] {
			let make = |backend| Interpolator::new(spline, Resolution::Milliseconds).backend(backend).gpu_precision(precision);
			let cpu = make(Backend::Cpu).run_f64(&ts, &vs, start, end).expect("cpu");
			let gpu = make(Backend::Gpu).run_f64(&ts, &vs, start, end).expect("gpu");
			assert_eq!(gpu.backend(), Backend::Gpu);
			assert_eq!(gpu.precision(), precision);
			assert_eq!(gpu.points_recomputed_in_f64(), 0, "{precision} {spline}");
			assert_eq!(gpu.len(), cpu.len());
			assert_eq!(gpu.kinds(), cpu.kinds());
			let tolerance = if precision == Precision::F64 { 1e-9 } else { 1e-2 };
			for (i, (g, c)) in gpu.values().iter().zip(cpu.values()).enumerate() {
				assert!((g - c).abs() <= tolerance, "{precision} {spline} point {i}: gpu {g} vs cpu {c}");
			}
		}
	}
}

#[test]
fn buffers_are_reused_across_calls() {
	let Some((info, _serial)) = gpu("buffers_are_reused_across_calls") else { return };
	let (ts, vs) = series(100);
	let precision = precisions(&info)[0];
	let run = || Interpolator::new(Spline::Linear, Resolution::Milliseconds).backend(Backend::Gpu).gpu_precision(precision).run_f64(&ts, &vs, ts[0], ts[0] + TimeDelta::milliseconds(4_999)).expect("gpu");
	run();
	let before = splimes::gpu_pool_stats().expect("GPU started");
	for _ in 0..5 {
		run();
	}
	let after = splimes::gpu_pool_stats().expect("GPU started");
	assert!(after.reused >= before.reused + 5, "{before:?} -> {after:?}");
	assert!(after.idle_bytes <= POOL_CAP, "the pool stays under its configured cap: {after:?}");
}

#[test]
fn concurrent_callers_get_their_own_answers() {
	let Some((info, _serial)) = gpu("concurrent_callers_get_their_own_answers") else { return };
	let precision = precisions(&info)[0];
	let handles: Vec<_> = (0..8)
		.map(|thread| {
			std::thread::spawn(move || {
				// Each thread sends its own series — a different curve and time span — and
				// checks every value against the CPU, so a crossed wire or a dropped result
				// shows. (A constant series would normalise to all-zero knot values, the same
				// for every thread.)
				let n = 20 + i64::from(thread) * 7;
				let ts: Vec<_> = (0..n).map(|i| epoch() + TimeDelta::milliseconds(i * (300 + i64::from(thread) * 50))).collect();
				let vs: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3 + f64::from(thread)).sin() * (1.0 + f64::from(thread))).collect();
				let make = |backend| Interpolator::new(Spline::Cubic, Resolution::Milliseconds).backend(backend).gpu_precision(precision);
				let cpu = make(Backend::Cpu).run_f64(&ts, &vs, ts[0], ts[ts.len() - 1]).expect("cpu");
				let tolerance = if precision == Precision::F64 { 1e-12 } else { 1e-3 } * (1.0 + f64::from(thread));
				for _ in 0..20 {
					let gpu = make(Backend::Gpu).run_f64(&ts, &vs, ts[0], ts[ts.len() - 1]).expect("gpu");
					assert_eq!(gpu.len(), cpu.len(), "thread {thread}");
					// Every value came from the GPU itself, not a CPU recomputation.
					assert_eq!(gpu.points_recomputed_in_f64(), 0, "thread {thread}");
					for (g, c) in gpu.values().iter().zip(cpu.values()) {
						assert!((g - c).abs() <= tolerance, "thread {thread}: gpu {g} vs cpu {c}");
					}
				}
			})
		})
		.collect();
	for h in handles {
		h.join().expect("thread finished");
	}
}

#[test]
fn bigdecimal_runs_on_the_gpu() {
	let Some((info, _serial)) = gpu("bigdecimal_runs_on_the_gpu") else { return };
	let points: Vec<splimes::Point> = (0..10).map(|i| splimes::Point::new(epoch() + TimeDelta::seconds(i * 10), (i * i).into())).collect();
	let precision = precisions(&info)[0];
	let make = |backend| Interpolator::new(Spline::Cubic, Resolution::Seconds).backend(backend).gpu_precision(precision);
	let gpu = make(Backend::Gpu).run(&points, points[0].timestamp, points[9].timestamp).expect("gpu");
	let cpu = make(Backend::Cpu).run(&points, points[0].timestamp, points[9].timestamp).expect("cpu");
	assert_eq!(gpu.backend(), Backend::Gpu);
	assert_eq!(gpu.points_recomputed_in_f64(), 0);
	assert_eq!(gpu.values()[20], bigdecimal::BigDecimal::from(4), "raw point");
	assert_eq!(gpu.len(), 91);
	// The cubic through squares is exact, so every interpolated value must match too.
	let tolerance = if precision == Precision::F64 { 1e-9 } else { 1e-2 };
	for (g, c) in gpu.values().iter().zip(cpu.values()) {
		assert!((common::to_f64(g) - common::to_f64(c)).abs() <= tolerance, "gpu {g} vs cpu {c}");
	}
}

/// Repeated calls with mixed grid sizes reach a steady state: after one warm-up round,
/// every call reuses pooled buffers, and the pool never exceeds its cap.
#[test]
fn pool_is_stable_under_mixed_sizes() {
	let Some((info, _serial)) = gpu("pool_is_stable_under_mixed_sizes") else { return };
	let (ts, vs) = series(50);
	let precision = precisions(&info)[0];
	let round = || {
		for points in [10_i64, 5_000, 300, 20_000, 1, 1_000, 777] {
			Interpolator::new(Spline::Cubic, Resolution::Milliseconds).backend(Backend::Gpu).gpu_precision(precision).run_f64(&ts, &vs, ts[0], ts[0] + TimeDelta::milliseconds(points - 1)).expect("gpu");
		}
		splimes::gpu_pool_stats().expect("GPU started")
	};
	round();
	let warm = round();
	for _ in 0..5 {
		let now = round();
		assert_eq!(now.created, warm.created, "no new buffers once warm: {warm:?} -> {now:?}");
		assert!(now.idle_bytes <= POOL_CAP, "{now:?}");
	}
}

/// A call holds buffers for at most two chunks however long its grid, so its device
/// memory is bounded by `chunk_points`, not by the grid.
#[test]
fn a_call_holds_at_most_two_chunks_of_buffers() {
	let Some((info, _serial)) = gpu("a_call_holds_at_most_two_chunks_of_buffers") else { return };
	let (ts, vs) = series(200);
	// 64 chunks of 1,000 points. Whether its sets come fresh or from the pool, the call
	// may take at most two of them: created + reused counts every set it acquired.
	let points = 64 * i64::from(CHUNK);
	let before = splimes::gpu_pool_stats().expect("GPU started");
	Interpolator::new(Spline::Linear, Resolution::Microseconds).backend(Backend::Gpu).gpu_precision(precisions(&info)[0]).run_f64(&ts, &vs, ts[0], ts[0] + TimeDelta::microseconds(points - 1)).expect("gpu");
	let after = splimes::gpu_pool_stats().expect("GPU started");
	let acquired = (after.created + after.reused) - (before.created + before.reused);
	assert!(acquired <= 2, "64 chunks acquired {acquired} buffer sets");
}

/// Single precision can't span windows that mix dense and sparse knots (it underflows to a
/// plausible 0); those points are recomputed in f64 and reported, never returned wrong.
#[test]
fn f32_hands_ill_scaled_windows_to_f64() {
	let Some((_, _serial)) = gpu("f32_hands_ill_scaled_windows_to_f64") else { return };
	// 10,000 one-second knots, then gaps of a minute, an hour, 2.5 days and 150 days.
	let mut ts: Vec<DateTime<Utc>> = (0..10_000).map(|i| epoch() + TimeDelta::seconds(i)).collect();
	for gap in [60, 3_600, 216_000, 12_960_000] {
		let last = ts[ts.len() - 1];
		ts.push(last + TimeDelta::seconds(gap));
	}
	let vs: Vec<f64> = (0..ts.len()).map(|i| (i as f64 * 0.37).sin()).collect();
	let (start, end) = (ts[9_990], ts[9_999] + TimeDelta::seconds(30));
	for spline in [Spline::Polynomial(7, None), Spline::Polynomial(8, Some(0.5))] {
		let make = |backend, precision| Interpolator::new(spline, Resolution::Milliseconds).backend(backend).gpu_precision(precision);
		let cpu = make(Backend::Cpu, Precision::F64).run_f64(&ts, &vs, start, end).expect("cpu");
		let gpu = make(Backend::Gpu, Precision::F32).run_f64(&ts, &vs, start, end).expect("gpu");
		assert!(gpu.points_recomputed_in_f64() > 0, "{spline}: the ill-scaled windows must go to f64");
		for (i, (g, c)) in gpu.values().iter().zip(cpu.values()).enumerate() {
			assert!((g - c).abs() <= 1e-5 * (1.0 + c.abs()), "{spline} point {i}: f32 GPU {g} vs CPU {c}");
		}
	}
}

/// The GPU names the same first overflowing point as the CPU.
#[test]
fn gpu_non_finite_results_name_the_first_point() {
	let Some((info, _serial)) = gpu("gpu_non_finite_results_name_the_first_point") else { return };
	let ts = vec![epoch(), epoch() + TimeDelta::seconds(1)];
	let vs = vec![-1e308, 1e308];
	let run = |backend, precision| Interpolator::new(Spline::Linear, Resolution::Milliseconds).backend(backend).gpu_precision(precision).run_f64(&ts, &vs, ts[0], ts[0] + TimeDelta::seconds(200)).expect_err("overflows");
	let cpu = run(Backend::Cpu, Precision::F64);
	for precision in precisions(&info) {
		assert_eq!(run(Backend::Gpu, precision), cpu, "{precision}");
	}
}

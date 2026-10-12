//! The GPU backend end to end: configure it, start it, run on it in `f64` and `f32`, see
//! what actually ran, and let `Backend::Auto` use it above a threshold.
//!
//! ```text
//! cargo run --release --example gpu
//! ```
//!
//! Without a usable adapter (or built without the `gpu` feature) it says why and stops.

use std::time::Instant;

use chrono::{DateTime, TimeDelta, TimeZone, Utc};
use splimes::{AutoThresholds, Backend, GpuConfig, Interpolation, Interpolator, Precision, Resolution, Spline};

fn main() -> splimes::Result<()> {
	// The GPU is process-wide and configured once, before anything starts it. The smaller
	// preset bounds the device memory a call holds (two 1 Mi-point buffer sets) and keeps
	// at most 128 MiB of idle buffers between calls.
	splimes::configure_gpu(GpuConfig::low_memory())?;
	// splimes never starts the GPU on its own: Backend::Auto ignores it until it has
	// started, here or by an explicit Backend::Gpu call.
	let started = Instant::now();
	let info = match splimes::prewarm_gpu() {
		Ok(info) => info,
		Err(e) => {
			println!("{e}");
			return Ok(());
		}
	};
	println!("GPU: {} ({}, {}), f64 {}; started in {:.0?}", info.name, info.api, info.device_type, if info.supports_f64 { "supported" } else { "unsupported" }, started.elapsed());

	// Ten days of irregular readings, resampled every second: 864,001 grid points.
	let (timestamps, values) = readings();
	let (start, end) = (timestamps[0], timestamps[0] + TimeDelta::days(10));
	let cubic = |backend, precision| Interpolator::new(Spline::Cubic, Resolution::Seconds).backend(backend).gpu_precision(precision);

	let reference = timed("Parallel (CPU)", || cubic(Backend::Parallel, Precision::F64).run_f64(&timestamps, &values, start, end))?;
	let range = values.iter().copied().fold(f64::NEG_INFINITY, f64::max) - values.iter().copied().fold(f64::INFINITY, f64::min);
	// f64 needs an adapter with f64 shaders; every adapter runs f32.
	let precisions: &[Precision] = if info.supports_f64 { &[Precision::F64, Precision::F32] } else { &[Precision::F32] };
	for &precision in precisions {
		let gpu = timed(&format!("Gpu, {precision}"), || cubic(Backend::Gpu, precision).run_f64(&timestamps, &values, start, end))?;
		let worst = gpu.values().iter().zip(reference.values()).map(|(g, c)| (g - c).abs()).fold(0.0, f64::max);
		// In f32, points the single-precision kernel can't compute reliably are recomputed in
		// f64 on the CPU, and counted: never returned wrong, never silently.
		println!("    ran in {}, {} point(s) recomputed in f64; worst difference from the CPU {:.1e} of the value range", gpu.precision(), gpu.points_recomputed_in_f64(), worst / range);
	}

	// Backend::Auto picks per call by grid size. Its GPU thresholds are "never" by default
	// (see BENCHMARKS.md); calibrate() measures this machine, or set them yourself.
	splimes::set_auto_thresholds(AutoThresholds::new(65_536, 500_000));
	let precision = precisions[0];
	for points in [10_000, 864_001] {
		let auto = cubic(Backend::Auto, precision).run_f64(&timestamps, &values, start, start + TimeDelta::seconds(points - 1))?;
		// What actually ran, and why: a GPU failure under Auto reruns on the CPU and says so.
		match auto.gpu_fallback() {
			Some(e) => println!("Auto, {points} points: GPU failed ({e}), reran on {}", auto.backend()),
			None => println!("Auto, {points} points: ran on {} in {}", auto.backend(), auto.precision()),
		}
	}

	if let Some(pool) = splimes::gpu_pool_stats() {
		println!("buffer pool: {} idle set(s), {} KiB; {} created, {} reused, {} evicted", pool.idle_sets, pool.idle_bytes / 1024, pool.created, pool.reused, pool.evicted);
	}
	Ok(())
}

/// Readings every 2–7 s for a little over ten days: a slow cycle of about six days plus a fast
/// wobble.
fn readings() -> (Vec<DateTime<Utc>>, Vec<f64>) {
	let first = Utc.with_ymd_and_hms(2026, 10, 1, 0, 0, 0).single().expect("valid date");
	let (mut timestamps, mut values) = (Vec::new(), Vec::new());
	let (mut t, mut i) = (first, 0_u32);
	while t <= first + TimeDelta::days(10) + TimeDelta::minutes(1) {
		let x = f64::from(i);
		timestamps.push(t);
		values.push(20.0 + 5.0 * (x / 18_000.0).sin() + 0.5 * (x / 7.0).cos());
		t += TimeDelta::milliseconds(2_000 + i64::from((i * 7_919) % 5_000));
		i += 1;
	}
	(timestamps, values)
}

fn timed(label: &str, run: impl FnOnce() -> splimes::Result<Interpolation<f64>>) -> splimes::Result<Interpolation<f64>> {
	let started = Instant::now();
	let result = run()?;
	println!("{label}: {} points in {:.1?}", result.len(), started.elapsed());
	Ok(result)
}

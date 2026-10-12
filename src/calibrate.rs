//! Measuring this machine's backend crossovers.

use std::time::{Duration, Instant};

use chrono::{DateTime, TimeDelta, Utc};

use crate::{AutoThresholds, Backend, Interpolator, Precision, Resolution, Result, Spline, set_auto_thresholds};

/// What [`calibrate`] measured, and the thresholds it chose.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct Calibration {
	/// The thresholds now in force.
	pub thresholds: AutoThresholds,
	/// One row per grid size, smallest first.
	pub samples: Vec<CalibrationSample>,
}

/// Best-of-three wall time for one grid size on each backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct CalibrationSample {
	/// Grid points.
	pub points: usize,
	/// [`Backend::Cpu`].
	pub cpu: Duration,
	/// [`Backend::Parallel`].
	pub parallel: Duration,
	/// [`Backend::Gpu`] in [`Precision::F64`], or `None` if the GPU can't run it.
	pub gpu_f64: Option<Duration>,
	/// [`Backend::Gpu`] in [`Precision::F32`], or `None` if there is no usable GPU.
	pub gpu_f32: Option<Duration>,
}

/// Input points in the calibration series.
const KNOTS: i32 = 4096;

/// Grid sizes measured: 1 Ki to 16 Mi points, by factors of 4.
const SIZES: [usize; 8] = [1 << 10, 1 << 12, 1 << 14, 1 << 16, 1 << 18, 1 << 20, 1 << 22, 1 << 24];

/// Times every backend on this machine and sets the [`AutoThresholds`] to the measured
/// crossovers.
///
/// Runs a cubic interpolation of 4,096 irregular `f64` points onto grids of 1 Ki to
/// 16 Mi points, best of three per backend and size, so expect it to take several
/// seconds; run it once at startup or in a setup step, not per request. The GPU is
/// measured in both precisions it supports, and each gets its own threshold. A backend
/// "wins" from the smallest size at which it is faster at that size and every larger one,
/// so a noisy single measurement can't flip the choice back and forth. If the GPU never
/// wins in a precision, or can't run it, `Auto` stays off the GPU for that precision.
///
/// It starts the GPU if it isn't running, which also lets `Auto` use it from then on;
/// call [`configure_gpu`](crate::configure_gpu) first if you want a non-default
/// configuration.
///
/// # Errors
///
/// Only an unexpected interpolation failure on a CPU backend; GPU failures just leave
/// the GPU out.
pub fn calibrate() -> Result<Calibration> {
	let gpu = crate::gpu::prewarm_gpu().is_ok();
	let (gpu_f64, gpu_f32) = (gpu && crate::gpu::ready(Precision::F64), gpu && crate::gpu::ready(Precision::F32));
	let origin = DateTime::<Utc>::UNIX_EPOCH;
	let values: Vec<f64> = (0..KNOTS).map(|i| (f64::from(i) * 0.01).sin() * 100.0).collect();

	let mut samples = Vec::with_capacity(SIZES.len());
	for points in SIZES {
		// A millisecond grid of `points` points, with the knots spread across it at an
		// irregular spacing (a deterministic jitter) so the whole grid interpolates.
		let span_ms = i64::try_from(points).unwrap_or(i64::MAX) - 1;
		let spacing = span_ms as f64 / f64::from(KNOTS);
		#[allow(clippy::cast_possible_truncation)]
		let timestamps: Vec<DateTime<Utc>> = (0..KNOTS).map(|i| origin + TimeDelta::microseconds((1000.0 * spacing * (f64::from(i) + 0.4 * f64::from((i * 7919) % 100) / 100.0)) as i64)).collect();
		let end = origin + TimeDelta::milliseconds(span_ms);
		let time = |backend, precision| -> Result<Duration> {
			let interpolator = Interpolator::new(Spline::Cubic, Resolution::Milliseconds).backend(backend).gpu_precision(precision);
			let mut best = Duration::MAX;
			for _ in 0..3 {
				let started = Instant::now();
				interpolator.run_f64(&timestamps, &values, origin, end)?;
				best = best.min(started.elapsed());
			}
			Ok(best)
		};
		samples.push(CalibrationSample { points, cpu: time(Backend::Cpu, Precision::F64)?, parallel: time(Backend::Parallel, Precision::F64)?, gpu_f64: if gpu_f64 { time(Backend::Gpu, Precision::F64).ok() } else { None }, gpu_f32: if gpu_f32 { time(Backend::Gpu, Precision::F32).ok() } else { None } });
	}

	let thresholds = thresholds(&samples);
	set_auto_thresholds(thresholds);
	Ok(Calibration { thresholds, samples })
}

/// The thresholds `samples` (smallest grid first) call for. A backend wins from the
/// smallest size at which it beats the alternatives there and at every larger size: rayon
/// against one thread, the GPU in each precision against the faster CPU backend. A
/// backend that never wins, or wasn't measured, gets `usize::MAX` (never).
fn thresholds(samples: &[CalibrationSample]) -> AutoThresholds {
	let first_win = |wins: &dyn Fn(&CalibrationSample) -> bool| (0..samples.len()).find(|&i| samples[i..].iter().all(wins)).map_or(usize::MAX, |i| samples[i].points);
	let parallel_min_points = first_win(&|s| s.parallel < s.cpu);
	let gpu_min_points = first_win(&|s| s.gpu_f64.is_some_and(|g| g < s.parallel.min(s.cpu)));
	let gpu_f32_min_points = first_win(&|s| s.gpu_f32.is_some_and(|g| g < s.parallel.min(s.cpu)));
	AutoThresholds::new(parallel_min_points, gpu_min_points).with_gpu_f32(gpu_f32_min_points)
}

#[cfg(test)]
mod tests {
	use super::*;

	/// A sample at `points` from times in microseconds; a GPU time of 0 means not measured.
	fn sample(points: usize, cpu: u64, parallel: u64, gpu_f64: u64, gpu_f32: u64) -> CalibrationSample {
		let gpu = |us| (us > 0).then(|| Duration::from_micros(us));
		CalibrationSample { points, cpu: Duration::from_micros(cpu), parallel: Duration::from_micros(parallel), gpu_f64: gpu(gpu_f64), gpu_f32: gpu(gpu_f32) }
	}

	#[test]
	fn each_backend_wins_from_its_crossover() {
		let samples = [sample(1 << 10, 10, 50, 900, 800), sample(1 << 12, 40, 60, 900, 800), sample(1 << 14, 160, 90, 900, 80), sample(1 << 16, 640, 200, 150, 70)];
		assert_eq!(thresholds(&samples), AutoThresholds::new(1 << 14, 1 << 16).with_gpu_f32(1 << 14));
	}

	#[test]
	fn a_win_must_hold_at_every_larger_size() {
		// rayon wins at 1 Ki by noise, loses at 4 Ki, and wins for good from 16 Ki.
		let samples = [sample(1 << 10, 50, 40, 0, 0), sample(1 << 12, 40, 60, 0, 0), sample(1 << 14, 160, 90, 0, 0), sample(1 << 16, 640, 200, 0, 0)];
		assert_eq!(thresholds(&samples).parallel_min_points, 1 << 14);
		// Losing at the largest size means never, whatever happened below it.
		let samples = [sample(1 << 10, 50, 40, 0, 0), sample(1 << 12, 160, 90, 0, 0), sample(1 << 14, 100, 100, 0, 0)];
		assert_eq!(thresholds(&samples).parallel_min_points, usize::MAX, "a tie is not a win");
	}

	#[test]
	fn the_gpu_must_beat_the_faster_cpu_backend() {
		// The GPU beats one thread everywhere but never beats rayon.
		let samples = [sample(1 << 10, 100, 50, 60, 60), sample(1 << 12, 400, 100, 120, 110)];
		let t = thresholds(&samples);
		assert_eq!((t.gpu_min_points, t.gpu_f32_min_points), (usize::MAX, usize::MAX));
		// And where one thread is the faster CPU backend, beating it is enough.
		let samples = [sample(1 << 10, 10, 50, 20, 20), sample(1 << 12, 40, 60, 30, 30)];
		let t = thresholds(&samples);
		assert_eq!((t.parallel_min_points, t.gpu_min_points, t.gpu_f32_min_points), (usize::MAX, 1 << 12, 1 << 12));
	}

	#[test]
	fn an_unmeasured_gpu_precision_is_never_used() {
		// f64 not measured at the largest size (no f64 kernel, or the run failed): never.
		let mut samples = [sample(1 << 10, 100, 100, 1, 1), sample(1 << 12, 400, 200, 1, 1)];
		samples[1].gpu_f64 = None;
		let t = thresholds(&samples);
		assert_eq!((t.gpu_min_points, t.gpu_f32_min_points), (usize::MAX, 1 << 10));
		assert_eq!(thresholds(&[]), AutoThresholds::new(usize::MAX, usize::MAX));
	}
}

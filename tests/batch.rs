//! `run_batch` and `run_f64_batch`: every job's result is exactly what `run` / `run_f64`
//! return for it alone — bit for bit, with the same report — and every refusal is the
//! one `run` gives, on every backend and GPU precision.
//!
//! Its own binary, with a small GPU chunk, so the batched GPU path packs the series into
//! several submissions and sends the series larger than a chunk down the single-call path.

mod common;

use bigdecimal::BigDecimal;
use chrono::{DateTime, TimeDelta, Utc};
use common::{Rng, epoch};
use splimes::{AutoThresholds, Backend, GpuConfig, Interpolation, Interpolator, Point, Precision, Resolution, Spline};

/// Grid points per GPU dispatch in this binary.
const CHUNK: u32 = 2048;

fn configure() {
	static ONCE: std::sync::Once = std::sync::Once::new();
	ONCE.call_once(|| {
		let _ = splimes::configure_gpu(GpuConfig::default().with_chunk_points(CHUNK));
	});
}

struct Job {
	timestamps: Vec<DateTime<Utc>>,
	values: Vec<f64>,
	start: DateTime<Utc>,
	end: DateTime<Utc>,
}

/// Mostly small irregular series, some longer than a chunk, plus one of each refusal.
fn jobs(seed: u64, count: usize) -> Vec<Job> {
	let mut rng = Rng::new(seed);
	let mut jobs: Vec<Job> = (0..count)
		.map(|_| {
			let knots = 1 + rng.below(200) as usize;
			let mut t = epoch() + TimeDelta::milliseconds(rng.below(1000) as i64);
			let mut timestamps = Vec::with_capacity(knots);
			let mut values = Vec::with_capacity(knots);
			for _ in 0..knots {
				timestamps.push(t);
				values.push((rng.unit() - 0.5) * 1e3);
				t += TimeDelta::milliseconds(1 + rng.below(40) as i64);
			}
			// Unsorted input, with the odd duplicate instant.
			if knots > 3 {
				timestamps.swap(0, knots - 1);
				values.swap(0, knots - 1);
				timestamps[1] = timestamps[2];
			}
			let span = t - timestamps.iter().min().copied().expect("non-empty");
			// Usually short grids; one in eight longer than a chunk.
			let points = if rng.below(8) == 0 { i64::from(CHUNK) + rng.below(3000) as i64 } else { 1 + rng.below(600) as i64 };
			let start = epoch() - TimeDelta::milliseconds(rng.below(50) as i64) + TimeDelta::milliseconds(rng.below(span.num_milliseconds().max(1) as u64) as i64 / 4);
			Job { timestamps, values, start, end: start + TimeDelta::milliseconds(points - 1) }
		})
		.collect();
	let ok = |j: &Job| (j.timestamps.clone(), j.values.clone());
	let (ts, vs) = ok(&jobs[0]);
	// No points.
	jobs.insert(3, Job { timestamps: Vec::new(), values: Vec::new(), start: epoch(), end: epoch() });
	// A non-finite value.
	let mut nan = vs.clone();
	nan[0] = f64::NAN;
	jobs.insert(7, Job { timestamps: ts.clone(), values: nan, start: epoch(), end: epoch() + TimeDelta::seconds(1) });
	// start after end.
	jobs.insert(11, Job { timestamps: ts.clone(), values: vs.clone(), start: epoch() + TimeDelta::seconds(1), end: epoch() });
	// More points than `max_points` allows.
	jobs.insert(13, Job { timestamps: ts.clone(), values: vs, start: epoch(), end: epoch() + TimeDelta::milliseconds(9_999) });
	// Extrapolation past f64::MAX.
	jobs.insert(17, Job { timestamps: vec![epoch(), epoch() + TimeDelta::milliseconds(1)], values: vec![0.0, f64::MAX], start: epoch(), end: epoch() + TimeDelta::milliseconds(10) });
	// One sample: every method steps down to a constant.
	jobs.insert(19, Job { timestamps: vec![epoch()], values: vec![42.0], start: epoch() - TimeDelta::milliseconds(5), end: epoch() + TimeDelta::milliseconds(5) });
	jobs
}

/// Equal, and every value bit-identical (`==` would equate 0.0 and -0.0).
fn same_f64(batch: &splimes::Result<Interpolation<f64>>, alone: &splimes::Result<Interpolation<f64>>, what: &str) {
	assert_eq!(batch, alone, "{what}");
	if let (Ok(b), Ok(a)) = (batch, alone) {
		assert!(b.values().iter().zip(a.values()).all(|(x, y)| x.to_bits() == y.to_bits()), "{what}: values differ in their bits");
	}
}

fn check_f64(interpolator: Interpolator, jobs: &[Job], what: &str) {
	let tuples: Vec<(&[DateTime<Utc>], &[f64], DateTime<Utc>, DateTime<Utc>)> = jobs.iter().map(|j| (j.timestamps.as_slice(), j.values.as_slice(), j.start, j.end)).collect();
	let batch = interpolator.run_f64_batch(&tuples);
	assert_eq!(batch.len(), jobs.len());
	for (i, (b, j)) in batch.iter().zip(jobs).enumerate() {
		let alone = interpolator.run_f64(&j.timestamps, &j.values, j.start, j.end);
		same_f64(b, &alone, &format!("{what}: job {i}"));
	}
	assert!(batch.iter().any(Result::is_ok) && batch.iter().any(Result::is_err), "{what}: the jobs cover both outcomes");
}

/// Every backend that can run here, as `(name, interpolator)`.
fn interpolators(spline: Spline) -> Vec<(String, Interpolator)> {
	let base = Interpolator::new(spline, Resolution::Milliseconds).max_points(9_000);
	let mut all = vec![("cpu".to_owned(), base.backend(Backend::Cpu)), ("parallel".to_owned(), base.backend(Backend::Parallel)), ("auto".to_owned(), base)];
	if let Some(info) = common::gpu_or_skip("batch") {
		if info.supports_f64 {
			all.push(("gpu-f64".to_owned(), base.backend(Backend::Gpu)));
		}
		all.push(("gpu-f32".to_owned(), base.backend(Backend::Gpu).gpu_precision(Precision::F32)));
	}
	all
}

/// One test, so the process-wide `Auto` thresholds it changes affect nothing else.
#[test]
fn a_batch_is_each_job_run_alone() {
	configure();
	let jobs = jobs(0xBA7C4, 150);
	for spline in [Spline::Linear, Spline::Cubic, Spline::Polynomial(5, Some(0.5))] {
		for (name, interpolator) in interpolators(spline) {
			check_f64(interpolator, &jobs, &format!("{spline} {name}"));
		}
		// Refusing to step down: the short series are refused, per job.
		check_f64(Interpolator::new(spline, Resolution::Milliseconds).backend(Backend::Parallel).exact(true), &jobs, &format!("{spline} exact"));
	}

	// `Auto` with the GPU from 500 points: some jobs go to the GPU, the rest stay on the
	// CPU, each by the choice `run` makes for it alone.
	if let Some(info) = common::gpu_or_skip("batch auto") {
		let before = splimes::auto_thresholds();
		splimes::set_auto_thresholds(AutoThresholds::new(200, 500));
		let precision = if info.supports_f64 { Precision::F64 } else { Precision::F32 };
		let interpolator = Interpolator::new(Spline::Cubic, Resolution::Milliseconds).gpu_precision(precision).max_points(9_000);
		let tuples: Vec<_> = jobs.iter().map(|j| (j.timestamps.as_slice(), j.values.as_slice(), j.start, j.end)).collect();
		let batch = interpolator.run_f64_batch(&tuples);
		let backends: Vec<Backend> = batch.iter().filter_map(|r| r.as_ref().ok().map(Interpolation::backend)).collect();
		check_f64(interpolator, &jobs, "auto with the GPU");
		splimes::set_auto_thresholds(before);
		assert!(backends.contains(&Backend::Gpu) && backends.contains(&Backend::Cpu), "{backends:?}");
	}

	// Mismatched columns are refused for that job only.
	let j = &jobs[0];
	let batch = Interpolator::new(Spline::Linear, Resolution::Milliseconds).run_f64_batch(&[(&j.timestamps, &j.values[1..], j.start, j.end), (&j.timestamps, &j.values, j.start, j.end)]);
	assert!(matches!(batch[0], Err(splimes::Error::LengthMismatch { .. })), "{:?}", batch[0]);
	assert!(batch[1].is_ok());
	// And an empty batch is an empty answer.
	assert!(Interpolator::new(Spline::Linear, Resolution::Milliseconds).backend(Backend::Gpu).run_f64_batch(&[]).is_empty());
}

/// Many tiny series: more than one submission's worth of bindings is split, not refused.
#[test]
fn thousands_of_tiny_series() {
	configure();
	let mut rng = Rng::new(7);
	let jobs: Vec<Job> = (0..5000)
		.map(|_| {
			let knots = 2 + rng.below(6) as usize;
			let timestamps: Vec<_> = (0..knots).map(|k| epoch() + TimeDelta::milliseconds(k as i64 * 3)).collect();
			let values = (0..knots).map(|_| rng.unit()).collect();
			Job { timestamps, values, start: epoch(), end: epoch() + TimeDelta::milliseconds(rng.below(20) as i64) }
		})
		.collect();
	for (name, interpolator) in interpolators(Spline::Cubic) {
		let tuples: Vec<_> = jobs.iter().map(|j| (j.timestamps.as_slice(), j.values.as_slice(), j.start, j.end)).collect();
		let batch = interpolator.run_f64_batch(&tuples);
		for (i, (b, j)) in batch.iter().zip(&jobs).enumerate() {
			same_f64(b, &interpolator.run_f64(&j.timestamps, &j.values, j.start, j.end), &format!("{name}: job {i}"));
		}
	}
}

/// `run_batch` returns, per job, exactly what `run` returns: the same decimals, digit for
/// digit, including the caller's own values for raw points.
#[test]
fn a_bigdecimal_batch_is_each_job_run_alone() {
	configure();
	let jobs = jobs(0xDEC, 40);
	let points: Vec<Vec<Point>> = jobs.iter().map(|j| j.timestamps.iter().zip(&j.values).map(|(&t, &v)| Point::new(t, if v.is_finite() { format!("{v:.6}").parse::<BigDecimal>().expect("decimal") } else { BigDecimal::from(0) })).collect()).collect();
	let tuples: Vec<(&[Point], DateTime<Utc>, DateTime<Utc>)> = points.iter().zip(&jobs).map(|(p, j)| (p.as_slice(), j.start, j.end)).collect();
	for (name, interpolator) in interpolators(Spline::Quadratic) {
		let batch = interpolator.run_batch(&tuples);
		for (i, (b, &(p, start, end))) in batch.iter().zip(&tuples).enumerate() {
			let alone = interpolator.run(p, start, end);
			assert_eq!(b, &alone, "{name}: job {i}");
			if let (Ok(b), Ok(a)) = (b, &alone) {
				// `==` on BigDecimal compares numerically; the representation must match too.
				assert!(b.values().iter().zip(a.values()).all(|(x, y)| x.as_bigint_and_exponent() == y.as_bigint_and_exponent()), "{name}: job {i}: representations differ");
			}
		}
	}
}

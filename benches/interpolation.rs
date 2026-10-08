//! Criterion benchmarks behind BENCHMARKS.md.
//!
//! ```text
//! cargo bench --bench interpolation                 # everything
//! cargo bench --bench interpolation -- backends     # one group
//! ```
//!
//! Every group interpolates a deterministic irregular series (a sine with jittered knot
//! spacing) onto a millisecond grid spanning it, so the whole grid is interpolation.

use std::hint::black_box;

use bigdecimal::BigDecimal;
use chrono::{DateTime, TimeDelta, Utc};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use splimes::{Backend, Interpolator, Point, Precision, Resolution, Spline};

struct Series {
	timestamps: Vec<DateTime<Utc>>,
	values: Vec<f64>,
	start: DateTime<Utc>,
	end: DateTime<Utc>,
}

/// `knots` irregular samples spread over a grid of `points` milliseconds.
fn series(knots: i64, points: i64) -> Series {
	let start = DateTime::<Utc>::UNIX_EPOCH;
	let spacing_us = 1000.0 * points as f64 / knots as f64;
	let timestamps = (0..knots).map(|i| start + TimeDelta::microseconds((spacing_us * (i as f64 + 0.4 * ((i * 7919) % 100) as f64 / 100.0)) as i64)).collect();
	let values = (0..knots).map(|i| (i as f64 * 0.01).sin() * 100.0).collect();
	Series { timestamps, values, start, end: start + TimeDelta::milliseconds(points - 1) }
}

fn gpu_precisions() -> Vec<Precision> {
	match splimes::prewarm_gpu() {
		Ok(info) if info.supports_f64 => vec![Precision::F64, Precision::F32],
		Ok(_) => vec![Precision::F32],
		Err(e) => {
			eprintln!("GPU benchmarks skipped: {e}");
			Vec::new()
		}
	}
}

/// Every backend, by grid size, f64 in and out.
fn backends(c: &mut Criterion) {
	let gpu = gpu_precisions();
	let mut group = c.benchmark_group("backends");
	group.sample_size(10);
	for points in [1_i64 << 12, 1 << 16, 1 << 20, 1 << 24] {
		let s = series(4096, points);
		group.throughput(Throughput::Elements(points as u64));
		let mut configs = vec![("cpu", Backend::Cpu, Precision::F64), ("parallel", Backend::Parallel, Precision::F64)];
		configs.extend(gpu.iter().map(|&p| (if p == Precision::F64 { "gpu-f64" } else { "gpu-f32" }, Backend::Gpu, p)));
		for (name, backend, precision) in configs {
			let interpolator = Interpolator::new(Spline::Cubic, Resolution::Milliseconds).backend(backend).gpu_precision(precision);
			group.bench_with_input(BenchmarkId::new(name, points), &s, |b, s| b.iter(|| black_box(interpolator.run_f64(&s.timestamps, &s.values, s.start, s.end).expect("runs"))));
		}
	}
	group.finish();
}

/// Each method on rayon's pool, 1 Mi points.
fn methods(c: &mut Criterion) {
	let mut group = c.benchmark_group("methods");
	group.sample_size(10);
	let s = series(4096, 1 << 20);
	group.throughput(Throughput::Elements(1 << 20));
	for spline in [Spline::Linear, Spline::Quadratic, Spline::Cubic, Spline::Polynomial(5, None), Spline::Polynomial(8, None)] {
		let interpolator = Interpolator::new(spline, Resolution::Milliseconds).backend(Backend::Parallel);
		group.bench_function(spline.to_string(), |b| b.iter(|| black_box(interpolator.run_f64(&s.timestamps, &s.values, s.start, s.end).expect("runs"))));
	}
	group.finish();
}

/// The cost of `BigDecimal` at the edges: the same interpolation through `run` and
/// `run_f64`, on rayon's pool.
fn bigdecimal(c: &mut Criterion) {
	let mut group = c.benchmark_group("bigdecimal");
	group.sample_size(10);
	for points in [1_i64 << 16, 1 << 20] {
		let s = series(4096, points);
		let decimal: Vec<Point> = s.timestamps.iter().zip(&s.values).map(|(&t, v)| Point::new(t, format!("{v:e}").parse::<BigDecimal>().expect("finite"))).collect();
		let interpolator = Interpolator::new(Spline::Cubic, Resolution::Milliseconds).backend(Backend::Parallel);
		group.throughput(Throughput::Elements(points as u64));
		group.bench_with_input(BenchmarkId::new("f64", points), &s, |b, s| b.iter(|| black_box(interpolator.run_f64(&s.timestamps, &s.values, s.start, s.end).expect("runs"))));
		group.bench_with_input(BenchmarkId::new("BigDecimal", points), &decimal, |b, d| b.iter(|| black_box(interpolator.run(d, s.start, s.end).expect("runs"))));
	}
	group.finish();
}

/// Input size at a fixed grid: the knot search is logarithmic, preparing the input linear.
fn inputs(c: &mut Criterion) {
	let mut group = c.benchmark_group("inputs");
	group.sample_size(10);
	for knots in [16_i64, 4096, 1 << 20] {
		let s = series(knots, 1 << 20);
		let interpolator = Interpolator::new(Spline::Cubic, Resolution::Milliseconds).backend(Backend::Parallel);
		group.bench_with_input(BenchmarkId::from_parameter(knots), &s, |b, s| b.iter(|| black_box(interpolator.run_f64(&s.timestamps, &s.values, s.start, s.end).expect("runs"))));
	}
	group.finish();
}

/// Preparing a million inputs (sorting, de-duplicating, converting) for a one-point grid,
/// so preparation is the whole cost: in order and shuffled, `f64` and `BigDecimal`.
fn prepare(c: &mut Criterion) {
	let mut group = c.benchmark_group("prepare");
	group.sample_size(10);
	let knots = 1_i64 << 20;
	let s = series(knots, 1 << 20);
	// An odd multiplier permutes indices modulo a power of two.
	let shuffle = |i: usize| (i * 7_919) % s.timestamps.len();
	let shuffled = Series { timestamps: (0..s.timestamps.len()).map(|i| s.timestamps[shuffle(i)]).collect(), values: (0..s.values.len()).map(|i| s.values[shuffle(i)]).collect(), start: s.start, end: s.end };
	let decimal: Vec<Point> = shuffled.timestamps.iter().zip(&shuffled.values).map(|(&t, v)| Point::new(t, format!("{v:e}").parse::<BigDecimal>().expect("finite"))).collect();
	let middle = s.start + (s.end - s.start) / 2;
	group.throughput(Throughput::Elements(knots as u64));
	for backend in [Backend::Cpu, Backend::Parallel] {
		let interpolator = Interpolator::new(Spline::Cubic, Resolution::Milliseconds).backend(backend);
		for (name, series) in [("f64-sorted", &s), ("f64-shuffled", &shuffled)] {
			group.bench_with_input(BenchmarkId::new(format!("{backend}/{name}"), knots), series, |b, s| b.iter(|| black_box(interpolator.run_f64(&s.timestamps, &s.values, middle, middle).expect("runs"))));
		}
		group.bench_with_input(BenchmarkId::new(format!("{backend}/BigDecimal-shuffled"), knots), &decimal, |b, d| b.iter(|| black_box(interpolator.run(d, middle, middle).expect("runs"))));
	}
	group.finish();
}

/// Small calls, where fixed overhead dominates: Auto's choice for a typical query.
fn small(c: &mut Criterion) {
	let mut group = c.benchmark_group("small");
	for points in [60_i64, 1440] {
		let s = series(32, points);
		let decimal: Vec<Point> = s.timestamps.iter().zip(&s.values).map(|(&t, v)| Point::new(t, format!("{v:e}").parse::<BigDecimal>().expect("finite"))).collect();
		let interpolator = Interpolator::new(Spline::Cubic, Resolution::Milliseconds);
		group.bench_with_input(BenchmarkId::new("auto-BigDecimal", points), &decimal, |b, d| b.iter(|| black_box(interpolator.run(d, s.start, s.end).expect("runs"))));
	}
	group.finish();
}

criterion_group!(benches, backends, methods, bigdecimal, inputs, prepare, small);
criterion_main!(benches);

//! One GPU against several: the same calls under each `GpuDevices` setting.
//!
//! The device set is process-wide and fixed once the GPU starts, so each setting is its
//! own run, chosen by `SPLIMES_BENCH_DEVICES`:
//!
//! ```text
//! SPLIMES_BENCH_DEVICES=one cargo bench --bench multi_gpu    # the default adapter alone
//! SPLIMES_BENCH_DEVICES=all cargo bench --bench multi_gpu    # GpuDevices::All
//! SPLIMES_BENCH_DEVICES=0,2 cargo bench --bench multi_gpu    # these gpu_adapters() indices
//! ```
//!
//! `SPLIMES_BENCH_MAX_POINTS` caps the grid sizes (for a quick smoke run). Two groups:
//! `split` is one large call, whose chunks are shared among the devices; `concurrent` is
//! eight 1 Mi-point calls at once from eight threads, each going whole to the least-loaded
//! device.

use std::hint::black_box;

use chrono::{DateTime, TimeDelta, Utc};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use splimes::{AdapterSelector, Backend, GpuConfig, GpuDevices, Interpolator, Precision, Resolution, Spline};

struct Series {
	timestamps: Vec<DateTime<Utc>>,
	values: Vec<f64>,
	start: DateTime<Utc>,
	end: DateTime<Utc>,
}

/// `knots` irregular samples spread over a grid of `points` milliseconds, as in the
/// `interpolation` bench.
fn series(knots: i64, points: i64) -> Series {
	let start = DateTime::<Utc>::UNIX_EPOCH;
	let spacing_us = 1000.0 * points as f64 / knots as f64;
	let timestamps = (0..knots).map(|i| start + TimeDelta::microseconds((spacing_us * (i as f64 + 0.4 * ((i * 7919) % 100) as f64 / 100.0)) as i64)).collect();
	let values = (0..knots).map(|i| (i as f64 * 0.01).sin() * 100.0).collect();
	Series { timestamps, values, start, end: start + TimeDelta::milliseconds(points - 1) }
}

/// Configures and starts the GPU for `SPLIMES_BENCH_DEVICES`; returns the run's label and
/// the precisions every device runs.
fn start() -> Option<(String, Vec<Precision>)> {
	let mode = std::env::var("SPLIMES_BENCH_DEVICES").unwrap_or_else(|_| "one".to_owned());
	let devices = match mode.as_str() {
		"one" => GpuDevices::One,
		"all" => GpuDevices::All,
		list => GpuDevices::List(list.split(',').map(|i| AdapterSelector::Index(i.trim().parse().expect("SPLIMES_BENCH_DEVICES: one, all or adapter indices"))).collect()),
	};
	for a in splimes::gpu_adapters() {
		eprintln!("adapter {}: {} ({}, {}, f64 {})", a.index, a.name, a.api, a.device_type, a.advertises_f64);
	}
	if let Err(e) = splimes::prewarm_gpu_with_config(GpuConfig::default().with_devices(devices)) {
		eprintln!("multi_gpu benchmarks skipped: {e}");
		return None;
	}
	let open = splimes::gpu_devices();
	for d in &open {
		eprintln!("device: {} ({}, {}), f64 {}", d.info.name, d.info.api, d.info.device_type, d.info.supports_f64);
	}
	let mut precisions = vec![Precision::F32];
	if open.iter().all(|d| d.info.supports_f64) {
		precisions.insert(0, Precision::F64);
	}
	Some((format!("{mode}x{}", open.len()), precisions))
}

fn max_points() -> i64 {
	std::env::var("SPLIMES_BENCH_MAX_POINTS").ok().and_then(|v| v.parse().ok()).unwrap_or(i64::MAX)
}

fn multi_gpu(c: &mut Criterion) {
	let Some((label, precisions)) = start() else { return };
	let cap = max_points();

	let mut group = c.benchmark_group("split");
	group.sample_size(10);
	for points in [1_i64 << 22, 1 << 24, 1 << 26].into_iter().filter(|&p| p <= cap) {
		let s = series(4096, points);
		group.throughput(Throughput::Elements(points as u64));
		for &precision in &precisions {
			let interpolator = Interpolator::new(Spline::Cubic, Resolution::Milliseconds).backend(Backend::Gpu).gpu_precision(precision);
			group.bench_with_input(BenchmarkId::new(format!("{label}/{precision}"), points), &s, |b, s| b.iter(|| black_box(interpolator.run_f64(&s.timestamps, &s.values, s.start, s.end).expect("runs"))));
		}
	}
	group.finish();

	let mut group = c.benchmark_group("concurrent");
	group.sample_size(10);
	let (calls, points) = (8_i64, (1_i64 << 20).min(cap));
	let s = series(4096, points);
	group.throughput(Throughput::Elements((calls * points) as u64));
	for &precision in &precisions {
		let interpolator = Interpolator::new(Spline::Cubic, Resolution::Milliseconds).backend(Backend::Gpu).gpu_precision(precision);
		group.bench_with_input(BenchmarkId::new(format!("{label}/{precision}"), format!("{calls}x{points}")), &s, |b, s| {
			b.iter(|| {
				std::thread::scope(|scope| {
					for _ in 0..calls {
						scope.spawn(|| black_box(interpolator.run_f64(&s.timestamps, &s.values, s.start, s.end).expect("runs")));
					}
				});
			});
		});
	}
	group.finish();
}

criterion_group!(benches, multi_gpu);
criterion_main!(benches);

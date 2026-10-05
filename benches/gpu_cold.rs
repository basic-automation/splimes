//! Benchmark GPU cold start (NO prewarm)
//!
//! This benchmark MUST be run as a separate process from other GPU benchmarks.
//! It measures the first `gpu_interpolate` call WITHOUT calling `prewarm_gpu()`,
//! capturing the full initialization overhead.
//!
//! Run with: `cargo bench --bench gpu_cold`
//!
//! Compare results with `cargo bench --bench gpu_prewarm` to see prewarm benefit.

use std::time::Instant;

use bigdecimal::BigDecimal;
use chrono::{Duration, Utc};
use criterion::{Criterion, criterion_group, criterion_main};
use splimes::{Point, Resolution, Spline, gpu_interpolate};
use tokio::runtime::Runtime;

fn create_test_data(count: usize) -> Vec<Point> {
	let start = Utc::now();
	(0..count).map(|i| Point { timestamp: start + Duration::seconds(i as i64), value: BigDecimal::from(i as i64) }).collect()
}

/// Benchmark WITHOUT prewarm - first call includes GPU initialization
///
/// Expected result: First call should be slow (~1.2s) because it includes
/// GPU device creation, shader compilation, etc.
fn bench_cold_start(c: &mut Criterion) {
	println!("\n=== NO prewarm_gpu() called - measuring cold start ===");

	let rt = Runtime::new().unwrap();
	const NUM_RUNS: usize = 10;
	const DATA_SIZE: usize = 10_000;

	let mut group = c.benchmark_group("GPU Cold Start");
	group.sample_size(10);

	// Measure first call WITHOUT prewarm - should be slow (includes init)
	group.bench_function("first_call_cold", |b| {
		b.iter_custom(|_iters| {
			let mut points = create_test_data(DATA_SIZE);
			let start_time = points[0].timestamp;
			let end_time = points[points.len() - 1].timestamp;

			let timer = Instant::now();
			rt.block_on(async { gpu_interpolate(&mut points, start_time, end_time, Resolution::Seconds, Spline::Linear).await.unwrap() });
			timer.elapsed()
		});
	});

	// Measure average of subsequent calls (GPU now initialized)
	group.bench_function(format!("subsequent_avg_{NUM_RUNS}_runs"), |b| {
		b.iter_custom(|_iters| {
			let timer = Instant::now();
			for _ in 0..NUM_RUNS {
				let mut points = create_test_data(DATA_SIZE);
				let start_time = points[0].timestamp;
				let end_time = points[points.len() - 1].timestamp;

				rt.block_on(async { gpu_interpolate(&mut points, start_time, end_time, Resolution::Seconds, Spline::Linear).await.unwrap() });
			}
			timer.elapsed() / NUM_RUNS as u32
		});
	});

	group.finish();

	println!("\n=== Results ===");
	println!("first_call_cold: Includes ~1.2s GPU initialization overhead");
	println!("subsequent_avg_{NUM_RUNS}_runs: Pure interpolation time");
	println!("Difference = init overhead that prewarm_gpu() eliminates from user-facing latency");
}

criterion_group!(benches, bench_cold_start);
criterion_main!(benches);

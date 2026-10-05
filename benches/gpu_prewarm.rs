//! Benchmark to verify `prewarm_gpu()` effectiveness
//!
//! This benchmark MUST be run as a separate process from other GPU benchmarks.
//! It calls `prewarm_gpu()` first, then measures if the first `gpu_interpolate`
//! call is as fast as subsequent calls (proving prewarm works).
//!
//! Run with: `cargo bench --bench gpu_prewarm`
//!
//! Compare results with `cargo bench --bench gpu_cold` to see the difference.

use std::time::Instant;

use bigdecimal::BigDecimal;
use chrono::{Duration, Utc};
use criterion::{Criterion, criterion_group, criterion_main};
use splimes::{Point, Resolution, Spline, gpu_interpolate, prewarm_gpu};
use tokio::runtime::Runtime;

fn create_test_data(count: usize) -> Vec<Point> {
	let start = Utc::now();
	(0..count).map(|i| Point { timestamp: start + Duration::seconds(i as i64), value: BigDecimal::from(i as i64) }).collect()
}

/// Benchmark with prewarm called BEFORE any GPU operations
///
/// Expected result: First call should be fast (same as subsequent calls)
/// because GPU init already happened in prewarm_gpu()
fn bench_with_prewarm(c: &mut Criterion) {
	// CRITICAL: Call prewarm BEFORE creating runtime or any GPU operations
	println!("\n=== Calling prewarm_gpu() BEFORE benchmark ===");
	let prewarm_start = Instant::now();
	match prewarm_gpu() {
		Ok(()) => println!("prewarm_gpu() completed in {:.3}s", prewarm_start.elapsed().as_secs_f64()),
		Err(e) => println!("prewarm_gpu() failed: {}", e),
	}

	let rt = Runtime::new().unwrap();
	const NUM_RUNS: usize = 10;
	const DATA_SIZE: usize = 10_000;

	let mut group = c.benchmark_group("GPU After Prewarm");
	group.sample_size(10);

	// Measure first call AFTER prewarm - should be fast
	group.bench_function("first_call_after_prewarm", |b| {
		b.iter_custom(|_iters| {
			let mut points = create_test_data(DATA_SIZE);
			let start_time = points[0].timestamp;
			let end_time = points[points.len() - 1].timestamp;

			let timer = Instant::now();
			rt.block_on(async { gpu_interpolate(&mut points, start_time, end_time, Resolution::Seconds, Spline::Linear).await.unwrap() });
			timer.elapsed()
		});
	});

	// Measure average of subsequent calls
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
	println!("If prewarm works: first_call_after_prewarm ≈ subsequent_avg_{NUM_RUNS}_runs");
	println!("Compare with `cargo bench --bench gpu_cold` to see cold start time");
}

criterion_group!(benches, bench_with_prewarm);
criterion_main!(benches);

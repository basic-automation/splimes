use bigdecimal::FromPrimitive;
use chrono::{DateTime, Duration, Utc};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use splimes::{Point, Resolution, Spline, auto_interpolate, cpu_interpolate, estimate_output_points, gpu_interpolate, parallel_interpolate};

// Helper function to generate test data
fn generate_test_data(input_size: usize, start: DateTime<Utc>, _resolution: Resolution) -> (Vec<Point>, DateTime<Utc>, DateTime<Utc>) {
	let mut rng = ChaCha8Rng::seed_from_u64(42);
	let mut points = Vec::with_capacity(input_size);
	let mut current_time = start;
	for _ in 0..input_size {
		let value = rng.gen_range(0.0..100.0);
		points.push(Point { timestamp: current_time, value: bigdecimal::BigDecimal::from_f64(value).unwrap() });
		current_time += Duration::seconds(rng.gen_range(1..60));
	}
	// Create a smaller time range to avoid massive output
	let end = current_time + Duration::minutes(180); // Much smaller range
	(points, start, end)
}

fn bench_interpolation(c: &mut Criterion) {
	let mut group = c.benchmark_group("Interpolation Strategies");
	group.warm_up_time(std::time::Duration::from_secs(1));
	group.measurement_time(std::time::Duration::from_secs(5));
	group.sample_size(10); // Fewer samples for faster benchmarking

	// Much smaller test sizes to avoid memory issues
	let sizes = vec![10, 50, 100, 500, 1_000, 10_000, 100_000, 1_000_000, 10_000_000];
	let resolution = Resolution::Minutes; // Use coarser resolution
	let spline = Spline::Cubic; // Start with linear for simpler testing

	for size in sizes {
		let start = Utc::now();
		let (_points, bench_start, bench_end) = generate_test_data(size, start, resolution);
		let estimated_output = estimate_output_points(bench_start, bench_end, resolution);

		println!("Testing size: {} input, {} estimated output", size, estimated_output);

		// Only benchmark CPU for smaller datasets (< 100K points)
		// CPU becomes impractically slow for larger datasets (397s vs 2.37s for parallel at 1M points)
		if size < 100_000 {
			// Benchmark plain CPU
			group.bench_with_input(BenchmarkId::new("CPU", format!("{size}_in_{estimated_output}_out")), &size, |b, &size| {
				b.to_async(criterion::async_executor::FuturesExecutor).iter(|| {
					let (mut points, bench_start, bench_end) = generate_test_data(size, start, resolution);
					async move {
						cpu_interpolate(&mut points, bench_start, bench_end, resolution, spline).await.unwrap();
						cpu_interpolate(&mut points, bench_start, bench_end, resolution, spline).await.unwrap();
					}
				});
			});
		}

		// Benchmark parallel
		group.bench_with_input(BenchmarkId::new("Parallel", format!("{size}_in_{estimated_output}_out")), &size, |b, &size| {
			b.to_async(criterion::async_executor::FuturesExecutor).iter(|| {
				let (mut points, bench_start, bench_end) = generate_test_data(size, start, resolution);
				async move {
					parallel_interpolate(&mut points, &bench_start, &bench_end, spline, resolution).await.unwrap();
					parallel_interpolate(&mut points, &bench_start, &bench_end, spline, resolution).await.unwrap();
				}
			});
		});

		// Benchmark GPU
		group.bench_with_input(BenchmarkId::new("GPU", format!("{size}_in_{estimated_output}_out")), &size, |b, &size| {
			b.to_async(criterion::async_executor::FuturesExecutor).iter(|| {
				let (mut points, bench_start, bench_end) = generate_test_data(size, start, resolution);
				async move {
					gpu_interpolate(&mut points, bench_start, bench_end, resolution, spline).await.unwrap();
					gpu_interpolate(&mut points, bench_start, bench_end, resolution, spline).await.unwrap();
				}
			});
		});

		// Benchmark Auto
		group.bench_with_input(BenchmarkId::new("Auto", format!("{size}_in_{estimated_output}_out")), &size, |b, &size| {
			b.to_async(criterion::async_executor::FuturesExecutor).iter(|| {
				let (mut points, bench_start, bench_end) = generate_test_data(size, start, resolution);
				async move {
					auto_interpolate(&mut points, bench_start, bench_end, resolution, spline).await.unwrap();
					auto_interpolate(&mut points, bench_start, bench_end, resolution, spline).await.unwrap();
				}
			});
		});
	}

	group.finish();
}

criterion_group!(benches, bench_interpolation);
criterion_main!(benches);

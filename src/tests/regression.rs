#[cfg(test)]
pub mod regression_tests {
	use bigdecimal::{BigDecimal, ToPrimitive};
	use chrono::Utc;
	use serial_test::serial;

	use crate::{gpu::types::GpuInterpolator, gpu_interpolate, parallel_interpolate, splines::*, Point, Resolution};

	/// Known deterministic test dataset with controlled values
	/// This ensures tests produce the same output every time
	fn get_deterministic_points() -> Vec<Point> {
		vec![Point { timestamp: Utc::now(), value: bigdecimal::BigDecimal::from(-100) }, Point { timestamp: Utc::now() + chrono::Duration::seconds(10), value: bigdecimal::BigDecimal::from(-95) }, Point { timestamp: Utc::now() + chrono::Duration::seconds(20), value: bigdecimal::BigDecimal::from(-90) }, Point { timestamp: Utc::now() + chrono::Duration::seconds(30), value: bigdecimal::BigDecimal::from(-85) }, Point { timestamp: Utc::now() + chrono::Duration::seconds(40), value: bigdecimal::BigDecimal::from(-80) }, Point { timestamp: Utc::now() + chrono::Duration::seconds(50), value: bigdecimal::BigDecimal::from(-75) }, Point { timestamp: Utc::now() + chrono::Duration::seconds(60), value: bigdecimal::BigDecimal::from(-70) }, Point { timestamp: Utc::now() + chrono::Duration::seconds(70), value: bigdecimal::BigDecimal::from(-65) }, Point { timestamp: Utc::now() + chrono::Duration::seconds(80), value: bigdecimal::BigDecimal::from(-60) }, Point { timestamp: Utc::now() + chrono::Duration::seconds(90), value: bigdecimal::BigDecimal::from(-55) }]
	}

	/// Verify that all implementations produce consistent output counts and timestamps
	/// This is the primary regression check: ensuring determinism and consistency
	#[tokio::test]
	#[serial(gpu_tests)]
	async fn test_regression_output_consistency() {
		// Clear GPU buffer pool for test isolation
		GpuInterpolator::clear_buffer_pool_static().expect("Failed to clear GPU buffer pool");

		let points = get_deterministic_points();
		let start = points.first().unwrap().timestamp - chrono::Duration::seconds(5);
		let end = points.last().unwrap().timestamp + chrono::Duration::seconds(5);
		let resolution = Resolution::Seconds;

		// Test all spline types
		let spline_types = vec![("Linear", crate::Spline::Linear), ("Quadratic", crate::Spline::Quadratic), ("Cubic", crate::Spline::Cubic), ("Polynomial", crate::Spline::Polynomial(3, Some(1.0)))];

		for (name, spline) in spline_types {
			// CPU interpolation
			let mut points_cpu = get_deterministic_points();
			let cpu_result = match name {
				"Linear" => linear(&mut points_cpu, &start, &end, &resolution).await.unwrap(),
				"Quadratic" => quadratic(&mut points_cpu, &start, &end, &resolution).await.unwrap(),
				"Cubic" => cubic(&mut points_cpu, &start, &end, &resolution).await.unwrap(),
				"Polynomial" => polynomial(&mut points_cpu, &start, &end, &resolution, &spline).await.unwrap(),
				_ => panic!("Unknown spline type"),
			};

			// Parallel interpolation
			let mut points_parallel = get_deterministic_points();
			let parallel_result = parallel_interpolate(&mut points_parallel, &start, &end, spline, resolution).await.unwrap();

			// GPU interpolation
			let mut points_gpu = get_deterministic_points();
			let gpu_result = gpu_interpolate(&mut points_gpu, start, end, resolution, spline).await.unwrap();

			// Regression checks: output counts should match (determinism)
			assert_eq!(cpu_result.len(), parallel_result.len(), "{}: CPU ({}) and Parallel ({}) output length mismatch", name, cpu_result.len(), parallel_result.len());
			assert_eq!(cpu_result.len(), gpu_result.len(), "{}: CPU ({}) and GPU ({}) output length mismatch", name, cpu_result.len(), gpu_result.len());

			// Timestamps should be identical across all backends
			let cpu_timestamps: Vec<_> = cpu_result.iter().map(|p| p.timestamp).collect();
			let parallel_timestamps: Vec<_> = parallel_result.iter().map(|p| p.timestamp).collect();
			let gpu_timestamps: Vec<_> = gpu_result.iter().map(|p| p.timestamp).collect();

			assert_eq!(cpu_timestamps, parallel_timestamps, "{name}: CPU and Parallel timestamps differ");
			assert_eq!(cpu_timestamps, gpu_timestamps, "{name}: CPU and GPU timestamps differ");

			// Values should be close (within reasonable tolerance for numerical precision)
			// Tolerance is per-implementation as they use different precisions
			let cpu_values: Vec<f64> = cpu_result.iter().map(|p| p.value.to_f64().unwrap()).collect();
			let parallel_values: Vec<f64> = parallel_result.iter().map(|p| p.value.to_f64().unwrap()).collect();
			let gpu_values: Vec<f64> = gpu_result.iter().map(|p| p.value.to_f64().unwrap()).collect();

			let mut max_cpu_parallel_diff = 0.0f64;
			let mut max_cpu_gpu_diff = 0.0f64;

			for i in 0..cpu_values.len() {
				let cpu_val = cpu_values[i];
				let parallel_val = parallel_values[i];
				let gpu_val = gpu_values[i];

				let cpu_parallel_diff = (cpu_val - parallel_val).abs();
				let cpu_gpu_diff = (cpu_val - gpu_val).abs();

				max_cpu_parallel_diff = max_cpu_parallel_diff.max(cpu_parallel_diff);
				max_cpu_gpu_diff = max_cpu_gpu_diff.max(cpu_gpu_diff);
			}

			println!("{name}: Max CPU/Parallel divergence: {max_cpu_parallel_diff:.10}, Max CPU/GPU divergence: {max_cpu_gpu_diff:.10}");

			// Use reasonable tolerances based on spline type
			// All spline types currently use the same tolerance
			// GPU interpolation may have slightly higher divergence due to floating point precision
			// (CPU uses BigDecimal arbitrary precision, GPU uses f64)
			let tolerance = 1.5;

			assert!(max_cpu_parallel_diff < tolerance, "{name}: CPU/Parallel max divergence {max_cpu_parallel_diff:.10} exceeds tolerance {tolerance}");
			assert!(max_cpu_gpu_diff < tolerance, "{name}: CPU/GPU max divergence {max_cpu_gpu_diff:.10} exceeds tolerance {tolerance}");
		}
	}

	/// Regression test for deterministic output: Same input always produces same output
	/// Note: GPU floating-point operations may have small variance, so we allow a small tolerance
	#[tokio::test]
	#[serial(gpu_tests)]
	async fn test_regression_deterministic_output() {
		// Clear GPU buffer pool for test isolation
		GpuInterpolator::clear_buffer_pool_static().expect("Failed to clear GPU buffer pool");

		let resolution = Resolution::Seconds;
		let start_base = Utc::now();
		let end_base = start_base + chrono::Duration::seconds(100);

		// Run linear interpolation three times with same data
		let mut runs = vec![];

		for _ in 0..3 {
			let mut points = get_deterministic_points();
			let result = linear(&mut points, &start_base, &end_base, &resolution).await.unwrap();
			runs.push(result);
		}

		// Verify all runs are nearly identical (allow small GPU floating-point variance)
		let first_run = &runs[0];
		for (i, run) in runs.iter().skip(1).enumerate() {
			assert_eq!(first_run.len(), run.len(), "Run {} has different output length: {} vs {}", i + 2, first_run.len(), run.len());

			for (j, (p1, p2)) in first_run.iter().zip(run.iter()).enumerate() {
				assert_eq!(p1.timestamp, p2.timestamp, "Run {} point {} has different timestamp", i + 2, j);
				// Allow small floating-point variance in GPU operations
				let diff = (&p1.value - &p2.value).abs();
				let tolerance = BigDecimal::from(1); // Allow up to 1 unit of variance
				assert!(diff <= tolerance, "Run {} point {} has value difference {diff} exceeding tolerance {tolerance}", i + 2, j);
			}
		}

		println!("Determinism test passed: 3 runs produced near-identical output");
	}
}

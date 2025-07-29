/// Determine if GPU should be used based on dataset characteristics
#[must_use]
pub const fn should_use_gpu(measurement_count: usize, estimated_output_points: usize) -> bool {
	// Updated thresholds based on the performance characteristics
	// GPU has high initialization overhead, so we need much larger datasets
	let min_measurements = 50_000;
	let min_output_points = 500_000;

	// GPU is beneficial for very dense output scenarios
	let density_factor = if measurement_count > 0 { estimated_output_points.div_euclid(measurement_count) } else { 0 };

	measurement_count >= min_measurements && estimated_output_points >= min_output_points && density_factor >= 5
}

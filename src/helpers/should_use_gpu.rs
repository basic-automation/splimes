/// Determine if GPU should be used based on dataset characteristics
#[must_use]
pub const fn should_use_gpu(measurement_count: usize, estimated_output_points: usize) -> bool {
	// Based on benchmark data, GPU has consistent ~800ms overhead
	// GPU only becomes beneficial for extremely large datasets where computation time exceeds this overhead
	// Even at 1000 input/525 output, parallel takes ~187ms vs GPU ~797ms
	// GPU would need datasets taking >1000ms on parallel to be worthwhile

	let min_measurements = 500_000; // Very high threshold based on 800ms overhead
	let min_output_points = 5_000_000; // Very high threshold

	// For GPU to be worthwhile, we need massive computational workloads
	let density_factor = if measurement_count > 0 { estimated_output_points.div_euclid(measurement_count) } else { 0 };

	// Extremely conservative - only use GPU for truly massive datasets
	measurement_count >= min_measurements && estimated_output_points >= min_output_points && density_factor >= 20
}

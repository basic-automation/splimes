/// Determine if GPU should be used based on dataset characteristics
///
/// Returns a strategy recommendation based on `LazyLock` GPU optimization benchmark results:
/// - GPU dominates at very large datasets (10M+ inputs: 14.8s vs 29.4s for parallel)
/// - GPU becomes competitive at 100K+ inputs due to reduced initialization overhead
/// - Parallel remains best for medium datasets (100-100K inputs)
/// - CPU is best for small datasets (< 100 inputs) to avoid parallel overhead
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationStrategy {
	/// Use GPU as primary with parallel fallback
	GpuPrimary,
	/// Try GPU first, quick fallback to parallel
	GpuThenParallel,
	/// Use parallel processing
	Parallel,
	/// Use single-threaded CPU
	Cpu,
}

#[must_use]
pub const fn should_use_gpu(measurement_count: usize, estimated_output_points: usize) -> InterpolationStrategy {
	// For very large datasets where GPU shows clear advantage
	if measurement_count >= 1_000_000 || estimated_output_points >= 500_000 {
		return InterpolationStrategy::GpuPrimary;
	}

	// For large datasets where GPU might be competitive but parallel is safer
	if measurement_count >= 100_000 || estimated_output_points >= 50_000 {
		return InterpolationStrategy::GpuThenParallel;
	}

	// For medium datasets (100-100K inputs), parallel is clearly optimal
	if measurement_count >= 100 || estimated_output_points >= 100 {
		return InterpolationStrategy::Parallel;
	}

	// For small datasets (< 100 inputs), use CPU to avoid parallel overhead
	InterpolationStrategy::Cpu
}

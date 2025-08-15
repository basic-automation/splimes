/// Determine if GPU should be used based on dataset characteristics
///
/// Returns a strategy recommendation based on latest benchmark results:
/// - GPU dominates at very large datasets (10M+ inputs: 14.8s vs 29.6s for parallel - 50% faster!)
/// - GPU becomes competitive at 1M+ inputs (2.25s vs 2.20s for parallel - nearly equal)
/// - GPU still has overhead at medium scales (1.02s vs 0.52s for parallel at 100K)
/// - Parallel remains best for medium datasets (100-1M inputs)
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
	// For very large datasets where GPU shows clear 50% performance advantage
	// 10M points: GPU 14.8s vs Parallel 29.6s
	if measurement_count >= 5_000_000 || estimated_output_points >= 2_500_000 {
		return InterpolationStrategy::GpuPrimary;
	}

	// For large datasets where GPU is competitive (within 10% of parallel)
	// 1M points: GPU 2.25s vs Parallel 2.20s - nearly equal performance
	if measurement_count >= 1_000_000 || estimated_output_points >= 500_000 {
		return InterpolationStrategy::GpuThenParallel;
	}

	// For medium datasets, parallel is clearly optimal
	// 100K points: Parallel 0.52s vs GPU 1.02s - parallel is 2x faster
	if measurement_count >= 100 || estimated_output_points >= 100 {
		return InterpolationStrategy::Parallel;
	}

	// For small datasets (< 100 inputs), use CPU to avoid parallel overhead
	// Small datasets: CPU ~300-400ms vs Parallel ~300-400ms (similar, but CPU avoids overhead)
	InterpolationStrategy::Cpu
}

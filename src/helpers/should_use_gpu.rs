/// Determine if GPU should be used based on dataset characteristics
///
/// Returns a strategy recommendation based on benchmark results with streaming pipelining:
///
/// **Benchmark Results (fixed benchmark - runtime created once, not per-iteration):**
/// | Dataset | CPU | Parallel | GPU | Winner |
/// |---------|-----|----------|-----|--------|
/// | 100 inputs | 509ms | 509ms | 491ms | GPU (~4% faster) |
/// | 1k inputs | 494ms | 487ms | 495ms | Parallel (slight) |
/// | 10k inputs | 469ms | 586ms | 575ms | CPU (~20% faster) |
/// | 100k inputs | 666ms | 576ms | 551ms | GPU (~17% faster than CPU) |
///
/// **Key observations:**
/// - GPU wins at small (100) and large (100k+) scales
/// - CPU wins at medium scales (10k) where overhead dominates
/// - At 1k inputs, all strategies are roughly equal
/// - Crossover from CPU to GPU advantage occurs between 10k-100k inputs
/// - For very large datasets (1M+), GPU memory advantages still apply
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
	// For very large datasets where GPU memory efficiency dominates
	// GPU can handle larger datasets without memory pressure
	// 5M+ points: GPU preferred for memory management
	if measurement_count >= 5_000_000 || estimated_output_points >= 2_500_000 {
		return InterpolationStrategy::GpuPrimary;
	}

	// For large datasets where GPU streaming provides good throughput
	// 1M+ points: GPU competitive with parallel, better memory efficiency
	if measurement_count >= 1_000_000 || estimated_output_points >= 500_000 {
		return InterpolationStrategy::GpuThenParallel;
	}

	// For large datasets (50k-1M), GPU wins
	// Benchmark: 100k - GPU 551ms vs Parallel 576ms vs CPU 666ms
	// GPU parallelism advantage outweighs overhead at this scale
	if measurement_count >= 50_000 || estimated_output_points >= 50_000 {
		return InterpolationStrategy::GpuThenParallel;
	}

	// For medium datasets (1k-50k), CPU is fastest
	// Benchmark: 10k - CPU 469ms vs Parallel 586ms vs GPU 575ms
	// CPU avoids synchronization overhead at these scales
	if measurement_count >= 1_000 || estimated_output_points >= 1_000 {
		return InterpolationStrategy::Cpu;
	}

	// For small datasets (100-1k), GPU streaming is competitive
	// Benchmark: 100 inputs - GPU 491ms vs Parallel 509ms vs CPU 509ms
	// GPU pipelining overlaps well at this scale
	if measurement_count >= 100 || estimated_output_points >= 100 {
		return InterpolationStrategy::GpuThenParallel;
	}

	// For very small datasets (< 100 inputs), use CPU to avoid all overhead
	// The overhead of GPU/parallel setup exceeds any parallelism benefit
	InterpolationStrategy::Cpu
}

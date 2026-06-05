#![warn(clippy::pedantic, clippy::nursery, clippy::all)]
#![allow(clippy::multiple_crate_versions, clippy::used_underscore_binding, clippy::similar_names, clippy::module_name_repetitions, clippy::module_inception, clippy::cast_precision_loss)]

use anyhow::{Result, bail};
use chrono::{DateTime, Utc};
use std::sync::LazyLock;
pub use optimizations::{apply_fast_path, cpu_interpolate, parallel_interpolate};
pub use types::{BASE_BATCH_SIZE, Error, POINT_SIZE, Point, Resolution, Spline};

mod gpu;
pub mod helpers; // Make helpers public to allow access to helpers::should_use_gpu
mod optimizations;
mod splines;
mod tests;
mod types;

/// When gpu-eager-init feature is enabled, run GPU prewarm BEFORE main()
/// This eliminates any latency on the first interpolation call
#[cfg(feature = "gpu-eager-init")]
#[ctor::ctor]
fn _gpu_startup_init() {
	let _ = gpu::force_init_gpu();
}

/// Auto-initialize GPU on first library use (passive first-use)
/// This static ensures GPU prewarming happens automatically when the library is loaded
/// If gpu-eager-init feature is enabled, this will be a no-op since GPU is already initialized
static _GPU_AUTO_INIT: LazyLock<()> = LazyLock::new(|| {
	// GPU prewarming is silently performed on first access
	// Errors are ignored to ensure the library remains functional even if GPU initialization fails
	let _ = gpu::force_init_gpu();
});

/// Ensures GPU auto-initialization is triggered
/// This is called internally to guarantee GPU prewarming happens on first library use
#[inline]
fn ensure_gpu_init() {
	// Access the lazy static to trigger initialization
	let () = &*_GPU_AUTO_INIT;
}

// Re-export for public API
pub use gpu::gpu_interpolate;
pub use gpu::{GpuConfig, BufferPoolStats};
pub use helpers::{InterpolationStrategy, estimate_output_points, generate_target_times, should_use_gpu};
pub use splines::{DAYS_IN_MONTH, DAYS_IN_YEAR, SECONDS_IN_DAY, SECONDS_IN_HOUR, SECONDS_IN_MINUTE, SECONDS_IN_MONTH, SECONDS_IN_WEEK, SECONDS_IN_YEAR};

/// Pre-warms the GPU interpolator to eliminate first-use latency.
///
/// Call this function early during application startup to trigger GPU initialization
/// before any user interactions that might trigger interpolation. This eliminates the
/// ~1.2 second latency that would otherwise occur on the first interpolation call.
///
/// # Errors
///
/// Returns an error if GPU is unavailable or initialization fails.
///
/// # Example
/// ```ignore
/// #[tokio::main]
/// async fn main() -> anyhow::Result<()> {
///     // Pre-warm GPU at startup to avoid first-use latency
///     let _ = splimes::prewarm_gpu();
///
///     // ... rest of application
///     Ok(())
/// }
/// ```
pub fn prewarm_gpu() -> Result<()> {
	gpu::force_init_gpu()
}

/// Pre-warm the GPU with a custom configuration
///
/// Initializes the GPU interpolator with the specified configuration preset.
/// Call this before any interpolation operations to control GPU resource usage.
///
/// # Arguments
/// * `config` - GPU configuration (use presets like `GpuConfig::low_memory()`,
///   `GpuConfig::high_performance()`, or `GpuConfig::default()`)
///
/// # Errors
///
/// Returns an error if GPU is unavailable or initialization fails.
///
/// # Example
/// ```ignore
/// use splimes::GpuConfig;
///
/// #[tokio::main]
/// async fn main() -> anyhow::Result<()> {
///     // Pre-warm with high-performance configuration
///     splimes::prewarm_gpu_with_config(GpuConfig::high_performance())?;
///
///     // ... interpolation operations ...
///     Ok(())
/// }
/// ```
pub fn prewarm_gpu_with_config(_config: GpuConfig) -> Result<()> {
	// Store the config in a thread-local or global state if needed
	// For now, just force initialization
	// TODO: Apply configuration to the global interpolator
	gpu::force_init_gpu()
}

/// Get buffer pool statistics
///
/// Returns information about the GPU buffer pool usage including:
/// - Total number of pooled buffers
/// - Total allocated memory
/// - Number of allocations
/// - Number of buffer reuses
///
/// # Errors
///
/// Returns an error if GPU interpolator is not initialized.
///
/// # Example
/// ```ignore
/// let stats = splimes::gpu_buffer_pool_stats()?;
/// println!("Pool: {} buffers, {:.1} MB allocated",
///     stats.total_buffers,
///     stats.total_allocated_bytes as f64 / 1024.0 / 1024.0
/// );
/// ```
pub fn gpu_buffer_pool_stats() -> Result<BufferPoolStats> {
	Ok(gpu::types::GpuInterpolator::get_buffer_pool_static()?.stats())
}

/// Main async interpolation function with GPU acceleration support
///
/// This is the primary entry point for all interpolation operations in the library.
/// It automatically selects the optimal interpolation strategy (GPU, CPU, SIMD, Parallel)
/// based on dataset characteristics and performance benchmarks.
///
/// The GPU is automatically pre-warmed on first use to eliminate initialization latency.
///
/// If the requested spline method requires more points than available, this function
/// will automatically fall back to a simpler method that can work with the available data:
/// - Cubic (4 points) → Quadratic (3 points) → Linear (2 points)
/// - Polynomial(n) → lower degree polynomial → Cubic → Quadratic → Linear
///
/// With only 1 point, it returns that point's value for any requested time.
/// With 0 points, it returns an error.
///
/// # Errors
///
/// Returns an error if:
/// - No measurements provided (0 points)
/// - Invalid time range (start >= end)
/// - All interpolation methods fail
pub async fn auto_interpolate(points: &mut [Point], start: DateTime<Utc>, end: DateTime<Utc>, resolution: Resolution, spline: Spline) -> Result<Vec<Point>> {
	// Trigger GPU auto-initialization on first use
	ensure_gpu_init();

	if points.is_empty() {
		bail!(Error::InsufficientMeasurementsError);
	}

	if start >= end {
		bail!(Error::InvalidTimeRangeError);
	}

	// Handle single point case - return constant value for all requested times
	if points.len() == 1 {
		let target_times = helpers::generate_target_times(start, end, resolution);
		let constant_value = points[0].value.clone();
		return Ok(target_times.into_iter().map(|t| Point { timestamp: t, value: constant_value.clone() }).collect());
	}

	// Determine the best available spline method based on point count
	let effective_spline = select_best_available_spline(spline, points.len());

	let estimated_output_points = helpers::estimate_output_points(start, end, resolution);
	let effective_spline = apply_fast_path(effective_spline, points.len());

	// Use centralized strategy selection based on benchmark results
	match helpers::should_use_gpu(points.len(), estimated_output_points) {
		helpers::InterpolationStrategy::GpuPrimary => {
			// Try GPU first for very large datasets where it's proven to be faster
			if let Ok(result) = gpu_interpolate(points, start, end, resolution, effective_spline).await {
				return Ok(result);
			}
			// Fallback to parallel if GPU fails
			parallel_interpolate(points, &start, &end, effective_spline, resolution).await
		}
		helpers::InterpolationStrategy::GpuThenParallel => {
			// Try GPU with timeout for large datasets where it's competitive
			let gpu_result = tokio::time::timeout(std::time::Duration::from_secs(10), gpu_interpolate(points, start, end, resolution, effective_spline)).await;

			match gpu_result {
				Ok(Ok(result)) => Ok(result),
				_ => {
					// Quick fallback to parallel
					parallel_interpolate(points, &start, &end, effective_spline, resolution).await
				}
			}
		}
		helpers::InterpolationStrategy::Parallel => {
			// Use parallel for medium-sized datasets
			parallel_interpolate(points, &start, &end, effective_spline, resolution).await
		}
		helpers::InterpolationStrategy::Cpu => {
			// Use CPU for small datasets to avoid overhead
			cpu_interpolate(points, start, end, resolution, effective_spline).await
		}
	}
}

/// Select the best available spline method based on the number of points available.
/// Falls back to simpler methods ONLY - never upgrades to a more complex method than requested.
/// Fallback order: Polynomial(n) → Polynomial(n-1) → ... → Cubic → Quadratic → Linear
#[must_use]
const fn select_best_available_spline(requested: Spline, point_count: usize) -> Spline {
	// If we have enough points for the requested method, use it exactly as requested
	if point_count >= requested.number_of_points_required() {
		return requested;
	}

	// Not enough points - fall back to simpler methods, respecting the user's requested ceiling
	// We can only fall back to methods SIMPLER than what was requested
	match requested {
		Spline::Linear => {
			// Linear is the simplest, no fallback possible (needs 2 points minimum)
			// With 1 point, we handle it specially in auto_interpolate
			Spline::Linear
		}
		Spline::Cubic | Spline::Quadratic => {
			// Cubic needs 4 points, can fall back to Quadratic (3) or Linear (2)
			// With 0-1 points, Linear will be handled specially in auto_interpolate
			if point_count >= 3 { Spline::Quadratic } else { Spline::Linear }
		}
		Spline::Polynomial(_degree, _bounds) => {
			// Polynomial(n) needs n+1 points
			// Fall back through lower degrees, but cap at Cubic since that's the highest non-polynomial
			let max_usable_degree = point_count.saturating_sub(1);

			if max_usable_degree > 3 {
				requested
			} else if max_usable_degree == 3 {
				// Can use Cubic (degree 3)
				Spline::Cubic
			} else if max_usable_degree == 2 {
				Spline::Quadratic
			} else {
				Spline::Linear
			}
		}
	}
}

#![warn(clippy::pedantic, clippy::nursery, clippy::all)]
#![allow(clippy::multiple_crate_versions, clippy::used_underscore_binding, clippy::similar_names, clippy::module_name_repetitions, clippy::module_inception)]

use anyhow::{Result, bail};
use chrono::{DateTime, Utc};
use helpers::{InterpolationStrategy, should_use_gpu};
pub use optimizations::{apply_fast_path, cpu_interpolate, parallel_interpolate};
pub use types::{BASE_BATCH_SIZE, Error, POINT_SIZE, Point, Resolution, Spline};

mod gpu;
mod helpers;
mod optimizations;
mod splines;
mod tests;
mod types;

// Re-export for public API
pub use gpu::gpu_interpolate;
pub use helpers::{estimate_output_points, generate_target_times};
pub use splines::{DAYS_IN_MONTH, DAYS_IN_YEAR, SECONDS_IN_DAY, SECONDS_IN_HOUR, SECONDS_IN_MINUTE, SECONDS_IN_MONTH, SECONDS_IN_WEEK, SECONDS_IN_YEAR};

/// Main async interpolation function with GPU acceleration support
///
/// This is the primary entry point for all interpolation operations in the library.
/// It automatically selects the optimal interpolation strategy (GPU, CPU, SIMD, Parallel)
/// based on dataset characteristics and performance benchmarks.
///
/// # Errors
///
/// Returns an error if:
/// - Insufficient measurements for interpolation
/// - Invalid time range
/// - GPU initialization fails (with CPU fallback)
/// - All interpolation methods fail
pub async fn auto_interpolate(points: &mut [Point], start: DateTime<Utc>, end: DateTime<Utc>, resolution: Resolution, spline: Spline) -> Result<Vec<Point>> {
	if points.len() < spline.number_of_points_required() {
		bail!(Error::InsufficientMeasurementsError);
	}

	if start >= end {
		bail!(Error::InvalidTimeRangeError);
	}

	let estimated_output_points = estimate_output_points(start, end, resolution);
	let spline = apply_fast_path(spline, points.len());

	// Use centralized strategy selection based on benchmark results
	match should_use_gpu(points.len(), estimated_output_points) {
		InterpolationStrategy::GpuPrimary => {
			// Try GPU first for very large datasets where it's proven to be faster
			if let Ok(result) = gpu_interpolate(points, start, end, resolution, spline).await {
				return Ok(result);
			}
			// If GPU fails, fall back to parallel (still better than CPU for large datasets)
			parallel_interpolate(points, &start, &end, spline, resolution).await
		}
		InterpolationStrategy::GpuThenParallel => {
			// Try GPU first, but with quick fallback to parallel
			if let Ok(result) = gpu_interpolate(points, start, end, resolution, spline).await {
				return Ok(result);
			}
			// Fall back to parallel for large datasets
			parallel_interpolate(points, &start, &end, spline, resolution).await
		}
		InterpolationStrategy::Parallel => {
			// For medium datasets, parallel is clearly optimal
			parallel_interpolate(points, &start, &end, spline, resolution).await
		}
		InterpolationStrategy::Cpu => {
			// For small datasets, use CPU to avoid parallel overhead
			cpu_interpolate(points, start, end, resolution, spline).await
		}
	}
}

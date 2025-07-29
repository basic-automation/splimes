#![warn(clippy::pedantic, clippy::nursery, clippy::all)]
#![allow(clippy::multiple_crate_versions, clippy::used_underscore_binding, clippy::similar_names, clippy::module_name_repetitions, clippy::module_inception)]

use anyhow::{Result, bail};
use chrono::{DateTime, Utc};
pub(crate) use gpu::gpu_interpolate;
pub(crate) use helpers::{InterpolationState, TargetTimesIterator, batch, estimate_output_points, generate_target_times, should_use_gpu};
pub use optimizations::{apply_fast_path, cpu_interpolate, parallel_interpolate};
pub(crate) use splines::{cubic, cubic_simd, linear, linear_simd, polynomial, polynomial_simd, quadratic, quadratic_simd};
pub use types::{BASE_BATCH_SIZE, Error, POINT_SIZE, Point, Resolution, Spline};

mod gpu;
mod helpers;
mod optimizations;
mod splines;
mod tests;
mod types;

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
	let use_gpu = should_use_gpu(points.len(), estimated_output_points);
	let spline = apply_fast_path(spline, points.len());

	if use_gpu {
		// Try GPU interpolation with fallback - clone measurements to avoid ownership issues
		if let Ok(result) = gpu_interpolate(points, start, end, resolution, spline).await {
			return Ok(result);
		}
	}

	// CPU implementation with optimal strategy selection
	cpu_interpolate(points, start, end, resolution, spline).await
}

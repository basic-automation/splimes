use std::io::Write;

use anyhow::{Context, Result, bail};
use bigdecimal::FromPrimitive;
use chrono::{DateTime, Utc};
use rayon::prelude::*;

use crate::{Error, InterpolationState, POINT_SIZE, Point, Resolution, Spline, batch, cubic, cubic_simd, generate_target_times, linear, linear_simd, polynomial, polynomial_simd, quadratic, quadratic_simd};

mod fast_path;
pub use fast_path::apply_fast_path; // Re-export apply_fast_path

const SIMD_THRESHOLD: usize = 200;
const SIMD_THRESHOLD_PLUS_ONE: usize = SIMD_THRESHOLD + 1;

/// # Errors
/// todo
pub async fn cpu_interpolate(points: &mut [Point], start: DateTime<Utc>, end: DateTime<Utc>, resolution: Resolution, spline: Spline) -> Result<Vec<Point>> {
	let input_count = points.len();
	let target_times = generate_target_times(start, end, resolution);
	let output_count = target_times.len();

	match spline {
		Spline::Linear => match (input_count, output_count) {
			(input, output) if input < SIMD_THRESHOLD && output < SIMD_THRESHOLD_PLUS_ONE => linear(points, &start, &end, &resolution).await,
			_ => parallel_interpolate(points, &start, &end, spline, resolution).await,
		},
		Spline::Quadratic => match (input_count, output_count) {
			(input, output) if input < SIMD_THRESHOLD && output < SIMD_THRESHOLD_PLUS_ONE => quadratic(points, &start, &end, &resolution).await,
			_ => parallel_interpolate(points, &start, &end, spline, resolution).await,
		},
		Spline::Cubic => match (input_count, output_count) {
			(input, output) if input < SIMD_THRESHOLD && output < SIMD_THRESHOLD_PLUS_ONE => cubic(points, &start, &end, &resolution).await,
			_ => parallel_interpolate(points, &start, &end, spline, resolution).await,
		},
		Spline::Polynomial(_, _) => match (input_count, output_count) {
			(input, output) if input < SIMD_THRESHOLD && output < SIMD_THRESHOLD_PLUS_ONE => polynomial(points, &start, &end, &resolution, &spline).await,
			_ => parallel_interpolate(points, &start, &end, spline, resolution).await,
		},
	}
}

/// # Errors
/// todo
pub async fn parallel_interpolate(points: &mut [Point], start: &DateTime<Utc>, end: &DateTime<Utc>, spline: Spline, resolution: Resolution) -> Result<Vec<Point>> {
	spline.pre_check(points, start, end)?;
	batch(points, start, end, &spline, &resolution, |state| Box::pin(p_interpolate(state))).await
}

// Fixed p_interpolate function in mod.rs
// The key fix is to use state.result instead of always creating temp files

pub async fn p_interpolate(state: &mut InterpolationState) -> Result<()> {
	let Some(input_points) = &state.input_points else {
		bail!("No input points provided for interpolation");
	};
	let Some(batch_times) = &state.batch_times else {
		bail!("No batch times provided for interpolation");
	};

	state.system.refresh_memory();
	let available_memory = usize::from_u64(state.system.available_memory()).context("Failed to get available memory")?;

	let batch_size = if available_memory < state.memory_threshold {
		let max_points = (available_memory / POINT_SIZE).max(50);
		batch_times.len().min(max_points).max(1)
	} else {
		batch_times.len()
	};

	// Collect chunks first, then process in parallel
	let chunks: Vec<Vec<DateTime<Utc>>> = batch_times.chunks(batch_size).map(<[chrono::DateTime<chrono::Utc>]>::to_vec).collect();

	let chunk_results: Result<Vec<Vec<Point>>> = chunks
		.par_iter()
		.map(|times| match state.spline {
			Spline::Linear => linear_simd(input_points, times, state.resolution),
			Spline::Quadratic => Ok(quadratic_simd(input_points, times, state.resolution)),
			Spline::Cubic => cubic_simd(input_points, times, state.resolution),
			Spline::Polynomial(_, _) => polynomial_simd(input_points, times, state.resolution, &state.spline),
		})
		.collect();

	let chunk_results = chunk_results?;

	// Flatten results in the correct order
	let all_points: Vec<Point> = chunk_results.into_iter().flatten().collect();

	// Verify we have the expected number of points
	assert_eq!(all_points.len(), batch_times.len(), "Parallel processing lost points: expected {}, got {}", batch_times.len(), all_points.len());

	// Check if we should use temp file or in-memory storage
	if state.temp_file.is_some() {
		// Use temp file (for large datasets)
		if let Some(writer) = state.temp_file.as_mut() {
			let mut writer = writer.lock().await;
			for point in all_points {
				writeln!(writer, "{},{}", point.timestamp.to_rfc3339(), point.value)?;
			}
			writer.flush().map_err(|e| Error::IOError(e.to_string()))?;
		}
	} else {
		// Use in-memory storage (for smaller datasets)
		state.result = Some(all_points);
	}

	Ok(())
}

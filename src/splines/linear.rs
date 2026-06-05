use std::io::Write;

use anyhow::{Result, bail};
use bigdecimal::{BigDecimal, FromPrimitive, ToPrimitive, Zero};
use chrono::{DateTime, Utc};
use wide::f64x4;

use super::SIMD_BATCH_SIZE;
use crate::{
	Error, Point, Resolution, Spline, helpers::{InterpolationState, batch}
};

pub async fn linear(points: &mut [Point], start: &DateTime<Utc>, end: &DateTime<Utc>, resolution: &Resolution) -> Result<Vec<Point>> {
	Spline::Linear.pre_check(points, start, end)?;
	batch(points, start, end, &Spline::Linear, resolution, |state| Box::pin(linear_interpolate(state))).await
}

pub async fn linear_interpolate(state: &mut InterpolationState) -> Result<()> {
	let Some(input_points) = &state.input_points else {
		bail!("No input points provided for interpolation");
	};
	let Some(batch_times) = &state.batch_times else {
		bail!("No batch times provided for interpolation");
	};

	let spline = crate::splines::LinearSpline::new(input_points, state.resolution)?;

	// Initialize result if storing in memory
	if state.temp_file.is_none() && state.result.is_none() {
		state.result = Some(Vec::new());
	}

	for current_time in batch_times {
		// Process all times, remove .take(batch_size)
		let value = spline.evaluate(current_time)?;

		if state.temp_file.is_none() {
			state.result.as_mut().unwrap().push(Point { timestamp: *current_time, value });
		} else if let Some(writer) = state.temp_file.as_mut() {
			writeln!(writer.lock().await, "{},{}", current_time.to_rfc3339(), value)?;
		}
	}

	if let Some(writer) = state.temp_file.as_mut() {
		writer.lock().await.flush().map_err(|e| Error::IOError(e.to_string()))?;
	}
	Ok(())
}

/// SIMD-optimized linear interpolation for batch processing
///
/// # Errors
///
/// Returns an error if:
/// - Insufficient points (< 2 points)
/// - Timestamp conversion fails
/// - `BigDecimal` operations fail
pub fn linear_simd(points: &[Point], target_times: &[DateTime<Utc>], _resolution: Resolution) -> Result<Vec<Point>> {
	if points.len() < 2 {
		bail!(Error::InsufficientPointsError);
	}
	if target_times.is_empty() {
		return Ok(Vec::new());
	}

	let base_time = points[0].timestamp;

	// Use nanoseconds for internal time calculations to avoid integer division issues
	// (e.g., minute data with Years resolution = 0). Nanoseconds provide sufficient
	// precision for interpolation while staying within f64 range.
	let input_times: Vec<f64> = points.iter()
		.map(|p| (p.timestamp - base_time).num_nanoseconds().unwrap_or(0) as f64)
		.collect();
	let input_values: Vec<f64> = points.iter().map(|p| round_to_places(p.value.to_f64().unwrap_or(0.0), 10)).collect();
	let targets: Vec<f64> = target_times.iter()
		.map(|t| (*t - base_time).num_nanoseconds().unwrap_or(0) as f64)
		.collect();

	let mut results = Vec::with_capacity(target_times.len());

	for chunk in targets.chunks(SIMD_BATCH_SIZE) {
		let mut padded_targets = [0.0; SIMD_BATCH_SIZE];
		let chunk_size = chunk.len();
		padded_targets[..chunk_size].copy_from_slice(chunk);
		if chunk_size < SIMD_BATCH_SIZE {
			let last_value = chunk.last().copied().unwrap_or(0.0);
			padded_targets[chunk_size..].fill(last_value);
		}

		let target_simd = f64x4::from(padded_targets);
		let result_simd = simd_linear_interpolate(&input_times, &input_values, target_simd);

		let result_array = result_simd.to_array();
		results.extend_from_slice(&result_array[..chunk_size]);
	}

	let interpolated = results.into_iter().enumerate().map(|(i, value)| Point { timestamp: target_times[i], value: BigDecimal::from_f64(round_to_places(value, 10)).unwrap_or_else(BigDecimal::zero) }).collect();

	Ok(interpolated)
}

fn simd_linear_interpolate(input_times: &[f64], input_values: &[f64], target_times: f64x4) -> f64x4 {
	let target_times_array = target_times.to_array();
	let mut results = [0.0; 4];
	for i in 0..4 {
		let t = target_times_array[i];
		let idx = if t <= input_times[0] {
			0
		} else if t >= input_times[input_times.len() - 1] {
			input_times.len().saturating_sub(2)
		} else {
			input_times.partition_point(|&it| it < t).saturating_sub(1)
		};
		let t0 = input_times[idx];
		let t1 = input_times[idx + 1];
		let v0 = input_values[idx];
		let v1 = input_values[idx + 1];
		let result = if t <= input_times[0] {
			let slope = (v1 - v0) / (t1 - t0);
			slope.mul_add(t - t0, v0)
		} else if t >= input_times[input_times.len() - 1] {
			let slope = (v1 - v0) / (t1 - t0);
			slope.mul_add(t - t1, v1)
		} else {
			let alpha = if (t1 - t0).abs() < f64::EPSILON { 0.0 } else { (t - t0) / (t1 - t0) };
			alpha.mul_add(v1 - v0, v0)
		};
		results[i] = round_to_places(result, 10);
	}
	f64x4::new(results)
}

fn round_to_places(value: f64, places: i32) -> f64 {
	let multiplier = 10.0_f64.powi(places);
	(value * multiplier).round() / multiplier
}

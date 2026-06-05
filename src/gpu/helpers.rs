use anyhow::{Context, Result};
use bigdecimal::{BigDecimal, FromPrimitive, ToPrimitive};
use chrono::{DateTime, Utc};

use crate::{Point, Resolution};

// NOTE: get_max_buffer_size removed - use GpuInterpolator::get_max_buffer_size_static() instead
// The old function was creating a new wgpu Instance per call, causing massive overhead.

pub fn convert_points_to_gpu_format_f64(points: &[Point], _resolution: Resolution) -> Result<(Vec<f64>, Vec<f64>)> {
	if points.is_empty() {
		return Ok((Vec::new(), Vec::new()));
	}

	// Sort points by timestamp to match CPU/parallel implementations
	let mut sorted_points = points.to_vec();
	sorted_points.sort_by_key(|p| p.timestamp);

	let base_time = sorted_points[0].timestamp;
	let mut times = Vec::with_capacity(sorted_points.len());
	let mut values = Vec::with_capacity(sorted_points.len());

	for point in &sorted_points {
		let duration = point.timestamp.signed_duration_since(base_time);
		// Use nanoseconds for internal time calculations to avoid integer division issues
		// (e.g., minute data with Years resolution = 0). Nanoseconds provide sufficient
		// precision for interpolation while staying within f64 range.
		let time_offset = duration.num_nanoseconds().context("Invalid duration")?;
		let time_offset = f64::from_i64(time_offset).context("Invalid time offset")?;
		times.push(time_offset);
		let value = point.value.to_f64().context("Invalid value")?;
		values.push(value); // Don't clamp - preserve full precision
	}

	Ok((times, values))
}

pub fn convert_points_to_gpu_format_f32(points: &[Point], _resolution: Resolution) -> Result<(Vec<f32>, Vec<f32>)> {
	if points.is_empty() {
		return Ok((Vec::new(), Vec::new()));
	}

	// Sort points by timestamp to match CPU/parallel implementations
	let mut sorted_points = points.to_vec();
	sorted_points.sort_by_key(|p| p.timestamp);

	let base_time = sorted_points[0].timestamp;
	let mut times = Vec::with_capacity(sorted_points.len());
	let mut values = Vec::with_capacity(sorted_points.len());

	for point in &sorted_points {
		let duration = point.timestamp.signed_duration_since(base_time);
		// Use nanoseconds for internal time calculations to avoid integer division issues
		// (e.g., minute data with Years resolution = 0). Nanoseconds provide sufficient
		// precision for interpolation while staying within f64 range, then convert to f32.
		let time_offset = duration.num_nanoseconds().context("Invalid duration")?;
		let time_offset = f32::from_i64(time_offset).context("Invalid time offset")?;
		times.push(time_offset);
		let value = point.value.to_f32().context("Invalid value")?;
		values.push(value); // Don't clamp - preserve full precision
	}

	Ok((times, values))
}

pub fn convert_datetimes_to_gpu_format_f64(target_times: &[DateTime<Utc>], _resolution: Resolution, base_time: DateTime<Utc>) -> Result<Vec<f64>> {
	let mut times = Vec::with_capacity(target_times.len());
	for &target_time in target_times {
		let duration = target_time.signed_duration_since(base_time);
		// Use nanoseconds for internal time calculations to avoid integer division issues
		// (e.g., minute data with Years resolution = 0). Nanoseconds provide sufficient
		// precision for interpolation while staying within f64 range.
		let time_offset = duration.num_nanoseconds().context("Invalid duration")?;
		let time_offset = f64::from_i64(time_offset).context("Invalid time offset")?;
		times.push(time_offset);
	}
	Ok(times)
}

pub fn convert_datetimes_to_gpu_format_f32(target_times: &[DateTime<Utc>], _resolution: Resolution, base_time: DateTime<Utc>) -> Result<Vec<f32>> {
	let mut times = Vec::with_capacity(target_times.len());
	for &target_time in target_times {
		let duration = target_time.signed_duration_since(base_time);
		// Use nanoseconds for internal time calculations to avoid integer division issues
		// (e.g., minute data with Years resolution = 0). Nanoseconds provide sufficient
		// precision for interpolation while staying within f64 range, then convert to f32.
		let time_offset = duration.num_nanoseconds().context("Invalid duration")?;
		let time_offset = f32::from_i64(time_offset).context("Invalid time offset")?;
		times.push(time_offset);
	}
	Ok(times)
}

pub fn convert_gpu_results_to_points_f64(results: Vec<f64>, target_times: &[DateTime<Utc>]) -> Vec<Point> {
	results.into_iter()
		.zip(target_times)
		.map(|(value, timestamp)| {
			let value = BigDecimal::from_f64(value).unwrap_or_default();
			Point { timestamp: *timestamp, value }
		})
		.collect()
}

pub fn convert_gpu_results_to_points_f32(results: Vec<f32>, target_times: &[DateTime<Utc>]) -> Vec<Point> {
	results.into_iter()
		.zip(target_times)
		.map(|(value, timestamp)| {
			let value = BigDecimal::from_f32(value).unwrap_or_default();
			Point { timestamp: *timestamp, value }
		})
		.collect()
}

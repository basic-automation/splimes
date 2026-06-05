use std::io::Write;

use anyhow::{bail, Result};
use bigdecimal::{BigDecimal, FromPrimitive, ToPrimitive};
use chrono::{DateTime, Utc};
use wide::f64x4;

use super::SIMD_BATCH_SIZE;
use crate::{
	helpers::{batch, InterpolationState}, Error, Point, Resolution, Spline
};

pub async fn cubic(points: &mut [Point], start: &DateTime<Utc>, end: &DateTime<Utc>, resolution: &Resolution) -> Result<Vec<Point>> {
	Spline::Cubic.pre_check(points, start, end)?;
	batch(points, start, end, &Spline::Cubic, resolution, |state| Box::pin(cubic_interpolate(state))).await
}

pub async fn cubic_interpolate(state: &mut InterpolationState) -> Result<()> {
	let Some(input_points) = &state.input_points else {
		bail!("No input points provided for interpolation");
	};
	let Some(batch_times) = &state.batch_times else {
		bail!("No batch times provided for interpolation");
	};

	if state.temp_file.is_none() && state.result.is_none() {
		state.result = Some(Vec::new());
	}

	// Sort points by timestamp to match GPU/SIMD behavior
	let mut sorted_points = input_points.clone();
	sorted_points.sort_by_key(|p| p.timestamp);

	for current_time in batch_times {
		let value = evaluate_lagrange_cubic(&sorted_points, *current_time, state.resolution)?;
		if let Some(result) = state.result.as_mut() {
			result.push(Point { timestamp: *current_time, value });
		} else if let Some(writer) = state.temp_file.as_mut() {
			writeln!(writer.lock().await, "{},{}", current_time.to_rfc3339(), value)?;
		}
	}

	if let Some(writer) = state.temp_file.as_mut() {
		let mut writer = writer.lock().await;
		writer.flush().map_err(|e| Error::IOError(e.to_string()))?;
	}
	Ok(())
}

pub fn cubic_simd(points: &[Point], target_times: &[DateTime<Utc>], resolution: Resolution) -> Result<Vec<Point>> {
	if points.len() < Spline::Cubic.number_of_points_required() {
		bail!(Error::InsufficientPointsError);
	}

	if target_times.is_empty() {
		return Ok(Vec::new());
	}

	let mut sorted_points = points.to_vec();
	sorted_points.sort_by_key(|m| m.timestamp);

	let base_time = sorted_points[0].timestamp;
	#[allow(clippy::cast_precision_loss)]
	let point_times: Vec<f64> = sorted_points.iter().map(|p| resolution.difference(&p.timestamp, &base_time).map(|d| d as f64)).collect::<Result<Vec<f64>>>()?;
	let point_values: Vec<f64> = sorted_points.iter().map(|p| p.value.to_f64().ok_or(Error::DecimalConversionError).map(|v| round_to_places(v, 10)).map_err(anyhow::Error::from)).collect::<Result<Vec<f64>>>()?;

	let mut results = Vec::with_capacity(target_times.len());

	for chunk in target_times.chunks(SIMD_BATCH_SIZE) {
		let mut padded_targets = [0.0; SIMD_BATCH_SIZE];
		let chunk_size = chunk.len();
		#[allow(clippy::cast_precision_loss)]
		for (i, target_time) in chunk.iter().enumerate() {
			padded_targets[i] = resolution.difference(target_time, &base_time)? as f64;
		}
		if chunk_size < SIMD_BATCH_SIZE {
			let last_value = padded_targets[chunk_size - 1];
			padded_targets[chunk_size..].fill(last_value);
		}

		let target_vec = f64x4::from(padded_targets);
		let result_vec = evaluate_lagrange_cubic_simd(&point_times, &point_values, target_vec);

		let result_array = result_vec.to_array();
		for (i, &value) in result_array[..chunk_size].iter().enumerate() {
			results.push(Point { timestamp: chunk[i], value: BigDecimal::from_f64(round_to_places(value, 10)).ok_or(Error::DecimalConversionError)? });
		}
	}

	Ok(results)
}

fn evaluate_lagrange_cubic(points: &[Point], target_time: DateTime<Utc>, resolution: Resolution) -> Result<BigDecimal> {
	let n = points.len();

	if target_time <= points[0].timestamp {
		return Ok(points[0].value.clone());
	}

	if target_time >= points[n - 1].timestamp {
		return Ok(points[n - 1].value.clone());
	}

	let segment_idx = find_segment(points, target_time);
	let i1 = (1_usize).max(segment_idx.min(n - 2));
	let i0 = i1 - 1;
	let i2 = i1 + 1;
	let i3 = i1 + 2;
	let (final_i0, final_i1, final_i2, final_i3) = if i3 >= n { (n - 4, n - 3, n - 2, n - 1) } else { (i0, i1, i2, i3) };

	let base_time = points[final_i0].timestamp;
	let t0 = resolution.difference(&points[final_i0].timestamp, &base_time)? as f64;
	let t1 = resolution.difference(&points[final_i1].timestamp, &base_time)? as f64;
	let t2 = resolution.difference(&points[final_i2].timestamp, &base_time)? as f64;
	let t3 = resolution.difference(&points[final_i3].timestamp, &base_time)? as f64;
	let target_t = resolution.difference(&target_time, &base_time)? as f64;
	let v0 = points[final_i0].value.to_f64().ok_or(Error::DecimalConversionError)?;
	let v1 = points[final_i1].value.to_f64().ok_or(Error::DecimalConversionError)?;
	let v2 = points[final_i2].value.to_f64().ok_or(Error::DecimalConversionError)?;
	let v3 = points[final_i3].value.to_f64().ok_or(Error::DecimalConversionError)?;

	let t = [t0, t1, t2, t3];
	let v = [v0, v1, v2, v3];
	let result = cubic_interpolate_lagrange(target_t, &t, &v);
	Ok(BigDecimal::from_f64(result).ok_or(Error::DecimalConversionError)?)
}

fn evaluate_lagrange_cubic_simd(times: &[f64], values: &[f64], target_times: f64x4) -> f64x4 {
	let n = times.len();
	let mut results = [0.0; 4];
	let target_array = target_times.to_array();

	for (lane, &target_time) in target_array.iter().enumerate() {
		if target_time <= times[0] {
			results[lane] = values[0];
			continue;
		}
		if target_time >= times[n - 1] {
			results[lane] = values[n - 1];
			continue;
		}

		let i1 = (1_usize).max(find_segment_f64_binary(times, target_time).min(n - 2));
		let i0 = i1 - 1;
		let i2 = i1 + 1;
		let i3 = i1 + 2;
		let (final_i0, final_i1, final_i2, final_i3) = if i3 >= n { (n - 4, n - 3, n - 2, n - 1) } else { (i0, i1, i2, i3) };

		let t = [times[final_i0], times[final_i1], times[final_i2], times[final_i3]];
		let v = [values[final_i0], values[final_i1], values[final_i2], values[final_i3]];
		results[lane] = cubic_interpolate_lagrange(target_time, &t, &v);
	}

	f64x4::new(results)
}

fn cubic_interpolate_lagrange(target: f64, t: &[f64; 4], v: &[f64; 4]) -> f64 {
	let denom0 = (t[0] - t[1]) * (t[0] - t[2]) * (t[0] - t[3]);
	let denom1 = (t[1] - t[0]) * (t[1] - t[2]) * (t[1] - t[3]);
	let denom2 = (t[2] - t[0]) * (t[2] - t[1]) * (t[2] - t[3]);
	let denom3 = (t[3] - t[0]) * (t[3] - t[1]) * (t[3] - t[2]);

	if denom0.abs() < 1e-10 || denom1.abs() < 1e-10 || denom2.abs() < 1e-10 || denom3.abs() < 1e-10 {
		return v[1];
	}

	let l0 = ((target - t[1]) * (target - t[2]) * (target - t[3])) / denom0;
	let l1 = ((target - t[0]) * (target - t[2]) * (target - t[3])) / denom1;
	let l2 = ((target - t[0]) * (target - t[1]) * (target - t[3])) / denom2;
	let l3 = ((target - t[0]) * (target - t[1]) * (target - t[2])) / denom3;

	l3.mul_add(v[3], l2.mul_add(v[2], l0.mul_add(v[0], l1 * v[1])))
}

fn find_segment(points: &[Point], target_time: DateTime<Utc>) -> usize {
	for (i, point) in points.iter().enumerate() {
		if target_time <= point.timestamp {
			return i.saturating_sub(1);
		}
	}
	points.len().saturating_sub(2)
}

fn find_segment_f64_binary(times: &[f64], target_time: f64) -> usize {
	if target_time <= times[0] {
		return 0;
	}

	let mut left = 0;
	let mut right = times.len() - 1;

	while left < right - 1 {
		let mid = left + (right - left) / 2;
		if target_time < times[mid] {
			right = mid;
		} else {
			left = mid;
		}
	}

	left
}

fn round_to_places(value: f64, places: i32) -> f64 {
	let multiplier = 10.0_f64.powi(places);
	(value * multiplier).round() / multiplier
}

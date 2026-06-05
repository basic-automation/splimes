use std::io::Write;

use anyhow::{bail, Result};
use bigdecimal::{BigDecimal, FromPrimitive, ToPrimitive};
use chrono::{DateTime, Utc};
use wide::f64x4;

use super::SIMD_BATCH_SIZE;
use crate::{
	helpers::{batch, InterpolationState}, Error, Point, Resolution, Spline
};

pub async fn quadratic(points: &mut [Point], start: &DateTime<Utc>, end: &DateTime<Utc>, resolution: &Resolution) -> Result<Vec<Point>> {
	Spline::Quadratic.pre_check(points, start, end)?;
	batch(points, start, end, &Spline::Quadratic, resolution, |state| Box::pin(quadratic_interpolate(state))).await
}

pub async fn quadratic_interpolate(state: &mut InterpolationState) -> Result<()> {
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

	let base_time = sorted_points[0].timestamp;
	let input_count = sorted_points.len();

	for current_time in batch_times {
		let value = if *current_time < sorted_points[0].timestamp {
			extrapolate_backward_quadratic(&sorted_points, *current_time, &base_time, state.resolution)?
		} else if *current_time > sorted_points[input_count - 1].timestamp {
			extrapolate_forward_quadratic(&sorted_points, *current_time, &base_time, state.resolution)?
		} else {
			interpolate_quadratic(&sorted_points, *current_time, &base_time, state.resolution)?
		};

		if state.temp_file.is_none() {
			state.result.as_mut().ok_or_else(|| Error::ConversionError("Result not initialized".to_string()))?.push(Point { timestamp: *current_time, value });
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

fn extrapolate_backward_quadratic(points: &[Point], target_time: DateTime<Utc>, base_time: &DateTime<Utc>, resolution: Resolution) -> Result<BigDecimal> {
	if points.len() < 3 {
		bail!(Error::InsufficientPointsError);
	}
	let t0 = BigDecimal::from_i64(resolution.difference(&points[0].timestamp, base_time)?).ok_or(Error::DecimalConversionError)?;
	let t1 = BigDecimal::from_i64(resolution.difference(&points[1].timestamp, base_time)?).ok_or(Error::DecimalConversionError)?;
	let t2 = BigDecimal::from_i64(resolution.difference(&points[2].timestamp, base_time)?).ok_or(Error::DecimalConversionError)?;
	let v0 = &points[0].value;
	let v1 = &points[1].value;
	let v2 = &points[2].value;
	let target_t = BigDecimal::from_i64(resolution.difference(&target_time, base_time)?).ok_or(Error::DecimalConversionError)?;

	let denom0 = (&t0 - &t1) * (&t0 - &t2);
	let denom1 = (&t1 - &t0) * (&t1 - &t2);
	let denom2 = (&t2 - &t0) * (&t2 - &t1);

	if denom0.abs() < BigDecimal::from_f64(1e-10).ok_or(Error::DecimalConversionError)? || denom1.abs() < BigDecimal::from_f64(1e-10).ok_or(Error::DecimalConversionError)? || denom2.abs() < BigDecimal::from_f64(1e-10).ok_or(Error::DecimalConversionError)? {
		return Ok(v1.clone());
	}

	let l0 = ((&target_t - &t1) * (&target_t - &t2)) / denom0;
	let l1 = ((&target_t - &t0) * (&target_t - &t2)) / denom1;
	let l2 = ((&target_t - &t0) * (&target_t - &t1)) / denom2;

	Ok(v0 * &l0 + v1 * &l1 + v2 * &l2)
}

fn extrapolate_forward_quadratic(points: &[Point], target_time: DateTime<Utc>, base_time: &DateTime<Utc>, resolution: Resolution) -> Result<BigDecimal> {
	if points.len() < 3 {
		bail!(Error::InsufficientPointsError);
	}
	let n = points.len();
	let t0 = BigDecimal::from_i64(resolution.difference(&points[n - 3].timestamp, base_time)?).ok_or(Error::DecimalConversionError)?;
	let t1 = BigDecimal::from_i64(resolution.difference(&points[n - 2].timestamp, base_time)?).ok_or(Error::DecimalConversionError)?;
	let t2 = BigDecimal::from_i64(resolution.difference(&points[n - 1].timestamp, base_time)?).ok_or(Error::DecimalConversionError)?;
	let v0 = &points[n - 3].value;
	let v1 = &points[n - 2].value;
	let v2 = &points[n - 1].value;
	let target_t = BigDecimal::from_i64(resolution.difference(&target_time, base_time)?).ok_or(Error::DecimalConversionError)?;

	let denom0 = (&t0 - &t1) * (&t0 - &t2);
	let denom1 = (&t1 - &t0) * (&t1 - &t2);
	let denom2 = (&t2 - &t0) * (&t2 - &t1);

	if denom0.abs() < BigDecimal::from_f64(1e-10).ok_or(Error::DecimalConversionError)? || denom1.abs() < BigDecimal::from_f64(1e-10).ok_or(Error::DecimalConversionError)? || denom2.abs() < BigDecimal::from_f64(1e-10).ok_or(Error::DecimalConversionError)? {
		return Ok(v1.clone());
	}

	let l0 = ((&target_t - &t1) * (&target_t - &t2)) / denom0;
	let l1 = ((&target_t - &t0) * (&target_t - &t2)) / denom1;
	let l2 = ((&target_t - &t0) * (&target_t - &t1)) / denom2;

	Ok(v0 * &l0 + v1 * &l1 + v2 * &l2)
}

fn interpolate_quadratic(points: &[Point], target_time: DateTime<Utc>, base_time: &DateTime<Utc>, resolution: Resolution) -> Result<BigDecimal> {
	let input_count = points.len();
	let norm_times: Vec<BigDecimal> = points.iter().map(|p| resolution.difference(&p.timestamp, base_time).and_then(|d| BigDecimal::from_i64(d).ok_or_else(|| anyhow::anyhow!(Error::DecimalConversionError)))).collect::<Result<Vec<BigDecimal>>>()?;
	let target_t = BigDecimal::from_i64(resolution.difference(&target_time, base_time)?).ok_or(Error::DecimalConversionError)?;

	let center_idx = match norm_times.binary_search_by(|t| t.partial_cmp(&target_t).unwrap_or(std::cmp::Ordering::Equal)) {
		Ok(idx) => idx,
		Err(idx) => idx.max(1).min(input_count - 2),
	};

	let (i0, i1, i2) = if center_idx == 0 {
		(0, 1, 2)
	} else if center_idx >= input_count - 1 {
		(input_count - 3, input_count - 2, input_count - 1)
	} else {
		(center_idx - 1, center_idx, center_idx + 1)
	};

	let t0 = &norm_times[i0];
	let t1 = &norm_times[i1];
	let t2 = &norm_times[i2];
	let v0 = &points[i0].value;
	let v1 = &points[i1].value;
	let v2 = &points[i2].value;

	let denom0 = (t0 - t1) * (t0 - t2);
	let denom1 = (t1 - t0) * (t1 - t2);
	let denom2 = (t2 - t0) * (t2 - t1);

	if denom0.abs() < BigDecimal::from_f64(1e-10).ok_or(Error::DecimalConversionError)? || denom1.abs() < BigDecimal::from_f64(1e-10).ok_or(Error::DecimalConversionError)? || denom2.abs() < BigDecimal::from_f64(1e-10).ok_or(Error::DecimalConversionError)? {
		return Ok(v1.clone());
	}

	let l0 = ((&target_t - t1) * (&target_t - t2)) / denom0;
	let l1 = ((&target_t - t0) * (&target_t - t2)) / denom1;
	let l2 = ((&target_t - t0) * (&target_t - t1)) / denom2;

	Ok(v0 * &l0 + v1 * &l1 + v2 * &l2)
}

pub fn quadratic_simd(points: &[Point], target_times: &[DateTime<Utc>], resolution: Resolution) -> Result<Vec<Point>> {
	if target_times.is_empty() {
		return Ok(Vec::new());
	}

	let mut sorted_points = points.to_vec();
	sorted_points.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));

	let base_time = sorted_points[0].timestamp;
	let input_times: Vec<f64> = sorted_points.iter().map(|m| resolution.difference(&m.timestamp, &base_time).map(|d| d as f64)).collect::<Result<Vec<f64>>>()?;
	let input_values: Vec<f64> = sorted_points.iter().map(|m| m.value.to_f64().ok_or(Error::DecimalConversionError).map(|v| round_to_places(v, 10)).map_err(anyhow::Error::from)).collect::<Result<Vec<f64>>>()?;
	let targets: Vec<f64> = target_times.iter().map(|t| resolution.difference(t, &base_time).map(|d| d as f64)).collect::<Result<Vec<f64>>>()?;

	let data_start = input_times[0];

	let mut results = Vec::with_capacity(target_times.len());

	for chunk in targets.chunks(SIMD_BATCH_SIZE) {
		let mut padded_targets = [0.0; SIMD_BATCH_SIZE];
		let chunk_size = chunk.len();
		padded_targets[..chunk_size].copy_from_slice(chunk);
		if chunk_size < SIMD_BATCH_SIZE {
			let last_value = chunk.last().copied().ok_or_else(|| Error::ConversionError("Empty chunk".to_string()))?;
			padded_targets[chunk_size..].fill(last_value);
		}

		let target_simd = f64x4::from(padded_targets);
		let result_simd = simd_quadratic_interpolate_general(&input_times, &input_values, target_simd, data_start);

		let result_array = result_simd.to_array();
		results.extend_from_slice(&result_array[..chunk_size]);
	}

	results.into_iter().enumerate().map(|(i, value)| Ok(Point { timestamp: target_times[i], value: BigDecimal::from_f64(round_to_places(value, 10)).ok_or(Error::DecimalConversionError)? })).collect::<Result<Vec<Point>>>()
}

fn simd_quadratic_interpolate_general(input_times: &[f64], input_values: &[f64], target_times: f64x4, base_time: f64) -> f64x4 {
	let mut results = [0.0; 4];
	let target_array = target_times.to_array();
	let input_count = input_times.len();

	for (i, &target_time) in target_array.iter().enumerate() {
		let norm_target_time = target_time - base_time;

		if norm_target_time < input_times[0] - base_time {
			// Backward extrapolation
			let t0 = input_times[0] - base_time;
			let t1 = input_times[1] - base_time;
			let t2 = input_times[2] - base_time;
			let v0 = input_values[0];
			let v1 = input_values[1];
			let v2 = input_values[2];

			let denom0 = (t0 - t1) * (t0 - t2);
			let denom1 = (t1 - t0) * (t1 - t2);
			let denom2 = (t2 - t0) * (t2 - t1);

			if denom0.abs() < 1e-10 || denom1.abs() < 1e-10 || denom2.abs() < 1e-10 {
				results[i] = v1;
				continue;
			}

			let l0 = ((norm_target_time - t1) * (norm_target_time - t2)) / denom0;
			let l1 = ((norm_target_time - t0) * (norm_target_time - t2)) / denom1;
			let l2 = ((norm_target_time - t0) * (norm_target_time - t1)) / denom2;

			results[i] = v2.mul_add(l2, v0.mul_add(l0, v1 * l1));
		} else if norm_target_time > input_times[input_count - 1] - base_time {
			// Forward extrapolation
			let t0 = input_times[input_count - 3] - base_time;
			let t1 = input_times[input_count - 2] - base_time;
			let t2 = input_times[input_count - 1] - base_time;
			let v0 = input_values[input_count - 3];
			let v1 = input_values[input_count - 2];
			let v2 = input_values[input_count - 1];

			let denom0 = (t0 - t1) * (t0 - t2);
			let denom1 = (t1 - t0) * (t1 - t2);
			let denom2 = (t2 - t0) * (t2 - t1);

			if denom0.abs() < 1e-10 || denom1.abs() < 1e-10 || denom2.abs() < 1e-10 {
				results[i] = v1;
				continue;
			}

			let l0 = ((norm_target_time - t1) * (norm_target_time - t2)) / denom0;
			let l1 = ((norm_target_time - t0) * (norm_target_time - t2)) / denom1;
			let l2 = ((norm_target_time - t0) * (norm_target_time - t1)) / denom2;

			results[i] = v2.mul_add(l2, v0.mul_add(l0, v1 * l1));
		} else {
			// Interpolation
			let center_idx = match input_times.binary_search_by(|t| t.partial_cmp(&target_time).unwrap_or(std::cmp::Ordering::Equal)) {
				Ok(idx) => idx,
				Err(idx) => idx.max(1).min(input_count - 2),
			};

			let (i0, i1, i2) = if center_idx == 0 {
				(0, 1, 2)
			} else if center_idx >= input_count - 1 {
				(input_count - 3, input_count - 2, input_count - 1)
			} else {
				(center_idx - 1, center_idx, center_idx + 1)
			};

			let t0 = input_times[i0] - base_time;
			let t1 = input_times[i1] - base_time;
			let t2 = input_times[i2] - base_time;
			let v0 = input_values[i0];
			let v1 = input_values[i1];
			let v2 = input_values[i2];

			let denom0 = (t0 - t1) * (t0 - t2);
			let denom1 = (t1 - t0) * (t1 - t2);
			let denom2 = (t2 - t0) * (t2 - t1);

			if denom0.abs() < 1e-10 || denom1.abs() < 1e-10 || denom2.abs() < 1e-10 {
				results[i] = v1;
				continue;
			}

			let l0 = ((norm_target_time - t1) * (norm_target_time - t2)) / denom0;
			let l1 = ((norm_target_time - t0) * (norm_target_time - t2)) / denom1;
			let l2 = ((norm_target_time - t0) * (norm_target_time - t1)) / denom2;

			results[i] = v2.mul_add(l2, v0.mul_add(l0, v1 * l1));
		}
	}

	f64x4::new(results)
}

fn round_to_places(value: f64, places: i32) -> f64 {
	let multiplier = 10.0_f64.powi(places);
	(value * multiplier).round() / multiplier
}

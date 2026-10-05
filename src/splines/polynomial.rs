use std::io::Write;

use anyhow::{Result, bail};
use bigdecimal::{BigDecimal, FromPrimitive, ToPrimitive, Zero};
use chrono::{DateTime, Utc};
use wide::f64x4;

use super::SIMD_BATCH_SIZE;
use crate::{
	Error, Point, Resolution, Spline, helpers::{InterpolationState, batch}
};

pub async fn polynomial(points: &mut [Point], start: &DateTime<Utc>, end: &DateTime<Utc>, resolution: &Resolution, spline: &Spline) -> Result<Vec<Point>> {
	spline.pre_check(points, start, end)?;
	batch(points, start, end, spline, resolution, |state| Box::pin(polynomial_interpolate(state))).await
}

pub async fn polynomial_interpolate(state: &mut InterpolationState) -> Result<()> {
	let Some(input_points) = &state.input_points else {
		bail!("No input points provided for interpolation");
	};
	let Some(batch_times) = &state.batch_times else {
		bail!("No batch times provided for interpolation");
	};

	if input_points.len() < state.spline.degree() + 1 {
		bail!(Error::InsufficientPointsError);
	}

	if state.temp_file.is_none() && state.result.is_none() {
		state.result = Some(Vec::new());
	}

	// Sort the input points just like polynomial_simd does to ensure consistent results
	let mut sorted_points = input_points.clone();
	sorted_points.sort_by_key(|p| p.timestamp);
	let deduplicated_points = deduplicate_points(sorted_points);
	if deduplicated_points.len() < state.spline.degree() + 1 {
		bail!(Error::InsufficientPointsError);
	}

	for current_time in batch_times {
		// Use the wrapper function instead of calculating safe_degree manually
		let value = evaluate_polynomial_safe(&deduplicated_points, *current_time, state.resolution, &state.spline)?;

		if state.temp_file.is_none() {
			state.result.as_mut().ok_or_else(|| Error::ConversionError("Result not initialized".to_string()))?.push(Point { timestamp: *current_time, value });
		} else if let Some(writer) = state.temp_file.as_mut() {
			writeln!(writer.lock().await, "{},{}", current_time.to_rfc3339(), value)?;
		}
	}

	if let Some(writer) = state.temp_file.as_mut() {
		writer.lock().await.flush().map_err(|e| Error::IOError(e.to_string()))?;
	}
	Ok(())
}

// Make the wrapper functions public so they can be used by external code
pub fn evaluate_polynomial_safe(points: &[Point], target_time: DateTime<Utc>, resolution: Resolution, spline: &Spline) -> Result<BigDecimal> {
	let safe_degree = determine_safe_degree(spline.degree(), points.len());
	evaluate_polynomial_safe_with_degree(points, target_time, resolution, safe_degree, spline)
}

pub fn evaluate_polynomial_simd(points: &[Point], times: &[f64], values: &[f64], target_times: f64x4, spline: &Spline) -> f64x4 {
	let safe_degree = determine_safe_degree(spline.degree(), points.len());
	evaluate_polynomial_simd_with_degree(points, times, values, target_times, safe_degree, spline)
}

pub fn polynomial_simd(points: &[Point], target_times: &[DateTime<Utc>], resolution: Resolution, spline: &Spline) -> Result<Vec<Point>> {
	if spline.degree() == 0 {
		bail!(Error::InsufficientPointsError);
	}
	if points.len() < spline.degree() + 1 {
		bail!(Error::InsufficientPointsError);
	}
	if target_times.is_empty() {
		return Ok(Vec::new());
	}

	let mut sorted_points = points.to_vec();
	sorted_points.sort_by_key(|p| p.timestamp);
	let deduplicated_points = deduplicate_points_simd(sorted_points);
	if deduplicated_points.len() < spline.degree() + 1 {
		bail!(Error::InsufficientPointsError);
	}

	let base_time = deduplicated_points[0].timestamp;

	let times: Vec<f64> = deduplicated_points.iter().map(|p| resolution.difference(&p.timestamp, &base_time).map(|d| d as f64)).collect::<Result<Vec<f64>>>()?;
	let values: Vec<f64> = deduplicated_points.iter().map(|p| p.value.to_f64().ok_or(Error::DecimalConversionError).map_err(anyhow::Error::from)).collect::<Result<Vec<f64>>>()?;

	let mut results = Vec::with_capacity(target_times.len());

	for chunk in target_times.chunks(SIMD_BATCH_SIZE) {
		let mut padded_targets = [0.0; SIMD_BATCH_SIZE];
		let chunk_size = chunk.len();

		for (i, target_time) in chunk.iter().enumerate() {
			padded_targets[i] = resolution.difference(target_time, &base_time).map(|d| d as f64)?;
		}

		if chunk_size < SIMD_BATCH_SIZE {
			let last_value = padded_targets[chunk_size - 1];
			padded_targets[chunk_size..].fill(last_value);
		}

		let target_simd = f64x4::from(padded_targets);
		// Use the wrapper function - it will calculate safe_degree internally
		let result_simd = evaluate_polynomial_simd(&deduplicated_points, &times, &values, target_simd, spline);

		let result_array = result_simd.to_array();
		for (i, &value) in result_array.iter().take(chunk_size).enumerate() {
			results.push(Point { timestamp: chunk[i], value: BigDecimal::from_f64(value).ok_or(Error::DecimalConversionError)? });
		}
	}

	Ok(results)
}

fn evaluate_polynomial_safe_with_degree(points: &[Point], target_time: DateTime<Utc>, resolution: Resolution, degree: usize, spline: &Spline) -> Result<BigDecimal> {
	if let Some(point) = points.iter().find(|p| p.timestamp == target_time) {
		return Ok(point.value.clone());
	}

	// Convert points to f64 arrays matching SIMD implementation
	let base_time = points[0].timestamp;
	let times: Vec<f64> = points.iter().map(|p| resolution.difference(&p.timestamp, &base_time).map(|d| d as f64)).collect::<Result<Vec<f64>, _>>()?;
	let values: Vec<f64> = points.iter().map(|p| p.value.to_f64().ok_or(Error::DecimalConversionError).map_err(anyhow::Error::from)).collect::<Result<Vec<f64>, _>>()?;

	let target_time_normalized = resolution.difference(&target_time, &base_time).map(|d| d as f64)?;
	let n = points.len();
	let num_points = degree + 1;

	// Use the same point selection logic as SIMD
	let nearest_idx = if target_time_normalized <= times[0] {
		0
	} else if target_time_normalized >= times[n - 1] {
		n.saturating_sub(num_points)
	} else {
		let insert_pos = times.partition_point(|&t| t < target_time_normalized);
		let half_window = num_points / 2;
		let start_idx = insert_pos.saturating_sub(half_window);
		(start_idx + num_points).min(n).saturating_sub(num_points)
	};

	let end_idx = (nearest_idx + num_points).min(n);
	let window_times = &times[nearest_idx..end_idx];
	let window_values = &values[nearest_idx..end_idx];

	let is_extrapolation = target_time_normalized < times[0] || target_time_normalized > times[n - 1];

	#[allow(clippy::comparison_chain)] // if-chain is more readable here than match with Ordering
	let mut total = if window_times.len() == 2 {
		// Linear interpolation for 2-point window
		let t0 = window_times[0];
		let t1 = window_times[1];
		let v0 = window_values[0];
		let v1 = window_values[1];

		if (t1 - t0).abs() > f64::EPSILON {
			let alpha = (target_time_normalized - t0) / (t1 - t0);
			BigDecimal::from_f64(alpha.mul_add(v1 - v0, v0)).ok_or(Error::DecimalConversionError)?
		} else {
			BigDecimal::from_f64(v0).ok_or(Error::DecimalConversionError)?
		}
	} else if window_times.len() > 2 {
		// Lagrange interpolation for 3+ points
		let mut total = BigDecimal::zero();
		for j in 0..window_times.len() {
			let mut term = BigDecimal::from_f64(window_values[j]).ok_or(Error::DecimalConversionError)?;
			for k in 0..window_times.len() {
				if j != k {
					let numerator = BigDecimal::from_f64(target_time_normalized - window_times[k]).ok_or(Error::DecimalConversionError)?;
					let denominator = BigDecimal::from_f64(window_times[j] - window_times[k]).ok_or(Error::DecimalConversionError)?;

					if !denominator.is_zero() {
						term *= &numerator / &denominator;
					}
				}
			}
			total += term;
		}
		total
	} else {
		BigDecimal::from_f64(window_values[0]).ok_or(Error::DecimalConversionError)?
	};

	if is_extrapolation && spline.bounds_factor().is_some() {
		total = apply_extrapolation_bounds(points, total, spline.bounds_factor().ok_or_else(|| Error::ConversionError("No bounds factor".to_string()))?)?;
	}

	Ok(total)
}

fn evaluate_polynomial_simd_with_degree(points: &[Point], times: &[f64], values: &[f64], target_times: f64x4, degree: usize, spline: &Spline) -> f64x4 {
	let mut results = [0.0; 4];
	let target_array = target_times.to_array();
	let n = points.len();
	let num_points = degree + 1;

	let (min_value, max_value, bounds_factor) = spline.bounds_factor().map_or((0.0, 0.0, None), |f| {
		let min_v = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
		let max_v = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
		(min_v, max_v, Some(f))
	});

	for (lane, &target_time) in target_array.iter().enumerate() {
		let nearest_idx = if target_time <= times[0] {
			0
		} else if target_time >= times[n - 1] {
			n.saturating_sub(num_points)
		} else {
			let insert_pos = times.partition_point(|&t| t < target_time);
			let half_window = num_points / 2;
			let start_idx = insert_pos.saturating_sub(half_window);
			(start_idx + num_points).min(n).saturating_sub(num_points)
		};

		let end_idx = (nearest_idx + num_points).min(n);
		let window_times = &times[nearest_idx..end_idx];
		let window_values = &values[nearest_idx..end_idx];

		if window_times.len() >= 2 {
			if window_times.len() == 2 {
				let t0 = window_times[0];
				let t1 = window_times[1];
				let v0 = window_values[0];
				let v1 = window_values[1];

				if (t1 - t0).abs() > f64::EPSILON {
					let alpha = (target_time - t0) / (t1 - t0);
					results[lane] = alpha.mul_add(v1 - v0, v0);
				} else {
					results[lane] = v0;
				}
			} else {
				let mut total = 0.0;
				for j in 0..window_times.len() {
					let mut term = window_values[j];
					for k in 0..window_times.len() {
						if j != k {
							let numerator = target_time - window_times[k];
							let denominator = window_times[j] - window_times[k];
							if denominator.abs() > f64::EPSILON {
								term *= numerator / denominator;
							}
						}
					}
					total += term;
				}
				results[lane] = total;
			}
		} else if !window_values.is_empty() {
			results[lane] = window_values[0];
		}

		// Apply bounds if extrapolating and bounds_factor is set
		if let Some(f) = bounds_factor {
			let is_extrapolating = target_time < times[0] || target_time > times[times.len() - 1];
			if is_extrapolating {
				let range = max_value - min_value;
				let lower_bound = range.mul_add(-f, min_value);
				let upper_bound = range.mul_add(f, max_value);
				results[lane] = results[lane].clamp(lower_bound, upper_bound);
			}
		}
	}

	f64x4::new(results)
}

fn deduplicate_points(points: Vec<Point>) -> Vec<Point> {
	let mut deduplicated = Vec::new();
	let mut last_timestamp = None;

	for point in points {
		if last_timestamp != Some(point.timestamp) {
			deduplicated.push(point.clone());
			last_timestamp = Some(point.timestamp);
		}
	}

	deduplicated
}

fn deduplicate_points_simd(points: Vec<Point>) -> Vec<Point> {
	deduplicate_points(points)
}

fn determine_safe_degree(requested_degree: usize, available_points: usize) -> usize {
	if available_points <= 1 {
		return 0; // Changed from 1 to 0 - can't do polynomial interpolation with <= 1 points
	}
	let max_degree_by_points = available_points - 1;
	let stability_cap = 8;
	max_degree_by_points.min(requested_degree).min(stability_cap)
}

fn apply_extrapolation_bounds(points: &[Point], value: BigDecimal, bounds_factor: f64) -> Result<BigDecimal> {
	let min_value = points.iter().map(|p| &p.value).min().ok_or_else(|| Error::ConversionError("No points".to_string()))?.clone();
	let max_value = points.iter().map(|p| &p.value).max().ok_or_else(|| Error::ConversionError("No points".to_string()))?.clone();
	let range = max_value.clone() - min_value.clone();
	let bounds_factor_bd = BigDecimal::from_f64(bounds_factor).ok_or(Error::DecimalConversionError)?;
	let lower_bound = min_value - &range * &bounds_factor_bd;
	let upper_bound = max_value + &range * &bounds_factor_bd;

	if value < lower_bound {
		Ok(lower_bound)
	} else if value > upper_bound {
		Ok(upper_bound)
	} else {
		Ok(value)
	}
}

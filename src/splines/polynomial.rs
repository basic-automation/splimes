use std::io::Write;

use anyhow::{bail, Result};
use bigdecimal::{BigDecimal, FromPrimitive, One, ToPrimitive, Zero};
use chrono::{DateTime, Utc};
use wide::f64x4;

use super::SIMD_BATCH_SIZE;
use crate::{
	helpers::{batch, InterpolationState}, Error, Point, Resolution, Spline
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

	let deduplicated_points = deduplicate_points(input_points.clone());
	if deduplicated_points.len() < state.spline.degree() + 1 {
		bail!(Error::InsufficientPointsError);
	}

	for current_time in batch_times {
		// Use the wrapper function instead of calculating safe_degree manually
		let value = evaluate_polynomial_safe(&deduplicated_points, *current_time, state.resolution, &state.spline)?;

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

	// Remove the redundant safe_degree calculation since the wrapper function calculates it
	let base_time = deduplicated_points[0].timestamp;

	let mut results = Vec::with_capacity(target_times.len());

	for chunk in target_times.chunks(SIMD_BATCH_SIZE) {
		let mut padded_targets = [0.0; SIMD_BATCH_SIZE];
		let chunk_size = chunk.len();

		for (i, target_time) in chunk.iter().enumerate() {
			padded_targets[i] = timestamp_to_f64(*target_time, base_time, resolution)?;
		}

		if chunk_size < SIMD_BATCH_SIZE {
			let last_value = padded_targets[chunk_size - 1];
			padded_targets[chunk_size..].fill(last_value);
		}

		let times: Vec<f64> = deduplicated_points.iter().map(|p| timestamp_to_f64(p.timestamp, base_time, resolution).unwrap_or(0.0)).collect();
		let values: Vec<f64> = deduplicated_points.iter().map(|p| p.value.to_f64().unwrap_or(0.0)).collect();

		let target_simd = f64x4::from(padded_targets);
		// Use the wrapper function - it will calculate safe_degree internally
		let result_simd = evaluate_polynomial_simd(&deduplicated_points, &times, &values, target_simd, spline);

		let result_array = result_simd.to_array();
		for (i, &value) in result_array.iter().take(chunk_size).enumerate() {
			results.push(Point { timestamp: chunk[i], value: BigDecimal::from_f64(value).unwrap_or_else(BigDecimal::zero) });
		}
	}

	Ok(results)
}

fn evaluate_polynomial_safe_with_degree(points: &[Point], target_time: DateTime<Utc>, resolution: Resolution, degree: usize, spline: &Spline) -> Result<BigDecimal> {
	if let Some(point) = points.iter().find(|p| p.timestamp == target_time) {
		return Ok(point.value.clone());
	}

	let is_extrapolation = is_extrapolating(points, target_time);
	let selected_points = select_nearest_points_adaptive(points, target_time, degree + 1);
	let time_values = get_normalized_time_values(selected_points, target_time, resolution)?;
	let target_x = time_values.target_normalized;
	let mut total = BigDecimal::zero();

	for j in 0..selected_points.len() {
		let mut numerator = BigDecimal::one();
		let mut denominator = BigDecimal::one();

		for k in 0..selected_points.len() {
			if j != k {
				// Use compound assignment operators
				numerator *= &target_x - &time_values.points_normalized[k];
				denominator *= &time_values.points_normalized[j] - &time_values.points_normalized[k];
			}
		}

		if !denominator.is_zero() {
			// Use compound assignment operator
			total += &selected_points[j].value * &numerator / &denominator;
		}
	}

	// Linear fallback for degenerate cases
	if selected_points.len() == 2 && degree > 1 {
		let p0 = &selected_points[0];
		let p1 = &selected_points[1];
		let t0 = time_values.points_normalized[0].to_f64().unwrap_or(0.0);
		let t1 = time_values.points_normalized[1].to_f64().unwrap_or(0.0);
		let v0 = p0.value.to_f64().unwrap_or(0.0);
		let v1 = p1.value.to_f64().unwrap_or(0.0);
		let t_norm = target_x.to_f64().unwrap_or(0.0);

		if (t1 - t0).abs() > f64::EPSILON {
			let normalized_t = (t_norm - t0) / (t1 - t0);
			total = BigDecimal::from_f64(normalized_t.mul_add(v1 - v0, v0)).unwrap_or_else(BigDecimal::zero);
		}
	}

	if is_extrapolation && spline.bounds_factor().is_some() {
		total = apply_extrapolation_bounds(points, total, spline.bounds_factor().unwrap());
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

fn is_extrapolating(points: &[Point], target_time: DateTime<Utc>) -> bool {
	target_time < points[0].timestamp || target_time > points[points.len() - 1].timestamp
}

fn select_nearest_points_adaptive(points: &[Point], target_time: DateTime<Utc>, num_points: usize) -> &[Point] {
	let n = points.len();
	if num_points >= n {
		return points;
	}

	let insert_pos = points.partition_point(|p| p.timestamp < target_time);
	let half_window = num_points / 2;
	let start_idx = insert_pos.saturating_sub(half_window);
	let adjusted_start = (start_idx + num_points).min(n).saturating_sub(num_points);
	let end_idx = (adjusted_start + num_points).min(n);
	&points[adjusted_start..end_idx]
}

fn get_normalized_time_values(points: &[Point], target_time: DateTime<Utc>, resolution: Resolution) -> Result<NormalizedTimeValues> {
	let base_time = points[0].timestamp;
	let target_normalized = BigDecimal::from_i64(resolution.difference(&target_time, &base_time).map_err(|_| Error::DecimalConversionError)?).ok_or(Error::DecimalConversionError)?;
	let points_normalized = points.iter().map(|p| BigDecimal::from_i64(resolution.difference(&p.timestamp, &base_time).map_err(|_| Error::DecimalConversionError)?).ok_or(Error::DecimalConversionError)).collect::<Result<Vec<BigDecimal>, _>>()?;
	Ok(NormalizedTimeValues { target_normalized, points_normalized })
}

struct NormalizedTimeValues {
	target_normalized: BigDecimal,
	points_normalized: Vec<BigDecimal>,
}

fn apply_extrapolation_bounds(points: &[Point], value: BigDecimal, bounds_factor: f64) -> BigDecimal {
	let min_value = points.iter().map(|p| &p.value).min_by(std::cmp::Ord::cmp).unwrap();
	let max_value = points.iter().map(|p| &p.value).max_by(std::cmp::Ord::cmp).unwrap();
	let range = max_value - min_value;
	let bounds_factor_bd = BigDecimal::from_f64(bounds_factor).unwrap_or_else(BigDecimal::zero);
	let lower_bound = min_value - &range * &bounds_factor_bd;
	let upper_bound = max_value + &range * &bounds_factor_bd;

	if value < lower_bound {
		lower_bound
	} else if value > upper_bound {
		upper_bound
	} else {
		value
	}
}

fn timestamp_to_f64(target_time: DateTime<Utc>, base_time: DateTime<Utc>, resolution: Resolution) -> Result<f64> {
	// Address the precision loss warning by making it explicit and documented
	#[allow(clippy::cast_precision_loss)]
	// Note: This cast may lose precision for very large time differences, but this is acceptable
	// for interpolation purposes where we need f64 for SIMD operations
	Ok(resolution.difference(&target_time, &base_time).map_err(|_| Error::DecimalConversionError)? as f64)
}

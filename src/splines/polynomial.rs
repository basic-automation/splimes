use std::io::Write;

use anyhow::{Result, bail};
use bigdecimal::{BigDecimal, FromPrimitive, One, ToPrimitive, Zero};
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

	let deduplicated_points = deduplicate_points(input_points.clone());
	if deduplicated_points.len() < state.spline.degree() + 1 {
		bail!(Error::InsufficientPointsError);
	}

	let safe_degree = determine_safe_degree(state.spline.degree(), deduplicated_points.len());

	for current_time in batch_times {
		let value = evaluate_polynomial_safe(&deduplicated_points, *current_time, state.resolution, &Spline::Polynomial(safe_degree, state.spline.bounds_factor()))?;
		if state.temp_file.is_none() {
			state.result.as_mut().unwrap().push(Point { timestamp: *current_time, value });
		} else if let Some(writer) = state.temp_file.as_mut() {
			let mut writer = writer.lock().await;
			writeln!(writer, "{},{}", current_time.to_rfc3339(), value)?;
			writer.flush()?;
		}
	}

	if let Some(writer) = state.temp_file.as_mut() {
		let mut writer = writer.lock().await;
		writer.flush().map_err(|e| Error::IOError(e.to_string()))?;
	}
	Ok(())
}

pub fn polynomial_simd(points: &[Point], target_times: &[DateTime<Utc>], resolution: Resolution, spline: &Spline) -> Result<Vec<Point>> {
	if spline.degree() == 0 {
		bail!(Error::InvalidDegreeError(spline.degree()));
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

	let safe_degree = determine_safe_degree(spline.degree(), deduplicated_points.len());
	let base_time = deduplicated_points[0].timestamp;

	let mut results = Vec::with_capacity(target_times.len());

	for chunk in target_times.chunks(SIMD_BATCH_SIZE) {
		let mut padded_targets = [0.0; SIMD_BATCH_SIZE];
		let chunk_size = chunk.len();
		for (i, target_time) in chunk.iter().enumerate() {
			let time_diff = resolution.difference(target_time, &base_time)?;
			let t_d = f64::from_i64(time_diff).unwrap_or_default();
			padded_targets[i] = t_d / 1_000_000_000.0;
		}
		if chunk_size < SIMD_BATCH_SIZE {
			let last_value = padded_targets[chunk_size - 1];
			padded_targets[chunk_size..].fill(last_value);
		}

		let point_times: Vec<f64> = deduplicated_points
			.iter()
			.map(|p| {
				resolution
					.difference(&p.timestamp, &base_time)
					.map(|t| {
						let t = f64::from_i64(t).unwrap_or_default();
						t / 1_000_000_000.0
					})
					.unwrap_or(0.0)
			})
			.collect();
		let point_values: Vec<f64> = deduplicated_points.iter().map(|p| p.value.to_f64().unwrap_or(0.0)).collect();

		let target_vec = f64x4::from(padded_targets);
		let result_vec = evaluate_polynomial_simd(&deduplicated_points, &point_times, &point_values, target_vec, &Spline::Polynomial(safe_degree, spline.bounds_factor()));

		let result_array = result_vec.to_array();
		for (i, &value) in result_array[..chunk_size].iter().enumerate() {
			results.push(Point { timestamp: chunk[i], value: BigDecimal::from_f64(value).unwrap_or_else(BigDecimal::zero) });
		}
	}

	Ok(results)
}

fn evaluate_polynomial_safe(points: &[Point], target_time: DateTime<Utc>, resolution: Resolution, spline: &Spline) -> Result<BigDecimal> {
	if let Some(point) = points.iter().find(|p| p.timestamp == target_time) {
		return Ok(point.value.clone());
	}

	let is_extrapolation = is_extrapolating(points, target_time);
	let selected_points = select_nearest_points_adaptive(points, target_time, spline.degree() + 1);
	let time_values = get_normalized_time_values(selected_points, target_time, resolution)?;
	let target_x = time_values.target_normalized;
	let mut total = BigDecimal::zero();

	for j in 0..selected_points.len() {
		let y_j = &selected_points[j].value;
		let x_j = &time_values.points_normalized[j];
		let mut lagrange_basis = BigDecimal::one();
		let mut has_near_zero_denominator = false;

		for m in 0..selected_points.len() {
			if m == j {
				continue;
			}
			let x_m = &time_values.points_normalized[m];
			let denominator = x_j - x_m;
			if denominator.abs() < BigDecimal::from_f64(1e-10).unwrap_or_else(BigDecimal::zero) {
				has_near_zero_denominator = true;
				break;
			}
			let numerator = &target_x - x_m;
			lagrange_basis *= numerator / denominator;
		}

		if !has_near_zero_denominator {
			total += y_j * lagrange_basis;
		} else if spline.degree() == 1 && selected_points.len() >= 2 {
			let t = timestamp_to_f64(target_time, selected_points[0].timestamp, resolution)?;
			let t1 = time_values.points_normalized[0].to_f64().unwrap_or(0.0);
			let t2 = time_values.points_normalized[1].to_f64().unwrap_or(0.0);
			let v1 = selected_points[0].value.to_f64().unwrap_or(0.0);
			let v2 = selected_points[1].value.to_f64().unwrap_or(0.0);
			let t_norm = (t - t1) / (t2 - t1);
			total = BigDecimal::from_f64(v1 + t_norm * (v2 - v1)).unwrap_or_else(BigDecimal::zero);
			break;
		}
	}

	if is_extrapolation && spline.bounds_factor().is_some() {
		total = apply_extrapolation_bounds(points, target_time, total, spline.bounds_factor().unwrap());
	}

	Ok(total)
}

fn evaluate_polynomial_simd(points: &[Point], times: &[f64], values: &[f64], target_times: f64x4, spline: &Spline) -> f64x4 {
	let mut results = [0.0; 4];
	let target_array = target_times.to_array();
	let n = points.len();
	let num_points = spline.degree() + 1;

	for (lane, &target_time) in target_array.iter().enumerate() {
		if n < num_points {
			results[lane] = 0.0;
			continue;
		}

		// Find the window of points for this target_time
		let insert_pos = times.iter().position(|&t| t > target_time).unwrap_or(times.len());
		let half_window = num_points / 2;
		let mut start_idx = insert_pos.saturating_sub(half_window);
		let end_idx = (start_idx + num_points).min(times.len());
		start_idx = end_idx.saturating_sub(num_points);
		let selected_times = &times[start_idx..end_idx];
		let selected_values = &values[start_idx..end_idx];

		if selected_times.len() < num_points {
			results[lane] = 0.0;
			continue;
		}

		// Lagrange interpolation
		let mut result = 0.0;
		let mut has_near_zero_denominator = false;

		for j in 0..selected_times.len() {
			let y_j = selected_values[j];
			let x_j = selected_times[j];
			let mut lagrange_basis = 1.0;

			for (m, _) in selected_times.iter().enumerate() {
				if m == j {
					continue;
				}
				let x_m = selected_times[m];
				let denominator = x_j - x_m;
				if denominator.abs() < 1e-10 {
					has_near_zero_denominator = true;
					break;
				}
				lagrange_basis *= (target_time - x_m) / denominator;
			}

			if !has_near_zero_denominator {
				result += y_j * lagrange_basis;
			}
		}

		// Fallback to linear interpolation for degree 1
		if has_near_zero_denominator && spline.degree() == 1 && selected_times.len() >= 2 {
			let t1 = selected_times[0];
			let t2 = selected_times[1];
			let v1 = selected_values[0];
			let v2 = selected_values[1];
			let t_norm = (target_time - t1) / (t2 - t1).max(1e-10);
			result = v1 + t_norm * (v2 - v1);
		}

		// Extrapolation bounds
		if spline.bounds_factor().is_some() && (target_time < times[0] || target_time > times[n - 1]) {
			let bounds_factor = spline.bounds_factor().unwrap();
			let min_value = points.iter().map(|p| p.value.to_f64().unwrap_or(0.0)).fold(f64::INFINITY, f64::min);
			let max_value = points.iter().map(|p| p.value.to_f64().unwrap_or(0.0)).fold(f64::NEG_INFINITY, f64::max);
			let range = max_value - min_value;
			let lower_bound = min_value - range * bounds_factor;
			let upper_bound = max_value + range * bounds_factor;
			result = if target_time < times[0] { lower_bound } else { upper_bound };
		}

		results[lane] = result;
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
		return 0;
	}
	let max_degree_by_points = available_points - 1;
	let stability_cap = 8;
	max_degree_by_points.min(requested_degree).min(stability_cap)
}

fn is_extrapolating(points: &[Point], target_time: DateTime<Utc>) -> bool {
	target_time < points[0].timestamp || target_time > points[points.len() - 1].timestamp
}

fn select_nearest_points_adaptive(points: &[Point], target_time: DateTime<Utc>, num_points: usize) -> &[Point] {
	if num_points >= points.len() {
		return points;
	}

	// Use point timestamps directly
	let insert_pos = points.iter().position(|p| p.timestamp > target_time).unwrap_or(points.len());
	let half_window = num_points / 2;
	let mut start_idx = insert_pos.saturating_sub(half_window);
	let end_idx = (start_idx + num_points).min(points.len());
	start_idx = end_idx.saturating_sub(num_points);
	let window = start_idx..end_idx;
	&points[window]
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

fn apply_extrapolation_bounds(points: &[Point], target_time: DateTime<Utc>, _value: BigDecimal, bounds_factor: f64) -> BigDecimal {
	let min_value = points.iter().map(|p| &p.value).min_by(std::cmp::Ord::cmp).unwrap();
	let max_value = points.iter().map(|p| &p.value).max_by(std::cmp::Ord::cmp).unwrap();
	let range = max_value - min_value;
	let lower_bound = min_value - &range * BigDecimal::from_f64(bounds_factor).unwrap_or_else(BigDecimal::zero);
	let upper_bound = max_value + &range * BigDecimal::from_f64(bounds_factor).unwrap_or_else(BigDecimal::zero);

	if target_time < points[0].timestamp { lower_bound } else { upper_bound }
}

fn timestamp_to_f64(target_time: DateTime<Utc>, base_time: DateTime<Utc>, resolution: Resolution) -> Result<f64> {
	Ok(resolution
		.difference(&target_time, &base_time)
		.map(|t| {
			let t = f64::from_i64(t).unwrap_or_default();
			t / 1_000_000_000.0
		})
		.map_err(|_| Error::DecimalConversionError)?)
}

use std::{
	fs::File, io::{BufRead, BufReader, BufWriter, Write}
};

use anyhow::bail;
use bigdecimal::{BigDecimal, FromPrimitive};
use chrono::{DateTime, Utc};
pub use helpers::get_max_buffer_size;
use sysinfo::System;
use tempfile::NamedTempFile;
pub use types::{GpuInterpolator, Method};
use wgpu::util::DeviceExt; // Added missing import

use crate::{BASE_BATCH_SIZE, Error, InterpolationState, POINT_SIZE, Point, Resolution, Result, Spline, TargetTimesIterator, batch};

mod helpers;
mod shaders;
mod types;

const F64_SIZE: usize = std::mem::size_of::<f64>();
const NUMBER_OF_BUFFERS: usize = 5;

pub async fn gpu_interpolate(points: &mut [Point], start: DateTime<Utc>, end: DateTime<Utc>, resolution: Resolution, spline: Spline) -> Result<Vec<Point>> {
	spline.pre_check(points, &start, &end)?;
	let method = match spline {
		Spline::Linear => Method::Linear,
		Spline::Quadratic => Method::Quadratic,
		Spline::Cubic => Method::Cubic,
		Spline::Polynomial(degree, _) => Method::Polynomial(points.len().min(degree + 1)),
	};
	let mut interpolator = GpuInterpolator::new().await?;
	if interpolator.supports_f64() { gpu_interpolate_f64(points, &start, &end, &resolution, &method, &mut interpolator).await } else { batch(points, &start, &end, &spline, &resolution, |state| Box::pin(gpu_interpolate_f32(state))).await }
}

pub async fn gpu_interpolate_f32(state: &mut InterpolationState) -> Result<()> {
	let Some(input_points) = &state.input_points else {
		bail!("No input points provided for interpolation");
	};
	let Some(batch_times) = &state.batch_times else {
		bail!("No batch times provided for interpolation");
	};

	if state.temp_file.is_none() && state.result.is_none() {
		state.result = Some(Vec::new());
	}

	let method = match state.spline {
		Spline::Linear => Method::Linear,
		Spline::Quadratic => Method::Quadratic,
		Spline::Cubic => Method::Cubic,
		Spline::Polynomial(degree, _) => Method::Polynomial(input_points.len().min(degree + 1)),
	};

	state.system.refresh_memory();
	let available_memory = usize::try_from(state.system.available_memory())?;
	let batch_size = if available_memory < state.memory_threshold {
		let max_points = (available_memory / POINT_SIZE).max(50);
		batch_times.len().min(max_points / 2)
	} else {
		batch_times.len()
	};

	let mut interpolator = GpuInterpolator::new().await?;
	let mut b = Vec::with_capacity(batch_size);
	let mut time_offset = 0;

	for batch in batch_times.chunks(batch_size) {
		let target_times_array = helpers::convert_datetimes_to_gpu_format_f32(batch, state.resolution, input_points[0].timestamp)?;
		let m_b = get_max_buffer_size().await?;
		let m_b = usize::from_u64(m_b).ok_or_else(|| Error::ConversionError("Failed to convert buffer size to usize".to_string()))?;
		let max_single_buffer_size = m_b / NUMBER_OF_BUFFERS;
		let max_targets_per_batch = (max_single_buffer_size / F64_SIZE).min(target_times_array.len());
		let (input_times, input_values) = helpers::convert_points_to_gpu_format_f32(input_points, state.resolution)?;

		let mut batch_results = Vec::new();
		for target_batch in target_times_array.chunks(max_targets_per_batch) {
			let config = match state.spline {
				Spline::Polynomial(degree, bounds_factor) => {
					let b_f = f32::from_f64(bounds_factor.unwrap_or_else(|| f64::from(f32::NAN))).ok_or_else(|| Error::ConversionError("Failed to convert bounds factor to f32".to_string()))?;
					let d = u32::from_usize(degree).ok_or_else(|| Error::ConversionError("Failed to convert degree to u32".to_string()))?;
					let t_o = u32::from_usize(time_offset).ok_or_else(|| Error::ConversionError("Failed to convert time offset to u32".to_string()))?;
					[t_o.to_le_bytes().to_vec(), d.to_le_bytes().to_vec(), b_f.to_bits().to_le_bytes().to_vec()].concat()
				}
				_ => [0u32.to_le_bytes().to_vec(), [0u32.to_le_bytes().to_vec(), f32::NAN.to_bits().to_le_bytes().to_vec()].concat()].concat(),
			};

			let config_buffer = interpolator.get_device().create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("content"), contents: &config, usage: wgpu::BufferUsages::UNIFORM });

			let results: Vec<f32> = interpolator.interpolate_f32(&input_times, &input_values, target_batch, &method, &config_buffer)?;
			let batch_slice = &batch[time_offset..time_offset + results.len()];
			let points = helpers::convert_gpu_results_to_points_f32(results, batch_slice);
			time_offset += points.len();

			if let Some(writer) = &mut state.temp_file {
				for point in points {
					writeln!(writer.lock().await, "{},{}", point.timestamp.to_rfc3339(), point.value)?;
				}
			} else if let Some(result) = &mut state.result {
				result.extend(points);
			} else {
				batch_results.extend_from_slice(&points);
			}
		}

		time_offset = 0;

		if let Some(writer) = &mut state.temp_file {
			let mut writer = writer.lock().await;
			writer.flush().map_err(|e| Error::IOError(e.to_string()))?;
		} else if let Some(result) = &mut state.result {
			result.extend(batch_results.clone());
		}

		if batch_results.is_empty() && state.temp_file.is_none() {
			b.extend(batch_results);
		}
	}
	if let Some(result) = &mut state.result {
		result.extend(b);
	} else if let Some(writer) = &mut state.temp_file {
		let mut writer = writer.lock().await;
		for point in b {
			writeln!(writer, "{},{}", point.timestamp.to_rfc3339(), point.value)?;
		}
		writer.flush().map_err(|e| Error::IOError(e.to_string()))?;
	} else if !b.is_empty() {
		state.result = Some(b);
	}

	Ok(())
}

pub async fn gpu_interpolate_f64(points: &[Point], start: &DateTime<Utc>, end: &DateTime<Utc>, resolution: &Resolution, method: &Method, interpolator: &mut GpuInterpolator) -> Result<Vec<Point>> {
	if points.len() < 2 {
		bail!(Error::InsufficientPointsError);
	}
	if start >= end {
		bail!(Error::InvalidTimeRangeError);
	}

	let mut system = System::new_all();
	let mut result = Vec::new();
	let mut temp_file: Option<BufWriter<File>> = None;
	let mut temp_path: Option<tempfile::TempPath> = None;

	let total_memory = system.total_memory();
	let t_m = f64::from_u64(total_memory).ok_or_else(|| Error::ConversionError("Failed to convert total memory to f64".to_string()))?;
	let memory_threshold = u64::from_f64(t_m * 0.8).ok_or_else(|| Error::ConversionError("Failed to convert memory threshold to u64".to_string()))?;
	let point_size = std::mem::size_of::<Point>() as u64;

	let element_size = F64_SIZE;
	let (input_times, input_values) = helpers::convert_points_to_gpu_format_f64(points, *resolution)?;
	let base_time = points[0].timestamp;

	let rounded_start = resolution.round(start)?;
	let rounded_end = resolution.round(end)?;
	let time_iter = TargetTimesIterator::new(rounded_start, rounded_end, *resolution);
	if let Ok(estimated_points) = time_iter.estimate_len() {
		if estimated_points < BASE_BATCH_SIZE * 2 {
			result.reserve(estimated_points);
		} else if estimated_points > 1_000_000 {
			let temp = NamedTempFile::new().map_err(|e| Error::IOError(e.to_string()))?;
			temp_path = Some(temp.into_temp_path());
			temp_file = Some(BufWriter::new(File::create(temp_path.as_ref().unwrap()).map_err(|e| Error::IOError(e.to_string()))?));
		}
	}
	for batch_times in time_iter {
		system.refresh_memory();
		let available_memory = system.available_memory();
		let batch_size = if available_memory < memory_threshold {
			let max_points = usize::try_from((available_memory / point_size).max(50))?;
			batch_times.len().min(max_points / 2)
		} else {
			batch_times.len()
		};
		let mut batch = Vec::with_capacity(batch_size);

		let target_times = helpers::convert_datetimes_to_gpu_format_f64(&batch_times[..batch_size], *resolution, base_time)?;
		let b_size = get_max_buffer_size().await?;
		let b_size = usize::from_u64(b_size).ok_or_else(|| Error::ConversionError("Failed to convert buffer size to usize".to_string()))?;
		let max_single_buffer_size = b_size / NUMBER_OF_BUFFERS / element_size;
		let max_targets_per_batch = (max_single_buffer_size / element_size).min(target_times.len());

		for target_batch in target_times.chunks(max_targets_per_batch) {
			let config = match method {
				Method::Polynomial(degree) => {
					let d = u32::from_usize(*degree).unwrap_or(1);
					[
						0u32.to_le_bytes().to_vec(), // time_offset (not used in f64 for now)
						d.to_le_bytes().to_vec(),
						f64::NAN.to_bits().to_le_bytes().to_vec(),
					]
					.concat()
				}
				_ => [0u32.to_le_bytes().to_vec(), [0u32.to_le_bytes().to_vec(), f64::NAN.to_bits().to_le_bytes().to_vec()].concat()].concat(),
			};

			let config_buffer = interpolator.get_device().create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("content"), contents: &config, usage: wgpu::BufferUsages::UNIFORM });

			let batch_results: Vec<f64> = interpolator.interpolate_f64(&input_times, &input_values, target_batch, method, &config_buffer)?;
			let points = helpers::convert_gpu_results_to_points_f64(batch_results, &batch_times[..batch_size.min(target_batch.len())]);

			if available_memory < memory_threshold && temp_file.is_none() {
				let temp = NamedTempFile::new().map_err(|e| Error::IOError(e.to_string()))?;
				temp_path = Some(temp.into_temp_path());
				temp_file = Some(BufWriter::new(File::create(temp_path.as_ref().unwrap()).map_err(|e| Error::IOError(e.to_string()))?));
			}

			if let Some(writer) = temp_file.as_mut() {
				for point in points {
					writeln!(writer, "{},{}", point.timestamp.to_rfc3339(), point.value)?;
				}
			} else {
				batch.extend_from_slice(&points);
			}
		}

		if temp_file.is_none() {
			result.extend(batch.into_iter());
		}
		if let Some(writer) = temp_file.as_mut() {
			writer.flush().map_err(|e| Error::IOError(e.to_string()))?;
		}
	}

	if let Some(path) = &temp_path {
		let file = File::open(path).map_err(|e| Error::IOError(e.to_string()))?;
		let reader = BufReader::new(file);
		for line in reader.lines() {
			let line = line.map_err(|e| Error::IOError(e.to_string()))?;
			let parts: Vec<&str> = line.split(',').collect();
			if parts.len() == 2 {
				let timestamp = DateTime::parse_from_rfc3339(parts[0]).map_err(|e| Error::IOError(format!("Failed to parse timestamp '{}': {}", parts[0], e)))?.with_timezone(&Utc);
				let value = parts[1].parse::<f64>().map_err(|e| Error::IOError(format!("Failed to parse value '{}': {}", parts[1], e)))?;
				let value = BigDecimal::from_f64(value).unwrap_or_default();
				result.push(Point { timestamp, value });
			}
		}
	}

	Ok(result)
}

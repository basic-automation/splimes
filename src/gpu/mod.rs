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
use wgpu::util::DeviceExt;

use crate::{
	Error, POINT_SIZE, Point, Resolution, Result, Spline, helpers::{InterpolationState, TargetTimesIterator, batch}
};

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
	// Use static method to check f64 support instead of creating new instance
	if GpuInterpolator::supports_f64_static()? { gpu_interpolate_f64(points, &start, &end, &resolution, &method, &spline).await } else { batch(points, &start, &end, &spline, &resolution, |state| Box::pin(gpu_interpolate_f32(state))).await }
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

	// Remove the local interpolator creation
	let mut b = Vec::with_capacity(batch_size);
	let mut time_offset = 0usize;

	for batch in batch_times.chunks(batch_size) {
		let target_times_array = helpers::convert_datetimes_to_gpu_format_f32(batch, state.resolution, input_points[0].timestamp)?;
		let m_b = get_max_buffer_size().await?;
		let m_b = usize::from_u64(m_b).ok_or_else(|| Error::ConversionError("Failed to convert buffer size to usize".to_string()))?;
		let max_single_buffer_size = m_b / NUMBER_OF_BUFFERS;
		let max_targets_per_batch = (max_single_buffer_size / F64_SIZE).min(target_times_array.len());

		for target_batch in target_times_array.chunks(max_targets_per_batch) {
			let (input_times, input_values) = helpers::convert_points_to_gpu_format_f32(input_points, state.resolution)?;
			let config = match method {
				Method::Linear => [time_offset as u32, 1, 0, 0],
				Method::Quadratic => [time_offset as u32, 2, 0, 0],
				Method::Cubic => [time_offset as u32, 3, 0, 0],
				Method::Polynomial(degree) => {
					let bounds_factor = match state.spline {
						Spline::Polynomial(_, Some(factor)) => factor as f32,
						_ => f32::NAN,
					};
					[time_offset as u32, degree as u32, bounds_factor.to_bits(), 0]
				}
			};
			// Use static method instead of instance method - convert config to bytes
			let config_bytes = bytemuck::cast_slice(&config);
			let config_buffer = GpuInterpolator::get_device_static()?.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("content"), contents: config_bytes, usage: wgpu::BufferUsages::UNIFORM });
			// Use static interpolation method
			let results: Vec<f32> = GpuInterpolator::interpolate_f32_static(&input_times, &input_values, target_batch, &method, &config_buffer)?;
			let target_slice = &batch[time_offset..time_offset + target_batch.len()];
			let batch_results = helpers::convert_gpu_results_to_points_f32(results, target_slice);
			b.extend(batch_results);
			time_offset += target_batch.len();
		}
		time_offset = 0;
	}
	if let Some(result) = &mut state.result {
		result.extend(b);
	} else if let Some(writer_mutex) = &state.temp_file {
		let mut writer = writer_mutex.lock().await;
		for point in &b {
			writeln!(writer, "{},{}", point.timestamp.timestamp_millis(), point.value)?;
		}
		writer.flush().map_err(|e| Error::IOError(e.to_string()))?;
	} else if !b.is_empty() {
		bail!("No output destination available");
	}

	Ok(())
}

// Remove the interpolator parameter since we're using the static global one
pub async fn gpu_interpolate_f64(points: &[Point], start: &DateTime<Utc>, end: &DateTime<Utc>, resolution: &Resolution, method: &Method, spline: &Spline) -> Result<Vec<Point>> {
	if points.len() < 2 {
		bail!("Insufficient points for interpolation");
	}
	if start >= end {
		bail!("Invalid time range");
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
		system.refresh_memory();
		let estimated_memory = estimated_points as u64 * point_size;
		if estimated_memory > memory_threshold {
			let temp = NamedTempFile::new().map_err(|e| Error::IOError(e.to_string()))?;
			let (file, path) = temp.into_parts();
			temp_file = Some(BufWriter::new(file));
			temp_path = Some(path);
		}
	}
	for batch_times in time_iter {
		let batch_size = batch_times.len();
		if batch_size == 0 {
			continue;
		}

		let target_times_array = helpers::convert_datetimes_to_gpu_format_f64(&batch_times, *resolution, base_time)?;
		let max_buffer_size = get_max_buffer_size().await?;
		let max_buffer_size_usize = usize::try_from(max_buffer_size)?;
		let max_single_buffer_size = max_buffer_size_usize / NUMBER_OF_BUFFERS;
		let max_targets_per_batch = (max_single_buffer_size / element_size).min(target_times_array.len());

		let mut time_offset = 0usize;
		for target_batch in target_times_array.chunks(max_targets_per_batch) {
			let config = match method {
				Method::Linear => [time_offset as u32, 1, 0, 0],
				Method::Quadratic => [time_offset as u32, 2, 0, 0],
				Method::Cubic => [time_offset as u32, 3, 0, 0],
				Method::Polynomial(degree) => {
					let bounds_factor = match spline {
						Spline::Polynomial(_, Some(factor)) => *factor,
						_ => f64::NAN,
					};
					[time_offset as u32, *degree as u32, bounds_factor.to_bits() as u32, 0]
				}
			};
			// Use static method instead of instance method - convert config to bytes
			let config_bytes = bytemuck::cast_slice(&config);
			let config_buffer = GpuInterpolator::get_device_static()?.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("content"), contents: config_bytes, usage: wgpu::BufferUsages::UNIFORM });
			// Use static interpolation method
			let batch_results: Vec<f64> = GpuInterpolator::interpolate_f64_static(&input_times, &input_values, target_batch, method, &config_buffer)?;
			let target_slice = &batch_times[time_offset..time_offset + target_batch.len()];
			let batch_points = helpers::convert_gpu_results_to_points_f64(batch_results, target_slice);

			if let Some(writer) = temp_file.as_mut() {
				for point in &batch_points {
					writeln!(writer, "{},{}", point.timestamp.timestamp_millis(), point.value)?;
				}
			} else {
				result.extend(batch_points);
			}
			time_offset += target_batch.len();
		}
	}

	if let Some(path) = &temp_path {
		drop(temp_file);
		let file = std::fs::File::open(path).map_err(|e| Error::IOError(e.to_string()))?;
		let reader = BufReader::new(file);
		for line in reader.lines() {
			let line = line.map_err(|e| Error::IOError(e.to_string()))?;
			let parts: Vec<&str> = line.split(',').collect();
			if parts.len() == 2 {
				let timestamp = parts[0].parse::<i64>().map_err(|e| Error::ConversionError(e.to_string()))?;
				let value = BigDecimal::from_f64(parts[1].parse::<f64>().map_err(|e| Error::ConversionError(e.to_string()))?).ok_or_else(|| Error::ConversionError("Failed to convert to BigDecimal".to_string()))?;
				result.push(Point { timestamp: DateTime::from_timestamp_millis(timestamp).ok_or_else(|| Error::ConversionError("Invalid timestamp".to_string()))?, value });
			}
		}
	}

	Ok(result)
}

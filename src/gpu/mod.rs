use std::{
	fs::File, io::{BufRead, BufWriter, Write}
};

use anyhow::{bail, Context, Result};
use chrono::{DateTime, Utc};
use sysinfo::System;
use tempfile::NamedTempFile;
use wgpu::util::DeviceExt;

use crate::{
	gpu::{types::GpuInterpolator, Method}, helpers::{batch, InterpolationState, TargetTimesIterator}, Error, Point, Resolution, Spline
};

mod helpers;
mod shaders;
pub mod types;
pub mod buffer_pool;
pub mod staging_buffer_manager;
pub mod async_handle;
pub mod config;

pub use types::*;
pub use buffer_pool::BufferPoolStats;
pub use staging_buffer_manager::StagingBufferManager;
pub use config::GpuConfig;

const F64_SIZE: usize = std::mem::size_of::<f64>();
const NUMBER_OF_BUFFERS: usize = 4;

/// Pre-warms the GPU interpolator to eliminate first-use latency.
///
/// # Errors
/// Returns the initialization error if GPU setup fails.
pub fn force_init_gpu() -> Result<()> {
	GpuInterpolator::force_init()
}

/// Performs GPU-accelerated interpolation on data points
///
/// # Errors
/// Returns an error if:
/// - The spline fails validation checks
/// - GPU device cannot be accessed or initialized
/// - Memory allocation fails
/// - GPU operations fail
/// - Buffer operations fail
pub async fn gpu_interpolate(points: &mut [Point], start: DateTime<Utc>, end: DateTime<Utc>, resolution: Resolution, spline: Spline) -> Result<Vec<Point>> {
	spline.pre_check(points, &start, &end)?;
	let method = match spline {
		Spline::Linear => Method::Linear,
		Spline::Quadratic => Method::Quadratic,
		Spline::Cubic => Method::Cubic,
		Spline::Polynomial(degree, _) => Method::Polynomial(points.len().min(degree + 1)),
	};
	// Use static method to check f64 support instead of creating new instance
	if GpuInterpolator::supports_f64_static()? {
		gpu_interpolate_f64(points, &start, &end, resolution, &method, &spline)
	} else {
		batch(points, &start, &end, &spline, &resolution, |state| Box::pin(gpu_interpolate_f32(state))).await
	}
}

/// Performs f32 GPU interpolation for batched processing
///
/// # Errors
/// Returns an error if:
/// - Input points or batch times are not provided
/// - Memory allocation fails
/// - GPU operations fail
/// - Buffer size conversions fail
/// - File I/O operations fail
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
	let available_memory = usize::try_from(state.system.available_memory()).context("Failed to convert available memory to usize")?;
	let batch_size = if available_memory < state.memory_threshold {
		let max_points = (available_memory / crate::POINT_SIZE).max(50);
		batch_times.len().min(max_points / 2)
	} else {
		batch_times.len()
	};

	// Remove the local interpolator creation
	let mut b = Vec::with_capacity(batch_size);
	#[allow(unused_mut)] // Used within the loop even if compiler thinks otherwise
	let mut time_offset = 0usize;

	// Fix 5: Move F32 conversion OUTSIDE the batch loop - input_points doesn't change
	let (input_times, input_values) = helpers::convert_points_to_gpu_format_f32(input_points, state.resolution)?;

	// FIX: Get max buffer size ONCE before the loop (was creating new GPU instance per iteration!)
	let max_buffer_size = GpuInterpolator::get_max_buffer_size_static()?;
	let max_single_buffer_size = max_buffer_size / NUMBER_OF_BUFFERS;

	for batch in batch_times.chunks(batch_size) {
		let target_times_array = helpers::convert_datetimes_to_gpu_format_f32(batch, state.resolution, input_points[0].timestamp)?;
		let max_targets_per_batch = (max_single_buffer_size / F64_SIZE).min(target_times_array.len());

		for target_batch in target_times_array.chunks(max_targets_per_batch) {
			let config = match method {
				Method::Linear => [u32::try_from(time_offset).context("time_offset exceeds u32 limit")?, 1, 0, 0],
				Method::Quadratic => [u32::try_from(time_offset).context("time_offset exceeds u32 limit")?, 2, 0, 0],
				Method::Cubic => [u32::try_from(time_offset).context("time_offset exceeds u32 limit")?, 3, 0, 0],
				Method::Polynomial(degree) => {
					#[allow(clippy::cast_possible_truncation)] // f64 to f32 conversion for GPU compatibility
					let bounds_factor = match state.spline {
						Spline::Polynomial(_, Some(factor)) => factor as f32,
						_ => f32::NAN,
					};
					[u32::try_from(time_offset).context("time_offset exceeds u32 limit")?, u32::try_from(degree).context("degree exceeds u32 limit")?, bounds_factor.to_bits(), 0]
				}
			};
			// Use create_buffer_init for config (small buffer, not worth pooling)
			let config_bytes = bytemuck::cast_slice(&config);
			let config_buffer = GpuInterpolator::get_device_static()?.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("Config Buffer"), contents: config_bytes, usage: wgpu::BufferUsages::UNIFORM });

			// Use static interpolation method with reused input data
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
			writeln!(writer, "{},{}", point.timestamp.to_rfc3339(), point.value)?;
		}
		writer.flush().map_err(|e| Error::IOError(e.to_string()))?;
	}
	Ok(())
}

// Remove the interpolator parameter since we're using the static global one
/// Performs f64 GPU interpolation for high-precision operations
///
/// # Errors
/// Returns an error if:
/// - Insufficient points for interpolation (< 2)
/// - Invalid time range (start >= end)
/// - Memory threshold calculations fail
/// - GPU operations fail
/// - Buffer operations fail
/// - File I/O operations fail
pub fn gpu_interpolate_f64(points: &[Point], start: &DateTime<Utc>, end: &DateTime<Utc>, resolution: Resolution, method: &Method, spline: &Spline) -> Result<Vec<Point>> {
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
	// For memory calculations, we need to use direct casting with proper bounds checking
	#[allow(clippy::cast_precision_loss)] // Memory calculations can safely lose precision for threshold purposes
	let t_m = total_memory as f64;
	#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // Memory threshold calculation - safe for positive values
	let memory_threshold = (t_m * 0.8) as u64;
	let point_size = std::mem::size_of::<Point>() as u64;

	let element_size = F64_SIZE;
	let (input_times, input_values) = helpers::convert_points_to_gpu_format_f64(points, resolution)?;
	let base_time = points[0].timestamp;

	let rounded_start = resolution.round(start)?;
	let rounded_end = resolution.round(end)?;
	let time_iter = TargetTimesIterator::new(rounded_start, rounded_end, resolution);
	if let Ok(estimated_points) = time_iter.estimate_len() {
		system.refresh_memory();
		let estimated_memory = u64::try_from(estimated_points).context("Failed to convert estimated_points to u64")? * point_size;
		if estimated_memory > memory_threshold {
			let temp = NamedTempFile::new().map_err(|e| Error::IOError(e.to_string()))?;
			let (file, path) = temp.into_parts();
			temp_file = Some(BufWriter::new(file));
			temp_path = Some(path);
		}
	}

	// FIX: Get max buffer size ONCE before the loop (was creating new GPU instance per iteration!)
	let max_buffer_size = GpuInterpolator::get_max_buffer_size_static()?;
	let max_single_buffer_size = max_buffer_size / NUMBER_OF_BUFFERS;

	for batch_times in time_iter {
		let batch_size = batch_times.len();
		if batch_size == 0 {
			continue;
		}

		let target_times_array = helpers::convert_datetimes_to_gpu_format_f64(&batch_times, resolution, base_time)?;
		let max_targets_per_batch = (max_single_buffer_size / element_size).min(target_times_array.len());

		let mut time_offset = 0usize;
		for target_batch in target_times_array.chunks(max_targets_per_batch) {
			let config = match method {
				Method::Linear => [u32::try_from(time_offset).context("time_offset exceeds u32 limit")?, 1, 0, 0],
				Method::Quadratic => [u32::try_from(time_offset).context("time_offset exceeds u32 limit")?, 2, 0, 0],
				Method::Cubic => [u32::try_from(time_offset).context("time_offset exceeds u32 limit")?, 3, 0, 0],
				Method::Polynomial(degree) => {
					let bounds_factor = match spline {
						Spline::Polynomial(_, Some(factor)) => *factor,
						_ => f64::NAN,
					};
					// Split the f64 bits into two u32 parts for the shader
					let bits = bounds_factor.to_bits();
					let bounds_factor_bits_low = u32::try_from(bits & 0xFFFF_FFFF).context("bounds_factor low bits exceed u32 limit")?;
					let bounds_factor_bits_high = u32::try_from(bits >> 32).context("bounds_factor high bits exceed u32 limit")?;
					[u32::try_from(time_offset).context("time_offset exceeds u32 limit")?, u32::try_from(*degree).context("degree exceeds u32 limit")?, bounds_factor_bits_low, bounds_factor_bits_high]
				}
			};
			// Use create_buffer_init for config (small buffer, not worth pooling)
			let config_bytes = bytemuck::cast_slice(&config);
			let config_buffer = GpuInterpolator::get_device_static()?.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("Config Buffer"), contents: config_bytes, usage: wgpu::BufferUsages::UNIFORM });

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

	if let Some(mut writer) = temp_file {
		writer.flush().map_err(|e| Error::IOError(e.to_string()))?;
	}

	if let Some(temp_path) = temp_path {
		// Read back the results from temp file
		let file = std::fs::File::open(&temp_path).map_err(|e| Error::IOError(e.to_string()))?;
		let reader = std::io::BufReader::new(file);
		for line in reader.lines() {
			let line = line.map_err(|e| Error::IOError(e.to_string()))?;
			let parts: Vec<&str> = line.split(',').collect();
			if parts.len() == 2 {
				let timestamp_millis = parts[0].parse::<i64>().map_err(|e| Error::IOError(e.to_string()))?;
				let timestamp = DateTime::from_timestamp_millis(timestamp_millis).ok_or_else(|| Error::IOError("Invalid timestamp".to_string()))?.with_timezone(&chrono::Utc);
				let value_f64 = parts[1].parse::<f64>().map_err(|e| Error::IOError(e.to_string()))?;
				let value = bigdecimal::BigDecimal::try_from(value_f64).map_err(|e| Error::IOError(e.to_string()))?;
				result.push(Point { timestamp, value });
			}
		}
		std::fs::remove_file(temp_path).unwrap_or(());
	}

	result.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
	Ok(result)
}

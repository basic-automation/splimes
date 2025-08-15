// Fixed version of batch.rs - the key changes are in result accumulation

use std::{
	fs::File, io::{BufRead, BufReader, BufWriter, Write}
};

use anyhow::Result;
use bigdecimal::FromPrimitive;
use chrono::{DateTime, Utc};
use sysinfo::System;
use tempfile::{NamedTempFile, TempPath};
use tokio::sync::Mutex;

use super::TargetTimesIterator;
use crate::{Error, Point, Resolution, Spline};

#[derive(Debug)]
pub struct InterpolationState {
	pub result: Option<Vec<Point>>,
	pub temp_path: Option<TempPath>,
	pub temp_file: Option<Mutex<BufWriter<File>>>,
	pub memory_threshold: usize,
	pub input_points: Option<Vec<Point>>,
	pub resolution: Resolution,
	pub system: System,
	pub batch_times: Option<Vec<DateTime<Utc>>>,
	pub spline: Spline,
}

/// Processes interpolation in batches with memory management and optional file-based storage
///
/// This function handles large datasets by processing them in manageable chunks, automatically
/// switching between in-memory and file-based storage based on memory availability.
///
/// # Arguments
/// * `points` - Input data points for interpolation (will be sorted by timestamp)
/// * `start` - Start time for interpolation range (will be rounded to resolution)
/// * `end` - End time for interpolation range (will be rounded to resolution)
/// * `spline` - Spline type to use for interpolation
/// * `resolution` - Time resolution for target points
/// * `f` - Interpolation function to apply to each batch
///
/// # Returns
/// Vector of interpolated points covering the specified time range
///
/// # Errors
/// Returns an error if:
/// - Time resolution rounding fails
/// - Memory threshold calculations fail
/// - Temporary file creation fails (when memory threshold exceeded)
/// - Interpolation function `f` returns an error
/// - File I/O operations fail during batch processing
/// - Target time generation fails
/// - Result consolidation from temporary files fails
/// - Batch times are empty when they shouldn't be
///
/// # Panics
/// Panics if:
/// - `batch_times.first()` is called on an empty batch (should not occur due to iterator design)
/// - Memory conversion operations fail unexpectedly
pub async fn batch<F>(points: &mut [Point], start: &DateTime<Utc>, end: &DateTime<Utc>, spline: &Spline, resolution: &Resolution, f: F) -> Result<Vec<Point>>
where
	F: for<'a> Fn(&'a mut InterpolationState) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + 'a>> + Send + Sync,
{
	let sorted_points = points;
	sorted_points.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
	let rounded_start = resolution.round(start)?;
	let rounded_end = resolution.round(end)?;
	let system = System::new_all();
	let mut result = Vec::new();
	let total_memory = system.total_memory();
	let t_m: f64 = f64::from_u64(total_memory).ok_or_else(|| Error::ConversionError("Failed to convert total memory".to_string()))?;
	let memory_threshold: usize = usize::from_f64(t_m * 0.8).ok_or_else(|| Error::ConversionError("Failed to convert memory threshold".to_string()))?;
	let time_iter = TargetTimesIterator::new(rounded_start, rounded_end, *resolution);
	let estimated_points = time_iter.estimate_len().unwrap_or(0);
	if estimated_points == 0 {
		return Ok(result);
	}

	result.reserve(estimated_points);

	// Always use in-memory processing for small datasets to avoid temp file overhead
	if estimated_points > 1_000_000 {
		let temp = NamedTempFile::new().map_err(|e| Error::IOError(e.to_string()))?;
		let temp_path = temp.into_temp_path();
		let temp_file = BufWriter::new(File::create(&temp_path).map_err(|e| Error::IOError(e.to_string()))?);

		let mut state = InterpolationState { result: None, temp_path: Some(temp_path), temp_file: Some(Mutex::new(temp_file)), memory_threshold, input_points: None, resolution: *resolution, system, batch_times: None, spline: *spline };

		for batch_times in time_iter {
			if batch_times.is_empty() {
				continue;
			}

			let batch_start = batch_times.first().ok_or_else(|| Error::ConversionError("Empty batch times".to_string()))?;
			let batch_end = batch_times.last().ok_or_else(|| Error::ConversionError("Empty batch times".to_string()))?;

			let start_idx = sorted_points.iter().position(|p| p.timestamp <= *batch_start).unwrap_or(0);
			let end_idx = sorted_points.iter().rposition(|p| p.timestamp >= *batch_end).unwrap_or(sorted_points.len() - 1);

			let input_start = start_idx.saturating_sub(1);
			let input_end = (end_idx + 1).min(sorted_points.len() - 1);

			state.input_points = Some(sorted_points[input_start..=input_end].to_vec());
			state.batch_times = Some(batch_times.clone());

			// Clear previous result to avoid accumulation
			state.result = None;

			f(&mut state).await.inspect_err(|_| {
				if let Some(ref temp_path) = state.temp_path {
					let _ = std::fs::remove_file(temp_path);
				}
			})?;

			// Flush temp file after each batch
			if let Some(ref mut temp_file) = state.temp_file {
				let mut temp_file = temp_file.lock().await;
				temp_file.flush().map_err(|e| Error::IOError(e.to_string()))?;
			}
		}

		// Read all results from temp file at the end
		if let Some(ref temp_path) = state.temp_path {
			let file = File::open(temp_path).map_err(|e| Error::IOError(e.to_string()))?;
			let reader = BufReader::new(file);
			for line in reader.lines() {
				let line = line.map_err(|e| Error::IOError(e.to_string()))?;
				let parts: Vec<&str> = line.split(',').collect();
				if parts.len() == 2 {
					let timestamp = DateTime::parse_from_rfc3339(parts[0]).map_err(|e| Error::IOError(e.to_string()))?.with_timezone(&Utc);
					let value_f64 = parts[1].parse::<f64>().map_err(|e| Error::IOError(e.to_string()))?;
					let value = bigdecimal::BigDecimal::from_f64(value_f64).ok_or_else(|| Error::ConversionError("Failed to convert f64 to BigDecimal".to_string()))?;
					result.push(Point { timestamp, value });
				}
			}
			let _ = std::fs::remove_file(temp_path);
		}
	} else {
		// In-memory processing for smaller datasets
		let mut state = InterpolationState { result: None, temp_path: None, temp_file: None, memory_threshold, input_points: None, resolution: *resolution, system, batch_times: None, spline: *spline };

		for batch_times in time_iter {
			if batch_times.is_empty() {
				continue;
			}

			let batch_start = batch_times.first().ok_or_else(|| Error::ConversionError("Empty batch times".to_string()))?;
			let batch_end = batch_times.last().ok_or_else(|| Error::ConversionError("Empty batch times".to_string()))?;

			let start_idx = sorted_points.iter().position(|p| p.timestamp <= *batch_start).unwrap_or(0);
			let end_idx = sorted_points.iter().rposition(|p| p.timestamp >= *batch_end).unwrap_or(sorted_points.len() - 1);

			let input_start = start_idx.saturating_sub(1);
			let input_end = (end_idx + 1).min(sorted_points.len() - 1);

			state.input_points = Some(sorted_points[input_start..=input_end].to_vec());
			state.batch_times = Some(batch_times.clone());

			// Clear previous result to avoid accumulation
			state.result = None;

			f(&mut state).await.inspect_err(|_| {
				if let Some(ref temp_path) = state.temp_path {
					let _ = std::fs::remove_file(temp_path);
				}
			})?;

			// FIXED: Always check for in-memory results first
			if let Some(res) = &state.result {
				result.extend(res.clone());
			}
		}
	}

	result.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
	Ok(result)
}

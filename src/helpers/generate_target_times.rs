use std::mem;

use anyhow::Result;
use chrono::{DateTime, Utc};
use sysinfo::System;

use crate::Resolution;

/// Iterator for generating target timestamps with memory management
pub struct TargetTimesIterator {
	current: DateTime<Utc>,
	end: DateTime<Utc>,
	resolution: Resolution,
	system: System,
}

impl TargetTimesIterator {
	#[must_use]
	pub fn new(start: DateTime<Utc>, end: DateTime<Utc>, resolution: Resolution) -> Self {
		Self { current: start, end, resolution, system: System::new_all() }
	}

	/// Estimates the total number of timestamps that will be generated
	///
	/// # Errors
	/// Returns an error if duration calculation fails or conversion to usize fails
	pub fn estimate_len(&self) -> Result<usize> {
		let duration = self.end.signed_duration_since(self.current);
		let total_seconds = duration.num_seconds();

		let interval_seconds = match self.resolution {
			Resolution::Nanoseconds => return Ok(usize::try_from(duration.num_nanoseconds().unwrap_or(0))?),
			Resolution::Microseconds => return Ok(usize::try_from(duration.num_microseconds().unwrap_or(0))?),
			Resolution::Milliseconds => return Ok(usize::try_from(duration.num_milliseconds())?),
			Resolution::Seconds => 1,
			Resolution::Minutes => 60,
			Resolution::Hours => 3600,
			Resolution::Days => 86400,
			Resolution::Weeks => 604_800,
			Resolution::Months => 2_629_746, // Average month
			Resolution::Years => 31_556_952, // Average year
		};

		Ok(usize::try_from(total_seconds / interval_seconds + 1)?)
	}

	fn next_impl(&mut self) -> Option<Vec<DateTime<Utc>>> {
		if self.current > self.end {
			return None;
		}

		self.system.refresh_memory();
		let available_memory = self.system.free_memory().max(self.system.available_memory()); // In bytes
		let m_t = self.system.total_memory();
		#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::cast_sign_loss)]
		let memory_threshold = (m_t as f64 * 0.8) as u64;
		let point_size = mem::size_of::<DateTime<Utc>>() as u64; // ~16 bytes

		// Adjust batch size: scale down if available memory is low
		let batch_size = {
			let max_memory = available_memory.min(memory_threshold);
			let max_points = max_memory / point_size;
			(max_points / 2).max(100) as usize // Ensure minimum batch size
		};

		let mut batch = Vec::with_capacity(batch_size);
		while batch.len() < batch_size && self.current <= self.end {
			batch.push(self.current);
			self.current += self.resolution.to_step();
		}

		if batch.is_empty() { None } else { Some(batch) }
	}
}

impl Iterator for TargetTimesIterator {
	type Item = Vec<DateTime<Utc>>;

	fn next(&mut self) -> Option<Self::Item> {
		self.next_impl()
	}
}

/// Generates a vector of target timestamps for interpolation
///
/// This function creates evenly spaced timestamps between start and end
/// at the specified resolution. It's useful for creating interpolation targets.
///
/// # Errors
/// This function is infallible and will not return errors, but the signature
/// includes Result for consistency with other interpolation functions.
#[must_use]
pub fn generate_target_times(start: DateTime<Utc>, end: DateTime<Utc>, resolution: Resolution) -> Vec<DateTime<Utc>> {
	let mut times = Vec::new();
	let mut current = start;

	while current <= end {
		times.push(current);
		current += resolution.to_step();
	}

	times
}

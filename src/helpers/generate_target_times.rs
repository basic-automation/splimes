use std::mem;

use anyhow::{Context, Result};
use bigdecimal::FromPrimitive;
use chrono::{DateTime, Duration, Utc};
use sysinfo::System;

use crate::Resolution;

// Iterator for paginated target times with dynamic batch size
pub struct TargetTimesIterator {
	current: DateTime<Utc>,
	end: DateTime<Utc>,
	step: Duration,
	system: System,
	resolution: Resolution,
}

impl TargetTimesIterator {
	pub fn new(start: DateTime<Utc>, end: DateTime<Utc>, resolution: Resolution) -> Self {
		let step = resolution.to_step();

		Self { current: start, end, step, system: System::new_all(), resolution }
	}

	// Estimate the total number of target times without generating them
	pub fn estimate_len(&self) -> Result<usize> {
		if self.current > self.end {
			return Ok(0);
		}

		let time_diff = self.resolution.difference(&self.end, &self.current)?;
		let step_base = self.resolution.to_step_base()?;
		let res = time_diff + step_base - 1;
		let res = res / step_base;
		let res: usize = usize::from_i64(res).context("Failed to convert estimated length to usize")?;
		Ok(res + 1)
	}
}

impl Iterator for TargetTimesIterator {
	type Item = Vec<DateTime<Utc>>;

	fn next(&mut self) -> Option<Self::Item> {
		if self.current > self.end {
			return None;
		}

		self.system.refresh_memory();
		let available_memory = self.system.free_memory().max(self.system.available_memory()); // In bytes
		let m_t = f64::from_u64(self.system.total_memory()).context("Failed to convert total memory").ok()?;
		let memory_threshold = u64::from_f64(m_t * 0.8).context("Failed to convert memory threshold").ok()?;
		let point_size = mem::size_of::<DateTime<Utc>>() as u64; // ~16 bytes

		// Adjust batch size: scale down if available memory is low
		let batch_size = {
			let max_memory = available_memory.min(memory_threshold);
			let max_points = max_memory / point_size;
			(max_points / 2).max(100) as usize // Ensure minimum batch size
		};

		let mut batch = Vec::with_capacity(batch_size);
		for _ in 0..batch_size {
			if self.current > self.end {
				break;
			}
			batch.push(self.current);
			self.current += self.step;
		}

		if batch.is_empty() { None } else { Some(batch) }
	}
}

// Wrapper function to maintain original signature
pub fn generate_target_times(start: DateTime<Utc>, end: DateTime<Utc>, resolution: Resolution) -> Vec<DateTime<Utc>> {
	TargetTimesIterator::new(start, end, resolution).flatten().collect()
}

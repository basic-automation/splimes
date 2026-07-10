use std::mem;

use anyhow::Result;
use chrono::{DateTime, Utc};
use sysinfo::System;

use crate::Resolution;

/// Upper bound on the number of timestamps a single [`TargetTimesIterator`] batch will
/// pre-allocate, regardless of how much system memory is free.
///
/// The batch size is otherwise derived from available memory (so a big batch is used when
/// there is headroom), but that figure alone sized the batch `Vec` to a *fraction of free
/// RAM* — tens of gigabytes on a workstation — rather than to the data. A single such
/// speculative allocation succeeds when RAM is idle (it is only ever filled up to the
/// actual grid size, then dropped), but several concurrent callers each requesting tens of
/// GiB sum past total RAM and abort the process. Capping at ~1 M timestamps (≈12–16 MiB)
/// keeps batching effective for genuinely huge ranges while making the per-batch
/// allocation proportional to the work, never to the machine.
const MAX_BATCH_POINTS: usize = 1 << 20;

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

	/// The number of timestamps that remain to be produced from `current` through `end`
	/// (inclusive) at the configured resolution — a saturating `u64` (never panics, never
	/// wraps). This bounds the per-batch pre-allocation so a batch is never sized larger
	/// than the timestamps that actually remain.
	fn remaining_points(&self) -> u64 {
		let duration = self.end.signed_duration_since(self.current);
		let interval_seconds = match self.resolution {
			#[allow(clippy::cast_sign_loss)] // clamped to >= 0 before the cast.
			Resolution::Nanoseconds => return duration.num_nanoseconds().unwrap_or(i64::MAX).max(0) as u64 + 1,
			#[allow(clippy::cast_sign_loss)]
			Resolution::Microseconds => return duration.num_microseconds().unwrap_or(i64::MAX).max(0) as u64 + 1,
			#[allow(clippy::cast_sign_loss)]
			Resolution::Milliseconds => return duration.num_milliseconds().max(0) as u64 + 1,
			Resolution::Seconds => 1,
			Resolution::Minutes => 60,
			Resolution::Hours => 3600,
			Resolution::Days => 86400,
			Resolution::Weeks => 604_800,
			Resolution::Months => 2_629_746,
			Resolution::Years => 31_556_952,
		};
		#[allow(clippy::cast_sign_loss)] // clamped to >= 0 before the cast.
		let total_seconds = duration.num_seconds().max(0) as u64;
		total_seconds / interval_seconds + 1
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

		// Batch size scales with available memory, but is capped by (a) the timestamps that
		// actually remain and (b) an absolute maximum — the memory figure alone sized the
		// `Vec` to a fraction of *free RAM* (tens of GiB), which under concurrent callers
		// aborted the process (see `MAX_BATCH_POINTS`). The allocation is now proportional
		// to the work, never to the machine.
		let batch_size = {
			let max_memory = available_memory.min(memory_threshold);
			let memory_points = (max_memory / point_size / 2).max(100);
			let bounded = memory_points.min(self.remaining_points()).min(MAX_BATCH_POINTS as u64);
			usize::try_from(bounded).unwrap_or(MAX_BATCH_POINTS).max(1)
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

#[cfg(test)]
mod tests {
	use chrono::TimeZone;

	use super::*;

	fn utc(secs: i64) -> DateTime<Utc> {
		Utc.timestamp_opt(secs, 0).single().expect("valid timestamp")
	}

	#[test]
	fn batch_capacity_tracks_the_data_not_free_ram() {
		// Regression for the parallel-only OOM: a tiny 11-second range must pre-allocate a
		// tiny batch. Before the fix `next_impl` sized the batch `Vec` to a fraction of free
		// RAM (~tens of GiB on a workstation), so concurrent callers aborted the process.
		let mut iter = TargetTimesIterator::new(utc(0), utc(10), Resolution::Seconds);
		let batch = iter.next().expect("first batch is non-empty");
		assert_eq!(batch.len(), 11, "11 inclusive one-second steps from 0..=10");
		assert!(batch.capacity() <= MAX_BATCH_POINTS, "capacity {} must never exceed the absolute cap", batch.capacity());
		// The whole grid fits one batch here, so capacity is bounded by the 11 remaining
		// points — proof it is sized by the data, not by free memory.
		assert!(batch.capacity() <= 64, "capacity {} must track the ~11 remaining points, not free RAM", batch.capacity());
	}

	#[test]
	fn iterator_reproduces_the_reference_grid() {
		// The batched iterator, flattened, must equal the simple reference generator across
		// several batches (the cap forces multiple batches only for huge ranges, so this
		// stays a correctness check that batching does not drop or duplicate a timestamp).
		let (start, end, res) = (utc(1_000), utc(1_600), Resolution::Seconds);
		let batched: Vec<DateTime<Utc>> = TargetTimesIterator::new(start, end, res).flatten().collect();
		assert_eq!(batched, generate_target_times(start, end, res));
	}

	#[test]
	fn remaining_points_matches_estimate_len() {
		// The new saturating `remaining_points` bound must agree with the existing
		// `estimate_len` on a normal range, so the batch cap is the true remaining count.
		let iter = TargetTimesIterator::new(utc(0), utc(3600), Resolution::Minutes);
		assert_eq!(iter.remaining_points(), iter.estimate_len().expect("estimates") as u64);
	}

	#[test]
	fn reversed_range_yields_no_timestamps() {
		// end before start: no points, and `remaining_points` saturates at 1 (the anchor)
		// rather than underflowing into a giant unsigned count.
		let mut iter = TargetTimesIterator::new(utc(100), utc(0), Resolution::Seconds);
		assert!(iter.next().is_none(), "a reversed range produces nothing");
		assert_eq!(TargetTimesIterator::new(utc(100), utc(0), Resolution::Seconds).remaining_points(), 1);
	}
}

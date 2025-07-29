use chrono::{DateTime, Utc};

use crate::Resolution;

/// Estimate output points based on time range and resolution (make public)
#[must_use]
pub fn estimate_output_points(start: DateTime<Utc>, end: DateTime<Utc>, resolution: Resolution) -> usize {
	let duration = end.signed_duration_since(start);

	let total_nanoseconds = duration.num_nanoseconds().unwrap_or(0).max(0); // Ensure non-negative

	#[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
	let total_nanoseconds_usize = total_nanoseconds as usize; // Safe after max(0) check

	match resolution {
		Resolution::Nanoseconds => total_nanoseconds_usize,
		Resolution::Microseconds => total_nanoseconds_usize / 1_000,
		Resolution::Milliseconds => total_nanoseconds_usize / 1_000_000,
		Resolution::Seconds => total_nanoseconds_usize / 1_000_000_000,
		Resolution::Minutes => total_nanoseconds_usize / 60_000_000_000,
		Resolution::Hours => total_nanoseconds_usize / 3_600_000_000_000,
		Resolution::Days => total_nanoseconds_usize / 86_400_000_000_000,
		Resolution::Weeks => total_nanoseconds_usize / 604_800_000_000_000,    // 7 days
		Resolution::Months => total_nanoseconds_usize / 2_592_000_000_000_000, // 30 days
		Resolution::Years => total_nanoseconds_usize / 31_536_000_000_000_000, // 365 days
	}
}

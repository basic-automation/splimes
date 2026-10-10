use std::{fmt, str::FromStr};

use chrono::TimeDelta;

use crate::Error;

/// The spacing of the output grid.
///
/// The grid starts at the requested `start` and steps by a fixed duration. `Months` and
/// `Years` are fixed lengths (30 and 365 days), not calendar months and years: a monthly
/// grid starting on 31 January lands on 2 March, not 28 February.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum Resolution {
	/// 1 nanosecond.
	Nanoseconds,
	/// 1 microsecond.
	Microseconds,
	/// 1 millisecond.
	Milliseconds,
	/// 1 second.
	Seconds,
	/// 60 seconds.
	Minutes,
	/// 3,600 seconds.
	Hours,
	/// 86,400 seconds. Time is on the POSIX scale, so a day never contains a leap second:
	/// a `23:59:60.x` timestamp is the same instant as `00:00:00.x` the next day.
	Days,
	/// 7 days.
	Weeks,
	/// 30 days.
	Months,
	/// 365 days.
	Years,
}

impl Resolution {
	/// Every resolution, finest first. A slice, so that adding a resolution in a minor
	/// release doesn't change its type.
	pub const ALL: &'static [Self] = &[Self::Nanoseconds, Self::Microseconds, Self::Milliseconds, Self::Seconds, Self::Minutes, Self::Hours, Self::Days, Self::Weeks, Self::Months, Self::Years];

	/// The grid step in nanoseconds.
	#[must_use]
	pub const fn step_nanos(self) -> i64 {
		const SECOND: i64 = 1_000_000_000;
		const DAY: i64 = 86_400 * SECOND;
		match self {
			Self::Nanoseconds => 1,
			Self::Microseconds => 1_000,
			Self::Milliseconds => 1_000_000,
			Self::Seconds => SECOND,
			Self::Minutes => 60 * SECOND,
			Self::Hours => 3_600 * SECOND,
			Self::Days => DAY,
			Self::Weeks => 7 * DAY,
			Self::Months => 30 * DAY,
			Self::Years => 365 * DAY,
		}
	}

	/// The grid step as a duration.
	#[must_use]
	pub const fn step(self) -> TimeDelta {
		TimeDelta::nanoseconds(self.step_nanos())
	}

	/// The lowercase plural name, e.g. `"seconds"`. [`FromStr`] accepts it back.
	#[must_use]
	pub const fn as_str(self) -> &'static str {
		match self {
			Self::Nanoseconds => "nanoseconds",
			Self::Microseconds => "microseconds",
			Self::Milliseconds => "milliseconds",
			Self::Seconds => "seconds",
			Self::Minutes => "minutes",
			Self::Hours => "hours",
			Self::Days => "days",
			Self::Weeks => "weeks",
			Self::Months => "months",
			Self::Years => "years",
		}
	}
}

impl fmt::Display for Resolution {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.write_str(self.as_str())
	}
}

impl FromStr for Resolution {
	type Err = Error;

	/// Parses the plural name, case-insensitively: `"seconds"`, `"Seconds"`, `"HOURS"`.
	fn from_str(s: &str) -> Result<Self, Self::Err> {
		Self::ALL.iter().copied().find(|r| r.as_str().eq_ignore_ascii_case(s)).ok_or_else(|| Error::Parse { kind: "resolution", input: s.to_owned() })
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn names_round_trip() {
		for &r in Resolution::ALL {
			assert_eq!(r.to_string().parse::<Resolution>(), Ok(r));
			assert_eq!(r.as_str().to_uppercase().parse::<Resolution>(), Ok(r));
		}
		assert!("fortnights".parse::<Resolution>().is_err());
	}

	#[test]
	fn all_lists_every_resolution() {
		// Adding a variant breaks this match until it is listed here, and the count below
		// until it is also in `ALL`.
		let index = |r: Resolution| match r {
			Resolution::Nanoseconds => 0,
			Resolution::Microseconds => 1,
			Resolution::Milliseconds => 2,
			Resolution::Seconds => 3,
			Resolution::Minutes => 4,
			Resolution::Hours => 5,
			Resolution::Days => 6,
			Resolution::Weeks => 7,
			Resolution::Months => 8,
			Resolution::Years => 9,
		};
		assert_eq!(Resolution::ALL.len(), 10);
		for (i, &r) in Resolution::ALL.iter().enumerate() {
			assert_eq!(index(r), i);
		}
	}

	#[test]
	fn steps_increase() {
		for pair in Resolution::ALL.windows(2) {
			assert!(pair[0].step_nanos() < pair[1].step_nanos());
		}
		// Every step, exactly: an increasing sequence alone allowed `Weeks` to be wrong.
		let steps = [TimeDelta::nanoseconds(1), TimeDelta::microseconds(1), TimeDelta::milliseconds(1), TimeDelta::seconds(1), TimeDelta::minutes(1), TimeDelta::hours(1), TimeDelta::days(1), TimeDelta::weeks(1), TimeDelta::days(30), TimeDelta::days(365)];
		for (&r, step) in Resolution::ALL.iter().zip(steps) {
			assert_eq!(r.step(), step, "{r}");
			assert_eq!(Some(r.step_nanos()), step.num_nanoseconds(), "{r}");
		}
	}
}

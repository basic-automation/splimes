//! Time as exact integer nanoseconds on the POSIX scale, in `i128` so no representable
//! range can overflow (chrono's own `num_nanoseconds` gives up beyond ±292 years).
//!
//! Every instant splimes sees — inputs, `start`, `end`, grid points — goes through
//! [`posix_nanos`] exactly once, so offsets, provenance and output timestamps all agree.
//! chrono can represent a leap second (`23:59:60.x`, stored as a nanosecond field of
//! 1e9 or more); on the POSIX scale it is the same instant as `00:00:00.x` the next
//! second, which is also what chrono's own addition assumes.

use chrono::{DateTime, NaiveDate, NaiveTime, Utc};

use crate::{Error, Resolution, Result};

const NANOS_PER_SECOND: i128 = 1_000_000_000;

/// `t` as nanoseconds since the Unix epoch, POSIX scale (leap seconds folded forward).
pub fn posix_nanos(t: DateTime<Utc>) -> i128 {
	i128::from(t.timestamp()) * NANOS_PER_SECOND + i128::from(t.timestamp_subsec_nanos())
}

/// The instant `nanos` after the Unix epoch, if chrono can represent it. Never a leap
/// second.
fn from_posix_nanos(nanos: i128) -> Option<DateTime<Utc>> {
	let secs = i64::try_from(nanos.div_euclid(NANOS_PER_SECOND)).ok()?;
	#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // In 0..1e9.
	let subsec = nanos.rem_euclid(NANOS_PER_SECOND) as u32;
	DateTime::from_timestamp(secs, subsec)
}

/// `nanos` as the nearest `f64`. `i128 as f64` is a slow software routine on most
/// targets; offsets almost always fit `i64`, whose conversion is one instruction.
#[inline]
pub fn nanos_to_f64(nanos: i128) -> f64 {
	#[allow(clippy::cast_precision_loss)]
	i64::try_from(nanos).map_or(nanos as f64, |n| n as f64)
}

/// The output grid: `start, start + step, …` up to and including `end`, on the POSIX
/// scale.
#[derive(Debug, Clone, Copy)]
pub struct Grid {
	/// The first grid point; `start` with any leap second folded forward.
	pub start: DateTime<Utc>,
	/// `start` in POSIX nanoseconds.
	pub start_nanos: i128,
	pub step_nanos: i64,
	pub len: usize,
}

impl Grid {
	/// # Errors
	///
	/// [`Error::InvalidTimeRange`] if `start` is after `end`; [`Error::OutputTooLarge`] if
	/// the grid has more points than a `Vec` can hold.
	pub fn new(start: DateTime<Utc>, end: DateTime<Utc>, resolution: Resolution) -> Result<Self> {
		// A leap second on chrono's very last day folds past the last instant it can
		// represent; the grid stops there.
		let last = posix_nanos(DateTime::<Utc>::MAX_UTC);
		let (start_nanos, end_nanos) = (posix_nanos(start), posix_nanos(end).min(last));
		if start_nanos > end_nanos {
			return Err(Error::InvalidTimeRange { start, end });
		}
		let step_nanos = resolution.step_nanos();
		// The span is non-negative here, so the quotient fits u128.
		let points = u128::try_from((end_nanos - start_nanos) / i128::from(step_nanos)).unwrap_or(0) + 1;
		// Each output point costs at least a timestamp, a value and a kind; refusing grids
		// whose index overflows `isize` keeps every later `k * step` and allocation in range.
		let len = usize::try_from(points).ok().filter(|&n| isize::try_from(n).is_ok()).ok_or(Error::OutputTooLarge { points })?;
		// `start_nanos` is at most `end_nanos`, which is at most chrono's last instant.
		let start = from_posix_nanos(start_nanos).unwrap_or(start);
		Ok(Self { start, start_nanos, step_nanos, len })
	}

	/// Grid point `k`.
	pub fn at(&self, k: usize) -> DateTime<Utc> {
		// Every grid point is at or before `end`, so it is representable.
		from_posix_nanos(self.start_nanos + i128::from(self.step_nanos) * k as i128).unwrap_or(self.start)
	}

	/// Grid points `first..first + len`, in order, never a leap second: each is built from
	/// its POSIX seconds and nanoseconds, kept by integer addition, on a calendar date
	/// computed once per day. That is several times cheaper per point than chrono's
	/// `checked_add_signed`, and gives the same instants as [`at`](Self::at).
	pub fn timestamps(&self, first: usize, len: usize) -> impl Iterator<Item = DateTime<Utc>> + '_ {
		const SECONDS_PER_DAY: i64 = 86_400;
		let nanos = self.start_nanos + i128::from(self.step_nanos) * first as i128;
		// Grid points are representable instants, so their seconds fit `i64`.
		let mut secs = i64::try_from(nanos.div_euclid(NANOS_PER_SECOND)).unwrap_or(i64::MAX);
		#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // In 0..1e9.
		let mut subsec = nanos.rem_euclid(NANOS_PER_SECOND) as u32;
		let step_secs = self.step_nanos.div_euclid(1_000_000_000);
		#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // In 0..1e9.
		let step_subsec = self.step_nanos.rem_euclid(1_000_000_000) as u32;
		let mut day: Option<(i64, NaiveDate)> = None;
		(0..len).map(move |_| {
			let (days, second_of_day) = (secs.div_euclid(SECONDS_PER_DAY), secs.rem_euclid(SECONDS_PER_DAY));
			let date = match day {
				Some((cached, date)) if cached == days => Some(date),
				_ => {
					// Midnight of a representable instant's day is representable.
					let date = DateTime::from_timestamp(days * SECONDS_PER_DAY, 0).map(|t| t.date_naive());
					day = date.map(|date| (days, date));
					date
				}
			};
			let time = u32::try_from(second_of_day).ok().and_then(|s| NaiveTime::from_num_seconds_from_midnight_opt(s, subsec));
			let t = date.zip(time).map_or(self.start, |(date, time)| date.and_time(time).and_utc());
			subsec += step_subsec;
			if subsec >= 1_000_000_000 {
				subsec -= 1_000_000_000;
				secs = secs.saturating_add(1);
			}
			secs = secs.saturating_add(step_secs);
			t
		})
	}

	/// Offset of grid point `k` from `origin` (POSIX nanoseconds), in nanoseconds.
	pub fn offset_nanos(&self, origin: i128, k: usize) -> i128 {
		self.start_nanos - origin + i128::from(self.step_nanos) * k as i128
	}

	/// Offsets from `origin` (POSIX nanoseconds) of grid points `first..`, by repeated
	/// addition: exact, and far cheaper than a multiply per point.
	pub fn offsets(&self, origin: i128, first: usize) -> impl Iterator<Item = i128> {
		let step = i128::from(self.step_nanos);
		std::iter::successors(Some(self.offset_nanos(origin, first)), move |o| Some(o + step))
	}
}

#[cfg(test)]
mod tests {
	use chrono::{TimeDelta, TimeZone};

	use super::*;

	fn at(secs: i64) -> DateTime<Utc> {
		Utc.timestamp_opt(secs, 0).single().expect("valid")
	}

	#[test]
	fn grid_is_inclusive_and_anchored_at_start() {
		let grid = Grid::new(at(1), at(11), Resolution::Seconds).expect("grid");
		assert_eq!(grid.len, 11);
		let ts: Vec<_> = grid.timestamps(0, grid.len).collect();
		assert_eq!(ts.first(), Some(&at(1)));
		assert_eq!(ts.last(), Some(&at(11)));

		// `end` off the grid: the last point is the last step that doesn't pass it.
		let grid = Grid::new(at(0), at(10), Resolution::Minutes).expect("grid");
		assert_eq!(grid.len, 1);
	}

	#[test]
	fn single_instant_grid() {
		assert_eq!(Grid::new(at(5), at(5), Resolution::Days).expect("grid").len, 1);
	}

	#[test]
	fn reversed_range_is_an_error() {
		assert!(matches!(Grid::new(at(5), at(4), Resolution::Seconds), Err(Error::InvalidTimeRange { .. })));
	}

	#[test]
	fn huge_grids_are_refused_not_allocated() {
		let err = Grid::new(DateTime::<Utc>::MIN_UTC, DateTime::<Utc>::MAX_UTC, Resolution::Nanoseconds).expect_err("too large");
		assert!(matches!(err, Error::OutputTooLarge { .. }));
	}

	#[test]
	fn posix_nanos_spans_the_whole_chrono_range() {
		let span = posix_nanos(DateTime::<Utc>::MAX_UTC) - posix_nanos(DateTime::<Utc>::MIN_UTC);
		assert!(span > i128::from(i64::MAX), "beyond what i64 nanoseconds can hold");
		assert_eq!(posix_nanos(at(1)) - posix_nanos(at(0)), 1_000_000_000);
		for t in [DateTime::<Utc>::MIN_UTC, DateTime::<Utc>::MAX_UTC, at(-1), at(1_700_000_000)] {
			assert_eq!(from_posix_nanos(posix_nanos(t)), Some(t));
		}
	}

	#[test]
	fn a_leap_second_on_the_last_day_is_capped() {
		let leap = chrono::NaiveDate::MAX.and_hms_nano_opt(23, 59, 59, 1_500_000_000).expect("chrono represents it").and_utc();
		let max = DateTime::<Utc>::MAX_UTC;
		let grid = Grid::new(max - TimeDelta::seconds(2), leap, Resolution::Seconds).expect("grid");
		let ts: Vec<_> = grid.timestamps(0, grid.len).collect();
		assert!(ts.windows(2).all(|w| w[0] < w[1]), "{ts:?}");
		assert!(ts.iter().all(|&t| t <= max));
		assert!(matches!(Grid::new(leap, leap, Resolution::Seconds), Err(Error::InvalidTimeRange { .. })));
	}

	#[test]
	fn leap_seconds_fold_onto_the_next_second() {
		let leap = DateTime::parse_from_rfc3339("2016-12-31T23:59:60.25Z").expect("chrono accepts leap seconds").to_utc();
		let next = DateTime::parse_from_rfc3339("2017-01-01T00:00:00.25Z").expect("valid").to_utc();
		assert_eq!(posix_nanos(leap), posix_nanos(next));
		// A grid starting in a leap second starts at the folded instant, and its timestamps
		// and offsets agree.
		let grid = Grid::new(leap, next + TimeDelta::seconds(2), Resolution::Seconds).expect("grid");
		assert_eq!(grid.start, next);
		assert_eq!(grid.len, 3);
		let ts: Vec<_> = grid.timestamps(0, grid.len).collect();
		for (k, (t, o)) in ts.iter().zip(grid.offsets(posix_nanos(next), 0)).enumerate() {
			assert_eq!(posix_nanos(*t) - posix_nanos(next), o, "point {k}");
			assert_eq!(*t, grid.at(k));
		}
		// An end inside a leap second never lets a point land past it.
		let before = DateTime::parse_from_rfc3339("2016-12-31T23:59:59Z").expect("valid").to_utc();
		let grid = Grid::new(before, leap, Resolution::Seconds).expect("grid");
		assert!(grid.timestamps(0, grid.len).all(|t| posix_nanos(t) <= posix_nanos(leap)));
	}

	#[test]
	fn timestamps_match_chrono_addition_everywhere() {
		let (min, max) = (DateTime::<Utc>::MIN_UTC, DateTime::<Utc>::MAX_UTC);
		let starts = [min, min + TimeDelta::nanoseconds(1_999_999_999), at(-86_401) + TimeDelta::nanoseconds(7), at(-1), at(0), at(1_700_000_000) + TimeDelta::nanoseconds(999_999_999), max - TimeDelta::days(800)];
		for resolution in Resolution::ALL.iter().copied() {
			for &start in &starts {
				let end = start.checked_add_signed(resolution.step() * 3_000).filter(|&e| e <= max).unwrap_or(max);
				let grid = Grid::new(start, end, resolution).expect("grid");
				let chrono: Vec<_> = std::iter::successors(Some(grid.start), |t| t.checked_add_signed(resolution.step())).take(grid.len).collect();
				// From the start, and from every offset into the grid.
				for first in [0, 1, grid.len / 3, grid.len - 1] {
					let ours: Vec<_> = grid.timestamps(first, grid.len - first).collect();
					assert_eq!(ours, chrono[first..], "{resolution:?} from {start}, point {first}");
				}
				for (k, t) in chrono.iter().enumerate().step_by(97) {
					assert_eq!(*t, grid.at(k), "{resolution:?} from {start}, point {k}");
				}
			}
		}
	}
}

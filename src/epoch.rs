//! Time given as integer ticks since the Unix epoch, the way columnar stores keep it.

use std::fmt;

use chrono::{DateTime, Utc};

use crate::{
	Error, Resolution, Result, prepare::Knots, time::{Grid, from_posix_nanos, posix_nanos}
};

/// The length of one epoch tick, for [`Interpolator::run_epoch`](crate::Interpolator::run_epoch)
/// and [`Interpolator::stream_epoch`](crate::Interpolator::stream_epoch): tick `t` is the
/// instant `t × unit` after the Unix epoch, on the POSIX scale (no leap seconds).
///
/// ```
/// use splimes::TickUnit;
///
/// assert_eq!(TickUnit::MILLISECONDS.nanos_per_tick(), 1_000_000);
/// assert_eq!(TickUnit::from_nanos(250), Some(TickUnit::from_nanos(250).unwrap()));
/// assert_eq!(TickUnit::from_nanos(0), None);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TickUnit {
	nanos: i64,
}

impl TickUnit {
	/// One-microsecond ticks.
	pub const MICROSECONDS: Self = Self { nanos: 1_000 };
	/// One-millisecond ticks.
	pub const MILLISECONDS: Self = Self { nanos: 1_000_000 };
	/// One-nanosecond ticks.
	pub const NANOSECONDS: Self = Self { nanos: 1 };
	/// One-second ticks.
	pub const SECONDS: Self = Self { nanos: 1_000_000_000 };

	/// Ticks of `nanos` nanoseconds each; `None` for zero or more than `i64::MAX`.
	#[must_use]
	pub const fn from_nanos(nanos: u64) -> Option<Self> {
		if nanos == 0 || nanos > i64::MAX as u64 {
			return None;
		}
		Some(Self { nanos: nanos.cast_signed() })
	}

	/// Nanoseconds per tick.
	#[must_use]
	pub const fn nanos_per_tick(self) -> u64 {
		self.nanos.cast_unsigned()
	}

	/// The ticks chrono can represent, as an inclusive range.
	fn range(self) -> (i64, i64) {
		let unit = i128::from(self.nanos);
		let (lo, hi) = (posix_nanos(DateTime::<Utc>::MIN_UTC), posix_nanos(DateTime::<Utc>::MAX_UTC));
		// Rounded inward; saturated to i64, which every in-range result of a whole-range
		// unit fits.
		let clamp = |t: i128| i64::try_from(t).unwrap_or(if t < 0 { i64::MIN } else { i64::MAX });
		(clamp(-((-lo).div_euclid(unit))), clamp(hi.div_euclid(unit)))
	}

	/// `tick` in POSIX nanoseconds. The caller has checked it is in range.
	#[inline]
	fn nanos(self, tick: i64) -> i128 {
		i128::from(tick) * i128::from(self.nanos)
	}

	/// `tick` as an instant.
	fn instant(self, tick: i64) -> Result<DateTime<Utc>> {
		from_posix_nanos(self.nanos(tick)).ok_or_else(|| Error::TickOutOfRange { tick, nanos_per_tick: self.nanos_per_tick() })
	}

	/// The grid from tick `start` to tick `end`.
	pub(crate) fn grid(self, start: i64, end: i64, resolution: Resolution) -> Result<Grid> {
		Grid::new(self.instant(start)?, self.instant(end)?, resolution)
	}

	/// The knots for `ticks` and `values`, which have the same length. Every tick is checked
	/// against chrono's range first, unless the unit is short enough that every `i64` is in
	/// it; then no instant is built for any input.
	pub(crate) fn knots<'v>(self, ticks: &[i64], values: &'v [f64], parallel: bool) -> Result<Knots<'v, f64>> {
		let (lo, hi) = self.range();
		if (lo, hi) != (i64::MIN, i64::MAX)
			&& let Some(&tick) = ticks.iter().find(|&&t| t < lo || t > hi)
		{
			return Err(Error::TickOutOfRange { tick, nanos_per_tick: self.nanos_per_tick() });
		}
		Knots::from_nanos(ticks.len(), |i| self.nanos(ticks[i]), |i| from_posix_nanos(self.nanos(ticks[i])).unwrap_or(DateTime::<Utc>::UNIX_EPOCH), |i| &values[i], parallel)
	}

	/// The grid's points as ticks of this unit: `(first tick, ticks per step)`, or
	/// [`Error::GridNotInTicks`] if a grid point falls between ticks.
	pub(crate) fn grid_ticks(self, grid: &Grid, resolution: Resolution) -> Result<(i64, i64)> {
		let unit = i128::from(self.nanos);
		let step = i128::from(grid.step_nanos);
		if step % unit != 0 || grid.start_nanos % unit != 0 {
			return Err(Error::GridNotInTicks { resolution, nanos_per_tick: self.nanos_per_tick() });
		}
		// The grid lies between two in-range ticks, so both fit.
		let first = i64::try_from(grid.start_nanos / unit).map_err(|_| Error::GridNotInTicks { resolution, nanos_per_tick: self.nanos_per_tick() })?;
		let per_step = i64::try_from(step / unit).map_err(|_| Error::GridNotInTicks { resolution, nanos_per_tick: self.nanos_per_tick() })?;
		Ok((first, per_step))
	}
}

impl fmt::Display for TickUnit {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self.nanos {
			1 => f.write_str("ns"),
			1_000 => f.write_str("us"),
			1_000_000 => f.write_str("ms"),
			1_000_000_000 => f.write_str("s"),
			n => write!(f, "{n}ns"),
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn range_is_chronos_range_in_ticks() {
		let (min, max) = (posix_nanos(DateTime::<Utc>::MIN_UTC), posix_nanos(DateTime::<Utc>::MAX_UTC));
		for unit in [TickUnit::NANOSECONDS, TickUnit::MICROSECONDS, TickUnit::MILLISECONDS, TickUnit::SECONDS, TickUnit::from_nanos(7).expect("unit"), TickUnit::from_nanos(i64::MAX as u64).expect("unit")] {
			let (lo, hi) = unit.range();
			for (tick, inside) in [(lo, true), (hi, true), (lo.saturating_sub(1), lo == i64::MIN), (hi.saturating_add(1), hi == i64::MAX)] {
				let nanos = unit.nanos(tick);
				assert_eq!((min..=max).contains(&nanos), inside, "{unit} tick {tick}");
				assert_eq!(unit.instant(tick).is_ok(), inside, "{unit} tick {tick}");
			}
		}
		// Nanosecond ticks cover all of i64; microsecond ticks don't.
		assert_eq!(TickUnit::NANOSECONDS.range(), (i64::MIN, i64::MAX));
		assert_ne!(TickUnit::MICROSECONDS.range(), (i64::MIN, i64::MAX));
	}

	#[test]
	fn out_of_range_ticks_are_refused() {
		let (_, hi) = TickUnit::SECONDS.range();
		let err = TickUnit::SECONDS.knots(&[0, hi + 1], &[1.0, 2.0], false).map(|k| k.len());
		assert_eq!(err, Err(Error::TickOutOfRange { tick: hi + 1, nanos_per_tick: 1_000_000_000 }));
		assert!(matches!(TickUnit::SECONDS.grid(0, i64::MAX, Resolution::Days), Err(Error::TickOutOfRange { tick: i64::MAX, .. })));
	}

	#[test]
	fn grid_ticks_need_whole_ticks() {
		let unit = TickUnit::MILLISECONDS;
		let grid = unit.grid(1_000, 5_000, Resolution::Seconds).expect("grid");
		assert_eq!(unit.grid_ticks(&grid, Resolution::Seconds), Ok((1_000, 1_000)));
		let grid = unit.grid(0, 5, Resolution::Microseconds).expect("grid");
		assert_eq!(unit.grid_ticks(&grid, Resolution::Microseconds), Err(Error::GridNotInTicks { resolution: Resolution::Microseconds, nanos_per_tick: 1_000_000 }));
		// Negative ticks are whole ticks too.
		let grid = unit.grid(-3_000, 0, Resolution::Milliseconds).expect("grid");
		assert_eq!(unit.grid_ticks(&grid, Resolution::Milliseconds), Ok((-3_000, 1)));
	}
}

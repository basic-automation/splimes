//! Turning caller input into the normalised knot arrays every backend computes on.

use chrono::{DateTime, Utc};

use crate::{Error, Result, time::posix_nanos, value::Value};

/// The input, sorted, de-duplicated and normalised.
///
/// Times are exact nanosecond offsets from the first knot `t0`; the kernel measures time
/// differences in units of the mean knot spacing `h`. Values are centred on the middle of
/// their range and scaled by its half-width, so they lie in `[-1, 1]`. Lagrange
/// interpolation commutes with both affine maps, so this changes no result
/// mathematically. It keeps the products in the Lagrange basis near 1 (instead of
/// `(10¹³ ns)³` for hourly data), which is what lets the `f32` GPU path work at all, and
/// gives the `f64` paths the same well-conditioned arithmetic.
pub struct Knots<'a, V> {
	/// The first knot, in POSIX nanoseconds.
	pub t0: i128,
	/// Nanoseconds per unit of normalised time: the mean knot spacing.
	pub h: f64,
	/// Normalised knot values.
	pub y: Vec<f64>,
	/// Exact knot offsets from `t0` in nanoseconds, strictly increasing, `offsets[0] == 0`.
	pub offsets: Vec<i128>,
	/// The caller's value for each knot, returned untouched for raw grid points.
	pub originals: Vec<&'a V>,
	/// Value normalisation: `value = y * scale + centre`.
	pub centre: f64,
	pub scale: f64,
	/// The smallest and largest knot values, exactly.
	pub min: f64,
	pub max: f64,
}

impl<'a, V: Value> Knots<'a, V> {
	/// Sorts by instant and keeps the **last** of any points at the same instant: a later
	/// observation replaces an earlier one. Instants are compared on the POSIX scale, so a
	/// leap second (`23:59:60.x`) is the same instant as `00:00:00.x` the next second.
	///
	/// # Errors
	///
	/// [`Error::NoPoints`] for empty input; [`Error::ValueOutOfRange`] for a value with no
	/// finite `f64` representation.
	pub fn new(timestamps: impl ExactSizeIterator<Item = DateTime<Utc>>, values: impl ExactSizeIterator<Item = &'a V>) -> Result<Self> {
		let mut samples: Vec<(i128, DateTime<Utc>, &V)> = timestamps.zip(values).map(|(t, v)| (posix_nanos(t), t, v)).collect();
		if samples.is_empty() {
			return Err(Error::NoPoints);
		}
		// Every value must be finite, including ones a later duplicate replaces: whether bad
		// input is reported mustn't depend on its position.
		if let Some(&(_, timestamp, _)) = samples.iter().find(|(_, _, v)| v.to_finite_f64().is_none()) {
			return Err(Error::ValueOutOfRange { timestamp });
		}
		// Stable, so equal instants keep their input order and the last one wins below.
		samples.sort_by_key(|&(n, _, _)| n);
		let mut distinct: Vec<(i128, DateTime<Utc>, &V)> = Vec::with_capacity(samples.len());
		for sample in samples {
			match distinct.last_mut() {
				Some(last) if last.0 == sample.0 => *last = sample,
				_ => distinct.push(sample),
			}
		}

		let mut raw_values = Vec::with_capacity(distinct.len());
		for &(_, timestamp, value) in &distinct {
			raw_values.push(value.to_finite_f64().ok_or(Error::ValueOutOfRange { timestamp })?);
		}
		let (min, max) = raw_values.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| (lo.min(v), hi.max(v)));
		// Halved before subtracting so values near ±f64::MAX can't overflow the width.
		let half_width = max / 2.0 - min / 2.0;
		let scale = if half_width > 0.0 { half_width } else { 1.0 };
		let centre = min / 2.0 + max / 2.0;

		let t0 = distinct[0].0;
		let offsets: Vec<i128> = distinct.iter().map(|&(n, _, _)| n - t0).collect();
		let n = offsets.len();
		#[allow(clippy::cast_precision_loss)] // A spacing only needs to be approximately the mean.
		let h = if n > 1 { offsets[n - 1] as f64 / (n - 1) as f64 } else { 1.0 };
		let y = raw_values.iter().map(|&v| (v - centre) / scale).collect();
		let originals = distinct.into_iter().map(|(_, _, v)| v).collect();
		Ok(Self { t0, h, y, offsets, originals, centre, scale, min, max })
	}

	pub const fn len(&self) -> usize {
		self.offsets.len()
	}

	/// The view the kernel reads.
	pub fn data(&self) -> crate::kernel::Data<'_> {
		crate::kernel::Data { offsets: &self.offsets, y: &self.y, inv_h: 1.0 / self.h }
	}

	/// Normalised `[min, max]` of the knot values.
	pub fn y_range(&self) -> (f64, f64) {
		self.y.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| (lo.min(v), hi.max(v)))
	}
}

#[cfg(test)]
mod tests {
	use bigdecimal::BigDecimal;
	use chrono::TimeZone;

	use super::*;

	fn at(secs: i64) -> DateTime<Utc> {
		Utc.timestamp_opt(secs, 0).single().expect("valid")
	}

	#[test]
	fn sorts_and_keeps_the_last_duplicate() {
		let ts = [at(20), at(0), at(10), at(10)];
		let vs = [2.0, 0.0, 1.0, 9.0];
		let knots = Knots::new(ts.into_iter(), vs.iter()).expect("knots");
		assert_eq!(knots.len(), 3);
		assert_eq!(knots.offsets, vec![0, 10_000_000_000, 20_000_000_000]);
		assert_eq!(knots.h, 10_000_000_000.0);
		assert_eq!(knots.originals.iter().map(|v| **v).collect::<Vec<_>>(), vec![0.0, 9.0, 2.0]);
		let restored: Vec<f64> = knots.y.iter().map(|y| y * knots.scale + knots.centre).collect();
		assert_eq!(restored, vec![0.0, 9.0, 2.0]);
	}

	#[test]
	fn constant_and_single_inputs_normalise() {
		let knots = Knots::new([at(0), at(5)].into_iter(), [7.0, 7.0].iter()).expect("knots");
		assert_eq!(knots.y, vec![0.0, 0.0]);
		assert_eq!(knots.centre, 7.0);
		let knots = Knots::new([at(0)].into_iter(), [3.0].iter()).expect("knots");
		assert_eq!(knots.offsets, vec![0]);
	}

	#[test]
	fn rejects_empty_and_unrepresentable() {
		let none: [f64; 0] = [];
		assert!(matches!(Knots::new(std::iter::empty(), none.iter()), Err(Error::NoPoints)));
		let huge: BigDecimal = "1e999".parse().expect("parse");
		assert!(matches!(Knots::new([at(0)].into_iter(), [huge].iter()), Err(Error::ValueOutOfRange { .. })));
		assert!(matches!(Knots::new([at(0)].into_iter(), [f64::NAN].iter()), Err(Error::ValueOutOfRange { .. })));
	}

	#[test]
	fn extreme_magnitudes_do_not_overflow_the_scale() {
		let knots = Knots::new([at(0), at(1)].into_iter(), [f64::MAX, -f64::MAX].iter()).expect("knots");
		assert!(knots.scale.is_finite() && knots.centre.is_finite());
		assert_eq!(knots.y, vec![1.0, -1.0]);
	}
}

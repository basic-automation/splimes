//! Turning caller input into the normalised knot arrays every backend computes on.

use chrono::{DateTime, Utc};
use rayon::prelude::*;

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
	/// The same offsets as `i64`, if they are all within `kernel::I64_SPAN`: the kernel's
	/// fast path.
	pub offsets64: Option<Vec<i64>>,
	/// The caller's value for each knot, returned untouched for raw grid points.
	pub originals: Vec<&'a V>,
	/// Value normalisation: `value = y * scale + centre`.
	pub centre: f64,
	pub scale: f64,
	/// The smallest and largest knot values, exactly.
	pub min: f64,
	pub max: f64,
}

/// Inputs from which preparation runs on rayon's pool when the caller allows it: below
/// this, splitting the work costs more than it saves.
const PARALLEL_FROM: usize = 1 << 14;

impl<'a, V: Value> Knots<'a, V> {
	/// The `len` samples `(timestamp(i), value(i))`, sorted by instant, keeping the
	/// **last** of any at the same instant: a later observation replaces an earlier one.
	/// Instants are compared on the POSIX scale, so a leap second (`23:59:60.x`) is the
	/// same instant as `00:00:00.x` the next second.
	///
	/// With `parallel`, large inputs are converted and sorted on rayon's pool; the result is
	/// the same either way.
	///
	/// # Errors
	///
	/// [`Error::NoPoints`] for empty input; [`Error::ValueOutOfRange`] for a value with no
	/// finite `f64` representation.
	pub fn new(len: usize, timestamp: impl Fn(usize) -> DateTime<Utc> + Sync, value: impl Fn(usize) -> &'a V + Sync, parallel: bool) -> Result<Self> {
		if len == 0 {
			return Err(Error::NoPoints);
		}
		// Each value is converted once: for `BigDecimal` that is most of the cost.
		let sample = |i: usize| {
			let v = value(i);
			v.to_finite_f64().map(|y| (posix_nanos(timestamp(i)), y, v)).ok_or(i)
		};
		// Every value must be finite, including ones a later duplicate replaces: whether bad
		// input is reported mustn't depend on its position. The first in input order is named.
		let out_of_range = |i: usize| Error::ValueOutOfRange { timestamp: timestamp(i) };
		let parallel = parallel && len >= PARALLEL_FROM;
		let mut samples = if parallel {
			// An indexed collect, which preallocates, with NaN marking a bad value: a converted
			// value never is NaN.
			let mut samples = Vec::new();
			(0..len).into_par_iter().map(|i| sample(i).unwrap_or_else(|_| (0, f64::NAN, value(i)))).collect_into_vec(&mut samples);
			if let Some(i) = samples.par_iter().position_first(|&(_, y, _)| y.is_nan()) {
				return Err(out_of_range(i));
			}
			samples
		} else {
			// Not `collect` into a `Result`, which can't preallocate.
			let mut samples = Vec::with_capacity(len);
			for i in 0..len {
				samples.push(sample(i).map_err(out_of_range)?);
			}
			samples
		};
		// Stable, so equal instants keep their input order and the last one wins below.
		// Already-sorted input, the usual case, is only checked.
		if !samples.is_sorted_by_key(|&(n, _, _)| n) {
			if parallel {
				samples.par_sort_by_key(|&(n, _, _)| n);
			} else {
				samples.sort_by_key(|&(n, _, _)| n);
			}
		}
		// In place: `dedup_by` keeps the earlier of two equal samples, so move the later one
		// into its slot first.
		samples.dedup_by(|later, kept| {
			let same = later.0 == kept.0;
			if same {
				std::mem::swap(later, kept);
			}
			same
		});
		let distinct = samples;

		let raw_values: Vec<f64> = distinct.iter().map(|&(_, y, _)| y).collect();
		let (min, max) = raw_values.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| (lo.min(v), hi.max(v)));
		// Halved before subtracting so values near ±f64::MAX can't overflow the width.
		let half_width = max / 2.0 - min / 2.0;
		let scale = if half_width > 0.0 { half_width } else { 1.0 };
		let centre = min / 2.0 + max / 2.0;

		let t0 = distinct[0].0;
		let offsets: Vec<i128> = distinct.iter().map(|&(n, _, _)| n - t0).collect();
		let n = offsets.len();
		let offsets64 = if offsets[n - 1] <= crate::kernel::I64_SPAN { offsets.iter().map(|&o| i64::try_from(o).ok()).collect() } else { None };
		#[allow(clippy::cast_precision_loss)] // A spacing only needs to be approximately the mean.
		let h = if n > 1 { offsets[n - 1] as f64 / (n - 1) as f64 } else { 1.0 };
		let y = raw_values.iter().map(|&v| (v - centre) / scale).collect();
		let originals = distinct.into_iter().map(|(_, _, v)| v).collect();
		Ok(Self { t0, h, y, offsets, offsets64, originals, centre, scale, min, max })
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

	fn from_slices<'a, V: Value>(ts: &[DateTime<Utc>], vs: &'a [V], parallel: bool) -> Result<Knots<'a, V>> {
		assert_eq!(ts.len(), vs.len());
		Knots::new(ts.len(), |i| ts[i], |i| &vs[i], parallel)
	}

	#[test]
	fn sorts_and_keeps_the_last_duplicate() {
		let ts = [at(20), at(0), at(10), at(10)];
		let vs = [2.0, 0.0, 1.0, 9.0];
		let knots = from_slices(&ts, &vs, false).expect("knots");
		assert_eq!(knots.len(), 3);
		assert_eq!(knots.offsets, vec![0, 10_000_000_000, 20_000_000_000]);
		assert_eq!(knots.h, 10_000_000_000.0);
		assert_eq!(knots.originals.iter().map(|v| **v).collect::<Vec<_>>(), vec![0.0, 9.0, 2.0]);
		let restored: Vec<f64> = knots.y.iter().map(|y| y * knots.scale + knots.centre).collect();
		assert_eq!(restored, vec![0.0, 9.0, 2.0]);
	}

	#[test]
	fn constant_and_single_inputs_normalise() {
		let knots = from_slices(&[at(0), at(5)], &[7.0, 7.0], false).expect("knots");
		assert_eq!(knots.y, vec![0.0, 0.0]);
		assert_eq!(knots.centre, 7.0);
		let knots = from_slices(&[at(0)], &[3.0], false).expect("knots");
		assert_eq!(knots.offsets, vec![0]);
	}

	#[test]
	fn rejects_empty_and_unrepresentable() {
		let none: [f64; 0] = [];
		assert!(matches!(from_slices(&[], &none, false), Err(Error::NoPoints)));
		let huge: BigDecimal = "1e999".parse().expect("parse");
		assert!(matches!(from_slices(&[at(0)], &[huge], false), Err(Error::ValueOutOfRange { .. })));
		assert!(matches!(from_slices(&[at(0)], &[f64::NAN], false), Err(Error::ValueOutOfRange { .. })));
	}

	#[test]
	fn extreme_magnitudes_do_not_overflow_the_scale() {
		let knots = from_slices(&[at(0), at(1)], &[f64::MAX, -f64::MAX], false).expect("knots");
		assert!(knots.scale.is_finite() && knots.centre.is_finite());
		assert_eq!(knots.y, vec![1.0, -1.0]);
	}

	#[test]
	fn parallel_preparation_matches_serial() {
		// Shuffled, with every fourth instant repeated with a different value, and large
		// enough to take the parallel path.
		let n = 3 * PARALLEL_FROM;
		let shuffled = |i: usize| i64::try_from((i * 7_919) % n / 4 * 4).expect("small");
		let ts: Vec<_> = (0..n).map(|i| at(shuffled(i))).collect();
		let vs: Vec<f64> = (0..n).map(|i| (f64::from(u32::try_from(i).expect("small")) * 0.37).sin()).collect();
		let serial = from_slices(&ts, &vs, false).expect("knots");
		let parallel = from_slices(&ts, &vs, true).expect("knots");
		assert_eq!(serial.len(), n / 4);
		assert_eq!(parallel.offsets, serial.offsets);
		assert_eq!(parallel.y, serial.y);
		assert!(parallel.originals.iter().zip(&serial.originals).all(|(a, b)| std::ptr::eq(*a, *b)), "the same input wins every instant");
		// And the first bad value in input order is the one named, either way, even when a
		// later input is the earlier instant (so naming the earliest in time would differ).
		let (first, later) = (1, 7);
		assert!(ts[first] > ts[later], "the later input must be the earlier instant");
		let mut bad = vs.clone();
		bad[first] = f64::NAN;
		bad[later] = f64::INFINITY;
		let expected = Err(Error::ValueOutOfRange { timestamp: ts[first] });
		assert_eq!(from_slices(&ts, &bad, true).map(|k| k.len()), expected);
		assert_eq!(from_slices(&ts, &bad, false).map(|k| k.len()), expected);
	}
}

use std::{fmt, str::FromStr};

use crate::Error;

/// The largest degree [`Spline::Polynomial`] accepts.
///
/// Local Lagrange polynomials of higher degree oscillate badly between the knots (Runge's
/// phenomenon), so the cap keeps the method useful rather than merely computable.
pub const MAX_POLYNOMIAL_DEGREE: usize = 8;

/// The interpolation method.
///
/// Every method is *local*: the value at a grid instant depends only on the few input
/// points around it, found by binary search, so the cost per output point is
/// `O(log n)` in the number of inputs whatever the method. Every method passes exactly
/// through the input points.
///
/// Each method fits a polynomial through a window of `m = min(degree + 1, n)`
/// consecutive knots, `n` being the number of distinct input timestamps. With `p` the
/// number of knots at or before the instant, the window starts at knot
/// `clamp(p − ⌊m/2⌋, 0, n − m)`. In the interior that centres the window on the
/// instant's segment — as many knots on each side for odd degrees, one more after than
/// before for even degrees — and near the first and last knots the window slides inward
/// rather than shrinking, so every segment gets a full-degree polynomial. For example,
/// with knots `k0 … k9`: `Cubic` uses `k0…k3` in the first segment (`k0`–`k1`),
/// `k3…k6` between `k4` and `k5`, and `k6…k9` in the last segment; `Quadratic` uses
/// `k4…k6` between `k4` and `k5`, and `k7…k9` in the last segment.
///
/// | Method | Window | Inside the data | Outside the data |
/// |--------|--------|-----------------|------------------|
/// | `Linear` | 2 | the line through the segment's ends | the first / last segment's line, extended |
/// | `Quadratic` | 3 | the parabola through the window (the knot before and the two after, in the interior) | the first / last three knots' parabola, extended |
/// | `Cubic` | 4 | the cubic through the window (two knots each side, in the interior) | **held** at the first / last input value, exactly |
/// | `Polynomial(d, b)` | d + 1 | the degree-*d* polynomial through the window | the first / last d + 1 knots' polynomial, extended, then clamped by `b` if set |
///
/// `bounds_factor` limits polynomial extrapolation: with input values spanning
/// `[min, max]` and `range = max - min`, an extrapolated value is clamped to
/// `[min - b·range, max + b·range]`. `None` leaves it unclamped. It has no effect inside
/// the data.
///
/// When there are fewer distinct input timestamps than a method needs, splimes steps down
/// and reports it (see [`Interpolation::spline`](crate::Interpolation::spline)):
/// `Cubic` → `Quadratic` → `Linear`, and `Polynomial(d, b)` → `Polynomial(n - 1, b)`. A
/// single input point gives a constant series, reported as `Linear` (or
/// `Polynomial(1, b)` if a polynomial was asked for).
///
/// With the `serde` feature, a `Polynomial`'s bounds factor is serialised as a plain
/// `f64`. `serde_json`'s default float parser can read one back an ulp off; enable
/// `serde_json`'s `float_roundtrip` feature when a stored method must come back bit for bit.
///
/// `#[non_exhaustive]`, so new methods can arrive in minor releases.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum Spline {
	/// Piecewise-linear.
	Linear,
	/// Local quadratic (three-point Lagrange).
	Quadratic,
	/// Local cubic (four-point Lagrange), held flat outside the data.
	Cubic,
	/// Local polynomial of the given degree, from 1 to [`MAX_POLYNOMIAL_DEGREE`], with an
	/// optional extrapolation `bounds_factor`.
	Polynomial(usize, Option<f64>),
}

impl Spline {
	/// The polynomial degree of the method.
	#[must_use]
	#[allow(clippy::trivially_copy_pass_by_ref)] // `&self`, as in 0.1, so method paths keep working.
	pub const fn degree(&self) -> usize {
		match *self {
			Self::Linear => 1,
			Self::Quadratic => 2,
			Self::Cubic => 3,
			Self::Polynomial(degree, _) => degree,
		}
	}

	/// The number of distinct input timestamps the method needs (its degree plus one).
	#[must_use]
	pub const fn min_points(self) -> usize {
		self.degree().saturating_add(1)
	}

	/// The polynomial extrapolation bounds factor, if any.
	#[must_use]
	#[allow(clippy::trivially_copy_pass_by_ref)] // `&self`, as in 0.1, so method paths keep working.
	pub const fn bounds_factor(&self) -> Option<f64> {
		match *self {
			Self::Polynomial(_, bounds) => bounds,
			_ => None,
		}
	}

	/// Checks the method's parameters.
	///
	/// # Errors
	///
	/// [`Error::InvalidDegree`] or [`Error::InvalidBoundsFactor`] for a `Polynomial` with a
	/// degree of 0 or above [`MAX_POLYNOMIAL_DEGREE`], or a negative or non-finite bounds factor.
	pub fn validate(self) -> Result<(), Error> {
		if let Self::Polynomial(degree, bounds) = self {
			if !(1..=MAX_POLYNOMIAL_DEGREE).contains(&degree) {
				return Err(Error::InvalidDegree { degree, max: MAX_POLYNOMIAL_DEGREE });
			}
			if let Some(b) = bounds
				&& !(b.is_finite() && b >= 0.0)
			{
				return Err(Error::InvalidBoundsFactor(b));
			}
		}
		Ok(())
	}

	/// The method splimes actually runs on `distinct` distinct timestamps: this one if
	/// there are enough, otherwise the step-down described on [`Spline`].
	#[must_use]
	pub const fn fallback_for(self, distinct: usize) -> Self {
		if distinct >= self.min_points() {
			return self;
		}
		match self {
			Self::Linear => Self::Linear,
			Self::Quadratic | Self::Cubic => {
				if distinct >= 3 {
					Self::Quadratic
				} else {
					Self::Linear
				}
			}
			Self::Polynomial(_, bounds) => Self::Polynomial(if distinct > 1 { distinct - 1 } else { 1 }, bounds),
		}
	}
}

impl fmt::Display for Spline {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self {
			Self::Linear => f.write_str("Linear"),
			Self::Quadratic => f.write_str("Quadratic"),
			Self::Cubic => f.write_str("Cubic"),
			Self::Polynomial(degree, Some(b)) => write!(f, "Polynomial(degree: {degree}, bounds_factor: {b})"),
			Self::Polynomial(degree, None) => write!(f, "Polynomial(degree: {degree}, bounds_factor: None)"),
		}
	}
}

impl FromStr for Spline {
	type Err = Error;

	/// Parses what [`Display`](fmt::Display) prints: `Linear`, `Quadratic`, `Cubic` (any
	/// case), or `Polynomial(degree: 5, bounds_factor: 1.5)` / `…bounds_factor: None)`.
	/// The result is [validated](Spline::validate).
	fn from_str(s: &str) -> Result<Self, Self::Err> {
		let fail = || Error::Parse { kind: "spline", input: s.to_owned() };
		let t = s.trim();
		let spline = if t.eq_ignore_ascii_case("linear") {
			Self::Linear
		} else if t.eq_ignore_ascii_case("quadratic") {
			Self::Quadratic
		} else if t.eq_ignore_ascii_case("cubic") {
			Self::Cubic
		} else {
			let args = t.strip_prefix("Polynomial(").and_then(|r| r.strip_suffix(')')).ok_or_else(fail)?;
			let (degree, bounds) = args.split_once(',').ok_or_else(fail)?;
			let degree = degree.trim().strip_prefix("degree:").ok_or_else(fail)?.trim().parse().map_err(|_| fail())?;
			let bounds = bounds.trim().strip_prefix("bounds_factor:").ok_or_else(fail)?.trim();
			let bounds = if bounds == "None" { None } else { Some(bounds.parse().map_err(|_| fail())?) };
			Self::Polynomial(degree, bounds)
		};
		spline.validate()?;
		Ok(spline)
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn display_round_trips() {
		for s in [Spline::Linear, Spline::Quadratic, Spline::Cubic, Spline::Polynomial(5, Some(1.5)), Spline::Polynomial(2, None)] {
			assert_eq!(s.to_string().parse::<Spline>(), Ok(s));
		}
		assert_eq!("cubic".parse::<Spline>(), Ok(Spline::Cubic));
		assert!("Akima".parse::<Spline>().is_err());
		assert!(matches!("Polynomial(degree: 0, bounds_factor: None)".parse::<Spline>(), Err(Error::InvalidDegree { .. })));
	}

	#[test]
	fn validate_rejects_bad_parameters() {
		assert!(Spline::Polynomial(0, None).validate().is_err());
		assert!(Spline::Polynomial(MAX_POLYNOMIAL_DEGREE + 1, None).validate().is_err());
		assert!(Spline::Polynomial(3, Some(-1.0)).validate().is_err());
		assert!(Spline::Polynomial(3, Some(f64::NAN)).validate().is_err());
		assert!(Spline::Polynomial(MAX_POLYNOMIAL_DEGREE, Some(0.0)).validate().is_ok());
	}

	#[test]
	fn absurd_degrees_do_not_overflow() {
		let s = Spline::Polynomial(usize::MAX, None);
		assert_eq!(s.min_points(), usize::MAX);
		assert_eq!(s.fallback_for(3), Spline::Polynomial(2, None));
		assert!(s.validate().is_err());
	}

	#[test]
	fn fallback_steps_down_within_the_family() {
		assert_eq!(Spline::Cubic.fallback_for(4), Spline::Cubic);
		assert_eq!(Spline::Cubic.fallback_for(3), Spline::Quadratic);
		assert_eq!(Spline::Cubic.fallback_for(2), Spline::Linear);
		assert_eq!(Spline::Quadratic.fallback_for(2), Spline::Linear);
		assert_eq!(Spline::Polynomial(6, Some(1.0)).fallback_for(4), Spline::Polynomial(3, Some(1.0)));
		assert_eq!(Spline::Polynomial(6, None).fallback_for(1), Spline::Polynomial(1, None));
	}
}

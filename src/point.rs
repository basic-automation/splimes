use std::fmt;

use bigdecimal::BigDecimal;
use chrono::{DateTime, Utc};

/// One sample: a value at an instant.
///
/// `Point` deliberately doesn't implement `Hash`: `BigDecimal`'s hash expands large
/// exponents, so hashing an untrusted value such as `1e100000000000` would exhaust memory.
#[derive(Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Point {
	/// When the value was observed (input) or the grid instant it was resampled to (output).
	pub timestamp: DateTime<Utc>,
	/// The value.
	pub value: BigDecimal,
}

impl Point {
	/// A point with the given timestamp and value.
	#[must_use]
	pub const fn new(timestamp: DateTime<Utc>, value: BigDecimal) -> Self {
		Self { timestamp, value }
	}
}

impl fmt::Debug for Point {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.debug_struct("Point").field("timestamp", &self.timestamp).field("value", &Bounded(&self.value)).finish()
	}
}

/// A `BigDecimal` formatted in space proportional to its digits, whatever its exponent.
struct Bounded<'a>(&'a BigDecimal);

impl fmt::Debug for Bounded<'_> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		let (digits, scale) = self.0.as_bigint_and_exponent();
		if scale.unsigned_abs() <= 64 {
			// Small exponents print plainly (`12.5`, `0.001`, `3000`).
			fmt::Display::fmt(self.0, f)
		} else {
			write!(f, "{digits}e{}", -i128::from(scale))
		}
	}
}

/// Where an output value came from.
///
/// Every output point carries one, so a caller never mistakes a synthetic value for an
/// observed one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize), serde(rename_all = "lowercase"))]
pub enum PointKind {
	/// The grid instant coincides with an input timestamp. The value is that input's
	/// value, exactly: a `BigDecimal` input comes back with every digit intact.
	Raw,
	/// The grid instant lies strictly between the first and last input timestamps.
	Interpolated,
	/// The grid instant lies before the first input timestamp or after the last.
	Extrapolated,
}

impl PointKind {
	/// The lowercase name: `"raw"`, `"interpolated"` or `"extrapolated"`.
	#[must_use]
	pub const fn as_str(self) -> &'static str {
		match self {
			Self::Raw => "raw",
			Self::Interpolated => "interpolated",
			Self::Extrapolated => "extrapolated",
		}
	}
}

impl fmt::Display for PointKind {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.write_str(self.as_str())
	}
}

#[cfg(test)]
mod tests {
	use chrono::{DateTime, Utc};

	use super::*;

	#[test]
	fn debug_is_bounded_by_the_digits() {
		let t = DateTime::<Utc>::UNIX_EPOCH;
		for (value, expected) in [(BigDecimal::new(125.into(), 1), "12.5"), (BigDecimal::new(1.into(), i64::MAX), "1e-9223372036854775807"), (BigDecimal::new(7.into(), i64::MIN), "7e9223372036854775808")] {
			let text = format!("{:?}", Point::new(t, value));
			assert!(text.contains(expected), "{text}");
			assert!(text.len() < 120, "{text}");
		}
	}
}

use std::fmt::Write as _;

use bigdecimal::BigDecimal;

mod sealed {
	pub trait Sealed {}
	impl Sealed for bigdecimal::BigDecimal {}
	impl Sealed for f64 {}
}

/// A value type splimes can interpolate: [`BigDecimal`] or [`f64`].
///
/// Computation is in `f64` either way. The difference is at the edges: a `BigDecimal`
/// input is converted to the nearest `f64`, and an interpolated `f64` comes back as the
/// shortest decimal that round-trips to it (`0.1`, not
/// `0.1000000000000000055511151231257827`). Output points that coincide with an input
/// ([`PointKind::Raw`](crate::PointKind::Raw)) return the input value itself, untouched.
///
/// This trait is sealed: it can't be implemented outside splimes.
pub trait Value: Clone + Send + Sync + sealed::Sealed + 'static {
	#[doc(hidden)]
	fn to_finite_f64(&self) -> Option<f64>;
	/// A cheap value to pre-fill output buffers with; every slot is overwritten.
	#[doc(hidden)]
	fn placeholder() -> Self;
	#[doc(hidden)]
	fn from_finite_f64(value: f64) -> Option<Self>;
}

impl Value for f64 {
	fn to_finite_f64(&self) -> Option<f64> {
		self.is_finite().then_some(*self)
	}

	fn placeholder() -> Self {
		0.0
	}

	fn from_finite_f64(value: f64) -> Option<Self> {
		value.is_finite().then_some(value)
	}
}

impl Value for BigDecimal {
	fn to_finite_f64(&self) -> Option<f64> {
		// Through text, because Rust's float parser rounds correctly to the nearest f64 and
		// `BigDecimal::to_f64` doesn't promise to (it refuses values near `f64::MAX`). The
		// text is the digits and the exponent, never a plain expansion: `1e-10000000000`
		// is 15 bytes here, not ten billion zeros.
		let (int, scale) = self.as_bigint_and_exponent();
		let text = int.to_string();
		let (sign, digits) = text.strip_prefix('-').map_or(("", text.as_str()), |d| ("-", d));
		// value = d.ddd… × 10^e10. Written that way the exponent stays small whatever the
		// number of digits, so the parser (which saturates very large exponents) is exact.
		let e10 = digits.len() as i128 - 1 - i128::from(scale);
		if e10 > 310 {
			return None;
		}
		if e10 < -400 {
			return Some(if sign.is_empty() { 0.0 } else { -0.0 });
		}
		format!("{sign}{}.{}e{e10}", &digits[..1], &digits[1..]).parse::<f64>().ok().filter(|v| v.is_finite())
	}

	fn placeholder() -> Self {
		// Zero owns no heap digits, so cloning it is a plain copy.
		Self::default()
	}

	fn from_finite_f64(value: f64) -> Option<Self> {
		shortest_decimal(value)
	}
}

/// The shortest decimal that round-trips to `value`, or `None` if it isn't finite.
fn shortest_decimal(value: f64) -> Option<BigDecimal> {
	if !value.is_finite() {
		return None;
	}
	// `{:e}` is Rust's shortest round-trip formatting: `-d.ddde-x`, at most 17 significant
	// digits. Read the digits and exponent straight into a BigDecimal, which is several
	// times faster than handing the text to `BigDecimal::from_str`.
	let mut buf = Buf { bytes: [0; 32], len: 0 };
	write!(buf, "{value:e}").ok()?;
	let text = &buf.bytes[..buf.len];
	let (negative, text) = match text.split_first() {
		Some((b'-', rest)) => (true, rest),
		_ => (false, text),
	};
	let e = text.iter().position(|&b| b == b'e')?;
	let mut mantissa: i64 = 0;
	let mut fraction_digits: i64 = 0;
	let mut seen_point = false;
	for &b in &text[..e] {
		if b == b'.' {
			seen_point = true;
		} else {
			mantissa = mantissa * 10 + i64::from(b - b'0');
			fraction_digits += i64::from(seen_point);
		}
	}
	let exponent: i64 = std::str::from_utf8(&text[e + 1..]).ok()?.parse().ok()?;
	let mantissa = if negative { -mantissa } else { mantissa };
	// value = mantissa × 10^(exponent − fraction_digits); BigDecimal's scale is the negation.
	Some(BigDecimal::new(mantissa.into(), fraction_digits - exponent))
}

/// A fixed buffer for formatting one `f64`; `{:e}` of an `f64` is at most 24 bytes.
struct Buf {
	bytes: [u8; 32],
	len: usize,
}

impl std::fmt::Write for Buf {
	fn write_str(&mut self, s: &str) -> std::fmt::Result {
		let end = self.len + s.len();
		self.bytes.get_mut(self.len..end).ok_or(std::fmt::Error)?.copy_from_slice(s.as_bytes());
		self.len = end;
		Ok(())
	}
}

#[cfg(test)]
mod tests {
	use std::str::FromStr;

	use super::*;

	#[test]
	fn fast_path_matches_parsing_the_text() {
		for v in [0.0, -0.0, 1.0, -1.0, 0.1, 123.456, -9.87e-12, 6.022_140_76e23, f64::MAX, f64::MIN_POSITIVE, 5e-324, 1.0 / 3.0, -2.5e300] {
			let expected = BigDecimal::from_str(&format!("{v:e}")).expect("parse");
			assert_eq!(shortest_decimal(v), Some(expected), "{v}");
		}
	}

	#[test]
	fn interpolated_values_come_back_as_short_decimals() {
		assert_eq!(BigDecimal::from_finite_f64(0.1), Some(BigDecimal::from_str("0.1").expect("parse")));
		assert_eq!(BigDecimal::from_finite_f64(-1234.5), Some(BigDecimal::from_str("-1234.5").expect("parse")));
		assert_eq!(BigDecimal::from_finite_f64(1e-300), Some(BigDecimal::from_str("1e-300").expect("parse")));
		assert_eq!(BigDecimal::from_finite_f64(0.0), Some(BigDecimal::from(0)));
		assert_eq!(BigDecimal::from_finite_f64(f64::INFINITY), None);
		assert_eq!(BigDecimal::from_finite_f64(f64::NAN), None);
	}

	#[test]
	fn decimal_round_trip_is_exact_in_f64() {
		for v in [std::f64::consts::PI, 1.0 / 3.0, 9_007_199_254_740_993.0, -2.5e-17, f64::MAX, f64::MIN_POSITIVE] {
			let d = BigDecimal::from_finite_f64(v).expect("finite");
			assert_eq!(d.to_finite_f64(), Some(v), "{v} must round-trip");
		}
	}

	#[test]
	fn long_mantissas_with_huge_exponents_convert_exactly() {
		use bigdecimal::num_bigint::BigInt;
		let ten = BigInt::from(10);
		// 10^92233 × 10^-(2^63 - 1): far below f64's range, so zero, not "overflow".
		assert_eq!(BigDecimal::new(ten.pow(92_233), i64::MAX).to_finite_f64(), Some(0.0));
		// 100,000 digits scaled back to about 1.
		assert_eq!(BigDecimal::new(ten.pow(100_000), 100_000).to_finite_f64(), Some(1.0));
		assert_eq!(BigDecimal::new(-ten.pow(100_000) * 25, 100_001).to_finite_f64(), Some(-2.5));
		assert_eq!(BigDecimal::new(BigInt::from(7), i64::MIN).to_finite_f64(), None);
	}

	#[test]
	fn out_of_range_decimals_are_refused() {
		let huge = BigDecimal::from_str("1e400").expect("parse");
		assert_eq!(huge.to_finite_f64(), None);
		assert_eq!(f64::NAN.to_finite_f64(), None);
	}
}

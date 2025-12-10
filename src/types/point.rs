use std::fmt::Debug;

use bigdecimal::{BigDecimal, FromPrimitive};
use chrono::{DateTime, Duration, Utc};
use fake::{Fake, Faker};
use serde::{Deserialize, Serialize};

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Point {
	pub timestamp: DateTime<Utc>,
	pub value: BigDecimal,
}

impl Point {
	#[must_use]
	pub const fn new(timestamp: DateTime<Utc>, value: BigDecimal) -> Self {
		Self { timestamp, value }
	}

	/// Generate a random Point for testing
	///
	/// # Panics
	/// Panics if the random f64 value cannot be converted to `BigDecimal` (should never happen)
	#[must_use]
	pub fn random() -> Self {
		let base_time = Utc::now();
		let offset_seconds: i64 = Faker.fake();
		let offset_seconds = offset_seconds.rem_euclid(61) - 30; // -30 to 30
		let timestamp = base_time + Duration::seconds(offset_seconds);
		let value_f64: f64 = Faker.fake();
		let value = BigDecimal::from_f64(value_f64.rem_euclid(200.0) - 100.0).unwrap();
		Self { timestamp, value }
	}
}

impl Debug for Point {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(f, "Point {{ timestamp: {}, value: {} }}", self.timestamp, self.value.to_plain_string())
	}
}

use std::fmt::Debug;

use bigdecimal::{BigDecimal, FromPrimitive};
use chrono::{DateTime, Duration, Utc};
use fake::{Dummy, Fake, Faker};
use rand::Rng;

#[derive(Clone, PartialEq, Eq)]
pub struct Point {
	pub timestamp: DateTime<Utc>,
	pub value: BigDecimal,
}

impl Dummy<Faker> for Point {
	fn dummy_with_rng<R: Rng + ?Sized>(_: &Faker, rng: &mut R) -> Self {
		// Base timestamp (e.g., current time or a fixed point)
		let base_time = Utc::now();
		// Generate random offset within ±30 seconds (adjust as needed)
		let offset_seconds = (-30..30).fake_with_rng(rng);
		let timestamp = base_time + Duration::seconds(offset_seconds);
		let value = BigDecimal::from_f64((-100.0..100.0).fake_with_rng(rng)).unwrap();
		Self { timestamp, value }
	}
}

impl Debug for Point {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(f, "Point {{ timestamp: {}, value: {} }}", self.timestamp, self.value.to_plain_string())
	}
}

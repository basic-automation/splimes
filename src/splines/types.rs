use anyhow::{Result, bail};
use bigdecimal::{BigDecimal, FromPrimitive, Zero};
use chrono::{DateTime, Utc};

use crate::{Error, Point, Resolution};

pub const SECONDS_IN_MINUTE: i64 = 60;
pub const SECONDS_IN_HOUR: i64 = 3_600;
pub const SECONDS_IN_DAY: i64 = 86_400;
pub const SECONDS_IN_WEEK: i64 = 604_800;
pub const SECONDS_IN_MONTH: i64 = 2_592_000;
pub const SECONDS_IN_YEAR: i64 = 31_536_000;
pub const DAYS_IN_MONTH: i64 = 30;
pub const DAYS_IN_YEAR: i64 = 365;

pub const SIMD_BATCH_SIZE: usize = 4;

/// Linear spline implementation for efficient interpolation
pub struct LinearSpline {
	segments: Vec<LinearSegment>,
	time_bounds: Vec<DateTime<Utc>>,
	resolution: Resolution,
}

impl LinearSpline {
	pub fn new(points: &[Point], resolution: Resolution) -> Result<Self> {
		let mut segments = Vec::with_capacity(points.len().saturating_sub(1));
		let mut time_bounds = Vec::with_capacity(points.len());

		for i in 0..points.len() - 1 {
			let segment = LinearSegment::fit_linear(&points[i], &points[i + 1], resolution)?;
			segments.push(segment);
			time_bounds.push(points[i].timestamp);
		}
		time_bounds.push(points[points.len() - 1].timestamp);

		Ok(Self { segments, time_bounds, resolution })
	}

	pub fn evaluate(&self, target_time: &DateTime<Utc>) -> Result<BigDecimal> {
		// Handle extrapolation backward - use first segment with proper linear extrapolation
		if *target_time <= self.time_bounds[0] {
			let Some(dt) = BigDecimal::from_i64(self.resolution.difference(target_time, &self.time_bounds[0])?) else {
				bail!(Error::InvalidTimeRangeError);
			};
			return Ok(self.segments[0].evaluate(&dt));
		}

		// Handle extrapolation forward - use last segment with proper linear extrapolation
		if *target_time >= self.time_bounds[self.time_bounds.len() - 1] {
			let last_segment_idx = self.segments.len() - 1;
			let Some(dt) = BigDecimal::from_i64(self.resolution.difference(target_time, &self.time_bounds[last_segment_idx])?) else {
				bail!(Error::InvalidTimeRangeError);
			};
			return Ok(self.segments[last_segment_idx].evaluate(&dt));
		}

		// Find the appropriate segment for interpolation
		for (i, &bound_time) in self.time_bounds.iter().enumerate().skip(1) {
			if *target_time <= bound_time {
				let segment_idx = i - 1;
				let Some(dt) = BigDecimal::from_i64(self.resolution.difference(target_time, &self.time_bounds[segment_idx])?) else {
					bail!(Error::InvalidTimeRangeError);
				};
				return Ok(self.segments[segment_idx].evaluate(&dt));
			}
		}

		// Fallback to last segment
		let last_idx = self.segments.len() - 1;
		let Some(dt) = BigDecimal::from_i64(self.resolution.difference(target_time, &self.time_bounds[last_idx])?) else {
			bail!(Error::InvalidTimeRangeError);
		};
		Ok(self.segments[last_idx].evaluate(&dt))
	}
}

/// Linear segment with slope and intercept
#[derive(Debug, Clone)]
struct LinearSegment {
	slope: BigDecimal,
	intercept: BigDecimal,
}

impl LinearSegment {
	/// Create a linear segment between two measurements
	fn fit_linear(p1: &Point, p2: &Point, resolution: Resolution) -> Result<Self> {
		let Some(dt) = BigDecimal::from_i64(resolution.difference(&p2.timestamp, &p1.timestamp)?) else {
			bail!(Error::InvalidTimeRangeError);
		};

		if dt == BigDecimal::zero() {
			// Handle identical timestamps - create constant segment
			return Ok(Self { slope: BigDecimal::zero(), intercept: p1.value.clone() });
		}

		let dy = &p2.value - &p1.value;
		let slope = dy / dt;

		// y = mx + b, where b is the y-intercept at the start time
		let intercept = p1.value.clone();

		Ok(Self { slope, intercept })
	}

	/// Evaluate the linear function at time offset dt (in milliseconds)
	fn evaluate(&self, dt: &BigDecimal) -> BigDecimal {
		&self.intercept + &self.slope * dt
	}
}

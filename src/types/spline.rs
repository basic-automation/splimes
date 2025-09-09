use anyhow::{Result, bail};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::Point;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Spline {
	Linear,
	Quadratic,
	Cubic,
	Polynomial(usize, Option<f64>), // (degree, bounds_factor)
}

impl Spline {
	#[must_use]
	pub const fn degree(&self) -> usize {
		match self {
			Self::Linear => 1,
			Self::Quadratic => 2,
			Self::Cubic => 3,
			Self::Polynomial(degree, _) => *degree,
		}
	}

	#[must_use]
	pub const fn number_of_points_required(&self) -> usize {
		match self {
			Self::Linear => 2,
			Self::Quadratic => 3,
			Self::Cubic => 4,
			Self::Polynomial(degree, _) => *degree + 1, // Degree n requires n+1 points
		}
	}

	#[must_use]
	pub const fn bounds_factor(&self) -> Option<f64> {
		match self {
			Self::Linear | Self::Quadratic | Self::Cubic => None, // These don't support bounds
			Self::Polynomial(_, bounds) => *bounds,
		}
	}

	/// # Errors
	/// todo
	pub fn pre_check(&self, points: &[Point], start: &DateTime<Utc>, end: &DateTime<Utc>) -> Result<()> {
		if points.len() < self.number_of_points_required() {
			bail!("Spline {:?} requires at least {} points, but got {}", self, self.number_of_points_required(), points.len());
		}

		if start >= end {
			bail!("Start time {:?} must be before end time {:?}", start, end);
		}

		Ok(())
	}
}

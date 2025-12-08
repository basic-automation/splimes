use std::{fmt::Display, str::FromStr};

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

impl Display for Spline {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self {
                        Self::Linear => write!(f, "Linear"),
                        Self::Quadratic => write!(f, "Quadratic"),
                        Self::Cubic => write!(f, "Cubic"),
                        Self::Polynomial(degree, bounds_factor) => {
                                if let Some(bounds) = bounds_factor {
                                        write!(f, "Polynomial(degree: {degree}, bounds_factor: {bounds})")
                                } else {
                                        write!(f, "Polynomial(degree: {degree}, bounds_factor: None)")
                                }
                        }
                }
        }
}

impl FromStr for Spline {
        type Err = anyhow::Error;

        fn from_str(s: &str) -> Result<Self, Self::Err> {
                match s {
                        "Linear" => Ok(Self::Linear),
                        "Quadratic" => Ok(Self::Quadratic),
                        "Cubic" => Ok(Self::Cubic),
                        _ if s.starts_with("Polynomial") => {
                                // Example format: "Polynomial(degree: 5, bounds_factor: 1.5)"
                                let parts: Vec<&str> = s.trim_start_matches("Polynomial(").trim_end_matches(')').split(',').collect();
                                if parts.len() != 2 {
                                        bail!("Invalid Polynomial format");
                                }

                                let degree_part = parts[0].trim().strip_prefix("degree: ").ok_or_else(|| anyhow::anyhow!("Invalid degree format"))?;
                                let bounds_part = parts[1].trim().strip_prefix("bounds_factor: ").ok_or_else(|| anyhow::anyhow!("Invalid bounds_factor format"))?;

                                let degree = degree_part.parse::<usize>()?;
                                let bounds_factor = if bounds_part == "None" {
                                        None
                                } else {
                                        Some(bounds_part.parse::<f64>()?)
                                };

                                Ok(Self::Polynomial(degree, bounds_factor))
                        }
                        _ => bail!("Unknown spline type: {s}"),
                }
        }
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
			bail!("Start time {start:?} must be before end time {end:?}");
		}

		Ok(())
	}
}

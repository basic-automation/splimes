use anyhow::{Result, bail};
use chrono::{DateTime, Utc};

use crate::{
	Error, splines::{DAYS_IN_MONTH, DAYS_IN_YEAR, SECONDS_IN_DAY, SECONDS_IN_HOUR, SECONDS_IN_MINUTE, SECONDS_IN_MONTH, SECONDS_IN_WEEK, SECONDS_IN_YEAR}
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Resolution {
	Nanoseconds,
	Microseconds,
	Milliseconds,
	Seconds,
	Minutes,
	Hours,
	Days,
	Weeks,
	Months,
	Years,
}

impl Resolution {
	#[must_use]
	pub const fn to_step(&self) -> chrono::Duration {
		match self {
			Self::Nanoseconds => chrono::Duration::nanoseconds(1),
			Self::Microseconds => chrono::Duration::microseconds(1),
			Self::Milliseconds => chrono::Duration::milliseconds(1),
			Self::Seconds => chrono::Duration::seconds(1),
			Self::Minutes => chrono::Duration::minutes(1),
			Self::Hours => chrono::Duration::hours(1),
			Self::Days => chrono::Duration::days(1),
			Self::Weeks => chrono::Duration::weeks(1),
			Self::Months => chrono::Duration::days(DAYS_IN_MONTH), // Approximate 30-day month
			Self::Years => chrono::Duration::days(DAYS_IN_YEAR),   // Approximate 365-day year
		}
	}

	/// # Errors
	/// todo
	pub fn to_base(&self, timestamp: &DateTime<Utc>) -> Result<i64> {
		let res = match self {
			Self::Nanoseconds => timestamp.timestamp_nanos_opt().ok_or_else(|| anyhow::anyhow!("Invalid timestamp for nanoseconds resolution"))?,
			Self::Microseconds => timestamp.timestamp_micros(),
			Self::Milliseconds => timestamp.timestamp_millis(),
			Self::Seconds => timestamp.timestamp(),
			Self::Minutes => timestamp.timestamp() / SECONDS_IN_MINUTE,
			Self::Hours => timestamp.timestamp() / SECONDS_IN_HOUR,
			Self::Days => timestamp.timestamp() / SECONDS_IN_DAY,
			Self::Weeks => timestamp.timestamp() / SECONDS_IN_WEEK,
			Self::Months => timestamp.timestamp() / SECONDS_IN_MONTH,
			Self::Years => timestamp.timestamp() / SECONDS_IN_YEAR,
		};
		Ok(res)
	}

	/// # Errors
	/// todo
	pub fn to_step_base(&self) -> Result<i64> {
		let step = self.to_step();
		let step_base = match self {
			Self::Nanoseconds => step.num_nanoseconds().ok_or_else(|| anyhow::anyhow!("Invalid nanosecond step"))?,
			Self::Microseconds => step.num_microseconds().ok_or_else(|| anyhow::anyhow!("Invalid microsecond step"))?,
			Self::Milliseconds => step.num_milliseconds(),
			Self::Seconds => step.num_seconds(),
			Self::Minutes => step.num_minutes(),
			Self::Hours => step.num_hours(),
			Self::Days => step.num_days(),
			Self::Weeks => step.num_weeks(),
			Self::Months => step.num_days() / DAYS_IN_MONTH,
			Self::Years => step.num_days() / DAYS_IN_YEAR,
		};
		Ok(step_base)
	}

	/// # Errors
	/// todo
	pub fn difference(&self, minuend: &DateTime<Utc>, subtrahend: &DateTime<Utc>) -> Result<i64> {
		let minuend = *minuend;
		match self {
			Self::Nanoseconds => match (minuend - subtrahend).num_nanoseconds() {
				Some(nanos) => Ok(nanos),
				None => bail!(Error::TimeError("Invalid nanosecond difference".to_string())),
			},
			Self::Microseconds => match (minuend - subtrahend).num_microseconds() {
				Some(micros) => Ok(micros),
				None => bail!(Error::TimeError("Invalid microsecond difference".to_string())),
			},
			Self::Milliseconds => Ok((minuend - subtrahend).num_milliseconds()),
			Self::Seconds => Ok((minuend - subtrahend).num_seconds()),
			Self::Minutes => Ok((minuend - subtrahend).num_minutes()),
			Self::Hours => Ok((minuend - subtrahend).num_hours()),
			Self::Days => Ok((minuend - subtrahend).num_days()),
			Self::Weeks => Ok((minuend - subtrahend).num_weeks()),
			Self::Months => Ok((minuend - subtrahend).num_days() / DAYS_IN_MONTH),
			Self::Years => Ok((minuend - subtrahend).num_days() / DAYS_IN_YEAR),
		}
	}

	/// # Errors
	/// todo
	pub fn round(&self, timestamp: &DateTime<Utc>) -> Result<DateTime<Utc>> {
		let base = self.to_base(timestamp)?;
		let step_base = self.to_step_base()?;
		let offset = base % step_base;
		if offset == 0 {
			return Ok(*timestamp);
		}
		match self {
			Self::Nanoseconds => Ok(*timestamp + chrono::Duration::nanoseconds(step_base - offset)),
			Self::Microseconds => Ok(*timestamp + chrono::Duration::microseconds(step_base - offset)),
			Self::Milliseconds => Ok(*timestamp + chrono::Duration::milliseconds(step_base - offset)),
			Self::Seconds => Ok(*timestamp + chrono::Duration::seconds(step_base - offset)),
			Self::Minutes => Ok(*timestamp + chrono::Duration::minutes(step_base - offset)),
			Self::Hours => Ok(*timestamp + chrono::Duration::hours(step_base - offset)),
			Self::Days => Ok(*timestamp + chrono::Duration::days(step_base - offset)),
			Self::Weeks => Ok(*timestamp + chrono::Duration::weeks(step_base - offset)),
			Self::Months => Ok(*timestamp + chrono::Duration::days((step_base - offset) / DAYS_IN_MONTH)),
			Self::Years => Ok(*timestamp + chrono::Duration::days((step_base - offset) / DAYS_IN_YEAR)),
		}
	}
}

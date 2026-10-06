//! Async wrappers that run an interpolation on tokio's blocking pool.

use bigdecimal::BigDecimal;
use chrono::{DateTime, Utc};

use crate::{Error, Interpolation, Interpolator, Point, Result};

impl Interpolator {
	/// [`run`](Self::run) on tokio's blocking thread pool, so a large interpolation doesn't
	/// stall the async worker it was called from. Takes the points by value because the
	/// task must own them.
	///
	/// Requires the `tokio` feature, and must be called inside a tokio runtime.
	///
	/// If this is the program's first use of the GPU ([`Backend::Gpu`](crate::Backend::Gpu)),
	/// the device opens on the blocking pool. Don't shut the runtime down abruptly
	/// (`shutdown_background`, `process::exit`) while that's in flight: a driver
	/// interrupted mid-initialisation can crash the process. Call
	/// [`prewarm_gpu`](crate::prewarm_gpu) at startup to avoid the question.
	///
	/// # Errors
	///
	/// As [`run`](Self::run), plus [`Error::Task`] if the blocking task panics.
	pub async fn run_async(self, points: Vec<Point>, start: DateTime<Utc>, end: DateTime<Utc>) -> Result<Interpolation<BigDecimal>> {
		tokio::task::spawn_blocking(move || self.run(&points, start, end)).await.map_err(|e| Error::Task(e.to_string()))?
	}

	/// [`run_f64`](Self::run_f64) on tokio's blocking thread pool.
	///
	/// # Errors
	///
	/// As [`run_f64`](Self::run_f64), plus [`Error::Task`] if the blocking task panics.
	pub async fn run_f64_async(self, timestamps: Vec<DateTime<Utc>>, values: Vec<f64>, start: DateTime<Utc>, end: DateTime<Utc>) -> Result<Interpolation<f64>> {
		tokio::task::spawn_blocking(move || self.run_f64(&timestamps, &values, start, end)).await.map_err(|e| Error::Task(e.to_string()))?
	}
}

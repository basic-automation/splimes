use thiserror::Error as ThisError;

#[derive(Debug, ThisError)]
pub enum Error {
	#[error("Insufficient measurements provided for interpolation")]
	InsufficientMeasurementsError,

	#[error("Invalid time range: start time must be before end time")]
	InvalidTimeRangeError,

	#[error("Invalid GPU input, {0}.")]
	InvalidGpuInputError(String),

	#[error("Invalid GPU output, {0}.")]
	InvalidGpuOutputError(String),

	#[error("Insufficient points provided for interpolation")]
	InsufficientPointsError,

	#[error("Decimal conversion error")]
	DecimalConversionError,

	#[error("Time error: {0}")]
	TimeError(String),

	#[error("Invalid timestamp error: {0}")]
	InvalidTimestampError(String),

	#[error("Invalid degree for polynomial interpolation: {0}. Must be at least 1.")]
	InvalidDegreeError(usize),

	#[error("IO Error: {0}")]
	IOError(String),

	#[error("Conversion error: {0}")]
	ConversionError(String),
}

pub use cubic::{cubic, cubic_simd};
pub use linear::{linear, linear_simd};
pub use polynomial::{polynomial, polynomial_simd};
pub use quadratic::{quadratic, quadratic_simd};
pub use types::{DAYS_IN_MONTH, DAYS_IN_YEAR, LinearSpline, SECONDS_IN_DAY, SECONDS_IN_HOUR, SECONDS_IN_MINUTE, SECONDS_IN_MONTH, SECONDS_IN_WEEK, SECONDS_IN_YEAR, SIMD_BATCH_SIZE};

pub mod cubic;
pub mod linear;
pub mod polynomial;
pub mod quadratic;
mod types;

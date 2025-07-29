pub use cubic::CUBIC_INTERPOLATION_SHADER;
pub use linear::{LINEAR_INTERPOLATION_SHADER_F32, LINEAR_INTERPOLATION_SHADER_F64};
pub use polynomial::POLYNOMIAL_INTERPOLATION_SHADER;
pub use quadratic::QUADRATIC_INTERPOLATION_SHADER;

mod cubic;
mod linear;
mod polynomial;
mod quadratic;

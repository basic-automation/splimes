use crate::gpu::shaders::{CUBIC_INTERPOLATION_SHADER, LINEAR_INTERPOLATION_SHADER_F32, LINEAR_INTERPOLATION_SHADER_F64, POLYNOMIAL_INTERPOLATION_SHADER, QUADRATIC_INTERPOLATION_SHADER};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Method {
	Linear,
	Quadratic,
	Cubic,
	Polynomial(usize), // Degree of polynomial
}

impl Method {
	pub const fn shader_source_f64(self) -> &'static str {
		match self {
			Self::Linear => LINEAR_INTERPOLATION_SHADER_F64,
			Self::Quadratic => QUADRATIC_INTERPOLATION_SHADER,
			Self::Cubic => CUBIC_INTERPOLATION_SHADER,
			Self::Polynomial(_) => POLYNOMIAL_INTERPOLATION_SHADER,
		}
	}

	pub const fn shader_source_f32(self) -> &'static str {
		match self {
			Self::Linear => LINEAR_INTERPOLATION_SHADER_F32,
			Self::Quadratic => QUADRATIC_INTERPOLATION_SHADER,
			Self::Cubic => CUBIC_INTERPOLATION_SHADER,
			Self::Polynomial(_) => POLYNOMIAL_INTERPOLATION_SHADER,
		}
	}
}

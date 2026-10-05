pub use cubic::{CUBIC_INTERPOLATION_SHADER, CUBIC_INTERPOLATION_SHADER_F64};
pub use linear::{LINEAR_INTERPOLATION_SHADER_F32, LINEAR_INTERPOLATION_SHADER_F64};
pub use polynomial::{POLYNOMIAL_INTERPOLATION_SHADER, POLYNOMIAL_INTERPOLATION_SHADER_F64};
pub use quadratic::{QUADRATIC_INTERPOLATION_SHADER, QUADRATIC_INTERPOLATION_SHADER_F64};

mod cubic;
mod linear;
mod polynomial;
mod quadratic;

#[cfg(test)]
mod tests {
	use naga::valid::{Capabilities, ValidationFlags, Validator};

	/// Every shader must parse and validate. The f64 shaders need `Capabilities::FLOAT64`
	/// (the `SHADER_F64` device feature); the f32 shaders are the fallback for adapters
	/// without it and must validate **without** that capability.
	///
	/// This runs on any machine, GPU or not. Without it, a broken f32 fallback goes
	/// unnoticed on f64-capable dev GPUs and only surfaces as a wgpu validation panic on
	/// Apple, integrated, or WARP adapters.
	#[test]
	fn all_shaders_parse_and_validate() {
		let shaders: [(&str, &str, Capabilities); 8] = [
			("linear f32", super::LINEAR_INTERPOLATION_SHADER_F32, Capabilities::empty()),
			("quadratic f32", super::QUADRATIC_INTERPOLATION_SHADER, Capabilities::empty()),
			("cubic f32", super::CUBIC_INTERPOLATION_SHADER, Capabilities::empty()),
			("polynomial f32", super::POLYNOMIAL_INTERPOLATION_SHADER, Capabilities::empty()),
			("linear f64", super::LINEAR_INTERPOLATION_SHADER_F64, Capabilities::FLOAT64),
			("quadratic f64", super::QUADRATIC_INTERPOLATION_SHADER_F64, Capabilities::FLOAT64),
			("cubic f64", super::CUBIC_INTERPOLATION_SHADER_F64, Capabilities::FLOAT64),
			("polynomial f64", super::POLYNOMIAL_INTERPOLATION_SHADER_F64, Capabilities::FLOAT64),
		];
		let mut failures = Vec::new();
		for (name, source, caps) in shaders {
			match naga::front::wgsl::parse_str(source) {
				Err(e) => failures.push(format!("{name}: parse error:\n{}", e.emit_to_string(source))),
				Ok(module) => {
					if let Err(e) = Validator::new(ValidationFlags::all(), caps).validate(&module) {
						failures.push(format!("{name}: validation error: {:?}", e.into_inner()));
					}
				}
			}
		}
		assert!(failures.is_empty(), "invalid shaders:\n{}", failures.join("\n\n"));
	}
}

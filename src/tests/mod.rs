pub mod cubic;
pub mod gpu_config;
pub mod linear;
pub mod plot;
pub mod polynomial;
pub mod quadratic;
pub mod regression;

#[cfg(test)]
pub use plot::plot::plot_terminal;

/// Whether a GPU-comparison test should run its GPU half.
///
/// Returns `true` when a GPU adapter is available. Without one, the test is skipped with a
/// notice, so `cargo test` works on any machine, unless `WEFT_REQUIRE_GPU` is set, in
/// which case a missing adapter fails the test. CI sets it (on a software Vulkan driver)
/// so GPU coverage can't silently disappear.
#[cfg(test)]
pub fn gpu_available_or_skip(test: &str) -> bool {
	match crate::gpu::types::GpuInterpolator::supports_f64_static() {
		Ok(_) => true,
		Err(e) if std::env::var_os("WEFT_REQUIRE_GPU").is_some() => panic!("{test}: WEFT_REQUIRE_GPU is set but no GPU adapter is available: {e:#}"),
		Err(e) => {
			eprintln!("{test}: skipped, no GPU adapter ({e:#}); set WEFT_REQUIRE_GPU=1 to make this a failure");
			false
		}
	}
}

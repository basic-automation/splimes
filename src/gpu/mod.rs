//! The GPU backend and its process-wide device.

use std::sync::OnceLock;

pub use config::{GpuConfig, GpuInfo, GpuPoolStats};

use crate::{Error, Precision, Result, kernel::Method, prepare::Knots, time::Grid, value::Value};

mod config;
#[cfg(feature = "gpu")]
mod context;
#[cfg(feature = "gpu")]
pub mod run;
#[cfg(feature = "gpu")]
mod shader;

/// The configuration the GPU starts with; unset means [`GpuConfig::default`].
static CONFIG: OnceLock<GpuConfig> = OnceLock::new();

/// The configuration set so far, or the default.
fn config() -> GpuConfig {
	CONFIG.get().cloned().unwrap_or_default()
}

/// The configuration the device opens with. Called as it opens, this also fixes it: a
/// `configure_gpu` racing with start-up then fails, instead of reporting success for a
/// configuration that was never used.
#[cfg(feature = "gpu")]
fn fix_config() -> GpuConfig {
	CONFIG.get_or_init(GpuConfig::default).clone()
}

/// Sets the [`GpuConfig`] the GPU will start with.
///
/// # Errors
///
/// [`Error::GpuAlreadyConfigured`] if the GPU has already started (by [`prewarm_gpu`],
/// [`calibrate`](crate::calibrate) or a [`Backend::Gpu`](crate::Backend::Gpu) call), or if
/// a configuration was already set.
pub fn configure_gpu(config: GpuConfig) -> Result<()> {
	// Opening the device fixes the configuration (see `fix_config`), so this `set` is the
	// only check needed, and it can't race with start-up.
	CONFIG.set(config).map_err(|_| {
		#[cfg(feature = "gpu")]
		if context::started() {
			return Error::GpuAlreadyConfigured("the GPU has already started; configure it before its first use");
		}
		Error::GpuAlreadyConfigured("a configuration was already set")
	})
}

/// The configuration the GPU started with, or will start with.
#[must_use]
pub fn gpu_config() -> GpuConfig {
	config()
}

/// Starts the GPU now, so the first interpolation doesn't pay for it, and lets
/// [`Backend::Auto`](crate::Backend::Auto) use it (above its thresholds).
///
/// Opening the device and compiling the kernels takes from tens of milliseconds to a
/// second or more, depending on the driver. Call this at startup, off any latency-critical
/// path. It's idempotent: later calls return the same answer immediately.
///
/// # Errors
///
/// [`Error::GpuUnavailable`] if there's no usable adapter or the crate was built without
/// the `gpu` feature.
pub fn prewarm_gpu() -> Result<GpuInfo> {
	#[cfg(feature = "gpu")]
	return context::get().map(|ctx| ctx.info.clone());
	#[cfg(not(feature = "gpu"))]
	Err(no_gpu_feature())
}

/// [`configure_gpu`] then [`prewarm_gpu`].
///
/// # Errors
///
/// Either function's errors.
pub fn prewarm_gpu_with_config(config: GpuConfig) -> Result<GpuInfo> {
	configure_gpu(config)?;
	prewarm_gpu()
}

/// The GPU in use, if it has started successfully. Never blocks or starts it.
#[must_use]
// Not `const`: whether a public fn is const mustn't depend on a feature.
#[cfg_attr(not(feature = "gpu"), allow(clippy::missing_const_for_fn))]
pub fn gpu_info() -> Option<GpuInfo> {
	#[cfg(feature = "gpu")]
	return context::try_get().map(|ctx| ctx.info.clone());
	#[cfg(not(feature = "gpu"))]
	None
}

/// Buffer-pool counters, if the GPU has started successfully.
#[must_use]
// Not `const`: whether a public fn is const mustn't depend on a feature.
#[cfg_attr(not(feature = "gpu"), allow(clippy::missing_const_for_fn))]
pub fn gpu_pool_stats() -> Option<GpuPoolStats> {
	#[cfg(feature = "gpu")]
	return context::try_get().map(context::Context::pool_stats);
	#[cfg(not(feature = "gpu"))]
	None
}

/// Runs the kernel on the GPU.
#[cfg_attr(not(feature = "gpu"), allow(unused_variables, clippy::needless_pass_by_ref_mut))]
pub fn eval<V: Value>(knots: &Knots<'_, V>, grid: &Grid, method: &Method, precision: Precision, out: &mut [f64]) -> Result<()> {
	#[cfg(feature = "gpu")]
	return run::eval(context::get()?, knots, grid, method, precision, out);
	#[cfg(not(feature = "gpu"))]
	Err(no_gpu_feature())
}

/// Whether the GPU is open and can run `precision`. Never blocks and never opens it:
/// [`Backend::Auto`](crate::Backend::Auto) uses only a GPU the caller has started.
#[cfg_attr(not(feature = "gpu"), allow(unused_variables, clippy::missing_const_for_fn))]
pub fn ready(precision: Precision) -> bool {
	#[cfg(feature = "gpu")]
	return context::ready(precision);
	#[cfg(not(feature = "gpu"))]
	false
}

#[cfg(not(feature = "gpu"))]
fn no_gpu_feature() -> Error {
	Error::GpuUnavailable("splimes was built without the `gpu` feature".to_owned())
}

#[cfg(all(test, feature = "gpu"))]
mod tests {
	/// Opening the device (successfully or not) fixes the configuration, so a later
	/// `configure_gpu` can't report success for a configuration that's never used.
	#[test]
	fn opening_the_device_fixes_the_configuration() {
		let _ = super::context::get();
		assert!(super::CONFIG.get().is_some());
		assert!(super::configure_gpu(super::GpuConfig::minimal()).is_err());
	}
}

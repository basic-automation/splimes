//! The GPU backend and its process-wide device.

use std::{ops::ControlFlow, sync::OnceLock};

pub use config::{GpuConfig, GpuInfo, GpuMemoryReport, GpuPoolStats};

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

/// The GPU device's allocated memory, if the GPU has started successfully and its backend
/// can report it. Never blocks or starts the GPU.
///
/// Unlike [`gpu_pool_stats`], which counts only the buffer pool's idle sets, this is the
/// device's own allocator: every live buffer, pooled or not. Use it to check that memory
/// returns to a steady level between calls, or falls when the pool evicts.
///
/// It first lets wgpu free the buffers of work that has already finished (a non-blocking
/// poll), so a buffer dropped by a completed call is not counted.
///
/// `None` when the GPU hasn't started, the crate was built without the `gpu` feature, or
/// the backend has no allocator report: wgpu 30 reports on Vulkan and DX12, not on Metal
/// or OpenGL. Memory the driver allocates outside wgpu's allocator is never included.
#[must_use]
// Not `const`: whether a public fn is const mustn't depend on a feature.
#[cfg_attr(not(feature = "gpu"), allow(clippy::missing_const_for_fn))]
pub fn gpu_memory_report() -> Option<GpuMemoryReport> {
	#[cfg(feature = "gpu")]
	return context::try_get().and_then(context::Context::memory_report);
	#[cfg(not(feature = "gpu"))]
	None
}

/// Whether anything has begun opening the GPU device: it may still be opening, or have
/// failed to. Never blocks.
#[cfg_attr(not(feature = "gpu"), allow(clippy::missing_const_for_fn))]
pub fn started() -> bool {
	#[cfg(feature = "gpu")]
	return context::started();
	#[cfg(not(feature = "gpu"))]
	false
}

/// Runs the kernel on the GPU.
#[cfg_attr(not(feature = "gpu"), allow(unused_variables, clippy::needless_pass_by_ref_mut))]
pub fn eval<V: Value>(knots: &Knots<'_, V>, grid: &Grid, method: &Method, precision: Precision, out: &mut [f64]) -> Result<()> {
	#[cfg(feature = "gpu")]
	return run::eval(context::get()?, knots, grid, method, precision, out);
	#[cfg(not(feature = "gpu"))]
	Err(no_gpu_feature())
}

/// Where the GPU delivers its results, a chunk at a time, in grid order.
pub trait ChunkSink {
	/// The buffer for the normalised results of grid points `first..first + len`.
	fn dest(&mut self, first: usize, len: usize) -> &mut [f64];

	/// [`dest`](Self::dest)'s buffer now holds the chunk. Runs while the GPU computes the
	/// next chunk. `Break` stops the call: no further chunk is dispatched.
	fn filled(&mut self, first: usize, len: usize) -> ControlFlow<()>;
}

/// Runs the kernel on the GPU in chunks of at most `max_chunk` grid points (and at most
/// the configured [`GpuConfig::chunk_points`]), handing each to `sink` as it is read back.
#[cfg_attr(not(feature = "gpu"), allow(unused_variables))]
pub fn eval_chunks<V: Value>(knots: &Knots<'_, V>, grid: &Grid, method: &Method, precision: Precision, max_chunk: usize, sink: &mut dyn ChunkSink) -> Result<()> {
	#[cfg(feature = "gpu")]
	return run::eval_chunks(context::get()?, knots, grid, method, precision, max_chunk, sink);
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

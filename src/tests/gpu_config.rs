//! `GpuConfig` is applied, not ignored.
//!
//! `prewarm_gpu_with_config` used to take a `GpuConfig` and drop it on the floor — it forced
//! GPU initialization and carried a `TODO: Apply configuration to the global interpolator`.
//! The documented API therefore claimed a capability the code did not have.
//!
//! These tests pin the behaviour that replaced it. They deliberately avoid requiring a GPU:
//! the configuration is recorded *before* the interpolator initializes, so what a configuration
//! is worth can be checked on any machine, including CI without an adapter.

use crate::{GpuConfig, effective_gpu_config, gpu_config_applied};

/// The presets are distinguishable, which is what makes applying one meaningful.
///
/// If every preset produced the same numbers, "applying" a configuration would be untestable
/// and the API pointless.
#[test]
fn presets_differ_from_each_other() {
	let default = GpuConfig::default();
	let low = GpuConfig::low_memory();
	let high = GpuConfig::high_performance();
	let minimal = GpuConfig::minimal();

	assert!(low.buffer_pool.max_pool_memory < default.buffer_pool.max_pool_memory, "low_memory must pool less than default ({} vs {})", low.buffer_pool.max_pool_memory, default.buffer_pool.max_pool_memory);
	assert!(high.buffer_pool.max_pool_memory > default.buffer_pool.max_pool_memory, "high_performance must pool more than default ({} vs {})", high.buffer_pool.max_pool_memory, default.buffer_pool.max_pool_memory);
	assert!(minimal.buffer_pool.max_pool_memory <= low.buffer_pool.max_pool_memory, "minimal must not pool more than low_memory");
	assert!(high.num_staging_buffers >= default.num_staging_buffers, "high_performance must not stage fewer buffers than default");
}

/// The effective configuration is readable, and with nothing requested it is the default.
///
/// This test must not itself request a configuration: the request is a process-wide
/// `OnceLock`, so doing so would make the outcome depend on test execution order.
#[test]
fn effective_config_is_readable_and_defaults_when_unset() {
	let effective = effective_gpu_config();
	if gpu_config_applied() {
		// Another test in this process configured first — the only guarantee left is that the
		// reported configuration is the one in force, which the assertion below cannot check
		// without knowing which. Reading it must still not panic, which it just did not.
		return;
	}
	let default = GpuConfig::default();
	assert_eq!(effective.buffer_pool.max_pool_memory, default.buffer_pool.max_pool_memory, "unset means default pool size");
	assert_eq!(effective.buffer_pool.eviction_timeout_secs, default.buffer_pool.eviction_timeout_secs, "unset means default eviction timeout");
	assert_eq!(effective.num_staging_buffers, default.num_staging_buffers, "unset means default staging buffer count");
}

/// A configuration supplied before the GPU is up is *recorded*, and reported as applied.
///
/// This is the regression guard for the original defect: the old implementation would have
/// left `gpu_config_applied()` false forever, because it never stored anything.
///
/// Requesting a configuration mutates process-wide state, so this test tolerates having lost
/// the race to another test — what it must never tolerate is the request silently succeeding
/// while `gpu_config_applied()` stays false, which is precisely the old bug.
#[test]
fn requesting_a_config_is_recorded_or_honestly_refused() {
	let wanted = GpuConfig::low_memory();
	match crate::prewarm_gpu_with_config(wanted.clone()) {
		Ok(()) => {
			eprintln!("GPU CONFIG PATH: applied (config reached the interpolator)");
			assert!(gpu_config_applied(), "a successful request must report the configuration as applied");
			let effective = effective_gpu_config();
			assert_eq!(effective.buffer_pool.max_pool_memory, wanted.buffer_pool.max_pool_memory, "the effective configuration must be the one that was requested");
			assert_eq!(effective.num_staging_buffers, wanted.num_staging_buffers);
		}
		Err(e) => {
			// The three legitimate refusals: the GPU was already up, another caller configured
			// first, or this machine has no usable adapter. All are reported, never silent.
			let msg = e.to_string();
			eprintln!("GPU CONFIG PATH: refused -> {msg}");
			assert!(msg.contains("already initialized") || msg.contains("already requested") || msg.contains("adapter") || msg.contains("GPU"), "a refusal must say why; got: {msg}");
		}
	}
}

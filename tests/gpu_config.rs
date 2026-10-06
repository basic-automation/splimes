//! Opening the GPU fixes its configuration, so a `configure_gpu` that loses a race with
//! start-up fails instead of reporting success for a configuration that is never used.
//! Its own binary, because the configuration is process-wide and set once.

#![cfg(feature = "gpu")]

use splimes::{Error, GpuConfig};

#[test]
fn starting_the_gpu_fixes_its_configuration() {
	// Start (or try to start) the GPU without configuring it. Even with no adapter, the
	// attempt settles the configuration.
	let started = splimes::prewarm_gpu();
	assert_eq!(splimes::gpu_config(), GpuConfig::default());
	assert!(matches!(splimes::configure_gpu(GpuConfig::minimal()), Err(Error::GpuAlreadyConfigured(_))), "configuring after start-up must fail");
	assert_eq!(splimes::gpu_config(), GpuConfig::default(), "and must not change what's reported");
	if started.is_ok() {
		assert!(splimes::gpu_info().is_some());
	}
}

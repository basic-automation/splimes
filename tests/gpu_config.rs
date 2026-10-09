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

/// `WGPU_ADAPTER_NAME` that names no adapter makes the GPU unavailable, with a message that
/// says why, rather than panicking (wgpu's own helper for the variable does) or quietly
/// opening some other adapter. In a child process, because the variable is read once, as
/// the GPU starts.
#[test]
fn an_adapter_name_that_matches_nothing_is_refused() {
	const CHILD: &str = "SPLIMES_TEST_ADAPTER_NAME_CHILD";
	if std::env::var_os(CHILD).is_some() {
		match splimes::prewarm_gpu() {
			Err(Error::GpuUnavailable(reason)) => assert!(reason.contains("WGPU_ADAPTER_NAME") && reason.contains("matches no adapter"), "{reason}"),
			other => panic!("expected GpuUnavailable, got {other:?}"),
		}
		assert!(splimes::gpu_info().is_none());
		return;
	}
	let exe = std::env::current_exe().expect("the test binary's path");
	let out = std::process::Command::new(exe).args(["--exact", "an_adapter_name_that_matches_nothing_is_refused", "--nocapture", "--test-threads", "1"]).env(CHILD, "1").env("WGPU_ADAPTER_NAME", "no such adapter 5f3c").output().expect("the child runs");
	let stdout = String::from_utf8_lossy(&out.stdout);
	assert!(out.status.success(), "child failed:\n{stdout}\n{}", String::from_utf8_lossy(&out.stderr));
	assert!(stdout.contains("1 passed"), "the child ran no test:\n{stdout}");
}

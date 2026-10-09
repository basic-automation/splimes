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

/// Set in a child process this binary starts, to run one test's GPU start-up there.
const CHILD: &str = "SPLIMES_TEST_GPU_CONFIG_CHILD";

/// Runs `test` alone in a child process with wgpu's adapter variables set to `env`, and
/// fails unless it passes there. The variables are read once, as the GPU starts, so each
/// setting needs a fresh process.
fn in_child(test: &str, env: &[(&str, &str)]) {
	let exe = std::env::current_exe().expect("the test binary's path");
	let mut command = std::process::Command::new(exe);
	command.args(["--exact", test, "--nocapture", "--test-threads", "1"]).env(CHILD, "1").env_remove("WGPU_ADAPTER_NAME").env_remove("WGPU_BACKEND");
	for (key, value) in env {
		command.env(key, value);
	}
	let out = command.output().expect("the child runs");
	let stdout = String::from_utf8_lossy(&out.stdout);
	assert!(out.status.success(), "child failed:\n{stdout}\n{}", String::from_utf8_lossy(&out.stderr));
	assert!(stdout.contains("1 passed"), "the child ran no test:\n{stdout}");
}

/// `WGPU_ADAPTER_NAME` that names no adapter makes the GPU unavailable, with a message that
/// says why, rather than panicking (wgpu's own helper for the variable does) or quietly
/// opening some other adapter.
#[test]
fn an_adapter_name_that_matches_nothing_is_refused() {
	if std::env::var_os(CHILD).is_none() {
		return in_child("an_adapter_name_that_matches_nothing_is_refused", &[("WGPU_ADAPTER_NAME", "no such adapter 5f3c")]);
	}
	match splimes::prewarm_gpu() {
		Err(Error::GpuUnavailable(reason)) => assert!(reason.contains("WGPU_ADAPTER_NAME") && reason.contains("matches no adapter"), "{reason}"),
		other => panic!("expected GpuUnavailable, got {other:?}"),
	}
	assert!(splimes::gpu_info().is_none());
}

/// `WGPU_BACKEND` limits the graphics APIs tried: naming one this platform doesn't have
/// leaves no adapter, even where another API would have found one.
#[test]
fn wgpu_backend_limits_the_apis_tried() {
	// Metal exists only on Apple platforms, DX12 only on Windows.
	let absent = if cfg!(target_vendor = "apple") { "dx12" } else { "metal" };
	if std::env::var_os(CHILD).is_none() {
		return in_child("wgpu_backend_limits_the_apis_tried", &[("WGPU_BACKEND", absent)]);
	}
	assert!(matches!(splimes::prewarm_gpu(), Err(Error::GpuUnavailable(_))), "WGPU_BACKEND={absent} must leave no adapter");
	assert!(splimes::gpu_info().is_none());
}

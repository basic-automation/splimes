//! GPU prewarm / buffer-pool integration tests.
//!
//! ATTRIBUTE ORDER MATTERS HERE: `#[serial(gpu_tests)]` must stay ABOVE `#[test]`.
//! With `#[test]` outermost, rustc's builtin harness injects its
//! `rustc_test_entrypoint_marker` and the `#[serial]` proc macro then re-emits the item
//! without that marker's tokens, and the compiler ICEs with "attribute is missing tokens"
//! (rustc_ast/src/attr/mod.rs) — which made this whole target uncompilable and blocked
//! `cargo test --workspace`. That is upstream rust-lang/rust#100263, open since 2022, so a
//! toolchain bump will not save you. Keep the order.

use serial_test::serial;
use splimes::{
    effective_gpu_config, gpu_buffer_pool_stats, gpu_config_applied, prewarm_gpu, prewarm_gpu_with_config, GpuConfig,
};

#[serial(gpu_tests)]
#[test]
fn test_gpu_prewarm_succeeds() {
    let result = prewarm_gpu();
    assert!(result.is_ok(), "GPU prewarming should succeed");
}

/// Every `GpuConfig` preset is an acceptable input, and a request is either **applied** or
/// **refused with a stated reason** — never silently ignored.
///
/// These four presets used to be four separate tests, each asserting `is_ok()`. They passed
/// vacuously: `prewarm_gpu_with_config` ignored its argument entirely (it carried a
/// `TODO: Apply configuration to the global interpolator`), so it behaved exactly like
/// `prewarm_gpu()` and could not fail. They therefore proved nothing about configuration.
///
/// Now that the configuration is genuinely applied, only the FIRST request in a process can
/// win — the interpolator is a singleton whose buffer pool and staging buffers are sized once
/// at construction. Asserting `is_ok()` for four different presets in one process would assert
/// something physically impossible. So this asserts the contract that is actually true
/// regardless of which test in this binary ran first.
#[serial(gpu_tests)]
#[test]
fn test_gpu_config_presets_are_applied_or_refused_with_a_reason() {
    for (name, config) in [
        ("default", GpuConfig::default()),
        ("low_memory", GpuConfig::low_memory()),
        ("high_performance", GpuConfig::high_performance()),
        ("minimal", GpuConfig::minimal()),
    ] {
        match prewarm_gpu_with_config(config.clone()) {
            Ok(()) => {
                assert!(gpu_config_applied(), "{name}: a successful request must report the config as applied");
                let effective = effective_gpu_config();
                assert_eq!(
                    effective.buffer_pool.max_pool_memory, config.buffer_pool.max_pool_memory,
                    "{name}: the effective pool size must be the one requested"
                );
                assert_eq!(
                    effective.num_staging_buffers, config.num_staging_buffers,
                    "{name}: the effective staging count must be the one requested"
                );
            }
            Err(e) => {
                let msg = e.to_string();
                assert!(
                    msg.contains("already initialized") || msg.contains("already requested") || msg.contains("GPU") || msg.contains("adapter"),
                    "{name}: a refusal must say why; got: {msg}"
                );
            }
        }
    }
}

/// The regression guard for the original defect: a configuration must never be accepted and
/// then quietly discarded.
///
/// If `prewarm_gpu_with_config` reports success, `gpu_config_applied()` must agree that a
/// configuration is in force. The old no-op implementation would fail this — it returned
/// `Ok(())` while storing nothing.
#[serial(gpu_tests)]
#[test]
fn test_gpu_config_is_never_silently_ignored() {
    if prewarm_gpu_with_config(GpuConfig::low_memory()).is_ok() {
        assert!(gpu_config_applied(), "a config accepted without error must actually be in force");
    }
}

#[serial(gpu_tests)]
#[test]
fn test_buffer_pool_stats_available() {
    // Ensure GPU is initialized
    let _ = prewarm_gpu();

    let stats_result = gpu_buffer_pool_stats();
    assert!(stats_result.is_ok(), "Buffer pool stats should be accessible");

    let stats = stats_result.unwrap();
    // Verify stats structure is valid
    // After prewarming, pool may be empty (buffers released back to pool)
    // Stats should show reasonable memory usage (under 1GB)
    assert!(stats.total_allocated_bytes < 1024 * 1024 * 1024,
        "Total allocated should be reasonable (< 1GB)");
}

#[serial(gpu_tests)]
#[test]
fn test_config_presets_are_different() {
    let default = GpuConfig::default();
    let low_mem = GpuConfig::low_memory();
    let high_perf = GpuConfig::high_performance();
    let minimal = GpuConfig::minimal();

    // Verify each preset has different pool sizes
    assert_ne!(
        default.buffer_pool.max_pool_memory,
        low_mem.buffer_pool.max_pool_memory,
        "Default and low_memory should have different pool sizes"
    );
    assert_ne!(
        default.buffer_pool.max_pool_memory,
        high_perf.buffer_pool.max_pool_memory,
        "Default and high_performance should have different pool sizes"
    );
    assert_ne!(
        low_mem.buffer_pool.max_pool_memory,
        minimal.buffer_pool.max_pool_memory,
        "Low_memory and minimal should have different pool sizes"
    );

    // Verify staging buffer counts are different
    assert_ne!(
        default.num_staging_buffers,
        low_mem.num_staging_buffers,
        "Default and low_memory should have different staging buffer counts"
    );
}

#[serial(gpu_tests)]
#[test]
fn test_config_presets_ordered_by_resources() {
    let minimal = GpuConfig::minimal();
    let low_mem = GpuConfig::low_memory();
    let default = GpuConfig::default();
    let high_perf = GpuConfig::high_performance();

    // Verify pool sizes increase monotonically
    assert!(
        minimal.buffer_pool.max_pool_memory < low_mem.buffer_pool.max_pool_memory,
        "Pool memory should increase from minimal to low_memory"
    );
    assert!(
        low_mem.buffer_pool.max_pool_memory < default.buffer_pool.max_pool_memory,
        "Pool memory should increase from low_memory to default"
    );
    assert!(
        default.buffer_pool.max_pool_memory < high_perf.buffer_pool.max_pool_memory,
        "Pool memory should increase from default to high_performance"
    );

    // Verify staging buffer counts increase monotonically
    assert!(
        minimal.num_staging_buffers <= low_mem.num_staging_buffers,
        "Staging buffers should increase from minimal to low_memory"
    );
    assert!(
        low_mem.num_staging_buffers <= default.num_staging_buffers,
        "Staging buffers should increase from low_memory to default"
    );
    assert!(
        default.num_staging_buffers <= high_perf.num_staging_buffers,
        "Staging buffers should increase from default to high_performance"
    );
}

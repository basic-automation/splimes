use serial_test::serial;
use splimes::{
    prewarm_gpu, gpu_buffer_pool_stats, GpuConfig, prewarm_gpu_with_config,
};

#[test]
#[serial(gpu_tests)]
fn test_gpu_prewarm_succeeds() {
    let result = prewarm_gpu();
    assert!(result.is_ok(), "GPU prewarming should succeed");
}

#[test]
#[serial(gpu_tests)]
fn test_gpu_prewarm_with_default_config() {
    let config = GpuConfig::default();
    let result = prewarm_gpu_with_config(config);
    assert!(result.is_ok(), "GPU prewarming with default config should succeed");
}

#[test]
#[serial(gpu_tests)]
fn test_gpu_prewarm_with_low_memory_config() {
    let config = GpuConfig::low_memory();
    let result = prewarm_gpu_with_config(config);
    assert!(result.is_ok(), "GPU prewarming with low_memory config should succeed");
}

#[test]
#[serial(gpu_tests)]
fn test_gpu_prewarm_with_high_performance_config() {
    let config = GpuConfig::high_performance();
    let result = prewarm_gpu_with_config(config);
    assert!(result.is_ok(), "GPU prewarming with high_performance config should succeed");
}

#[test]
#[serial(gpu_tests)]
fn test_gpu_prewarm_with_minimal_config() {
    let config = GpuConfig::minimal();
    let result = prewarm_gpu_with_config(config);
    assert!(result.is_ok(), "GPU prewarming with minimal config should succeed");
}

#[test]
#[serial(gpu_tests)]
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

#[test]
#[serial(gpu_tests)]
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

#[test]
#[serial(gpu_tests)]
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

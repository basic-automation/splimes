use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use std::hint::black_box;
use splimes::{prewarm_gpu_with_config, GpuConfig};

criterion_group!(
    name = gpu_benches;
    config = Criterion::default().sample_size(10);
    targets = bench_gpu_config_impact
);
criterion_main!(gpu_benches);

fn bench_gpu_config_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_config_presets");
    group.sample_size(10);

    let configs = vec![
        ("minimal", GpuConfig::minimal()),
        ("low_memory", GpuConfig::low_memory()),
        ("default", GpuConfig::default()),
        ("high_performance", GpuConfig::high_performance()),
    ];

    for (name, config) in configs {
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            name,
            |b, _| {
                b.iter(|| {
                    // Each benchmark iteration tests config application
                    let _ = prewarm_gpu_with_config(black_box(config.clone()));
                });
            },
        );
    }

    group.finish();
}

/// Benchmark showing pool memory configurations
#[allow(dead_code)]
fn bench_pool_memory_configs() {
    println!("\n=== Buffer Pool Memory Configurations ===");

    let configs = vec![
        ("minimal", GpuConfig::minimal()),
        ("low_memory", GpuConfig::low_memory()),
        ("default", GpuConfig::default()),
        ("high_performance", GpuConfig::high_performance()),
    ];

    for (name, config) in configs {
        let pool_mb = config.buffer_pool.max_pool_memory as f64 / 1024.0 / 1024.0;
        println!(
            "{:20} | Pool: {:6.1}MB | Staging Buffers: {} | Command Batch: {}",
            name,
            pool_mb,
            config.num_staging_buffers,
            config.max_command_batch_size
        );
    }
}

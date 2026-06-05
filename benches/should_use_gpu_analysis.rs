use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, PlotConfiguration, AxisScale};
use splimes::{Point, Resolution, Spline, cpu_interpolate, parallel_interpolate, gpu_interpolate, prewarm_gpu};
use chrono::{DateTime, Utc, Duration};
use bigdecimal::BigDecimal;

criterion_group!(
    name = gpu_analysis;
    config = Criterion::default().sample_size(10);
    targets = analyze_should_use_gpu_thresholds
);
criterion_main!(gpu_analysis);

/// Benchmark suite to determine optimal GPU usage thresholds
///
/// Tests should_use_gpu logic across different dataset sizes:
/// - Small: < 100 inputs
/// - Medium: 100 to 100K inputs
/// - Large: 1M inputs (GPU competitive with Parallel)
/// - Very Large: 5M+ inputs (GPU clearly better)
fn analyze_should_use_gpu_thresholds(c: &mut Criterion) {
    // Pre-warm GPU before benchmarks to ensure fair comparison
    let _ = prewarm_gpu();

    let mut group = c.benchmark_group("should_use_gpu_thresholds");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    group.sample_size(10);

    // Test dataset sizes that correspond to should_use_gpu thresholds
    // Note: Larger sizes take significantly longer. Start with smaller sizes for quick feedback.
    let test_sizes = vec![
        (100, "100_boundary"),
        (1_000, "1k_medium"),
        (10_000, "10k_medium"),
        (100_000, "100k_boundary"),
    ];

    for (input_count, label) in test_sizes {
        // Generate test data
        let start = Utc::now();
        let points = generate_test_points(input_count, start);
        let end = start + Duration::hours(1);
        let resolution = Resolution::Minutes;
        let spline = Spline::Cubic;

        // Estimate output points based on time range (60 minutes at minute resolution)
        let _estimated_output = 60; // 1 hour at minute resolution = 60 output points

        group.bench_with_input(
            BenchmarkId::new("CPU", label),
            &input_count,
            |b, &_count| {
                let rt = tokio::runtime::Runtime::new().unwrap();
                b.iter(|| {
                    rt.block_on(async {
                        let _ = cpu_interpolate(
                            &mut points.clone(),
                            start,
                            end,
                            resolution,
                            spline,
                        ).await;
                    });
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("Parallel", label),
            &input_count,
            |b, &_count| {
                let rt = tokio::runtime::Runtime::new().unwrap();
                b.iter(|| {
                    rt.block_on(async {
                        let _ = parallel_interpolate(
                            &mut points.clone(),
                            &start,
                            &end,
                            spline,
                            resolution,
                        ).await;
                    });
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("GPU", label),
            &input_count,
            |b, &_count| {
                let rt = tokio::runtime::Runtime::new().unwrap();
                b.iter(|| {
                    rt.block_on(async {
                        let _ = gpu_interpolate(
                            &mut points.clone(),
                            start,
                            end,
                            resolution,
                            spline,
                        ).await;
                    });
                });
            },
        );
    }

    group.finish();

    print_threshold_analysis();
}

/// Print analysis of should_use_gpu thresholds
fn print_threshold_analysis() {
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  should_use_gpu Threshold Analysis                          ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║                                                              ║");
    println!("║ Threshold 1: Small Datasets (< 100 inputs)                  ║");
    println!("║   Current Strategy: CPU                                      ║");
    println!("║   Rationale: Parallel overhead dominates                     ║");
    println!("║   Expected: CPU ≈ Parallel                                   ║");
    println!("║                                                              ║");
    println!("║ Threshold 2: Medium Datasets (100 - 100K inputs)            ║");
    println!("║   Current Strategy: Parallel                                 ║");
    println!("║   Rationale: Parallel 2x faster than GPU                     ║");
    println!("║   Expected: Parallel ≈ 0.5x GPU time                         ║");
    println!("║                                                              ║");
    println!("║ Threshold 3: Large Datasets (1M inputs)                      ║");
    println!("║   Current Strategy: GpuThenParallel                          ║");
    println!("║   Rationale: GPU competitive, within 10% of Parallel         ║");
    println!("║   Expected: GPU ≈ 1.0x-1.1x Parallel time                    ║");
    println!("║                                                              ║");
    println!("║ Threshold 4: Very Large Datasets (5M-10M inputs)            ║");
    println!("║   Current Strategy: GpuPrimary                               ║");
    println!("║   Rationale: GPU 50% faster than Parallel                    ║");
    println!("║   Expected: GPU ≈ 0.5x Parallel time                         ║");
    println!("║                                                              ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!("\nBenchmark Results Above Show Actual Performance");
    println!("If thresholds need adjustment, modify should_use_gpu() accordingly\n");
}

/// Generate synthetic test points
fn generate_test_points(count: usize, start: DateTime<Utc>) -> Vec<Point> {
    (0..count)
        .map(|i| Point {
            timestamp: start + Duration::minutes(i as i64),
            #[allow(clippy::cast_precision_loss)]
            value: BigDecimal::try_from((i as f64).sin()).unwrap_or_else(|_| BigDecimal::from(0)),
        })
        .collect()
}

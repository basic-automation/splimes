//! Command batching: many independent series through one `run_f64_batch` call against
//! the same series through one `run_f64` call each.
//!
//! ```text
//! cargo bench --bench batch                        # everything
//! cargo bench --bench batch -- 'batch/gpu-f64'     # one backend
//! cargo bench --bench batch -- --quick             # smoke run
//! ```
//!
//! Each case is `series` series of `points` grid points, each grid a millisecond grid
//! across its own irregular series of `points / 4` knots (at least 4), Cubic, f64 in and
//! out. Throughput is grid points over the whole batch. `SPLIMES_BATCH_MAX_POINTS` caps
//! `series × points` (default 2²⁴), skipping larger cases, so the full matrix (up to
//! 4,096 × 64 Ki = 2²⁸ points, about 10 GB of output) doesn't exhaust memory.

use std::hint::black_box;

use chrono::{DateTime, TimeDelta, Utc};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use splimes::{Backend, Interpolator, Precision, Resolution, Spline};

struct Series {
	timestamps: Vec<DateTime<Utc>>,
	values: Vec<f64>,
	start: DateTime<Utc>,
	end: DateTime<Utc>,
}

/// Series `seed`: `points / 4` (at least 4) jittered knots across a `points`-millisecond grid.
fn series(seed: i64, points: i64) -> Series {
	let knots = (points / 4).max(4);
	let start = DateTime::<Utc>::UNIX_EPOCH + TimeDelta::seconds(seed);
	let spacing_us = 1000.0 * points as f64 / knots as f64;
	let timestamps = (0..knots).map(|i| start + TimeDelta::microseconds((spacing_us * (i as f64 + 0.4 * (((i + seed) * 7919) % 100) as f64 / 100.0)) as i64)).collect();
	let values = (0..knots).map(|i| ((i + seed) as f64 * 0.01).sin() * 100.0).collect();
	Series { timestamps, values, start, end: start + TimeDelta::milliseconds(points - 1) }
}

fn max_points() -> u64 {
	std::env::var("SPLIMES_BATCH_MAX_POINTS").ok().and_then(|v| v.parse().ok()).unwrap_or(1 << 24)
}

fn batch(c: &mut Criterion) {
	let mut configs = vec![("cpu", Backend::Cpu, Precision::F64), ("parallel", Backend::Parallel, Precision::F64)];
	match splimes::prewarm_gpu() {
		Ok(info) => {
			if info.supports_f64 {
				configs.push(("gpu-f64", Backend::Gpu, Precision::F64));
			}
			configs.push(("gpu-f32", Backend::Gpu, Precision::F32));
		}
		Err(e) => eprintln!("GPU benchmarks skipped: {e}"),
	}
	let cap = max_points();
	let mut group = c.benchmark_group("batch");
	group.sample_size(10);
	for count in [1_usize, 16, 256, 4096] {
		for points in [1_i64 << 6, 1 << 10, 1 << 13, 1 << 16] {
			let total = count as u64 * points as u64;
			if total > cap {
				continue;
			}
			let all: Vec<Series> = (0..count as i64).map(|s| series(s, points)).collect();
			let jobs: Vec<_> = all.iter().map(|s| (s.timestamps.as_slice(), s.values.as_slice(), s.start, s.end)).collect();
			group.throughput(Throughput::Elements(total));
			let size = format!("{count}x{points}");
			for &(name, backend, precision) in &configs {
				let interpolator = Interpolator::new(Spline::Cubic, Resolution::Milliseconds).backend(backend).gpu_precision(precision);
				group.bench_with_input(BenchmarkId::new(format!("{name}/batched"), &size), &jobs, |b, jobs| {
					b.iter(|| {
						let out = interpolator.run_f64_batch(jobs);
						assert!(out.iter().all(Result::is_ok));
						black_box(out)
					});
				});
				group.bench_with_input(BenchmarkId::new(format!("{name}/per-call"), &size), &jobs, |b, jobs| b.iter(|| black_box(jobs.iter().map(|&(t, v, s, e)| interpolator.run_f64(t, v, s, e).expect("runs")).collect::<Vec<_>>())));
			}
		}
	}
	group.finish();
}

criterion_group!(benches, batch);
criterion_main!(benches);

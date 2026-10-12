//! The GPU buffer pool's accounting, exactly: which calls create a buffer set, which reuse
//! one, what is evicted when the idle sets exceed `max_pool_bytes`, and what
//! `gpu_pool_stats` reports.
//!
//! Its own binary, because the pool is process-wide: every count below is from a pool
//! that starts empty, with a cap smaller than two buffer sets.

mod common;

use chrono::{DateTime, TimeDelta, Utc};
use common::epoch;
use splimes::{Backend, GpuConfig, GpuPoolStats, Interpolator, Precision, Resolution, Spline};

const CHUNK: u32 = 1_024;

/// A buffer set's bytes for `capacity` grid points: output and read-back values, a 12-byte
/// time per point, and 64 bytes of parameters.
const fn set_bytes(precision: Precision, capacity: u64) -> u64 {
	let value = if matches!(precision, Precision::F64) { 8 } else { 4 };
	capacity * (2 * value + 12) + 64
}

/// Room for exactly one `f32` set of a full chunk and one of a 100-point call (rounded up
/// to 128 points): less than two full sets.
const CAP: u64 = set_bytes(Precision::F32, CHUNK as u64) + set_bytes(Precision::F32, 128);

fn series() -> (Vec<DateTime<Utc>>, Vec<f64>) {
	let ts = (0..40).map(|i| epoch() + TimeDelta::microseconds(i * 100 + (i * 37) % 60)).collect();
	let vs = (0..40).map(|i| (f64::from(i) * 0.3).sin() * 10.0).collect();
	(ts, vs)
}

/// One GPU call over `points` microsecond grid points, checked against the CPU, and the
/// pool's counters afterwards.
fn call(precision: Precision, points: i64) -> GpuPoolStats {
	let (ts, vs) = series();
	let (start, end) = (ts[0], ts[0] + TimeDelta::microseconds(points - 1));
	let make = |backend| Interpolator::new(Spline::Cubic, Resolution::Microseconds).backend(backend).gpu_precision(precision);
	let gpu = make(Backend::Gpu).run_f64(&ts, &vs, start, end).expect("gpu");
	let cpu = make(Backend::Cpu).run_f64(&ts, &vs, start, end).expect("cpu");
	assert_eq!(gpu.backend(), Backend::Gpu);
	assert_eq!(gpu.len(), usize::try_from(points).expect("small"));
	let tolerance = if precision == Precision::F64 { 1e-9 } else { 1e-3 };
	for (i, (g, c)) in gpu.values().iter().zip(cpu.values()).enumerate() {
		assert!((g - c).abs() <= tolerance, "{precision} {points} points, point {i}: gpu {g} vs cpu {c}");
	}
	let stats = splimes::gpu_pool_stats().expect("GPU started");
	assert!(stats.idle_bytes <= CAP, "the idle sets stay under the cap: {stats:?}");
	stats
}

fn expect(stats: GpuPoolStats, idle_sets: usize, idle_bytes: u64, created: u64, reused: u64, evicted: u64, step: &str) {
	let want = (idle_sets, idle_bytes, created, reused, evicted);
	assert_eq!((stats.idle_sets, stats.idle_bytes, stats.created, stats.reused, stats.evicted), want, "{step}: {stats:?}, want (idle_sets, idle_bytes, created, reused, evicted) = {want:?}");
}

#[test]
fn pool_accounting_is_exact() {
	splimes::configure_gpu(GpuConfig::default().with_chunk_points(CHUNK).with_max_pool_bytes(CAP)).expect("first configuration succeeds");
	let Some(info) = common::gpu_or_skip("pool_accounting_is_exact") else { return };
	let (f32_small, f32_chunk) = (set_bytes(Precision::F32, 128), set_bytes(Precision::F32, CHUNK.into()));
	expect(splimes::gpu_pool_stats().expect("GPU started"), 0, 0, 0, 0, 0, "fresh pool");

	// A 100-point call creates a set rounded up to 128 points and keeps it.
	expect(call(Precision::F32, 100), 1, f32_small, 1, 0, 0, "first small call");

	// Three full chunks: the idle 128-point set is too small, so the first two chunks
	// create a set each, and the third reuses the first chunk's set within the call. Both
	// go back to the pool, which is then over the cap by one full set: the largest goes,
	// leaving exactly the cap, which is not over it.
	expect(call(Precision::F32, 3 * i64::from(CHUNK)), 2, CAP, 3, 0, 1, "three-chunk call");

	// A small call reuses an idle set and frees nothing.
	expect(call(Precision::F32, 100), 2, CAP, 3, 1, 1, "second small call");

	// A call one point over a chunk takes the idle full set for its first chunk and the
	// idle small one for its one-point tail: nothing created.
	expect(call(Precision::F32, i64::from(CHUNK) + 1), 2, CAP, 3, 3, 1, "chunk-and-one call");

	if !info.supports_f64 {
		eprintln!("pool_accounting_is_exact: {} has no f64; the f64 half is skipped", info.name);
		return;
	}
	// Sets are never shared between precisions: a 300-point f64 call creates its own
	// 512-point set, although the idle 1,024-point f32 set is big enough. Over the cap
	// again, the largest set goes, whatever its precision.
	let f64_512 = set_bytes(Precision::F64, 512);
	let kept = f32_small + f64_512;
	assert!(kept + f32_chunk > CAP && kept <= CAP);
	expect(call(Precision::F64, 300), 2, kept, 4, 3, 2, "300-point f64 call");
	expect(call(Precision::F64, 100), 2, kept, 4, 4, 2, "100-point f64 call");

	// Three full f64 chunks: two new sets, as the idle 512-point one is too small. One
	// release is then over the cap by both, and must free both, not stop after the first.
	assert!(kept + set_bytes(Precision::F64, CHUNK.into()) > CAP);
	expect(call(Precision::F64, 3 * i64::from(CHUNK)), 2, kept, 6, 4, 4, "three-chunk f64 call");
}

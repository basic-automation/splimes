//! `gpu_memory_report`: the device's own allocator, not just the pool's idle sets. It rises
//! when a call leaves a buffer set in the pool, holds steady across repeated calls (nothing
//! leaks outside the pool), and falls when the pool evicts.
//!
//! Its own binary, because the pool and its cap are process-wide.

mod common;

use chrono::{DateTime, TimeDelta, Utc};
use common::epoch;
use splimes::{Backend, GpuConfig, GpuMemoryReport, Interpolator, Precision, Resolution, Spline};

const CHUNK: u32 = 1 << 16;

/// The pool's own size for a set of `capacity` points (see `tests/gpu_pool.rs`).
const fn set_bytes(precision: Precision, capacity: u64) -> u64 {
	let value = if matches!(precision, Precision::F64) { 8 } else { 4 };
	capacity * (2 * value + 12) + 64
}

/// The bytes of a set's three per-point buffers, which the allocator must hold at least.
const fn per_point_bytes(precision: Precision, capacity: u64) -> u64 {
	set_bytes(precision, capacity) - 64
}

/// Room for one full f32 chunk set and nothing more, so any second set evicts it.
const CAP: u64 = set_bytes(Precision::F32, CHUNK as u64);

fn call(precision: Precision, points: i64) {
	let ts: Vec<DateTime<Utc>> = (0..40).map(|i| epoch() + TimeDelta::microseconds(i * 100_000 + (i * 37) % 60)).collect();
	let vs: Vec<f64> = (0..40).map(|i| (f64::from(i) * 0.3).sin()).collect();
	let out = Interpolator::new(Spline::Cubic, Resolution::Microseconds).backend(Backend::Gpu).gpu_precision(precision).run_f64(&ts, &vs, ts[0], ts[0] + TimeDelta::microseconds(points - 1)).expect("gpu");
	assert_eq!(out.backend(), Backend::Gpu);
}

fn report(step: &str) -> GpuMemoryReport {
	let r = splimes::gpu_memory_report().unwrap_or_else(|| panic!("{step}: no report"));
	assert!(r.reserved_bytes >= r.allocated_bytes, "{step}: {r:?}");
	eprintln!("{step}: {r:?}, pool {:?}", splimes::gpu_pool_stats());
	r
}

#[test]
fn device_memory_rises_holds_and_falls_with_the_pool() {
	splimes::configure_gpu(GpuConfig::default().with_chunk_points(CHUNK).with_max_pool_bytes(CAP)).expect("first configuration succeeds");
	assert_eq!(splimes::gpu_memory_report(), None, "no report before the GPU starts");
	let Some(info) = common::gpu_or_skip("device_memory_rises_holds_and_falls_with_the_pool") else {
		assert_eq!(splimes::gpu_memory_report(), None);
		return;
	};
	if splimes::gpu_memory_report().is_none() {
		// wgpu 30 has an allocator report on Vulkan and DX12 only.
		assert!(!matches!(info.api.as_str(), "vulkan" | "dx12"), "{} ({}) should report its allocator", info.name, info.api);
		eprintln!("device_memory_rises_holds_and_falls_with_the_pool: skipped, {} ({}) has no allocator report", info.name, info.api);
		return;
	}
	let start = report("started");

	// A full chunk leaves one set in the pool: the device holds its buffers more. (Less a
	// few small allocations wgpu made at start-up and has freed since: 4 KiB covers them.)
	call(Precision::F32, CHUNK.into());
	let held = report("one f32 set pooled");
	assert_eq!(splimes::gpu_pool_stats().expect("started").idle_sets, 1);
	let set = per_point_bytes(Precision::F32, CHUNK.into());
	assert!(held.allocated_bytes + 4096 >= start.allocated_bytes + set, "the pooled set's {set} bytes show: {start:?} -> {held:?}");

	// The same call again reuses that set, and everything else it allocated (knots, params
	// upload) is freed: the device's allocation comes back to exactly the same level.
	for i in 0..5 {
		call(Precision::F32, CHUNK.into());
		assert_eq!(report(&format!("repeat {i}")).allocated_bytes, held.allocated_bytes, "repeat {i} leaked outside the pool");
	}

	if !info.supports_f64 {
		eprintln!("device_memory_rises_holds_and_falls_with_the_pool: {} has no f64; the eviction half is skipped", info.name);
		return;
	}
	// A small f64 call can't reuse the f32 set, so it adds its own, which puts the pool over
	// its cap; the largest set (the f32 one) is evicted, and the device lets go of it.
	call(Precision::F64, 100);
	let pool = splimes::gpu_pool_stats().expect("started");
	assert_eq!((pool.idle_sets, pool.evicted), (1, 1), "{pool:?}");
	let after = report("f32 set evicted");
	let small = per_point_bytes(Precision::F64, 128);
	assert!(after.allocated_bytes < held.allocated_bytes, "eviction frees device memory: {held:?} -> {after:?}");
	// The evicted set's buffers go, less the small set that replaced it; 16 KiB covers the
	// allocator's alignment of four small buffers.
	assert!(held.allocated_bytes - after.allocated_bytes + small + 16 * 1024 >= set, "fell by too little: {held:?} -> {after:?}");
}

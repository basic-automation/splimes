//! Device-memory stability: many GPU calls of mixed sizes and precisions, with the device's
//! allocated memory (`gpu_memory_report`) and the pool's idle bytes (`gpu_pool_stats`)
//! after each round. Memory that keeps rising round after round once the pool is warm is a
//! leak, inside the pool or outside it.
//!
//! ```text
//! cargo run --release --example gpu_memory -- [rounds] [max_points]
//! ```
//!
//! Defaults: 20 rounds, grids up to 4 Mi points. Prints one tab-separated line per round,
//! then a summary; exits with an error if allocated memory after the last round is above
//! its level after the first (warm-up) round. No timings.

use chrono::{DateTime, TimeDelta, Utc};
use splimes::{Backend, GpuConfig, Interpolator, Precision, Resolution, Spline};

fn main() -> Result<(), Box<dyn std::error::Error>> {
	let mut args = std::env::args().skip(1);
	let rounds: usize = args.next().map_or(Ok(20), |a| a.parse())?;
	let max_points: i64 = args.next().map_or(Ok(4 << 20), |a| a.parse())?;
	// A pool smaller than the largest calls, so rounds both reuse and evict.
	splimes::configure_gpu(GpuConfig::low_memory())?;
	let info = match splimes::prewarm_gpu() {
		Ok(info) => info,
		Err(e) => {
			println!("{e}");
			return Ok(());
		}
	};
	println!("GPU: {} ({}), f64 {}", info.name, info.api, if info.supports_f64 { "supported" } else { "unsupported" });
	let Some(start) = splimes::gpu_memory_report() else {
		println!("{} ({}) has no allocator report; nothing to measure", info.name, info.api);
		return Ok(());
	};
	println!("start\tallocated {}\treserved {}", start.allocated_bytes, start.reserved_bytes);

	let precisions: &[Precision] = if info.supports_f64 { &[Precision::F64, Precision::F32] } else { &[Precision::F32] };
	// Sizes from one point to `max_points`, by factors of four, plus an odd one in between.
	let mut sizes = vec![1_i64];
	while let Some(&last) = sizes.last()
		&& last * 4 <= max_points
	{
		sizes.push(last * 4);
		sizes.push(last * 4 + last / 2 + 1);
	}
	sizes.retain(|&s| s <= max_points);

	let (ts, vs) = series();
	println!("round\tcalls\tallocated_bytes\treserved_bytes\tallocations\tblocks\tpool_idle_bytes\tpool_created\tpool_reused\tpool_evicted");
	let (mut first, mut peak, mut last) = (None, 0_u64, 0_u64);
	for round in 1..=rounds {
		let mut calls = 0;
		for &precision in precisions {
			for &points in &sizes {
				let end = ts[0] + TimeDelta::microseconds(points - 1);
				Interpolator::new(Spline::Cubic, Resolution::Microseconds).backend(Backend::Gpu).gpu_precision(precision).run_f64(&ts, &vs, ts[0], end)?;
				calls += 1;
			}
		}
		let m = splimes::gpu_memory_report().ok_or("the allocator report disappeared")?;
		let p = splimes::gpu_pool_stats().ok_or("the GPU stopped")?;
		println!("{round}\t{calls}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}", m.allocated_bytes, m.reserved_bytes, m.allocations, m.blocks, p.idle_bytes, p.created, p.reused, p.evicted);
		first.get_or_insert(m.allocated_bytes);
		peak = peak.max(m.allocated_bytes);
		last = m.allocated_bytes;
	}
	let Some(first) = first else { return Ok(()) };
	println!("summary\tafter round 1 {first}\tpeak {peak}\tlast {last}\tdrift {}", i128::from(last) - i128::from(first));
	if last > first {
		return Err(format!("allocated device memory grew from {first} to {last} bytes after warm-up").into());
	}
	Ok(())
}

/// Irregular knots spanning the largest grid.
fn series() -> (Vec<DateTime<Utc>>, Vec<f64>) {
	let epoch = DateTime::<Utc>::UNIX_EPOCH;
	let ts = (0..4_096_i64).map(|i| epoch + TimeDelta::microseconds(i * 2_000 + (i * 7_919) % 900)).collect();
	let vs = (0..4_096).map(|i| (f64::from(i) * 0.01).sin() * 100.0).collect();
	(ts, vs)
}

//! `GpuConfig::with_chunk_points(0)` is documented as one point per dispatch: the smallest
//! chunk, where every grid point is its own dispatch, read back while the next computes.
//! Its own binary, because the configuration is process-wide and set once.

mod common;

use chrono::{DateTime, TimeDelta, Utc};
use common::epoch;
use splimes::{Backend, GpuConfig, Interpolator, Precision, Resolution, Spline};

fn series() -> (Vec<DateTime<Utc>>, Vec<f64>) {
	let ts = (0..12).map(|i| epoch() + TimeDelta::seconds(i * 10 + (i * 7) % 5)).collect();
	let vs = (0..12).map(|i| (f64::from(i) * 0.9).cos() * 3.0 + f64::from(i)).collect();
	(ts, vs)
}

#[test]
fn a_zero_chunk_is_one_point_per_dispatch() {
	splimes::configure_gpu(GpuConfig::default().with_chunk_points(0)).expect("first configuration succeeds");
	let Some(info) = common::gpu_or_skip("a_zero_chunk_is_one_point_per_dispatch") else { return };
	assert_eq!(splimes::gpu_config().chunk_points, 0, "the configuration is reported as given");
	let (ts, vs) = series();
	// 161 grid points, from before the first knot to after the last: raw, interpolated and
	// extrapolated points, each in its own dispatch.
	let (start, end) = (ts[0] - TimeDelta::seconds(25), ts[0] + TimeDelta::seconds(135));
	let precisions = if info.supports_f64 { vec![Precision::F64, Precision::F32] } else { vec![Precision::F32] };
	for &precision in &precisions {
		for spline in [Spline::Linear, Spline::Cubic, Spline::Polynomial(4, None)] {
			let make = |backend| Interpolator::new(spline, Resolution::Seconds).backend(backend).gpu_precision(precision);
			let gpu = make(Backend::Gpu).run_f64(&ts, &vs, start, end).expect("gpu");
			let cpu = make(Backend::Cpu).run_f64(&ts, &vs, start, end).expect("cpu");
			assert_eq!(gpu.len(), 161);
			assert_eq!(gpu.kinds(), cpu.kinds());
			let tolerance = if precision == Precision::F64 { 1e-9 } else { 1e-2 };
			for (i, (g, c)) in gpu.values().iter().zip(cpu.values()).enumerate() {
				assert!((g - c).abs() <= tolerance * c.abs().max(1.0), "{precision} {spline} point {i}: gpu {g} vs cpu {c}");
			}
		}
	}
	// One-point sets, and at most two per precision however many dispatches ran.
	let stats = splimes::gpu_pool_stats().expect("GPU started");
	// Each set holds one point: its output and read-back values, a 12-byte time and 64
	// bytes of parameters.
	let set_bytes = |p: &Precision| if *p == Precision::F64 { 2 * 8 + 12 + 64 } else { 2 * 4 + 12 + 64 };
	assert_eq!(stats.created, 2 * precisions.len() as u64, "{stats:?}");
	assert_eq!(stats.idle_bytes, precisions.iter().map(|p| 2 * set_bytes(p)).sum::<u64>(), "{stats:?}");
}

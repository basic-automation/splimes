//! `GpuDevices::All` through the public API: every device that opens is reported, and a
//! large call split among them returns, for every point, a value one of them computes
//! alone. Its own binary, because the device set is process-wide and set once.

#![cfg(feature = "gpu")]

mod common;

use chrono::{DateTime, TimeDelta, Utc};
use splimes::{Backend, GpuConfig, GpuDevices, Interpolator, Precision, Resolution, Spline};

#[test]
fn all_devices_open_report_and_agree_with_the_cpu() {
	let adapters = splimes::gpu_adapters();
	for a in &adapters {
		eprintln!("adapter {}: {} ({}, {}, f64 advertised: {})", a.index, a.name, a.api, a.device_type, a.advertises_f64);
	}
	assert!(adapters.iter().enumerate().all(|(i, a)| a.index == i));
	splimes::configure_gpu(GpuConfig::minimal().with_chunk_points(4096).with_devices(GpuDevices::All)).expect("configures");
	let Some(primary) = common::gpu_or_skip("all_devices_open_report_and_agree_with_the_cpu") else { return };
	let devices = splimes::gpu_devices();
	assert!(!devices.is_empty());
	assert_eq!(devices[0].info, primary, "gpu_info reports the first device");
	eprintln!("open: {:?}", devices.iter().map(|d| &d.info.name).collect::<Vec<_>>());
	if devices.len() < 2 {
		eprintln!("all_devices_open_report_and_agree_with_the_cpu: only one device opens, so nothing is split");
	}

	let start = DateTime::<Utc>::UNIX_EPOCH;
	let timestamps: Vec<_> = (0..2000_i64).map(|i| start + TimeDelta::milliseconds(i * 37 + (i * 7919) % 11)).collect();
	let values: Vec<f64> = (0..2000).map(|i| (f64::from(i) * 0.05).sin() * 1000.0).collect();
	let end = timestamps[timestamps.len() - 1];
	let precision = if devices.iter().all(|d| d.info.supports_f64) { Precision::F64 } else { Precision::F32 };
	let gpu = Interpolator::new(Spline::Cubic, Resolution::Milliseconds).backend(Backend::Gpu).gpu_precision(precision).run_f64(&timestamps, &values, start, end).expect("runs on the GPU");
	let cpu = Interpolator::new(Spline::Cubic, Resolution::Milliseconds).backend(Backend::Cpu).run_f64(&timestamps, &values, start, end).expect("runs on the CPU");
	assert_eq!(gpu.timestamps(), cpu.timestamps());
	let bound = if precision == Precision::F64 { 1e-9 } else { 1e-3 };
	for (g, c) in gpu.values().iter().zip(cpu.values()) {
		assert!((g - c).abs() <= bound * c.abs().max(1.0), "{g} vs {c}");
	}
	let after = splimes::gpu_devices();
	let chunks: Vec<u64> = after.iter().map(|d| d.chunks).collect();
	eprintln!("chunks per device: {chunks:?}");
	assert!(chunks.iter().sum::<u64>() >= 2, "a grid of {} points is more than one 4096-point chunk", gpu.values().len());
	assert!(after.iter().all(|d| d.in_flight == 0 && !d.lost));
}

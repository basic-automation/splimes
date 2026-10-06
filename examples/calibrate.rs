//! Measures this machine's backend crossovers and prints them.
//!
//! ```text
//! cargo run --release --example calibrate
//! ```

fn main() -> splimes::Result<()> {
	let started = std::time::Instant::now();
	match splimes::prewarm_gpu() {
		Ok(info) => println!("GPU: {} ({}, {}), f64 {}; started in {:.0?}", info.name, info.api, info.device_type, info.supports_f64, started.elapsed()),
		Err(e) => println!("GPU: {e}"),
	}
	println!("CPU threads: {}", rayon::current_num_threads());
	let calibration = splimes::calibrate()?;
	println!("\ncubic, 4,096 inputs, f64 values, best of 3\n");
	println!("{:>10} {:>12} {:>12} {:>12} {:>12}", "points", "cpu", "parallel", "gpu f64", "gpu f32");
	let show_time = |d: Option<std::time::Duration>| d.map_or_else(|| "-".to_owned(), |d| format!("{d:.2?}"));
	for s in &calibration.samples {
		println!("{:>10} {:>12} {:>12} {:>12} {:>12}", s.points, show_time(Some(s.cpu)), show_time(Some(s.parallel)), show_time(s.gpu_f64), show_time(s.gpu_f32));
	}
	let t = calibration.thresholds;
	let show = |n: usize| if n == usize::MAX { "never".to_owned() } else { format!("from {n} points") };
	println!("\nparallel {}, GPU f64 {}, GPU f32 {}", show(t.parallel_min_points), show(t.gpu_min_points), show(t.gpu_f32_min_points));
	Ok(())
}

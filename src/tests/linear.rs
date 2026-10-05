#[cfg(test)]
pub mod tests {
	// Make it pub so other test modules can access it
	use std::sync::LazyLock;

	use bigdecimal::{BigDecimal, FromPrimitive, ToPrimitive, Zero};
	use chrono::{DateTime, Utc};
	use serial_test::serial;

	use crate::{Point, Resolution, auto_interpolate, gpu::types::GpuInterpolator, gpu_interpolate, helpers::TargetTimesIterator, parallel_interpolate, splines::linear, tests::plot_terminal};

	/// The shared interpolation fixture: ten points with **random values on strictly
	/// separated knots**.
	///
	/// The timestamps are deliberately *not* random. `Point::random` draws its offset from
	/// just 61 integer-second buckets (-30..=30) around `Utc::now()`, so ten independent
	/// draws collide in the same second about half the time — and because each draw re-reads
	/// `Utc::now()`, a collision yields two knots microseconds apart rather than identical
	/// ones. At `Resolution::Seconds` that is a near-degenerate knot pair: the spline's
	/// divided differences divide by a ~1e-6 s gap, the result blows up, and the run produces
	/// garbage values (norms hundreds of times too large, cosine similarities going negative).
	///
	/// That is the mechanism behind the "flaky GPU interpolation tests" — the flakiness is
	/// **input-driven, not GPU-specific**: every path (CPU-vs-SIMD included) fails on a bad
	/// draw, and because this static is shared process-wide, one bad draw fails several tests
	/// at once, which is exactly the observed clustering. Fixing the knots keeps the values
	/// random (so the comparison still exercises real arithmetic) while removing the
	/// degeneracy the engine was never being asked to handle.
	pub static POINTS: LazyLock<Vec<Point>> = LazyLock::new(|| {
		let base = Utc::now();
		let mut points: Vec<Point> = Vec::new();
		for i in 0..10 {
			// Random value, deterministic well-separated knot.
			let mut point = Point::random();
			point.timestamp = base + chrono::Duration::seconds(i * 10);
			points.push(point);
		}
		points.sort_by_key(|p| p.timestamp);
		points
	});

	// Z-score threshold for outlier detection. Set higher to accommodate
	// extrapolated edge values which can naturally deviate from the mean.
	pub const Z_THRESHOLD: f64 = 10.0;
	// Cosine similarity threshold for every CPU / SIMD / GPU / auto-dispatch comparison.
	//
	// This was 8e-5, which made the gate inert — it accepted anything short of a ~10,000x
	// error. That number was never a tolerance; it was an artifact of `pow` computing
	// `v^(2^n)` (so `pow(v, 2)` returned `v⁴`), which inflated both norms and dragged the
	// reported "cosine similarity" of two near-identical vectors down to ~1e-4. The
	// threshold had been tuned to the broken metric instead of the metric being fixed.
	//
	// With `pow` corrected and the fixture's degenerate knots removed (see POINTS), all
	// four paths agree to ~13 decimal places — measured CPU-vs-SIMD, CPU-vs-GPU and
	// CPU-vs-auto all in [0.9999999999999, 1.0000000000001]. 0.999 is therefore a real
	// gate with enormous headroom over observed agreement, rather than a number chosen to
	// let the suite pass.
	pub const COS_THRESHOLD: f64 = 0.999;
	pub const RESOLUTION: Resolution = Resolution::Seconds;

	pub fn mean(values: &[BigDecimal]) -> BigDecimal {
		let sum: BigDecimal = values.iter().cloned().sum();
		sum / BigDecimal::from_usize(values.len()).unwrap()
	}

	/// `value` raised to `exponent`.
	///
	/// NB the obvious-looking loop `value = value * value` computes `value^(2^exponent)`,
	/// not `value^exponent` — it squares the *accumulator* each pass. That bug made
	/// `pow(v, 2)` return `v⁴`, which inflated every norm in [`cosine_similarity`] and
	/// every variance in [`std_dev`]; the multiplication must be by the untouched base.
	pub fn pow(value: BigDecimal, exponent: u64) -> BigDecimal {
		let mut acc = BigDecimal::from(1);
		for _ in 0..exponent {
			acc = acc * value.clone();
		}

		acc
	}

	pub fn std_dev(values: &[BigDecimal], mean: &BigDecimal) -> BigDecimal {
		let variance: BigDecimal = values.iter().map(|v| pow(v - mean, 2)).sum::<BigDecimal>() / BigDecimal::from_usize(values.len()).unwrap();
		variance.sqrt().unwrap()
	}

	fn cosine_similarity(a: &[BigDecimal], b: &[BigDecimal]) -> f64 {
		let dot_product: BigDecimal = a.iter().zip(b.iter()).map(|(a, b)| a.round(10) * b.round(10)).sum::<BigDecimal>().round(10);
		let cpu_norm: BigDecimal = a.iter().map(|v| pow(v.clone().round(10), 2)).sum::<BigDecimal>().sqrt().unwrap_or_else(BigDecimal::zero).round(10);
		let parallel_norm: BigDecimal = b.iter().map(|v| pow(v.clone().round(10), 2)).sum::<BigDecimal>().sqrt().unwrap_or_else(BigDecimal::zero).round(10);
		println!("Dot product: {dot_product}, CPU norm: {cpu_norm}, Parallel norm: {parallel_norm}");
		if cpu_norm == BigDecimal::zero() || parallel_norm == BigDecimal::zero() {
			return 0.0;
		}
		let similarity = (dot_product / (cpu_norm * parallel_norm)).to_f64().unwrap_or(0.0);
		println!("Cosine similarity: {similarity}");
		similarity
	}

	// Detect regressions with z-scores and cosine similarity
	pub fn check_similarity(a: &[BigDecimal], b: &[BigDecimal]) -> (Vec<f64>, Vec<f64>, f64) {
		// Z-scores for outlier detection
		let cpu_mean = mean(a);
		let cpu_std = std_dev(a, &cpu_mean);
		let parallel_mean = mean(b);
		let parallel_std = std_dev(b, &parallel_mean);

		// If standard deviation is zero, all values are the same, so z-scores are zero.
		let cpu_z_scores: Vec<f64> = if cpu_std.is_zero() { vec![0.0; a.len()] } else { a.iter().map(|v| ((v - &cpu_mean) / &cpu_std).to_f64().unwrap_or(0.0)).collect() };

		let parallel_z_scores: Vec<f64> = if parallel_std.is_zero() { vec![0.0; b.len()] } else { b.iter().map(|v| ((v - &parallel_mean) / &parallel_std).to_f64().unwrap_or(0.0)).collect() };

		// Cosine similarity for overall comparison
		let similarity = cosine_similarity(a, b);

		(cpu_z_scores, parallel_z_scores, similarity)
	}

	#[tokio::test]
	#[serial(gpu_tests)]
	async fn test_linear_interpolation() {
		// Clear GPU buffer pool for test isolation
		GpuInterpolator::clear_buffer_pool_static().expect("Failed to clear GPU buffer pool");

		let mut points = POINTS.clone();
		let start = {
			let s = points.first().map_or_else(Utc::now, |p| p.timestamp);
			match RESOLUTION {
				Resolution::Nanoseconds => s - chrono::Duration::nanoseconds(10),
				Resolution::Microseconds => s - chrono::Duration::microseconds(10),
				Resolution::Milliseconds | Resolution::Seconds => s - chrono::Duration::seconds(10),
				Resolution::Minutes => s - chrono::Duration::minutes(10),
				Resolution::Hours => s - chrono::Duration::hours(10),
				Resolution::Days => s - chrono::Duration::days(10),
				Resolution::Weeks => s - chrono::Duration::weeks(10),
				Resolution::Months => s - chrono::Duration::days(30 * 10),
				Resolution::Years => s - chrono::Duration::days(365 * 10),
			}
		};

		let end = {
			let e = points.last().map_or_else(Utc::now, |p| p.timestamp);
			match RESOLUTION {
				Resolution::Nanoseconds => e + chrono::Duration::nanoseconds(10),
				Resolution::Microseconds => e + chrono::Duration::microseconds(10),
				Resolution::Milliseconds | Resolution::Seconds => e + chrono::Duration::seconds(10),
				Resolution::Minutes => e + chrono::Duration::minutes(10),
				Resolution::Hours => e + chrono::Duration::hours(10),
				Resolution::Days => e + chrono::Duration::days(10),
				Resolution::Weeks => e + chrono::Duration::weeks(10),
				Resolution::Months => e + chrono::Duration::days(30 * 10),
				Resolution::Years => e + chrono::Duration::days(365 * 10),
			}
		};

		let target_times_iter = TargetTimesIterator::new(start, end, RESOLUTION);
		let spline = crate::Spline::Linear; // Using linear spline for this test
		let input_count = points.len();
		let output_count = target_times_iter.estimate_len().unwrap();
		println!("Linear: Input count: {input_count}, Output count: {output_count}");

		// cpu interpolation
		println!("Linear: CPU Test Starting...");
		let timer = tokio::time::Instant::now();
		let cpu = linear(&mut points, &start, &end, &RESOLUTION).await.unwrap();
		let cpu_time = timer.elapsed();
		println!("Linear: CPU Interpolation took: {cpu_time:?}");
		let cpu_len = cpu.len();
		let (cpu_timestamps, cpu_values): (Vec<_>, Vec<_>) = cpu.iter().map(|p| (p.timestamp, p.value.clone())).unzip();
		plot_terminal("Linear: CPU Interpolation Results", cpu.clone()).unwrap();
		//println!("CPU results: {:?}", cpu);
		drop(cpu);

		// parallel interpolation
		println!("Linear: Parallel Test Starting...");
		let timer = tokio::time::Instant::now();
		let parallel = parallel_interpolate(&mut points, &start, &end, spline, RESOLUTION).await.unwrap();
		let parallel_time = timer.elapsed();
		println!("Linear: Parallel Interpolation took: {parallel_time:?}");
		let parallel_len = parallel.len();
		let (parallel_timestamps, parallel_values): (Vec<_>, Vec<_>) = parallel.iter().map(|p| (p.timestamp, p.value.clone())).unzip();
		plot_terminal("Linear: Parallel Interpolation Results", parallel.clone()).unwrap();
		drop(parallel);

		// compare cpu and parallel results
		println!("Linear: Comparing CPU and Parallel results...");
		assert_eq!(cpu_len, parallel_len);

		let (cpu_z_scores, parallel_z_scores, similarity) = check_similarity(&cpu_values, &parallel_values);
		assert!(similarity >= COS_THRESHOLD, "Cosine similarity is below threshold: {COS_THRESHOLD}, similarity: {similarity}");
		assert!(similarity >= COS_THRESHOLD, "Cosine similarity is below threshold: {COS_THRESHOLD}, similarity: {similarity}");
		for (i, (cpu_z, parallel_z)) in cpu_z_scores.iter().zip(parallel_z_scores.iter()).enumerate() {
			assert!(cpu_z.abs() < Z_THRESHOLD, "CPU value at index {i} is an outlier: {cpu_z}");
			assert!(parallel_z.abs() < Z_THRESHOLD, "Parallel value at index {i} is an outlier: {parallel_z}");
		}

		assert_eq!(cpu_timestamps, parallel_timestamps);
		drop(parallel_timestamps);
		drop(parallel_values);

		// gpu interpolation
		println!("Linear: GPU Test Starting...");
		let timer = tokio::time::Instant::now();
		let gpu = gpu_interpolate(&mut points, start, end, RESOLUTION, spline).await.unwrap();
		let gpu_time = timer.elapsed();
		println!("Linear: GPU Interpolation took: {gpu_time:?}");
		let gpu_len = gpu.len();
		let (gpu_timestamps, gpu_values): (Vec<_>, Vec<_>) = gpu.iter().map(|p| (p.timestamp, p.value.clone())).unzip();
		plot_terminal("Linear: GPU Interpolation Results", gpu.clone()).unwrap();
		//println!("GPU results: {:?}", gpu);
		drop(gpu);

		// compare cpu and gpu results
		println!("Linear: Comparing CPU and GPU results...");
		assert_eq!(cpu_len, gpu_len);

		let (cpu_z_scores, gpu_z_scores, similarity) = check_similarity(&cpu_values, &gpu_values);
		assert!(similarity >= COS_THRESHOLD, "GPU: Cosine similarity is below threshold: {similarity}");
		for (i, (cpu_z, gpu_z)) in cpu_z_scores.iter().zip(gpu_z_scores.iter()).enumerate() {
			assert!(cpu_z.abs() < Z_THRESHOLD, "CPU value at index {i} is an outlier: {cpu_z}");
			assert!(gpu_z.abs() < Z_THRESHOLD, "GPU value at index {i} is an outlier: {gpu_z}");
		}

		assert_eq!(cpu_timestamps, gpu_timestamps);
		drop(gpu_timestamps);
		drop(gpu_values);

		// automatically determine the fastest method
		println!("Linear: Auto Test Starting...");
		let auto_timer = tokio::time::Instant::now();
		let auto = auto_interpolate(&mut points, start, end, RESOLUTION, spline).await.unwrap();
		let auto_time = auto_timer.elapsed();
		println!("Linear: Auto Interpolation took: {auto_time:?}");
		let auto_len = auto.len();
		let (auto_timestamps, auto_values): (Vec<_>, Vec<_>) = auto.iter().map(|p| (p.timestamp, p.value.clone())).unzip();
		plot_terminal("Linear: Auto Interpolation Results", auto.clone()).unwrap();
		drop(auto);

		// compare cpu and auto results
		println!("Linear: Comparing CPU and Auto results...");
		assert_eq!(cpu_len, auto_len);

		let (cpu_z_scores, auto_z_scores, similarity) = check_similarity(&cpu_values, &auto_values);
		assert!(similarity >= COS_THRESHOLD, "Auto: Cosine similarity is below threshold: {similarity}");
		for (i, (cpu_z, auto_z)) in cpu_z_scores.iter().zip(auto_z_scores.iter()).enumerate() {
			assert!(cpu_z.abs() < Z_THRESHOLD, "CPU value at index {i} is an outlier: {cpu_z}");
			assert!(auto_z.abs() < Z_THRESHOLD, "Auto value at index {i} is an outlier: {auto_z}");
		}

		assert_eq!(cpu_timestamps, auto_timestamps);
		drop(auto_timestamps);
		drop(auto_values);

		if auto_time < gpu_time && auto_time < cpu_time && auto_time < parallel_time {
			let percentage_difference = (gpu_time.checked_sub(auto_time).unwrap().as_nanos() as f64 / auto_time.as_nanos() as f64) * 100.0;
			println!("Linear: Auto was faster by {percentage_difference:.2}%");
		} else if gpu_time < auto_time && gpu_time < cpu_time && gpu_time < parallel_time {
			let percentage_difference = (auto_time.checked_sub(gpu_time).unwrap().as_nanos() as f64 / gpu_time.as_nanos() as f64) * 100.0;
			println!("Linear: GPU was faster by {percentage_difference:.2}%");
		} else if cpu_time < auto_time && cpu_time < gpu_time && cpu_time < parallel_time {
			let percentage_difference = (auto_time.checked_sub(cpu_time).unwrap().as_nanos() as f64 / cpu_time.as_nanos() as f64) * 100.0;
			println!("Linear: CPU was faster by {percentage_difference:.2}%");
		} else if parallel_time < auto_time && parallel_time < gpu_time && parallel_time < cpu_time && parallel_time < parallel_time {
			let percentage_difference = (auto_time.checked_sub(parallel_time).unwrap().as_nanos() as f64 / parallel_time.as_nanos() as f64) * 100.0;
			println!("Linear: SIMD was faster by {percentage_difference:.2}%");
		} else {
			let st = parallel_time.as_nanos();
			let gt = gpu_time.as_nanos();
			let dif = gt.abs_diff(st);
			let percentage_difference = (dif as f64 / gpu_time.as_nanos() as f64) * 100.0;
			println!("Linear: Parallel SIMD was faster by {percentage_difference:.2}%");
		}
	}

	#[tokio::test]
	#[serial(gpu_tests)]
	async fn test_target_times() {
		let points = POINTS.clone();
		let start = {
			let s = points.first().map_or_else(Utc::now, |p| p.timestamp);
			match RESOLUTION {
				Resolution::Nanoseconds => s - chrono::Duration::nanoseconds(10),
				Resolution::Microseconds => s - chrono::Duration::microseconds(10),
				Resolution::Milliseconds | Resolution::Seconds => s - chrono::Duration::seconds(10),
				Resolution::Minutes => s - chrono::Duration::minutes(10),
				Resolution::Hours => s - chrono::Duration::hours(10),
				Resolution::Days => s - chrono::Duration::days(10),
				Resolution::Weeks => s - chrono::Duration::weeks(10),
				Resolution::Months => s - chrono::Duration::days(30 * 10),
				Resolution::Years => s - chrono::Duration::days(365 * 10),
			}
		};

		let end = {
			let e = points.last().map_or_else(Utc::now, |p| p.timestamp);
			match RESOLUTION {
				Resolution::Nanoseconds => e + chrono::Duration::nanoseconds(10),
				Resolution::Microseconds => e + chrono::Duration::microseconds(10),
				Resolution::Milliseconds | Resolution::Seconds => e + chrono::Duration::seconds(10),
				Resolution::Minutes => e + chrono::Duration::minutes(10),
				Resolution::Hours => e + chrono::Duration::hours(10),
				Resolution::Days => e + chrono::Duration::days(10),
				Resolution::Weeks => e + chrono::Duration::weeks(10),
				Resolution::Months => e + chrono::Duration::days(30 * 10),
				Resolution::Years => e + chrono::Duration::days(365 * 10),
			}
		};
		let target_times_1: Vec<DateTime<Utc>> = TargetTimesIterator::new(start, end, RESOLUTION).flatten().collect();
		let target_times_2: Vec<DateTime<Utc>> = TargetTimesIterator::new(start, end, RESOLUTION).flatten().collect();
		println!("Target times 1: {}, Target times 2: {}", target_times_1.len(), target_times_2.len());
		assert_eq!(target_times_1, target_times_2, "Target times iterators should produce the same results");
	}
}

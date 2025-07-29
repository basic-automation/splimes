#[cfg(test)]
mod tests {
	use chrono::{DateTime, Utc};

	use super::super::plot_terminal;
	use crate::{
		Resolution, TargetTimesIterator, auto_interpolate, gpu_interpolate, parallel_interpolate, polynomial, tests::linear::tests::{COS_THRESHOLD, POINTS, RESOLUTION, Z_THRESHOLD, check_similarity}
	};

	#[tokio::test]
	async fn test_polynomial_interpolation() {
		let mut points = POINTS.clone();
		let start = {
			let s = points.first().map_or_else(|| Utc::now(), |p| p.timestamp);
			match RESOLUTION {
				Resolution::Nanoseconds => s - chrono::Duration::nanoseconds(10),
				Resolution::Microseconds => s - chrono::Duration::microseconds(10),
				Resolution::Milliseconds => s - chrono::Duration::seconds(10),
				Resolution::Seconds => s - chrono::Duration::seconds(10),
				Resolution::Minutes => s - chrono::Duration::minutes(10),
				Resolution::Hours => s - chrono::Duration::hours(10),
				Resolution::Days => s - chrono::Duration::days(10),
				Resolution::Weeks => s - chrono::Duration::weeks(10),
				Resolution::Months => s - chrono::Duration::days(30 * 10),
				Resolution::Years => s - chrono::Duration::days(365 * 10),
			}
		};

		let end = {
			let e = points.last().map_or_else(|| Utc::now(), |p| p.timestamp);
			match RESOLUTION {
				Resolution::Nanoseconds => e + chrono::Duration::nanoseconds(10),
				Resolution::Microseconds => e + chrono::Duration::microseconds(10),
				Resolution::Milliseconds => e + chrono::Duration::seconds(10),
				Resolution::Seconds => e + chrono::Duration::seconds(10),
				Resolution::Minutes => e + chrono::Duration::minutes(10),
				Resolution::Hours => e + chrono::Duration::hours(10),
				Resolution::Days => e + chrono::Duration::days(10),
				Resolution::Weeks => e + chrono::Duration::weeks(10),
				Resolution::Months => e + chrono::Duration::days(30 * 10),
				Resolution::Years => e + chrono::Duration::days(365 * 10),
			}
		};

		let spline = crate::Spline::Polynomial(9, Some(1.0));
		let target_times_iter = TargetTimesIterator::new(start, end, RESOLUTION);
		let input_count = points.len();
		let output_count = target_times_iter.estimate_len().unwrap();
		println!("Polynomial: Input count: {}, Output count: {}", input_count, output_count);

		// cpu interpolation
		println!("Polynomial: CPU Test Starting...");
		let timer = tokio::time::Instant::now();
		let cpu = polynomial(&mut points, &start, &end, &RESOLUTION, &spline).await.unwrap();
		let cpu_time = timer.elapsed();
		println!("Polynomial: CPU Interpolation took: {:?}", cpu_time);
		let cpu_len = cpu.len();
		let (cpu_timestamps, cpu_values): (Vec<_>, Vec<_>) = cpu.iter().map(|p| (p.timestamp, p.value.clone())).unzip();
		plot_terminal("Polynomial: CPU Interpolation Results", cpu.clone()).unwrap();
		//println!("CPU results: {:?}", cpu);
		drop(cpu);

		// parallel interpolation
		println!("Polynomial: Parallel Test Starting...");
		let timer = tokio::time::Instant::now();
		let parallel = parallel_interpolate(&mut points, &start, &end, spline, RESOLUTION).await.unwrap();
		let parallel_time = timer.elapsed();
		println!("Polynomial: Parallel Interpolation took: {:?}", parallel_time);
		let parallel_len = parallel.len();
		let (parallel_timestamps, parallel_values): (Vec<_>, Vec<_>) = parallel.iter().map(|p| (p.timestamp, p.value.clone())).unzip();
		plot_terminal("Polynomial: Parallel Interpolation Results", parallel.clone()).unwrap();
		//println!("Parallel results: {:?}", parallel);
		drop(parallel);

		// compare cpu and parallel results
		println!("Polynomial: Comparing CPU and Parallel results...");
		assert_eq!(cpu_len, parallel_len);

		let (cpu_z_scores, parallel_z_scores, similarity) = check_similarity(&cpu_values, &parallel_values);
		assert!(similarity >= COS_THRESHOLD, "Parallel: Cosine similarity is below threshold: {}, similarity: {}", COS_THRESHOLD, similarity);
		for (i, (cpu_z, parallel_z)) in cpu_z_scores.iter().zip(parallel_z_scores.iter()).enumerate() {
			assert!(cpu_z.abs() < Z_THRESHOLD, "CPU value at index {} is an outlier: {}", i, cpu_z);
			assert!(parallel_z.abs() < Z_THRESHOLD, "Parallel value at index {} is an outlier: {}", i, parallel_z);
		}

		assert_eq!(cpu_timestamps, parallel_timestamps);
		drop(parallel_timestamps);
		drop(parallel_values);

		// gpu interpolation
		println!("Polynomial: GPU Test Starting...");
		let timer = tokio::time::Instant::now();
		let gpu = gpu_interpolate(&mut points, start, end, RESOLUTION, spline).await.unwrap();
		let gpu_time = timer.elapsed();
		println!("Polynomial: GPU Interpolation took: {:?}", gpu_time);
		let gpu_len = gpu.len();
		let (gpu_timestamps, gpu_values): (Vec<_>, Vec<_>) = gpu.iter().map(|p| (p.timestamp, p.value.clone())).unzip();
		plot_terminal("Polynomial: GPU Interpolation Results", gpu.clone()).unwrap();
		//println!("GPU results: {:?}", gpu);
		drop(gpu);

		// compare cpu and gpu results
		println!("Polynomial: Comparing CPU and GPU results...");
		assert_eq!(cpu_len, gpu_len);

		let (cpu_z_scores, gpu_z_scores, similarity) = check_similarity(&cpu_values, &gpu_values);
		assert!(similarity >= COS_THRESHOLD, "GPU: Cosine similarity is below threshold: {}", similarity);
		for (i, (cpu_z, gpu_z)) in cpu_z_scores.iter().zip(gpu_z_scores.iter()).enumerate() {
			assert!(cpu_z.abs() < Z_THRESHOLD, "CPU value at index {} is an outlier: {}", i, cpu_z);
			assert!(gpu_z.abs() < Z_THRESHOLD, "GPU value at index {} is an outlier: {}", i, gpu_z);
		}

		assert_eq!(cpu_timestamps, gpu_timestamps);
		drop(gpu_timestamps);
		drop(gpu_values);

		// automatically determine the fastest method
		println!("Polynomial: Auto Test Starting...");
		let auto_timer = tokio::time::Instant::now();
		let auto = auto_interpolate(&mut points, start, end, RESOLUTION, spline).await.unwrap();
		let auto_time = auto_timer.elapsed();
		println!("Polynomial: Auto Interpolation took: {:?}", auto_time);
		let auto_len = auto.len();
		let (auto_timestamps, auto_values): (Vec<_>, Vec<_>) = auto.iter().map(|p| (p.timestamp, p.value.clone())).unzip();
		plot_terminal("Polynomial: Auto Interpolation Results", auto.clone()).unwrap();
		drop(auto);

		// compare cpu and auto results
		println!("Polynomial: Comparing CPU and Auto results...");
		assert_eq!(cpu_len, auto_len);

		let (cpu_z_scores, auto_z_scores, similarity) = check_similarity(&cpu_values, &auto_values);
		assert!(similarity >= COS_THRESHOLD, "Auto: Cosine similarity is below threshold: {}", similarity);
		for (i, (cpu_z, auto_z)) in cpu_z_scores.iter().zip(auto_z_scores.iter()).enumerate() {
			assert!(cpu_z.abs() < Z_THRESHOLD, "CPU value at index {} is an outlier: {}", i, cpu_z);
			assert!(auto_z.abs() < Z_THRESHOLD, "Auto value at index {} is an outlier: {}", i, auto_z);
		}

		assert_eq!(cpu_timestamps, auto_timestamps);
		drop(auto_timestamps);
		drop(auto_values);

		if auto_time < gpu_time && auto_time < cpu_time && auto_time < parallel_time && auto_time < parallel_time {
			let percentage_difference = ((gpu_time - auto_time).as_nanos() as f64 / auto_time.as_nanos() as f64) * 100.0;
			println!("Polynomial: Auto was faster by {:.2}%", percentage_difference);
		} else if gpu_time < auto_time && gpu_time < cpu_time && gpu_time < parallel_time && gpu_time < parallel_time {
			let percentage_difference = ((auto_time - gpu_time).as_nanos() as f64 / gpu_time.as_nanos() as f64) * 100.0;
			println!("Polynomial: GPU was faster by {:.2}%", percentage_difference);
		} else if cpu_time < auto_time && cpu_time < gpu_time && cpu_time < parallel_time && cpu_time < parallel_time {
			let percentage_difference = ((auto_time - cpu_time).as_nanos() as f64 / cpu_time.as_nanos() as f64) * 100.0;
			println!("Polynomial: CPU was faster by {:.2}%", percentage_difference);
		} else if parallel_time < auto_time && parallel_time < gpu_time && parallel_time < cpu_time && parallel_time < parallel_time {
			let percentage_difference = ((auto_time - parallel_time).as_nanos() as f64 / parallel_time.as_nanos() as f64) * 100.0;
			println!("Polynomial: SIMD was faster by {:.2}%", percentage_difference);
		} else {
			let st = parallel_time.as_nanos();
			let gt = gpu_time.as_nanos();
			let dif = if st < gt { gt - st } else { st - gt };
			let percentage_difference = (dif as f64 / gpu_time.as_nanos() as f64) * 100.0;
			println!("Polynomial: Parallel SIMD was faster by {:.2}%", percentage_difference);
		}
	}

	#[tokio::test]
	async fn test_target_times() {
		let points = POINTS.clone();
		let start = {
			let s = points.first().map_or_else(|| Utc::now(), |p| p.timestamp);
			match RESOLUTION {
				Resolution::Nanoseconds => s - chrono::Duration::nanoseconds(10),
				Resolution::Microseconds => s - chrono::Duration::microseconds(10),
				Resolution::Milliseconds => s - chrono::Duration::seconds(10),
				Resolution::Seconds => s - chrono::Duration::seconds(10),
				Resolution::Minutes => s - chrono::Duration::minutes(10),
				Resolution::Hours => s - chrono::Duration::hours(10),
				Resolution::Days => s - chrono::Duration::days(10),
				Resolution::Weeks => s - chrono::Duration::weeks(10),
				Resolution::Months => s - chrono::Duration::days(30 * 10),
				Resolution::Years => s - chrono::Duration::days(365 * 10),
			}
		};

		let end = {
			let e = points.last().map_or_else(|| Utc::now(), |p| p.timestamp);
			match RESOLUTION {
				Resolution::Nanoseconds => e + chrono::Duration::nanoseconds(10),
				Resolution::Microseconds => e + chrono::Duration::microseconds(10),
				Resolution::Milliseconds => e + chrono::Duration::seconds(10),
				Resolution::Seconds => e + chrono::Duration::seconds(10),
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

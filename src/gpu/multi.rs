//! Spreading calls over several devices: whole calls to the least-loaded device, or one
//! large call's chunks shared out among all of them.

use std::sync::{
	Mutex, PoisonError, atomic::{AtomicBool, Ordering}
};

use super::{
	context::{Context, Devices}, run
};
use crate::{Error, Precision, Result, kernel::Method, prepare::Knots, time::Grid, value::Value};

/// Runs the kernel on `devices`. With one device this is exactly [`run::eval`].
pub fn eval<V: Value>(devices: &Devices, knots: &Knots<'_, V>, grid: &Grid, method: &Method, precision: Precision, out: &mut [f64]) -> Result<()> {
	if let [only] = devices.list.as_slice() {
		return run::eval(only, knots, grid, method, precision, out);
	}
	// A device lost during the call fails it; retry on the devices left, at most once per
	// device. A device that fails without being lost fails the call, as one device would.
	let mut attempts = 0;
	loop {
		let usable: Vec<&Context> = devices.list.iter().filter(|ctx| ctx.can_run(precision, knots.len())).collect();
		let Some(first) = usable.first() else {
			// Report why the primary can't, as with one device.
			let primary = devices.primary();
			primary.kernel(precision)?;
			return Err(Error::Gpu(format!("{} knots exceed every device's storage-binding limit", knots.len())));
		};
		let chunk = usable.iter().map(|ctx| ctx.chunk_len()).min().unwrap_or(1);
		let split = usable.len() > 1 && grid.len > chunk && grid.len as u64 >= first.config.split_min_points;
		let result = if split { split_eval(&usable, chunk, knots, grid, method, precision, out) } else { run::eval(least_loaded(devices, &usable), knots, grid, method, precision, out) };
		attempts += 1;
		match result {
			Err(e) if attempts < devices.list.len() && usable.iter().any(|ctx| ctx.is_lost()) && devices.list.iter().any(|ctx| ctx.can_run(precision, knots.len())) => {
				log::warn!("splimes: a GPU device was lost during a call ({e}); retrying on the devices left");
			}
			other => return other,
		}
	}
}

/// The device with the fewest calls in flight; ties go to each device in turn.
fn least_loaded<'a>(devices: &Devices, usable: &[&'a Context]) -> &'a Context {
	let start = devices.next.fetch_add(1, Ordering::Relaxed) % usable.len();
	let rotated = usable[start..].iter().chain(&usable[..start]);
	rotated.min_by_key(|ctx| ctx.in_flight.load(Ordering::Acquire)).copied().unwrap_or(usable[0])
}

/// Shares one call's chunks among `usable`: each device takes the next chunk when it has
/// room for one, so a faster device takes more. The calling thread drives the first
/// device and a scoped thread each of the others; all have finished when this returns.
fn split_eval<V: Value>(usable: &[&Context], chunk: usize, knots: &Knots<'_, V>, grid: &Grid, method: &Method, precision: Precision, out: &mut [f64]) -> Result<()> {
	let queue = Mutex::new(out.chunks_mut(chunk).enumerate());
	let failed = AtomicBool::new(false);
	let worker = |ctx: &Context| {
		let next = || {
			if failed.load(Ordering::Acquire) {
				return None;
			}
			queue.lock().unwrap_or_else(PoisonError::into_inner).next().map(|(i, dest)| (i * chunk, dest))
		};
		let result = run::run_chunks(ctx, knots, grid, method, precision, next);
		if result.is_err() {
			// The others stop taking chunks; the call fails either way.
			failed.store(true, Ordering::Release);
		}
		result
	};
	let results: Vec<Result<()>> = std::thread::scope(|scope| {
		// Collected so every other device starts before this thread drives the first.
		let mut others = Vec::with_capacity(usable.len() - 1);
		for &ctx in &usable[1..] {
			others.push(scope.spawn(move || worker(ctx)));
		}
		let mut results = vec![worker(usable[0])];
		results.extend(others.into_iter().map(|h| h.join().unwrap_or_else(|_| Err(Error::Gpu("a GPU worker thread panicked".to_owned())))));
		results
	});
	// Report a lost device's error first: it is what explains the others stopping.
	let mut errors: Vec<(usize, Error)> = results.into_iter().enumerate().filter_map(|(i, r)| r.err().map(|e| (i, e))).collect();
	errors.sort_by_key(|(i, _)| !usable[*i].is_lost());
	errors.into_iter().next().map_or(Ok(()), |(_, e)| Err(e))
}

#[cfg(test)]
mod tests {
	use chrono::{DateTime, Utc};

	use super::eval;
	use crate::{
		Precision, Resolution, Spline, gpu::{
			AdapterSelector, GpuConfig, GpuDevices, context::{self, Devices}
		}, kernel::Method, prepare::Knots, time::Grid
	};

	fn at(s: i64) -> DateTime<Utc> {
		DateTime::<Utc>::from_timestamp(s, 0).expect("valid")
	}

	/// An irregular series of `n` knots and its grid at one-second resolution.
	fn series(n: usize) -> (Vec<DateTime<Utc>>, Vec<f64>) {
		let ts: Vec<_> = (0..i64::try_from(n).expect("small")).map(|i| at(i * 10 + (i * 7919) % 7)).collect();
		let values = (0..u32::try_from(n).expect("small")).map(f64::from).map(|x| (x * 0.37).sin().mul_add(100.0, x)).collect();
		(ts, values)
	}

	fn run(devices: &Devices, spline: Spline, precision: Precision, n: usize) -> Vec<f64> {
		let (ts, values) = series(n);
		let knots = Knots::new(n, |i| ts[i], |i| &values[i], false).expect("knots");
		let grid = Grid::new(at(-50), ts[n - 1] + chrono::TimeDelta::seconds(50), Resolution::Seconds).expect("grid");
		let mut out = vec![f64::NAN; grid.len];
		eval(devices, &knots, &grid, &Method::new(spline, &knots), precision, &mut out).expect("runs");
		out
	}

	/// The index of every adapter that opens, or `None` (with a notice) when there is no GPU.
	fn opening() -> Option<Vec<usize>> {
		let indices: Vec<usize> = context::adapters().iter().map(|a| a.index).filter(|&i| Devices::open(&config(GpuDevices::List(vec![AdapterSelector::Index(i)]))).is_ok()).collect();
		if indices.is_empty() {
			eprintln!("multi-GPU tests: skipped, no adapter opens");
			return None;
		}
		Some(indices)
	}

	/// Small chunks so that a small grid splits into many.
	fn config(devices: GpuDevices) -> GpuConfig {
		GpuConfig::minimal().with_chunk_points(64).with_devices(devices)
	}

	const BITS: fn(&[f64]) -> Vec<u64> = |v| v.iter().map(|x| x.to_bits()).collect();

	/// Two devices on the same adapter split a call chunk by chunk, and the result matches
	/// one device's bit for bit, in both precisions and for every method family; both
	/// devices take part.
	#[test]
	fn two_devices_on_one_adapter_match_one_device_bit_for_bit() {
		let Some(indices) = opening() else { return };
		for index in indices {
			let one = Devices::open(&config(GpuDevices::List(vec![AdapterSelector::Index(index)]))).expect("opens");
			let two = Devices::open(&config(GpuDevices::List(vec![AdapterSelector::Index(index), AdapterSelector::Index(index)]))).expect("opens twice");
			let name = &one.primary().info.name;
			let precisions = if one.primary().f64.is_some() { vec![Precision::F64, Precision::F32] } else { vec![Precision::F32] };
			for precision in precisions {
				for spline in [Spline::Linear, Spline::Quadratic, Spline::Cubic, Spline::Polynomial(5, None), Spline::Polynomial(5, Some(2.0))] {
					let single = run(&one, spline, precision, 300);
					let split = run(&two, spline, precision, 300);
					assert!(split.iter().all(|v| v.is_finite()), "{name}: every point written");
					assert_eq!(BITS(&split), BITS(&single), "{name} {spline:?} {precision}");
				}
			}
			let chunks: Vec<u64> = two.list.iter().map(|c| c.chunks.load(std::sync::atomic::Ordering::Relaxed)).collect();
			eprintln!("{name}: chunks per device {chunks:?}");
			assert!(chunks.iter().all(|&c| c > 0), "{name}: both devices took chunks: {chunks:?}");
		}
	}

	/// On two different adapters, every chunk of a split call is exactly what one of the
	/// two computes alone. Needs two adapters that open; skipped otherwise.
	#[test]
	fn a_call_split_across_two_adapters_is_each_adapters_own_result() {
		let Some(indices) = opening() else { return };
		// One adapter per physical device: the same GPU on two APIs isn't a second device.
		let infos = context::adapters();
		let mut distinct: Vec<usize> = Vec::new();
		for i in indices {
			if !distinct.iter().any(|&j| (infos[j].vendor, infos[j].device, &infos[j].name) == (infos[i].vendor, infos[i].device, &infos[i].name)) {
				distinct.push(i);
			}
		}
		let [a, b, ..] = distinct[..] else {
			eprintln!("a_call_split_across_two_adapters_is_each_adapters_own_result: skipped, only {} distinct adapter(s) open", distinct.len());
			return;
		};
		let alone = |i| Devices::open(&config(GpuDevices::List(vec![AdapterSelector::Index(i)]))).expect("opens");
		let (da, db) = (alone(a), alone(b));
		let both = Devices::open(&config(GpuDevices::List(vec![AdapterSelector::Index(a), AdapterSelector::Index(b)]))).expect("opens both");
		let names = format!("{} + {}", da.primary().info.name, db.primary().info.name);
		let f64_both = da.primary().f64.is_some() && db.primary().f64.is_some();
		let precisions = if f64_both { vec![Precision::F64, Precision::F32] } else { vec![Precision::F32] };
		for precision in precisions {
			for spline in [Spline::Linear, Spline::Cubic, Spline::Polynomial(5, Some(2.0))] {
				let (ra, rb, split) = (run(&da, spline, precision, 300), run(&db, spline, precision, 300), run(&both, spline, precision, 300));
				let differ = ra.iter().zip(&rb).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
				eprintln!("{names} {spline:?} {precision}: the adapters differ at {differ} of {} points", ra.len());
				for (c, ((s, x), y)) in split.chunks(64).zip(ra.chunks(64)).zip(rb.chunks(64)).enumerate() {
					assert!(BITS(s) == BITS(x) || BITS(s) == BITS(y), "{names} {spline:?} {precision}: chunk {c} is neither adapter's result");
				}
			}
		}
	}

	/// A device lost mid-call is skipped: the call is retried on the other device, and later
	/// calls never touch the lost one.
	#[test]
	fn a_lost_device_is_skipped_and_the_call_retried() {
		use std::sync::atomic::Ordering::Relaxed;
		let Some(indices) = opening() else { return };
		let index = indices[0];
		let one = Devices::open(&config(GpuDevices::List(vec![AdapterSelector::Index(index)]))).expect("opens");
		let two = Devices::open(&config(GpuDevices::List(vec![AdapterSelector::Index(index), AdapterSelector::Index(index)]))).expect("opens twice");
		let expected = BITS(&run(&one, Spline::Cubic, Precision::F32, 300));
		// The calling thread drives the first device; its first read-back finds it lost, as
		// after a driver reset, with its buffers intact.
		crate::gpu::run::tests::LOSE_BEFORE_POLL.set(Some(crate::gpu::run::tests::Loss::Signalled));
		let retried = run(&two, Spline::Cubic, Precision::F32, 300);
		crate::gpu::run::tests::LOSE_BEFORE_POLL.set(None);
		assert!(two.list[0].is_lost());
		assert_eq!(BITS(&retried), expected, "retried on the device left");
		let before = two.list[0].calls.load(Relaxed);
		assert_eq!(BITS(&run(&two, Spline::Cubic, Precision::F32, 300)), expected);
		assert_eq!(two.list[0].calls.load(Relaxed), before, "a lost device gets no more work");
	}

	/// Calls too small to split go whole to the device with the fewest in flight, in turn
	/// when they tie.
	#[test]
	fn small_calls_alternate_between_idle_devices() {
		let Some(indices) = opening() else { return };
		let index = indices[0];
		let devices = Devices::open(&config(GpuDevices::List(vec![AdapterSelector::Index(index), AdapterSelector::Index(index)])).with_split_min_points(u64::MAX)).expect("opens twice");
		for _ in 0..4 {
			run(&devices, Spline::Linear, Precision::F32, 300);
		}
		let calls: Vec<u64> = devices.list.iter().map(|c| c.calls.load(std::sync::atomic::Ordering::Relaxed)).collect();
		assert_eq!(calls, vec![2, 2], "{calls:?}");
		assert!(devices.list.iter().all(|c| c.in_flight.load(std::sync::atomic::Ordering::Relaxed) == 0));
	}

	/// A selector that matches nothing is refused, naming the adapters found.
	#[test]
	fn an_unmatched_selector_is_refused() {
		for selector in [AdapterSelector::Index(usize::MAX), AdapterSelector::Name("no such adapter 5f3c".to_owned()), AdapterSelector::Api("no such api".to_owned())] {
			let Err(reason) = Devices::open(&config(GpuDevices::List(vec![selector.clone()]))) else { panic!("{selector:?} opened") };
			assert!(reason.contains("matches no adapter") && reason.contains("found:"), "{reason}");
			let Err(reason) = Devices::open(&GpuConfig::default().with_adapter(selector.clone())) else { panic!("{selector:?} opened") };
			assert!(reason.contains("matches no adapter"), "{reason}");
		}
		assert!(Devices::open(&config(GpuDevices::List(Vec::new()))).is_err());
	}
}

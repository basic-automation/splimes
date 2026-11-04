//! Epoch-tick input (`run_epoch`) and streamed output (`stream_f64`, `stream_epoch`) are
//! bit-identical to `run_f64` on every backend and precision, across chunk boundaries.
//! Its own binary, because it configures the GPU with small chunks, and the configuration
//! is process-wide and set once.

mod common;

use std::{ops::ControlFlow, sync::Once};

use chrono::{DateTime, TimeDelta, Utc};
use common::{Rng, epoch};
use splimes::{Backend, Error, GpuConfig, Interpolation, Interpolator, PointKind, Precision, Resolution, Spline, StreamSummary, TickUnit};

/// The GPU's dispatch size in this binary: small, so a few-thousand-point grid crosses
/// several dispatch boundaries.
const GPU_CHUNK: u32 = 1_000;

fn setup() {
	static ONCE: Once = Once::new();
	ONCE.call_once(|| splimes::configure_gpu(GpuConfig::default().with_chunk_points(GPU_CHUNK)).expect("first configuration succeeds"));
}

/// Backend and precision pairs to check: the CPU backends, `Auto`, and the GPU in each
/// precision it has.
fn targets(test: &str) -> Vec<(Backend, Precision)> {
	setup();
	let mut targets = vec![(Backend::Cpu, Precision::F64), (Backend::Parallel, Precision::F64), (Backend::Auto, Precision::F64)];
	if let Some(info) = common::gpu_or_skip(test) {
		if info.supports_f64 {
			targets.push((Backend::Gpu, Precision::F64));
		}
		targets.push((Backend::Gpu, Precision::F32));
	}
	targets
}

/// Irregular, unsorted input with repeated instants, whole milliseconds apart, and its
/// grid's ends (a 1 ms grid of about 3,000 points from before the first knot to after
/// the last).
fn series(seed: u64) -> (Vec<DateTime<Utc>>, Vec<f64>, DateTime<Utc>, DateTime<Utc>) {
	let mut rng = Rng::new(seed);
	let mut ms = 0_i64;
	let mut ts = Vec::new();
	for _ in 0..60 {
		ms += 1 + i64::try_from(rng.below(90)).expect("small");
		ts.push(epoch() + TimeDelta::milliseconds(ms));
	}
	// Repeat a few instants (the later value wins) and shuffle.
	for i in [3, 17, 40] {
		ts.push(ts[i]);
	}
	let vs: Vec<f64> = (0..ts.len()).map(|_| rng.unit().mul_add(200.0, -100.0)).collect();
	let mut order: Vec<usize> = (0..ts.len()).collect();
	for i in (1..order.len()).rev() {
		order.swap(i, usize::try_from(rng.below(i as u64 + 1)).expect("small"));
	}
	let (ts, vs) = order.iter().map(|&i| (ts[i], vs[i])).unzip();
	(ts, vs, epoch() - TimeDelta::milliseconds(250), epoch() + TimeDelta::milliseconds(ms + 250))
}

const METHODS: [Spline; 4] = [Spline::Linear, Spline::Cubic, Spline::Polynomial(4, None), Spline::Polynomial(6, Some(0.5))];

/// Everything a stream delivered, concatenated, checking each chunk's position and size.
struct Collected<T> {
	times: Vec<T>,
	values: Vec<f64>,
	kinds: Vec<PointKind>,
	sizes: Vec<usize>,
}

impl<T: Copy> Collected<T> {
	fn new() -> Self {
		Self { times: Vec::new(), values: Vec::new(), kinds: Vec::new(), sizes: Vec::new() }
	}

	fn take(&mut self, chunk: &splimes::Chunk<'_, T>) -> ControlFlow<()> {
		assert_eq!(chunk.first(), self.values.len(), "chunks arrive in order, without gaps");
		assert!(!chunk.is_empty());
		assert_eq!((chunk.times().len(), chunk.kinds().len()), (chunk.len(), chunk.len()));
		self.times.extend_from_slice(chunk.times());
		self.values.extend_from_slice(chunk.values());
		self.kinds.extend_from_slice(chunk.kinds());
		self.sizes.push(chunk.len());
		ControlFlow::Continue(())
	}
}

fn same_values(a: &[f64], b: &[f64]) -> bool {
	a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// The summary reports what the whole-grid run reports.
fn same_report(summary: &StreamSummary, whole: &Interpolation<f64>, label: &str) {
	assert!(summary.is_complete(), "{label}");
	assert_eq!(summary.len(), whole.len(), "{label}");
	assert_eq!((summary.backend(), summary.precision(), summary.spline(), summary.requested_spline()), (whole.backend(), whole.precision(), whole.spline(), whole.requested_spline()), "{label}");
	assert_eq!(summary.points_recomputed_in_f64(), whole.points_recomputed_in_f64(), "{label}");
	assert_eq!(summary.gpu_fallback(), whole.gpu_fallback(), "{label}");
}

/// The largest chunk a stream may deliver with `requested` points per chunk.
fn chunk_limit(backend: Backend, requested: usize) -> usize {
	let requested = if requested == 0 { GPU_CHUNK as usize } else { requested };
	if backend == Backend::Gpu { requested.min(GPU_CHUNK as usize) } else { requested }
}

#[test]
fn streamed_f64_matches_run_f64() {
	let targets = targets("streamed_f64_matches_run_f64");
	let (ts, vs, start, end) = series(1);
	for (backend, precision) in targets {
		for spline in METHODS {
			let make = || Interpolator::new(spline, Resolution::Milliseconds).backend(backend).gpu_precision(precision);
			let whole = make().run_f64(&ts, &vs, start, end).expect("runs");
			assert!(whole.len() > 2 * GPU_CHUNK as usize);
			for requested in [7, 0, 999, 1_000, 1_001, 1 << 20] {
				let label = format!("{backend} {precision} {spline}, {requested}-point chunks");
				let mut got = Collected::new();
				let summary = make().stream_chunk_points(requested).stream_f64(&ts, &vs, start, end, |c| got.take(&c)).expect("streams");
				same_report(&summary, &whole, &label);
				assert_eq!(got.times, whole.timestamps(), "{label}");
				assert_eq!(got.kinds, whole.kinds(), "{label}");
				assert!(same_values(&got.values, whole.values()), "{label}");
				// Every chunk but the last is full.
				let limit = chunk_limit(summary.backend(), requested);
				assert_eq!(summary.chunks(), got.sizes.len(), "{label}");
				assert_eq!(got.sizes.len(), whole.len().div_ceil(limit), "{label}: {:?}", got.sizes);
				assert!(got.sizes[..got.sizes.len() - 1].iter().all(|&s| s == limit), "{label}: {:?}", got.sizes);
			}
		}
	}
}

/// One point per chunk, on every backend: the smallest chunk, where every point is its own
/// dispatch on the GPU.
#[test]
fn one_point_chunks() {
	let targets = targets("one_point_chunks");
	let (ts, vs, _, _) = series(2);
	let (start, end) = (epoch() + TimeDelta::milliseconds(40), epoch() + TimeDelta::milliseconds(200));
	for (backend, precision) in targets {
		for spline in [Spline::Linear, Spline::Cubic] {
			let make = || Interpolator::new(spline, Resolution::Milliseconds).backend(backend).gpu_precision(precision);
			let whole = make().run_f64(&ts, &vs, start, end).expect("runs");
			let mut got = Collected::new();
			let summary = make().stream_chunk_points(1).stream_f64(&ts, &vs, start, end, |c| got.take(&c)).expect("streams");
			same_report(&summary, &whole, "one point");
			assert_eq!(got.sizes, vec![1; whole.len()]);
			assert!(same_values(&got.values, whole.values()), "{backend} {precision} {spline}");
			assert_eq!(got.kinds, whole.kinds());
		}
	}
}

/// `run_epoch` and `stream_epoch` on ticks of any unit compute what `run_f64` computes on
/// the same instants, and `stream_epoch`'s times are those instants as ticks.
#[test]
fn epoch_ticks_match_run_f64() {
	let targets = targets("epoch_ticks_match_run_f64");
	let (ts, vs, _, _) = series(3);
	for (backend, precision) in targets {
		for (unit, resolution) in [(TickUnit::NANOSECONDS, Resolution::Milliseconds), (TickUnit::MICROSECONDS, Resolution::Milliseconds), (TickUnit::MILLISECONDS, Resolution::Milliseconds), (TickUnit::from_nanos(250_000).expect("unit"), Resolution::Milliseconds), (TickUnit::SECONDS, Resolution::Seconds)] {
			let per = i64::try_from(unit.nanos_per_tick()).expect("small");
			let to_tick = |t: DateTime<Utc>| i64::try_from(common::posix(t) / i128::from(per)).expect("fits");
			let from_tick = |tick: i64| DateTime::<Utc>::UNIX_EPOCH + TimeDelta::nanoseconds(tick * per);
			// The instants the ticks name (seconds ticks truncate the input).
			let ticks: Vec<i64> = ts.iter().map(|&t| to_tick(t)).collect();
			let instants: Vec<DateTime<Utc>> = ticks.iter().map(|&t| from_tick(t)).collect();
			let (start, end) = (to_tick(epoch()) - 3, to_tick(epoch() + TimeDelta::seconds(4)));
			for spline in METHODS {
				let label = format!("{backend} {precision} {spline} in {unit} ticks");
				let make = || Interpolator::new(spline, resolution).backend(backend).gpu_precision(precision);
				let whole = make().run_f64(&instants, &vs, from_tick(start), from_tick(end)).expect("runs");
				let epoch = make().run_epoch(&ticks, unit, &vs, start, end).expect("runs");
				assert_eq!(epoch.timestamps(), whole.timestamps(), "{label}");
				assert_eq!(epoch.kinds(), whole.kinds(), "{label}");
				assert!(same_values(epoch.values(), whole.values()), "{label}");
				assert_eq!((epoch.backend(), epoch.precision(), epoch.points_recomputed_in_f64()), (whole.backend(), whole.precision(), whole.points_recomputed_in_f64()), "{label}");

				let mut got = Collected::new();
				let summary = make().stream_chunk_points(333).stream_epoch(&ticks, unit, &vs, start, end, |c| got.take(&c)).expect("streams");
				same_report(&summary, &whole, &label);
				assert_eq!(got.times, whole.timestamps().iter().map(|&t| to_tick(t)).collect::<Vec<_>>(), "{label}");
				assert_eq!(got.kinds, whole.kinds(), "{label}");
				assert!(same_values(&got.values, whole.values()), "{label}");
			}
		}
	}
}

/// Windows the `f32` kernel can't compute are recomputed in `f64` chunk by chunk, to the
/// same values and the same count as the whole-grid run.
#[test]
fn f32_repairs_match_across_chunks() {
	setup();
	let Some(_) = common::gpu_or_skip("f32_repairs_match_across_chunks") else { return };
	// 2,000 knots a second apart, a gap of 400,000 s, then 5 more: the big gap is far
	// beyond the f32 limit of 1,024 mean spacings.
	let secs: Vec<i64> = (0..2_000).chain((0..5).map(|i| 402_000 + i)).collect();
	let ts: Vec<_> = secs.iter().map(|&s| epoch() + TimeDelta::seconds(s)).collect();
	let vs: Vec<f64> = secs.iter().map(|&s| (s as f64 * 0.37).sin()).collect();
	let (start, end) = (ts[0], ts[ts.len() - 1]);
	let make = || Interpolator::new(Spline::Cubic, Resolution::Minutes).backend(Backend::Gpu).gpu_precision(Precision::F32);
	let whole = make().run_f64(&ts, &vs, start, end).expect("runs");
	assert!(whole.points_recomputed_in_f64() > 0);
	let mut got = Collected::new();
	let summary = make().stream_chunk_points(257).stream_f64(&ts, &vs, start, end, |c| got.take(&c)).expect("streams");
	same_report(&summary, &whole, "f32 repair");
	assert!(same_values(&got.values, whole.values()));
}

/// Breaking out stops the run: nothing more is delivered, the summary says how much was,
/// and the next run is unaffected.
#[test]
fn breaking_stops_the_stream() {
	let targets = targets("breaking_stops_the_stream");
	let (ts, vs, start, end) = series(4);
	for (backend, precision) in targets {
		let make = || Interpolator::new(Spline::Cubic, Resolution::Milliseconds).backend(backend).gpu_precision(precision).stream_chunk_points(500);
		let mut calls = 0;
		let summary = make()
			.stream_f64(&ts, &vs, start, end, |_| {
				calls += 1;
				if calls == 2 { ControlFlow::Break(()) } else { ControlFlow::Continue(()) }
			})
			.expect("streams");
		assert_eq!(calls, 2, "{backend} {precision}");
		assert_eq!((summary.delivered(), summary.chunks(), summary.is_complete()), (1_000, 2, false), "{backend} {precision}");
		let whole = make().run_f64(&ts, &vs, start, end).expect("runs");
		let mut got = Collected::new();
		make().stream_f64(&ts, &vs, start, end, |c| got.take(&c)).expect("streams");
		assert!(same_values(&got.values, whole.values()), "{backend} {precision}: a run after a break");
	}
}

/// An overflow fails the stream with the error `run_f64` returns, naming the same point,
/// after delivering only the chunks before it.
#[test]
fn non_finite_results_stop_the_stream() {
	let targets = targets("non_finite_results_stop_the_stream");
	let ts = [epoch(), epoch() + TimeDelta::seconds(1)];
	let vs = [-1e308, 1e308];
	let (start, end) = (epoch(), epoch() + TimeDelta::seconds(3));
	for (backend, precision) in targets {
		let make = || Interpolator::new(Spline::Linear, Resolution::Milliseconds).backend(backend).gpu_precision(precision).stream_chunk_points(100);
		let expected = make().run_f64(&ts, &vs, start, end).expect_err("overflows");
		let Error::NonFiniteResult { timestamp } = expected else { panic!("{expected:?}") };
		let k = usize::try_from((timestamp - start).num_milliseconds()).expect("on the grid");
		if precision == Precision::F64 {
			// −1e308 + 2e308·t passes f64::MAX ≈ 1.798e308 at t ≈ 1.3989 s.
			assert_eq!(k, 1_399, "{backend} {precision}");
		}
		let mut got = Collected::new();
		let err = make().stream_f64(&ts, &vs, start, end, |c| got.take(&c)).expect_err("overflows");
		assert_eq!(err, expected, "{backend} {precision}");
		assert_eq!(got.values.len(), k / 100 * 100, "{backend} {precision}: every chunk before the failing one, none after");
		let ticks = [0, 1_000];
		let err = make().stream_epoch(&ticks, TickUnit::MILLISECONDS, &vs, 0, 3_000, |_| ControlFlow::Continue(())).expect_err("overflows");
		assert_eq!(err, Error::NonFiniteResult { timestamp: DateTime::<Utc>::UNIX_EPOCH + TimeDelta::milliseconds(i64::try_from(k).expect("small")) }, "{backend} {precision}");
	}
}

/// A panic in the callback reaches the caller, and the GPU is still usable after it.
#[test]
fn a_panicking_callback_panics_the_caller() {
	let targets = targets("a_panicking_callback_panics_the_caller");
	let (ts, vs, start, end) = series(5);
	for (backend, precision) in targets {
		let make = || Interpolator::new(Spline::Cubic, Resolution::Milliseconds).backend(backend).gpu_precision(precision).stream_chunk_points(500);
		let result = std::panic::catch_unwind(|| make().stream_f64(&ts, &vs, start, end, |_| -> ControlFlow<()> { panic!("callback panic") }));
		let payload = result.expect_err("the panic propagates");
		assert_eq!(payload.downcast_ref::<&str>(), Some(&"callback panic"));
		let whole = make().run_f64(&ts, &vs, start, end).expect("runs after the panic");
		assert_eq!(whole.len(), make().stream_f64(&ts, &vs, start, end, |_| ControlFlow::Continue(())).expect("streams").len());
	}
}

#[test]
fn refusals() {
	setup();
	let cpu = Interpolator::new(Spline::Linear, Resolution::Milliseconds).backend(Backend::Cpu);
	let go = |_: splimes::Chunk<'_, i64>| ControlFlow::Continue(());
	// A millisecond grid in second ticks has points between ticks.
	assert_eq!(cpu.stream_epoch(&[0, 1], TickUnit::SECONDS, &[0.0, 1.0], 0, 1, go).map(|s| s.len()), Err(Error::GridNotInTicks { resolution: Resolution::Milliseconds, nanos_per_tick: 1_000_000_000 }));
	// But `run_epoch` returns `DateTime`s, so it accepts that grid.
	assert_eq!(cpu.run_epoch(&[0, 1], TickUnit::SECONDS, &[0.0, 1.0], 0, 1).map(|r| r.len()), Ok(1_001));
	// Ticks beyond chrono's range.
	let far = i64::MAX / 2;
	assert_eq!(cpu.run_epoch(&[0, far], TickUnit::SECONDS, &[0.0, 1.0], 0, 1).map(|r| r.len()), Err(Error::TickOutOfRange { tick: far, nanos_per_tick: 1_000_000_000 }));
	assert_eq!(cpu.stream_epoch(&[0, 1], TickUnit::SECONDS, &[0.0, 1.0], 0, far, go).map(|s| s.len()), Err(Error::TickOutOfRange { tick: far, nanos_per_tick: 1_000_000_000 }));
	// Columns of different lengths, and the checks every entry point makes, in order.
	assert_eq!(cpu.stream_epoch(&[0, 1], TickUnit::MILLISECONDS, &[0.0], 0, 1, go).map(|s| s.len()), Err(Error::LengthMismatch { timestamps: 2, values: 1 }));
	assert_eq!(cpu.run_epoch(&[0], TickUnit::MILLISECONDS, &[], 0, 1).map(|r| r.len()), Err(Error::LengthMismatch { timestamps: 1, values: 0 }));
	assert_eq!(cpu.stream_epoch(&[], TickUnit::MILLISECONDS, &[], 0, 1, go).map(|s| s.len()), Err(Error::NoPoints));
	assert_eq!(cpu.stream_epoch(&[0], TickUnit::MILLISECONDS, &[0.0], 2, 1, go).map(|s| s.len()), Err(Error::InvalidTimeRange { start: DateTime::<Utc>::UNIX_EPOCH + TimeDelta::milliseconds(2), end: DateTime::<Utc>::UNIX_EPOCH + TimeDelta::milliseconds(1) }));
	assert_eq!(cpu.max_points(10).stream_epoch(&[0], TickUnit::MILLISECONDS, &[0.0], 0, 100, go).map(|s| s.len()), Err(Error::OutputTooLarge { points: 101 }));
	assert_eq!(cpu.stream_epoch(&[0], TickUnit::MILLISECONDS, &[f64::NAN], 0, 1, go).map(|s| s.len()), Err(Error::ValueOutOfRange { timestamp: DateTime::<Utc>::UNIX_EPOCH }));
	let at = |ms| DateTime::<Utc>::UNIX_EPOCH + TimeDelta::milliseconds(ms);
	assert_eq!(cpu.stream_f64(&[at(0)], &[0.0, 1.0], at(0), at(1), |_| ControlFlow::Continue(())).map(|s| s.len()), Err(Error::LengthMismatch { timestamps: 1, values: 2 }));
	assert_eq!(cpu.exact(true).stream_f64(&[at(0)], &[0.0], at(0), at(1), |_| ControlFlow::Continue(())).map(|s| s.len()), Err(Error::InsufficientPoints { spline: Spline::Linear, required: 2, available: 1 }));
}

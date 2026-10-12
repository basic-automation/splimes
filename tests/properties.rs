//! Randomised properties of the input and provenance contract, on every backend.
//!
//! `contract.rs` measures accuracy on clean, sorted series. This file feeds every backend
//! the input the crate docs promise to accept — unsorted, with duplicate instants, a
//! single point, constant values, decimals with more digits than `f64` holds — and checks
//! what the docs say about the result:
//!
//! - the result depends only on the distinct, last-wins input, not on its order;
//! - every point is labelled by the definition of [`PointKind`], and raw points return
//!   the input's own value, every digit of it;
//! - the method reported is the documented step-down, and `exact(true)` refuses it;
//! - the grid is `start, start + step, …`, never past `end`;
//! - `run` and `run_f64` agree, and `spawn` and `spawn_f64` return exactly what they do;
//! - every value is within the published bound of the exact reference;
//! - a request the docs refuse gets the same error, naming the same point, on every
//!   backend and precision and through every entry point (`run`, `run_f64`, `spawn`,
//!   `spawn_f64`, and the `tokio` feature's `run_async` and `run_f64_async`).

mod common;

use bigdecimal::BigDecimal;
use chrono::{DateTime, TimeDelta, Utc};
use common::{Rng, epoch, exact_with_lebesgue, to_f64};
use splimes::{Backend, Error, Interpolation, Interpolator, MAX_POLYNOMIAL_DEGREE, Point, PointKind, Precision, Resolution, Spline};

const CASES: u64 = 1024;

/// One randomised input and the grid it is resampled onto.
struct Case {
	label: String,
	spline: Spline,
	/// As the caller passes it: shuffled, with duplicate instants.
	points: Vec<Point>,
	start: DateTime<Utc>,
	end: DateTime<Utc>,
}

fn case(seed: u64) -> Case {
	let mut rng = Rng::new(seed);
	let n = 1 + rng.below(24) as usize;
	// Instants on a whole-second lattice, so the grid lands on them, drawn with
	// replacement so small lattices repeat instants; a third are nudged off it by a few
	// milliseconds, so they are never raw.
	let lattice = n as u64 * [1, 2, 4][rng.below(3) as usize];
	let shape = rng.below(4);
	let constant = rng.below(8) == 0;
	let points: Vec<Point> = (0..n)
		.map(|_| {
			let secs = rng.below(lattice) as i64;
			let millis = if rng.below(3) == 0 { 1 + rng.below(999) as i64 } else { 0 };
			let t = epoch() + TimeDelta::seconds(secs) + TimeDelta::milliseconds(millis);
			// Up to 24 significant digits: beyond f64, so raw points show whether the input
			// came back untouched.
			let digits: String = (0..1 + rng.below(24)).map(|_| char::from(b'0' + rng.below(10) as u8)).collect();
			let sign = if rng.below(2) == 0 { "" } else { "-" };
			let value = match (constant, shape) {
				(true, _) => "7.25".to_owned(),
				(_, 0) => format!("{sign}{digits}e-3"),
				(_, 1) => format!("{sign}{digits}e-120"),
				(_, 2) => format!("{sign}{digits}e80"),
				// Large offset, small variation: what breaks naive single precision.
				_ => format!("1000000000.{digits}"),
			};
			Point::new(t, value.parse().expect("decimal"))
		})
		.collect();
	let spline = match rng.below(6) {
		0 => Spline::Linear,
		1 => Spline::Quadratic,
		2 => Spline::Cubic,
		3 => Spline::Polynomial(1 + rng.below(8) as usize, None),
		_ => Spline::Polynomial(1 + rng.below(8) as usize, Some(rng.unit() * 2.0)),
	};
	// From a few seconds before the lattice to a few after, or a window inside it.
	let (from, to) = if rng.below(4) == 0 {
		let a = rng.below(lattice) as i64;
		(a, a + rng.below(lattice) as i64)
	} else {
		(-(rng.below(6) as i64), lattice as i64 + rng.below(6) as i64)
	};
	let label = format!("seed {seed}: {n} points, {spline}, grid {from}..={to} s");
	Case { label, spline, points, start: epoch() + TimeDelta::seconds(from), end: epoch() + TimeDelta::seconds(to) }
}

/// The backends and precisions to check: the CPU always, the GPU in every precision the
/// adapter has, when there is one.
fn configurations() -> Vec<(Backend, Precision)> {
	let mut out = vec![(Backend::Cpu, Precision::F64), (Backend::Parallel, Precision::F64)];
	if let Some(info) = common::gpu_or_skip("properties: GPU configurations") {
		if info.supports_f64 {
			out.push((Backend::Gpu, Precision::F64));
		}
		out.push((Backend::Gpu, Precision::F32));
	}
	out
}

const fn bound(precision: Precision) -> f64 {
	match precision {
		Precision::F64 => 1e-13,
		_ => 1e-5,
	}
}

/// Checks every documented property of `out`, the result of running `c` on `backend`.
fn check(c: &Case, distinct: &[Point], backend: Backend, precision: Precision, out: &Interpolation<BigDecimal>, reference: &[(f64, f64)]) {
	let label = format!("{} on {backend} {precision}", c.label);
	assert_eq!(out.requested_spline(), c.spline, "{label}");
	assert_eq!(out.spline(), c.spline.fallback_for(distinct.len()), "{label}: the documented step-down");
	assert_eq!(out.backend(), backend, "{label}");
	assert_eq!(out.precision(), if backend == Backend::Gpu { precision } else { Precision::F64 }, "{label}");
	assert!(out.gpu_fallback().is_none(), "{label}");

	// The grid: anchored at start, one step apart, never past end, and no shorter.
	let ts = out.timestamps();
	assert_eq!(ts.first(), Some(&c.start), "{label}");
	assert!(ts.windows(2).all(|w| w[1] - w[0] == TimeDelta::seconds(1)), "{label}");
	let last = *ts.last().expect("a grid has its start");
	assert!(last <= c.end && last + TimeDelta::seconds(1) > c.end, "{label}");

	let first_knot = distinct[0].timestamp;
	let last_knot = distinct[distinct.len() - 1].timestamp;
	let (min, max) = common::value_range(distinct);
	let range = to_f64(&(&max - &min));
	let range = if range > 0.0 { range } else { to_f64(&max).abs().max(f64::MIN_POSITIVE) };
	for (k, ((t, v), kind)) in ts.iter().zip(out.values()).zip(out.kinds()).enumerate() {
		match distinct.iter().find(|p| p.timestamp == *t) {
			Some(input) => {
				assert_eq!(*kind, PointKind::Raw, "{label}: {t} is an input instant");
				// Every digit, not just the same number.
				assert_eq!(v.to_string(), input.value.to_string(), "{label}: raw value at {t}");
			}
			None if *t < first_knot || *t > last_knot => assert_eq!(*kind, PointKind::Extrapolated, "{label} at {t}"),
			None => assert_eq!(*kind, PointKind::Interpolated, "{label} at {t}"),
		}
		if *kind != PointKind::Raw {
			let (exact, lebesgue) = reference[k];
			let got = to_f64(v);
			let err = ((got - exact).abs() - f64::EPSILON * exact.abs()).max(0.0) / (range * lebesgue);
			assert!(err <= bound(precision), "{label} at {t}: {got} vs exact {exact}, {err:.2e} of range · Λ");
		}
	}
}

#[test]
fn every_backend_keeps_the_input_contract() {
	let configurations = configurations();
	// Any executor can await `spawn`'s future; a single-threaded one shows it needs none.
	let executor = tokio::runtime::Builder::new_current_thread().build().expect("a tokio runtime");
	let mut checked = 0;
	for seed in 0..CASES {
		let c = case(seed);
		let distinct = common::distinct(&c.points);
		let grid_len = usize::try_from((c.end - c.start).num_seconds()).expect("ascending") + 1;
		let reference: Vec<(f64, f64)> = (0..grid_len)
			.map(|k| {
				let (value, lebesgue) = exact_with_lebesgue(&c.points, c.spline, c.start + TimeDelta::seconds(k as i64));
				(to_f64(&value), lebesgue)
			})
			.collect();
		let ts: Vec<_> = c.points.iter().map(|p| p.timestamp).collect();
		let vs: Vec<_> = c.points.iter().map(|p| to_f64(&p.value)).collect();
		for &(backend, precision) in &configurations {
			let interpolator = Interpolator::new(c.spline, Resolution::Seconds).backend(backend).gpu_precision(precision);
			let out = interpolator.run(&c.points, c.start, c.end).unwrap_or_else(|e| panic!("{} on {backend} {precision}: {e}", c.label));
			check(&c, &distinct, backend, precision, &out, &reference);
			// The same series as a caller with clean data would pass it, sorted with one point
			// per instant, gives bit for bit the same answer: order and duplicates don't matter.
			let canonical = interpolator.run(&distinct, c.start, c.end).unwrap_or_else(|e| panic!("{} on {backend} {precision}, canonical: {e}", c.label));
			assert_eq!(out, canonical, "{} on {backend} {precision}: order and duplicates must not matter", c.label);
			// run_f64 on the same samples computes the same f64s; raw points are the input's
			// own f64.
			let f = interpolator.run_f64(&ts, &vs, c.start, c.end).unwrap_or_else(|e| panic!("{} on {backend} {precision}, run_f64: {e}", c.label));
			let decimal: Vec<f64> = out.values().iter().map(to_f64).collect();
			assert_eq!(f.values(), decimal.as_slice(), "{} on {backend} {precision}: run and run_f64 agree", c.label);
			assert_eq!(f.kinds(), out.kinds(), "{} on {backend} {precision}", c.label);
			// spawn and spawn_f64 are run and run_f64 on rayon's pool: the same result, bit for bit.
			assert_eq!(executor.block_on(interpolator.spawn(c.points.clone(), c.start, c.end)), Ok(out.clone()), "{} on {backend} {precision}: spawn", c.label);
			assert_eq!(executor.block_on(interpolator.spawn_f64(ts.clone(), vs.clone(), c.start, c.end)), Ok(f), "{} on {backend} {precision}: spawn_f64", c.label);
			checked += out.len();
		}

		// exact(true) refuses exactly when the documented step-down would apply.
		let cpu = Interpolator::new(c.spline, Resolution::Seconds).backend(Backend::Cpu);
		let exact = cpu.exact(true).run(&c.points, c.start, c.end);
		assert_eq!(executor.block_on(cpu.exact(true).spawn(c.points.clone(), c.start, c.end)), exact, "{}: spawn, exact", c.label);
		if distinct.len() < c.spline.min_points() {
			assert_eq!(exact, Err(Error::InsufficientPoints { spline: c.spline, required: c.spline.min_points(), available: distinct.len() }), "{}", c.label);
		} else {
			assert_eq!(exact, cpu.run(&c.points, c.start, c.end), "{}", c.label);
		}
	}
	println!("{CASES} cases, {} configurations, {checked} grid points checked", configurations.len());
}

/// The generator covers what it claims to: single points, duplicates, unsorted input,
/// step-downs, raw, interpolated and extrapolated points, and constant series.
#[test]
fn the_generator_covers_the_edge_cases() {
	let (mut single, mut duplicates, mut unsorted, mut stepped, mut constant) = (0, 0, 0, 0, 0);
	let mut kinds = [0; 3];
	for seed in 0..CASES {
		let c = case(seed);
		let distinct = common::distinct(&c.points);
		single += usize::from(distinct.len() == 1);
		duplicates += usize::from(distinct.len() < c.points.len());
		unsorted += usize::from(c.points.windows(2).any(|w| w[0].timestamp > w[1].timestamp));
		stepped += usize::from(c.spline.fallback_for(distinct.len()) != c.spline);
		constant += usize::from(distinct.len() > 1 && distinct.iter().all(|p| p.value == distinct[0].value));
		let out = Interpolator::new(c.spline, Resolution::Seconds).backend(Backend::Cpu).run(&c.points, c.start, c.end).expect("runs");
		for kind in out.kinds() {
			kinds[*kind as usize] += 1;
		}
	}
	for (what, count) in [("single-point", single), ("duplicate", duplicates), ("unsorted", unsorted), ("stepped-down", stepped), ("constant", constant), ("raw", kinds[0]), ("interpolated", kinds[1]), ("extrapolated", kinds[2])] {
		assert!(count >= 5, "only {count} {what} cases");
	}
}

/// Every way of running `interpolator` on one input: `run`, `run_f64`, `spawn`,
/// `spawn_f64` and, with the `tokio` feature, `run_async` and `run_f64_async`, each
/// reduced to its error or the result's length. `points` feeds the decimal entry points
/// and `columns` the `f64` ones; either may be absent where the input has no form for it
/// (a NaN isn't a `BigDecimal`, 1e400 isn't an `f64`).
fn every_entry_point(interpolator: Interpolator, points: Option<&[Point]>, columns: Option<(&[DateTime<Utc>], &[f64])>, start: DateTime<Utc>, end: DateTime<Utc>) -> Vec<(&'static str, Result<usize, Error>)> {
	let mut out = Vec::new();
	#[cfg(feature = "tokio")]
	let runtime = tokio::runtime::Builder::new_current_thread().build().expect("a runtime");
	if let Some(points) = points {
		out.push(("run", interpolator.run(points, start, end).map(|r| r.len())));
		out.push(("spawn", common::block_on(interpolator.spawn(points.to_vec(), start, end)).map(|r| r.len())));
		#[cfg(feature = "tokio")]
		out.push(("run_async", runtime.block_on(interpolator.run_async(points.to_vec(), start, end)).map(|r| r.len())));
	}
	if let Some((ts, vs)) = columns {
		out.push(("run_f64", interpolator.run_f64(ts, vs, start, end).map(|r| r.len())));
		out.push(("spawn_f64", common::block_on(interpolator.spawn_f64(ts.to_vec(), vs.to_vec(), start, end)).map(|r| r.len())));
		#[cfg(feature = "tokio")]
		out.push(("run_f64_async", runtime.block_on(interpolator.run_f64_async(ts.to_vec(), vs.to_vec(), start, end)).map(|r| r.len())));
	}
	out
}

/// A request that must be refused, and the error it must be refused with.
struct Refusal<'a> {
	label: &'static str,
	interpolator: Interpolator,
	points: Option<&'a [Point]>,
	columns: Option<(&'a [DateTime<Utc>], &'a [f64])>,
	start: DateTime<Utc>,
	end: DateTime<Utc>,
	expected: Error,
}

/// A request the docs say is refused gets the same error, naming the same point, on every
/// backend and precision and through every entry point; the documented one, checked on
/// the CPU, is in the table.
#[test]
fn every_backend_refuses_the_same_input() {
	let at = |s: i64| epoch() + TimeDelta::seconds(s);
	let decimal = |pairs: &[(i64, &str)]| pairs.iter().map(|&(s, v)| Point::new(at(s), v.parse().expect("decimal"))).collect::<Vec<_>>();
	let two = decimal(&[(0, "0"), (1, "1")]);
	let (two_ts, two_vs) = ([at(0), at(1)], [0.0, 1.0]);
	// Big enough to be prepared on rayon's pool off `Cpu` (16 Ki inputs), shuffled, with two
	// bad values: the first in input order is at the later instant.
	let n = 20_000_i64;
	let big_ts: Vec<_> = (0..n).map(|i| at((i * 7919) % n)).collect();
	let mut big_vs: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
	let (first_bad, second_bad) = (5_000, 15_000);
	assert!(big_ts[first_bad] > big_ts[second_bad], "the first bad value in input order is the later one in time");
	big_vs[first_bad] = f64::NAN;
	big_vs[second_bad] = f64::INFINITY;
	let mut big_points: Vec<Point> = big_ts.iter().zip(&big_vs).map(|(&t, &v)| Point::new(t, BigDecimal::try_from(if v.is_finite() { v } else { 0.0 }).expect("finite"))).collect();
	big_points[first_bad].value = "1e400".parse().expect("decimal");
	big_points[second_bad].value = "-1e400".parse().expect("decimal");
	let steep = decimal(&[(0, "-1e308"), (1, "1e308")]);

	let linear = Interpolator::new(Spline::Linear, Resolution::Seconds);
	let cases = [Refusal { label: "no points", interpolator: linear, points: Some(&[][..]), columns: Some((&[], &[])), start: at(0), end: at(1), expected: Error::NoPoints }, Refusal { label: "start after end", interpolator: linear, points: Some(&two), columns: Some((&two_ts, &two_vs)), start: at(1), end: at(0), expected: Error::InvalidTimeRange { start: at(1), end: at(0) } }, Refusal { label: "max_points", interpolator: linear.max_points(10), points: Some(&two), columns: Some((&two_ts, &two_vs)), start: at(0), end: at(10), expected: Error::OutputTooLarge { points: 11 } }, Refusal { label: "a grid beyond memory", interpolator: Interpolator::new(Spline::Linear, Resolution::Nanoseconds), points: Some(&two), columns: Some((&two_ts, &two_vs)), start: at(0), end: at(0) + TimeDelta::days(365 * 1000), expected: Error::OutputTooLarge { points: 31_536_000_000_000_000_001 } }, Refusal { label: "degree 0", interpolator: Interpolator::new(Spline::Polynomial(0, None), Resolution::Seconds), points: Some(&two), columns: Some((&two_ts, &two_vs)), start: at(0), end: at(1), expected: Error::InvalidDegree { degree: 0, max: MAX_POLYNOMIAL_DEGREE } }, Refusal { label: "degree too high", interpolator: Interpolator::new(Spline::Polynomial(MAX_POLYNOMIAL_DEGREE + 1, None), Resolution::Seconds), points: Some(&two), columns: Some((&two_ts, &two_vs)), start: at(0), end: at(1), expected: Error::InvalidDegree { degree: MAX_POLYNOMIAL_DEGREE + 1, max: MAX_POLYNOMIAL_DEGREE } }, Refusal { label: "negative bounds factor", interpolator: Interpolator::new(Spline::Polynomial(2, Some(-1.0)), Resolution::Seconds), points: Some(&two), columns: Some((&two_ts, &two_vs)), start: at(0), end: at(1), expected: Error::InvalidBoundsFactor(-1.0) }, Refusal { label: "columns of different lengths", interpolator: linear, points: None, columns: Some((&two_ts, &[0.0])), start: at(0), end: at(1), expected: Error::LengthMismatch { timestamps: 2, values: 1 } }, Refusal { label: "bad values, first in input order", interpolator: linear, points: Some(&big_points), columns: Some((&big_ts, &big_vs)), start: at(0), end: at(10), expected: Error::ValueOutOfRange { timestamp: big_ts[first_bad] } }, Refusal { label: "exact(true), too few for cubic", interpolator: Interpolator::new(Spline::Cubic, Resolution::Seconds).exact(true), points: Some(&two), columns: Some((&two_ts, &two_vs)), start: at(0), end: at(1), expected: Error::InsufficientPoints { spline: Spline::Cubic, required: 4, available: 2 } }, Refusal { label: "exact(true), one point", interpolator: linear.exact(true), points: Some(&two[..1]), columns: Some((&two_ts[..1], &two_vs[..1])), start: at(0), end: at(1), expected: Error::InsufficientPoints { spline: Spline::Linear, required: 2, available: 1 } }, Refusal { label: "an overflowing extrapolation", interpolator: Interpolator::new(Spline::Linear, Resolution::Milliseconds), points: Some(&steep), columns: None, start: at(0), end: at(200), expected: Error::NonFiniteResult { timestamp: at(0) + TimeDelta::milliseconds(1399) } }];

	let mut configurations = configurations();
	configurations.push((Backend::Auto, Precision::F64));
	let mut checked = 0;
	for c in &cases {
		for &(backend, precision) in &configurations {
			for (entry, outcome) in every_entry_point(c.interpolator.backend(backend).gpu_precision(precision), c.points, c.columns, c.start, c.end) {
				assert_eq!(outcome, Err(c.expected.clone()), "{}: {entry} on {backend} {precision}", c.label);
				checked += 1;
			}
		}
	}
	println!("{} refusals, {} configurations, {checked} outcomes checked", cases.len(), configurations.len());
}

/// A GPU chunk larger than one 16 Ki-point block of staged grid times (the default chunk is
/// 4 Mi points) computes what the CPU does at every point, across block boundaries and in
/// the tail. Grids in the other GPU tests fit in one block.
#[test]
fn gpu_chunks_of_many_staging_blocks_match_the_cpu() {
	let Some(info) = common::gpu_or_skip("gpu_chunks_of_many_staging_blocks_match_the_cpu") else { return };
	let ts: Vec<DateTime<Utc>> = (0..400_i64).map(|i| epoch() + TimeDelta::milliseconds(i * 250 + (i * 7_919) % 200)).collect();
	let vs: Vec<f64> = (0..400).map(|i| (f64::from(i) * 0.21).sin()).collect();
	// Three whole blocks and a tail, on a millisecond grid.
	let (start, end) = (ts[0], ts[0] + TimeDelta::milliseconds(3 * 16 * 1024 + 123 - 1));
	let mut precisions = vec![Precision::F32];
	if info.supports_f64 {
		precisions.push(Precision::F64);
	}
	for spline in [Spline::Linear, Spline::Cubic, Spline::Polynomial(5, None)] {
		let cpu = Interpolator::new(spline, Resolution::Milliseconds).backend(Backend::Cpu).run_f64(&ts, &vs, start, end).expect("cpu");
		for &precision in &precisions {
			let gpu = Interpolator::new(spline, Resolution::Milliseconds).backend(Backend::Gpu).gpu_precision(precision).run_f64(&ts, &vs, start, end).expect("gpu");
			assert_eq!((gpu.len(), gpu.backend(), gpu.points_recomputed_in_f64()), (3 * 16 * 1024 + 123, Backend::Gpu, 0), "{spline} {precision}");
			let tolerance = bound(precision) * 100.0;
			for (k, (g, c)) in gpu.values().iter().zip(cpu.values()).enumerate() {
				assert!((g - c).abs() <= tolerance, "{spline} {precision}, grid point {k}: GPU {g} vs CPU {c}");
			}
		}
	}
}

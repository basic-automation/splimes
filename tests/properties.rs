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
//! - `run` and `run_f64` agree;
//! - every value is within the published bound of the exact reference.

mod common;

use bigdecimal::BigDecimal;
use chrono::{DateTime, TimeDelta, Utc};
use common::{Rng, epoch, exact_with_lebesgue, to_f64};
use splimes::{Backend, Error, Interpolation, Interpolator, Point, PointKind, Precision, Resolution, Spline};

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
			checked += out.len();
		}

		// exact(true) refuses exactly when the documented step-down would apply.
		let cpu = Interpolator::new(c.spline, Resolution::Seconds).backend(Backend::Cpu);
		let exact = cpu.exact(true).run(&c.points, c.start, c.end);
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

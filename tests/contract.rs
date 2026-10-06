//! The numerical contract in the crate docs, enforced.
//!
//! Every backend and precision is compared against `common::exact` — the method's
//! definition evaluated in 60-digit decimal arithmetic on the raw timestamps and values —
//! over randomised series: knot spacings from microseconds to months, regular and
//! irregular; values tiny, huge, and large-offset-small-variation (the case that breaks
//! naive single precision); 1 to 2,000 points; grids across both edges and the middle.
//!
//! An error `|got − exact|` is measured, after subtracting the half-ulp any `f64` of that
//! magnitude must carry, in units of `range · Λ(t)`: the input value range (`max − min`, or
//! the largest magnitude if constant) times the window's Lebesgue function at the grid
//! point (see `common::exact_with_lebesgue`). That is the yardstick for rounding in any
//! Lagrange evaluation, inside or outside the data. The bounds asserted here are the ones
//! the crate docs publish; a failure means either a regression or a contract change, and
//! the latter is a breaking change.

mod common;

use std::sync::LazyLock;

use bigdecimal::BigDecimal;
use chrono::{DateTime, TimeDelta, Utc};
use common::{Rng, epoch, exact, exact_with_lebesgue, to_f64};
use rayon::prelude::*;
use splimes::{Backend, Interpolator, Point, PointKind, Precision, Resolution, Spline};

/// The published bounds, in units of `range · Λ`: the "Numerical contract" section of the
/// crate docs. Rounding in the Lagrange form is at most about `3m·ε` of that for an
/// `m`-point window, so they hold with wide headroom for every degree.
const fn bound(precision: Precision, _degree: usize) -> f64 {
	match precision {
		Precision::F64 => 1e-13,
		_ => 1e-5,
	}
}

const SPLINES: [Spline; 6] = [Spline::Linear, Spline::Quadratic, Spline::Cubic, Spline::Polynomial(5, None), Spline::Polynomial(4, Some(0.5)), Spline::Polynomial(8, None)];
/// Grid points per grid; three grids per series.
const GRID_POINTS: i32 = 300;

/// One series, one grid, one method, and the exact answer at every grid point.
struct Fixture {
	label: String,
	spline: Spline,
	points: Vec<Point>,
	start: DateTime<Utc>,
	end: DateTime<Utc>,
	resolution: Resolution,
	/// The exact value at each grid point, as the nearest f64, and the window's Lebesgue
	/// function there.
	expected: Vec<(f64, f64)>,
	range: f64,
}

static FIXTURES: LazyLock<Vec<Fixture>> = LazyLock::new(|| {
	let mut jobs = Vec::new();
	for (label, points) in series() {
		let knots = common::distinct(&points);
		let n = knots.len();
		let o: Vec<i128> = knots.iter().map(|p| common::nanos_between(p.timestamp, knots[0].timestamp)).collect();
		let first = knots[0].timestamp;
		let last = knots[n - 1].timestamp;
		// The coarsest resolution with at least 20 grid steps across the smallest knot gap,
		// so every grid samples between the knots it covers.
		let min_gap = o.windows(2).map(|w| w[1] - w[0]).min().unwrap_or(1_000_000_000).max(20);
		let resolution = Resolution::ALL.iter().copied().rev().find(|r| i128::from(r.step_nanos()) * 20 <= min_gap).unwrap_or(Resolution::Nanoseconds);
		let step = resolution.step();
		let width = step * (GRID_POINTS - 1);
		let span = o[n - 1].max(1_000);
		let middle = first + TimeDelta::nanoseconds((span / 2) as i64);
		// And one grid across the densest stretch, where irregularity bites hardest.
		let densest = o.windows(2).enumerate().min_by_key(|(_, w)| w[1] - w[0]).map_or(first, |(i, _)| knots[i].timestamp);
		let (min, max) = common::value_range(&knots);
		let range = to_f64(&(&max - &min));
		let range = if range > 0.0 { range } else { to_f64(&max).abs().max(f64::MIN_POSITIVE) };
		for spline in SPLINES {
			// The outside-the-data bound is published for up to two spacings of the edge
			// window (the mean gap among the first or last degree + 1 knots); the edge grids
			// reach exactly that far.
			let m = spline.fallback_for(n).min_points().min(n);
			let edge_pad = |a: i128, b: i128| TimeDelta::nanoseconds(if m >= 2 { (2 * (b - a) / (m as i128 - 1)) as i64 } else { 2_000 });
			let (pad_start, pad_end) = (edge_pad(o[0], o[m - 1]), edge_pad(o[n - m], o[n - 1]));
			// The bound holds at any distance (Λ grows with it); one grid far out, 100 edge
			// spacings past the last knot, keeps the scaled far-extrapolation path honest.
			let far = last + pad_end * 50;
			for (start, end) in [(first - pad_start, first - pad_start + width), (middle, middle + width), (densest, densest + width), (last + pad_end - width, last + pad_end), (far, far + width)] {
				jobs.push((label.clone(), spline, points.clone(), start, end, resolution, range));
			}
		}
	}
	jobs.into_par_iter()
		.map(|(label, spline, points, start, end, resolution, range)| {
			let expected = (0..GRID_POINTS)
				.map(|k| {
					let (value, lebesgue) = exact_with_lebesgue(&points, spline, start + resolution.step() * k);
					(to_f64(&value), lebesgue)
				})
				.collect();
			Fixture { label, spline, points, start, end, resolution, expected, range }
		})
		.collect()
});

fn series() -> Vec<(String, Vec<Point>)> {
	let mut rng = Rng::new(0x5EED);
	let spacings: [(&str, i64); 4] = [("µs", 1_000), ("s", 1_000_000_000), ("h", 3_600_000_000_000), ("30d", 2_592_000_000_000_000)];
	let mut out = Vec::new();
	for &(spacing_name, spacing) in &spacings {
		for n in [1, 2, 3, 4, 5, 9, 40, 2_000] {
			for shape in ["walk", "offset", "tiny", "huge"] {
				for irregular in [false, true] {
					let mut t = epoch();
					let mut v = 0.0_f64;
					let points = (0..n)
						.map(|_| {
							// Irregular gaps range over 0.05× to 3× the nominal spacing.
							let gap = if irregular { (spacing as f64 * 2.95_f64.mul_add(rng.unit(), 0.05)) as i64 } else { spacing };
							t += TimeDelta::nanoseconds(gap.max(1));
							v += rng.unit() - 0.5;
							let value = match shape {
								"walk" => v,
								"offset" => v.mul_add(1e-3, 1e9),
								"tiny" => v * 1e-200,
								_ => v * 1e200,
							};
							Point::new(t, format!("{value:e}").parse().expect("finite"))
						})
						.collect();
					out.push((format!("{spacing_name}/{n}/{shape}/{}", if irregular { "irregular" } else { "regular" }), points));
				}
			}
		}
	}
	// Two deterministic stress shapes, both within "neighbouring gaps vary by up to 60×".
	let at = |secs: f64| epoch() + TimeDelta::nanoseconds((secs * 1e9) as i64);
	let point = |secs: f64, value: f64| Point::new(at(secs), format!("{value:e}").parse().expect("finite"));
	// A burst: 32 samples 3 s apart, then 8 more 50 ms apart, on a smooth curve. The edge
	// window is 60× denser than the series' mean spacing.
	let mut t = 0.0;
	let burst = (0..40)
		.map(|i| {
			t += if i < 32 { 3.0 } else { 0.05 };
			point(t, (t / 10.0).sin() * 10.0 + t)
		})
		.collect();
	out.push(("burst/smooth".to_owned(), burst));
	// Geometric gaps, each 60× the next, alternating values: the hardest case the
	// irregularity limit allows.
	let mut t = 0.0;
	let geometric = [0.0, 216_000.0, 3_600.0, 60.0, 1.0, 33.9, 1.0, 1.0, 1.0]
		.iter()
		.enumerate()
		.map(|(i, gap)| {
			t += gap;
			point(t, if i % 2 == 0 { 1.0 } else { -1.0 })
		})
		.collect();
	out.push(("geometric/alternating".to_owned(), geometric));
	out
}

/// Worst relative errors found for one method.
#[derive(Default, Clone, Copy)]
struct Worst {
	inside: f64,
	outside: f64,
}

/// Runs every fixture on one backend configuration and returns the worst errors per
/// method. Raw points must equal the input exactly.
fn sweep(backend: Backend, precision: Precision, f64_api: bool) -> Vec<(Spline, Worst)> {
	let mut worst = [Worst::default(); SPLINES.len()];
	for f in FIXTURES.iter() {
		let interpolator = Interpolator::new(f.spline, f.resolution).backend(backend).gpu_precision(precision);
		// A NonFiniteResult is only correct where the exact value really leaves f64.
		let overflowed = |timestamp| {
			let exact = to_f64(&exact(&f.points, f.spline, timestamp));
			assert!(!exact.is_finite() || exact.abs() > 0.999 * f64::MAX, "{} {}: NonFiniteResult at {timestamp}, but the exact value {exact} is finite", f.label, f.spline);
		};
		let (got, kinds): (Vec<f64>, Vec<PointKind>) = if f64_api {
			let ts: Vec<_> = f.points.iter().map(|p| p.timestamp).collect();
			let vs: Vec<_> = f.points.iter().map(|p| to_f64(&p.value)).collect();
			match interpolator.run_f64(&ts, &vs, f.start, f.end) {
				Ok(r) => (r.values().to_vec(), r.kinds().to_vec()),
				Err(splimes::Error::NonFiniteResult { timestamp }) => {
					overflowed(timestamp);
					continue;
				}
				Err(e) => panic!("{} {}: {e}", f.label, f.spline),
			}
		} else {
			match interpolator.run(&f.points, f.start, f.end) {
				Ok(r) => {
					for ((t, v), k) in r.timestamps().iter().zip(r.values()).zip(r.kinds()) {
						if *k == PointKind::Raw {
							let input = &common::distinct(&f.points).into_iter().find(|p| p.timestamp == *t).expect("raw point has an input").value;
							assert_eq!(v, input, "{} {}: raw point at {t} must be the input, exactly", f.label, f.spline);
						}
					}
					(r.values().iter().map(to_f64).collect(), r.kinds().to_vec())
				}
				Err(splimes::Error::NonFiniteResult { timestamp }) => {
					overflowed(timestamp);
					continue;
				}
				Err(e) => panic!("{} {}: {e}", f.label, f.spline),
			}
		};
		assert_eq!(got.len(), f.expected.len(), "{} {}: grid length", f.label, f.spline);
		let w = &mut worst[SPLINES.iter().position(|s| *s == f.spline).expect("known spline")];
		for ((&g, &(e, lebesgue)), &kind) in got.iter().zip(&f.expected).zip(&kinds) {
			assert!(g.is_finite(), "{} {}: a non-finite value came back as a value", f.label, f.spline);
			// The contract: |got − exact| ≤ bound · range · Λ + ε·|exact|. The ε term is the
			// rounding any f64 of that magnitude carries.
			let err = ((g - e).abs() - f64::EPSILON * e.abs()).max(0.0) / (f.range * lebesgue);
			assert!(!err.is_nan(), "{} {}: NaN error against {e}", f.label, f.spline);
			let slot = match kind {
				// Raw values are checked for exact equality above (BigDecimal) or are the
				// input's own f64.
				PointKind::Raw => continue,
				PointKind::Interpolated => &mut w.inside,
				PointKind::Extrapolated => &mut w.outside,
			};
			if err > *slot {
				*slot = err;
				if std::env::var_os("CONTRACT_WORST").is_some() {
					eprintln!("worst so far {} {} {kind:?}: {err:.2e} (Λ {lebesgue:.2e}, got {g}, exact {e})", f.label, f.spline);
				}
			}
		}
	}
	SPLINES.into_iter().zip(worst).collect()
}

fn report(name: &str, rows: &[(Spline, Worst)], precision: Precision) {
	println!("\n{name}");
	println!("{:<44} {:>10} {:>10} {:>10}", "method", "inside", "outside", "bound");
	for (spline, w) in rows {
		println!("{:<44} {:>10.2e} {:>10.2e} {:>10.0e}", spline.to_string(), w.inside, w.outside, bound(precision, spline.degree()));
	}
	for (spline, w) in rows {
		let bound = bound(precision, spline.degree());
		assert!(w.inside <= bound, "{name}: {spline} interpolation error {:e} exceeds the published bound {bound:e}", w.inside);
		assert!(w.outside <= bound, "{name}: {spline} extrapolation error {:e} exceeds the published bound {bound:e}", w.outside);
	}
}

#[test]
fn cpu_meets_the_contract() {
	report("Cpu, BigDecimal", &sweep(Backend::Cpu, Precision::F64, false), Precision::F64);
}

#[test]
fn parallel_meets_the_contract() {
	report("Parallel, f64", &sweep(Backend::Parallel, Precision::F64, true), Precision::F64);
}

#[test]
fn gpu_f64_meets_the_contract() {
	let Some(info) = common::gpu_or_skip("gpu_f64_meets_the_contract") else { return };
	if !info.supports_f64 {
		eprintln!("gpu_f64_meets_the_contract: skipped, {} has no f64 support", info.name);
		return;
	}
	report(&format!("Gpu f64 on {} ({})", info.name, info.api), &sweep(Backend::Gpu, Precision::F64, true), Precision::F64);
}

#[test]
fn gpu_f32_meets_the_contract() {
	let Some(info) = common::gpu_or_skip("gpu_f32_meets_the_contract") else { return };
	report(&format!("Gpu f32 on {} ({})", info.name, info.api), &sweep(Backend::Gpu, Precision::F32, true), Precision::F32);
}

/// The bounds hold however long the series. Time differences are taken exactly, so
/// two million knots in, a millisecond grid still sees its local spacing to one rounding.
/// (Subtracting rounded absolute times instead breaks the f64 bound past about a million
/// knots.)
#[test]
fn long_series_keep_the_bound() {
	const N: i64 = 2_000_000;
	let timestamps: Vec<DateTime<Utc>> = (0..N).map(|i| epoch() + TimeDelta::seconds(i)).collect();
	let values: Vec<f64> = (0..N).map(|i| (i % 2) as f64).collect();
	let points: Vec<Point> = timestamps.iter().zip(&values).map(|(&t, &v)| Point::new(t, BigDecimal::from(v as i64))).collect();
	// The exact answer near the end needs only the last knots: the window there is
	// clamped to the end of the data, and the slice ends where the data does.
	let tail = &points[points.len() - 40..];
	let (start, end) = (timestamps[timestamps.len() - 4], timestamps[timestamps.len() - 1] + TimeDelta::seconds(1));
	let mut configs = vec![(Backend::Cpu, Precision::F64), (Backend::Parallel, Precision::F64)];
	if let Some(info) = common::gpu_or_skip("long_series_keep_the_bound (GPU part)") {
		if info.supports_f64 {
			configs.push((Backend::Gpu, Precision::F64));
		}
		configs.push((Backend::Gpu, Precision::F32));
	}
	for spline in [Spline::Linear, Spline::Cubic, Spline::Polynomial(5, None)] {
		for &(backend, precision) in &configs {
			let out = Interpolator::new(spline, Resolution::Milliseconds).backend(backend).gpu_precision(precision).run_f64(&timestamps, &values, start, end).expect("runs");
			let mut worst = 0.0_f64;
			for ((&t, &got), kind) in out.timestamps().iter().zip(out.values()).zip(out.kinds()) {
				assert!(got.is_finite(), "{spline} on {backend} {precision}: non-finite value at {t}");
				if *kind == PointKind::Interpolated {
					let expected = to_f64(&exact(tail, spline, t));
					worst = worst.max(((got - expected).abs() - f64::EPSILON * expected.abs()).max(0.0));
				}
			}
			let bound = bound(precision, spline.degree());
			println!("{N} knots, {spline} on {backend} {precision}: {worst:.2e} (bound {bound:.0e})");
			assert!(worst <= bound, "{spline} on {backend} {precision}: error {worst:e} exceeds {bound:e} two million knots in");
		}
	}
}

/// Knots a nanosecond apart, far from the first knot, stay distinct: no 0/0, no
/// spurious `NonFiniteResult`, and linear interpolation keeps its bound.
#[test]
fn nearly_coincident_knots_stay_distinct() {
	let day = TimeDelta::days(1);
	let t0 = epoch();
	let timestamps = vec![t0, t0 + day * 200, t0 + day * 200 + TimeDelta::nanoseconds(1), t0 + day * 400, t0 + day * 600];
	let values = vec![0.0, 1.0, 1.0, 0.0, 1.0];
	let points: Vec<Point> = timestamps.iter().zip(&values).map(|(&t, &v)| Point::new(t, BigDecimal::from(v as i64))).collect();
	let mut configs = vec![(Backend::Cpu, Precision::F64), (Backend::Parallel, Precision::F64)];
	if let Some(info) = common::gpu_or_skip("nearly_coincident_knots_stay_distinct (GPU part)")
		&& info.supports_f64
	{
		configs.push((Backend::Gpu, Precision::F64));
	}
	for (backend, precision) in configs {
		let run = |spline| Interpolator::new(spline, Resolution::Hours).backend(backend).gpu_precision(precision).run_f64(&timestamps, &values, t0, t0 + day * 600);
		let linear = run(Spline::Linear).expect("linear runs");
		for ((&t, &got), kind) in linear.timestamps().iter().zip(linear.values()).zip(linear.kinds()) {
			if *kind != PointKind::Raw {
				let expected = to_f64(&exact(&points, Spline::Linear, t));
				assert!((got - expected).abs() <= 1e-11, "linear on {backend} at {t}: {got} vs {expected}");
			}
		}
		// A cubic through two knots a nanosecond apart is ill-conditioned (the contract
		// says so), but it is finite: computing it must not fail.
		let cubic = run(Spline::Cubic).expect("cubic runs");
		assert!(cubic.values().iter().all(|v| v.is_finite()), "cubic on {backend}");
	}
}

/// Far outside the data, f32 Lagrange terms overflow long before the value does. The
/// bounds clamp must still land on the right side, and an unbounded value must still come
/// back right (the GPU recomputes overflowing points in f64), never as a wrong finite
/// number.
#[test]
fn far_bounded_extrapolation_lands_on_the_right_bound() {
	let t0 = epoch();
	let timestamps: Vec<DateTime<Utc>> = (0..9).map(|i| t0 + TimeDelta::seconds(i)).collect();
	let rising: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.5];
	let falling: Vec<f64> = rising.iter().map(|v| -v).collect();
	let mut configs = vec![(Backend::Cpu, Precision::F64)];
	if let Some(info) = common::gpu_or_skip("far_bounded_extrapolation_lands_on_the_right_bound (GPU part)") {
		if info.supports_f64 {
			configs.push((Backend::Gpu, Precision::F64));
		}
		configs.push((Backend::Gpu, Precision::F32));
	}
	for (values, lo, hi) in [(&rising, -4.25, 12.75), (&falling, -12.75, 4.25)] {
		for far in [TimeDelta::days(3), TimeDelta::days(7), TimeDelta::days(365)] {
			let at = timestamps[8] + far;
			let points: Vec<Point> = timestamps.iter().zip(values.iter()).map(|(&t, &v)| Point::new(t, format!("{v:e}").parse().expect("finite"))).collect();
			let (exact_free, lebesgue) = exact_with_lebesgue(&points, Spline::Polynomial(8, None), at);
			let exact_free = to_f64(&exact_free);
			for &(backend, precision) in &configs {
				let bounded = Interpolator::new(Spline::Polynomial(8, Some(0.5)), Resolution::Seconds).backend(backend).gpu_precision(precision).run_f64(&timestamps, values, at, at).expect("runs").values()[0];
				let expected = if values[8] > 0.0 { hi } else { lo };
				assert_eq!(bounded, expected, "bounded, {far} out, on {backend} {precision}");
				assert!(bounded >= lo && bounded <= hi);
				// Unbounded, the value is enormous (and in f32 near the top of its range), but it
				// must still be finite and within the published contract — not a wrong number.
				let free = Interpolator::new(Spline::Polynomial(8, None), Resolution::Seconds).backend(backend).gpu_precision(precision).run_f64(&timestamps, values, at, at).expect("runs").values()[0];
				let allowed = bound(precision, 8) * 8.5 * lebesgue + f64::EPSILON * exact_free.abs();
				assert!((free - exact_free).abs() <= allowed, "unbounded, {far} out, on {backend} {precision}: {free} vs exact {exact_free}");
				assert_eq!(free.signum(), exact_free.signum());
			}
		}
	}
}

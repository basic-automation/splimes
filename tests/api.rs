//! The documented behaviour of the public API, one property per test.

mod common;

use std::str::FromStr;

use bigdecimal::BigDecimal;
use chrono::{DateTime, TimeDelta, Utc};
use common::epoch;
use splimes::{Backend, Error, Interpolator, MAX_POLYNOMIAL_DEGREE, Point, PointKind, Precision, Resolution, Spline};

fn at(secs: i64) -> DateTime<Utc> {
	epoch() + TimeDelta::seconds(secs)
}

fn dec(s: &str) -> BigDecimal {
	BigDecimal::from_str(s).expect("decimal")
}

fn points(pairs: &[(i64, &str)]) -> Vec<Point> {
	pairs.iter().map(|&(t, v)| Point::new(at(t), dec(v))).collect()
}

fn cpu(spline: Spline) -> Interpolator {
	Interpolator::new(spline, Resolution::Seconds).backend(Backend::Cpu)
}

#[test]
fn every_output_point_is_labelled() {
	let input = points(&[(0, "0"), (10, "10"), (20, "0")]);
	let out = cpu(Spline::Linear).run(&input, at(-2), at(22)).expect("runs");
	let kind = |t: i64| out.kinds()[usize::try_from(t + 2).expect("index")];
	assert_eq!(kind(-2), PointKind::Extrapolated);
	assert_eq!(kind(0), PointKind::Raw);
	assert_eq!(kind(5), PointKind::Interpolated);
	assert_eq!(kind(10), PointKind::Raw);
	assert_eq!(kind(20), PointKind::Raw);
	assert_eq!(kind(21), PointKind::Extrapolated);
	assert_eq!(out.kinds().iter().filter(|k| **k == PointKind::Raw).count(), 3);
}

#[test]
fn raw_points_keep_every_digit() {
	let precise = "3.14159265358979323846264338327950288419716939937510";
	let input = points(&[(0, "1"), (5, precise), (10, "2"), (15, "1.5")]);
	for backend in [Backend::Cpu, Backend::Parallel] {
		let out = Interpolator::new(Spline::Cubic, Resolution::Seconds).backend(backend).run(&input, at(0), at(15)).expect("runs");
		assert_eq!(out.values()[5], dec(precise), "{backend}: a raw point is the input, not its f64 rounding");
	}
}

#[test]
fn interpolated_values_are_short_decimals() {
	let input = points(&[(0, "0"), (3, "1")]);
	let out = cpu(Spline::Linear).run(&input, at(0), at(3)).expect("runs");
	assert_eq!(out.values()[1].to_string(), "0.3333333333333333");
}

#[test]
fn input_order_does_not_matter() {
	let sorted = points(&[(0, "1"), (4, "3"), (9, "-2"), (13, "5"), (20, "0")]);
	let mut shuffled = sorted.clone();
	shuffled.swap(0, 3);
	shuffled.swap(1, 4);
	let a = cpu(Spline::Cubic).run(&sorted, at(-5), at(25)).expect("runs");
	let b = cpu(Spline::Cubic).run(&shuffled, at(-5), at(25)).expect("runs");
	assert_eq!(a, b);
}

#[test]
fn the_last_duplicate_wins() {
	let input = points(&[(0, "0"), (10, "1"), (10, "7"), (20, "0")]);
	let out = cpu(Spline::Linear).run(&input, at(10), at(10)).expect("runs");
	assert_eq!(out.values(), &[dec("7")]);
}

#[test]
fn fallback_is_reported_or_refused() {
	let three = points(&[(0, "0"), (1, "1"), (2, "4")]);
	let out = cpu(Spline::Cubic).run(&three, at(0), at(2)).expect("runs");
	assert_eq!(out.requested_spline(), Spline::Cubic);
	assert_eq!(out.spline(), Spline::Quadratic);

	let err = cpu(Spline::Cubic).exact(true).run(&three, at(0), at(2)).expect_err("refuses to fall back");
	assert_eq!(err, Error::InsufficientPoints { spline: Spline::Cubic, required: 4, available: 3 });

	// Duplicates don't count towards the minimum.
	let dupes = points(&[(0, "0"), (0, "1"), (1, "1"), (2, "4")]);
	assert_eq!(cpu(Spline::Cubic).run(&dupes, at(0), at(2)).expect("runs").spline(), Spline::Quadratic);

	let out = cpu(Spline::Polynomial(6, Some(1.0))).run(&three, at(0), at(2)).expect("runs");
	assert_eq!(out.spline(), Spline::Polynomial(2, Some(1.0)));

	// Linear can't step down, but one point is still too few for it.
	let one = points(&[(0, "5")]);
	assert_eq!(cpu(Spline::Linear).exact(true).run(&one, at(0), at(2)), Err(Error::InsufficientPoints { spline: Spline::Linear, required: 2, available: 1 }));
	assert_eq!(cpu(Spline::Polynomial(1, None)).run(&one, at(0), at(2)).expect("runs").spline(), Spline::Polynomial(1, None));
}

#[test]
fn one_point_is_a_constant() {
	let input = points(&[(5, "42.5")]);
	let out = cpu(Spline::Cubic).run(&input, at(0), at(10)).expect("runs");
	assert_eq!(out.spline(), Spline::Linear);
	assert!(out.values().iter().all(|v| *v == dec("42.5")));
	assert_eq!(out.kinds()[5], PointKind::Raw);
	assert_eq!(out.kinds()[0], PointKind::Extrapolated);
}

#[test]
fn grid_shape() {
	let input = points(&[(0, "0"), (100, "1")]);
	assert_eq!(cpu(Spline::Linear).run(&input, at(7), at(7)).expect("runs").len(), 1);
	let out = cpu(Spline::Linear).run(&input, at(0), at(10)).expect("runs");
	assert_eq!(out.timestamps().first(), Some(&at(0)));
	assert_eq!(out.timestamps().last(), Some(&at(10)));
	// Anchored at `start`, not the epoch, and never past `end`.
	let out = Interpolator::new(Spline::Linear, Resolution::Minutes).backend(Backend::Cpu).run(&input, at(1), at(150)).expect("runs");
	assert_eq!(out.timestamps(), &[at(1), at(61), at(121)]);
	let monthly = Interpolator::new(Spline::Linear, Resolution::Months).backend(Backend::Cpu).run(&input, at(0), at(0) + TimeDelta::days(95)).expect("runs");
	assert_eq!(monthly.timestamps()[1] - monthly.timestamps()[0], TimeDelta::days(30));
	assert_eq!(monthly.len(), 4);
}

#[test]
fn extrapolation_semantics() {
	let input = points(&[(0, "0"), (1, "1"), (2, "4"), (3, "9")]);
	let value = |spline, t| -> f64 { common::to_f64(&cpu(spline).run(&input, at(t), at(t)).expect("runs").values()[0]) };
	assert_eq!(value(Spline::Linear, 5), 19.0, "linear extends the last segment");
	assert_eq!(value(Spline::Quadratic, 5), 25.0, "quadratic extends the last parabola");
	assert_eq!(value(Spline::Cubic, 5), 9.0, "cubic holds the last value");
	assert_eq!(value(Spline::Cubic, -5), 0.0, "cubic holds the first value");
	// Held exactly: every digit of the edge input, not its round trip through the kernel.
	let precise = points(&[(0, "0.1000000000000000000001"), (1, "1"), (2, "4"), (3, "9.33333333333333333333333")]);
	let held = cpu(Spline::Cubic).run(&precise, at(-2), at(5)).expect("runs");
	assert_eq!(held.values()[0], dec("0.1000000000000000000001"));
	assert_eq!(held.values()[7], dec("9.33333333333333333333333"));
	assert_eq!(held.kinds()[0], PointKind::Extrapolated);
	assert_eq!(value(Spline::Polynomial(2, None), 5), 25.0);
	// range 9, bounds factor 0.5: clamp to [-4.5, 13.5].
	assert_eq!(value(Spline::Polynomial(2, Some(0.5)), 5), 13.5);
	assert_eq!(value(Spline::Polynomial(2, Some(0.5)), -5), 13.5, "the parabola t² is 25 at -5 too");
}

#[test]
fn errors() {
	let two = points(&[(0, "0"), (1, "1")]);
	assert_eq!(cpu(Spline::Linear).run(&[], at(0), at(1)), Err(Error::NoPoints));
	assert_eq!(cpu(Spline::Linear).run(&two, at(1), at(0)), Err(Error::InvalidTimeRange { start: at(1), end: at(0) }));
	assert!(matches!(Interpolator::new(Spline::Linear, Resolution::Nanoseconds).run(&two, at(0), at(0) + TimeDelta::days(365 * 1000)), Err(Error::OutputTooLarge { .. })));
	assert_eq!(cpu(Spline::Linear).max_points(10).run(&two, at(0), at(10)), Err(Error::OutputTooLarge { points: 11 }));
	assert_eq!(cpu(Spline::Linear).max_points(11).run(&two, at(0), at(10)).map(|r| r.len()), Ok(11));
	assert_eq!(cpu(Spline::Polynomial(0, None)).run(&two, at(0), at(1)), Err(Error::InvalidDegree { degree: 0, max: MAX_POLYNOMIAL_DEGREE }));
	assert_eq!(cpu(Spline::Polynomial(MAX_POLYNOMIAL_DEGREE + 1, None)).run(&two, at(0), at(1)), Err(Error::InvalidDegree { degree: MAX_POLYNOMIAL_DEGREE + 1, max: MAX_POLYNOMIAL_DEGREE }));
	assert!(matches!(cpu(Spline::Polynomial(2, Some(f64::INFINITY))).run(&two, at(0), at(1)), Err(Error::InvalidBoundsFactor(_))));
	assert_eq!(cpu(Spline::Linear).run(&points(&[(0, "1"), (1, "1e400")]), at(0), at(1)), Err(Error::ValueOutOfRange { timestamp: at(1) }));
	assert_eq!(cpu(Spline::Linear).run_f64(&[at(0)], &[1.0, 2.0], at(0), at(1)), Err(Error::LengthMismatch { timestamps: 1, values: 2 }));
	assert!(matches!(cpu(Spline::Linear).run_f64(&[at(0), at(1)], &[1.0, f64::NAN], at(0), at(1)), Err(Error::ValueOutOfRange { .. })));
	// A steep line, extended far enough, leaves f64.
	let steep = points(&[(0, "-1e308"), (1, "1e308")]);
	assert_eq!(cpu(Spline::Linear).run(&steep, at(5), at(5)), Err(Error::NonFiniteResult { timestamp: at(5) }));
}

#[test]
fn errors_display_usefully() {
	let msg = Error::InsufficientPoints { spline: Spline::Cubic, required: 4, available: 3 }.to_string();
	assert_eq!(msg, "Cubic needs at least 4 distinct points, got 3");
}

#[test]
fn cpu_and_parallel_are_bit_identical() {
	let input: Vec<Point> = (0..500).map(|i| Point::new(at(i * 37 + (i * i) % 11), format!("{}", (f64::from(i32::try_from(i).expect("small")) * 0.37).sin()).parse().expect("decimal"))).collect();
	for spline in [Spline::Linear, Spline::Quadratic, Spline::Cubic, Spline::Polynomial(7, Some(0.2))] {
		let make = |backend| Interpolator::new(spline, Resolution::Milliseconds).backend(backend);
		let (start, end) = (at(-100), at(1_900));
		let a = make(Backend::Cpu).run(&input, start, end).expect("cpu");
		let b = make(Backend::Parallel).run(&input, start, end).expect("parallel");
		assert_eq!(a.len(), 2_000_001);
		assert_eq!(a.values(), b.values(), "{spline}");
		assert_eq!(a.kinds(), b.kinds(), "{spline}");
		assert_eq!(b.backend(), Backend::Parallel);
	}
}

/// Large shuffled input with duplicates takes the parallel preparation path on every
/// backend but `Cpu`; the results are the same, bit for bit.
#[test]
fn parallel_preparation_changes_nothing() {
	let n: i64 = 50_000;
	// An odd multiplier permutes 0..n when n is coprime to it; every instant appears twice.
	let shuffled = |i: i64| (i * 7_919) % n;
	let points: Vec<Point> = (0..2 * n).map(|i| Point::new(at(shuffled(i % n) * 2), format!("{}", (i as f64 * 0.37).sin()).parse().expect("decimal"))).collect();
	for spline in [Spline::Linear, Spline::Cubic, Spline::Polynomial(6, Some(0.5))] {
		let make = |backend| Interpolator::new(spline, Resolution::Seconds).backend(backend);
		let cpu = make(Backend::Cpu).run(&points, at(-10), at(2 * n + 10)).expect("cpu");
		let parallel = make(Backend::Parallel).run(&points, at(-10), at(2 * n + 10)).expect("parallel");
		assert_eq!(cpu.values(), parallel.values(), "{spline}");
		assert_eq!(cpu.kinds(), parallel.kinds(), "{spline}");
		assert_eq!(cpu.kinds().iter().filter(|k| **k == PointKind::Raw).count(), usize::try_from(n).expect("small"));
	}
}

/// The kernel takes time differences in `i64` when every knot and grid point is within
/// about 146 years of the first knot, and in `i128` otherwise. The same instants get the
/// same values either way.
#[test]
fn the_i64_and_i128_paths_agree() {
	let year = TimeDelta::days(365);
	// Forty knots over forty years, irregular.
	let points: Vec<Point> = (0..40).map(|i: i32| Point::new(at(0) + year * i + TimeDelta::seconds(i64::from(i * i) * 86_413), format!("{}", f64::from(i).cos() * 10.0).parse().expect("decimal"))).collect();
	for spline in [Spline::Linear, Spline::Quadratic, Spline::Cubic, Spline::Polynomial(8, None)] {
		for backend in [Backend::Cpu, Backend::Parallel] {
			let make = Interpolator::new(spline, Resolution::Days).backend(backend);
			// A grid over the data (i64), and one running 300 years past it (i128).
			let near = make.run(&points, at(0) - year * 2, at(0) + year * 42).expect("near");
			let far = make.run(&points, at(0) - year * 2, at(0) + year * 300).expect("far");
			assert!(far.len() > near.len());
			assert_eq!(near.values(), &far.values()[..near.len()], "{spline} {backend}");
			assert_eq!(near.kinds(), &far.kinds()[..near.len()], "{spline} {backend}");
		}
	}
}

#[test]
fn f64_and_bigdecimal_apis_agree() {
	let input = points(&[(0, "1.5"), (3, "-2"), (7, "4.25"), (12, "0")]);
	let ts: Vec<_> = input.iter().map(|p| p.timestamp).collect();
	let vs: Vec<_> = input.iter().map(|p| common::to_f64(&p.value)).collect();
	let a = cpu(Spline::Cubic).run(&input, at(-3), at(15)).expect("decimal");
	let b = cpu(Spline::Cubic).run_f64(&ts, &vs, at(-3), at(15)).expect("f64");
	let a: Vec<f64> = a.values().iter().map(common::to_f64).collect();
	assert_eq!(a, b.values());
}

#[test]
fn report_and_accessors() {
	let input = points(&[(0, "0"), (2, "2")]);
	let out = cpu(Spline::Linear).run(&input, at(0), at(2)).expect("runs");
	assert_eq!(out.backend(), Backend::Cpu);
	assert_eq!(out.precision(), Precision::F64);
	assert!(out.gpu_fallback().is_none());
	assert!(!out.is_empty());
	let rows: Vec<_> = out.iter().map(|(t, v, k)| (t, v.clone(), k)).collect();
	assert_eq!(rows[1], (at(1), dec("1"), PointKind::Interpolated));
	let (ts, vs, ks) = out.clone().into_parts();
	assert_eq!((ts.len(), vs.len(), ks.len()), (3, 3, 3));
	assert_eq!(out.into_points()[2], Point::new(at(2), dec("2")));
}

#[test]
fn the_whole_chrono_range_works() {
	let input = vec![Point::new(DateTime::<Utc>::MIN_UTC, dec("0")), Point::new(DateTime::<Utc>::MAX_UTC, dec("1"))];
	let out = Interpolator::new(Spline::Linear, Resolution::Years).backend(Backend::Parallel).run(&input, DateTime::<Utc>::MIN_UTC, DateTime::<Utc>::MAX_UTC).expect("runs");
	assert!(out.len() > 500_000);
	let middle = common::to_f64(&out.values()[out.len() / 2]);
	assert!((middle - 0.5).abs() < 1e-6, "{middle}");
}

#[test]
fn convenience_function_matches_interpolator() {
	let input = points(&[(0, "0"), (1, "1"), (2, "4"), (3, "9")]);
	let a = splimes::interpolate(&input, at(0), at(3), Resolution::Seconds, Spline::Cubic).expect("runs");
	let b = Interpolator::new(Spline::Cubic, Resolution::Seconds).run(&input, at(0), at(3)).expect("runs");
	assert_eq!(a.values(), b.values());
}

/// The serde formats are a stability promise: WeftDB persists `Spline` and `Resolution`.
#[cfg(feature = "serde")]
#[test]
fn serde_wire_formats_are_stable() {
	let json = |v: &dyn erased::Ser| v.to_json();
	assert_eq!(json(&Resolution::Seconds), r#""Seconds""#);
	assert_eq!(json(&Spline::Cubic), r#""Cubic""#);
	assert_eq!(json(&Spline::Polynomial(3, Some(1.5))), r#"{"Polynomial":[3,1.5]}"#);
	assert_eq!(json(&Spline::Polynomial(2, None)), r#"{"Polynomial":[2,null]}"#);
	assert_eq!(json(&PointKind::Extrapolated), r#""extrapolated""#);
	let point = Point::new(DateTime::<Utc>::UNIX_EPOCH, dec("1.25"));
	assert_eq!(json(&point), r#"{"timestamp":"1970-01-01T00:00:00Z","value":"1.25"}"#);
	assert_eq!(serde_json::from_str::<Point>(&json(&point)).expect("round trip"), point);
	assert_eq!(serde_json::from_str::<Spline>(r#"{"Polynomial":[3,1.5]}"#).expect("parses"), Spline::Polynomial(3, Some(1.5)));
}

/// `Spline`'s text and serde forms round-trip every valid value bit for bit, bounds factors
/// included (subnormal, huge, negative zero), and its parser refuses every invalid one.
#[test]
fn spline_text_forms_round_trip() {
	let mut rng = common::Rng::new(0x5_911E);
	let mut bounds: Vec<Option<f64>> = vec![None, Some(0.0), Some(-0.0), Some(f64::MIN_POSITIVE / 8.0), Some(f64::MAX), Some(1.0 / 3.0), Some(1e-300)];
	// Random finite, non-negative doubles across every exponent.
	bounds.extend((0..500).map(|_| Some(f64::from_bits(rng.next_u64() >> 1))).filter(|b| b.is_some_and(f64::is_finite)));
	for degree in 1..=MAX_POLYNOMIAL_DEGREE {
		for &b in &bounds {
			let spline = Spline::Polynomial(degree, b);
			let parsed: Spline = spline.to_string().parse().unwrap_or_else(|e| panic!("{spline}: {e}"));
			assert_eq!(parsed.degree(), degree);
			assert_eq!(parsed.bounds_factor().map(f64::to_bits), b.map(f64::to_bits), "{spline}");
			#[cfg(feature = "serde")]
			{
				let json = serde_json::to_string(&spline).expect("serialises");
				let back: Spline = serde_json::from_str(&json).expect("deserialises");
				assert_eq!(back.bounds_factor().map(f64::to_bits), b.map(f64::to_bits), "{json}");
			}
		}
	}
	for invalid in ["Polynomial(degree: 0, bounds_factor: None)", "Polynomial(degree: 9, bounds_factor: None)", "Polynomial(degree: 3, bounds_factor: -1)", "Polynomial(degree: 3, bounds_factor: NaN)", "Polynomial(degree: 3, bounds_factor: inf)", "Polynomial(degree: 3)", "Polynomial(3, None)", "Akima", ""] {
		assert!(invalid.parse::<Spline>().is_err(), "{invalid:?} parsed");
	}
}

#[cfg(feature = "serde")]
mod erased {
	pub trait Ser {
		fn to_json(&self) -> String;
	}
	impl<T: serde::Serialize> Ser for T {
		fn to_json(&self) -> String {
			serde_json::to_string(self).expect("serialises")
		}
	}
}

/// On a single-threaded runtime, another task keeps running while an interpolation is in
/// flight: the work happens on the blocking pool, not on the executor.
#[cfg(feature = "tokio")]
#[tokio::test(flavor = "current_thread")]
async fn async_wrappers_run_off_the_executor() {
	use std::sync::{
		Arc, atomic::{AtomicBool, AtomicU64, Ordering}
	};
	let ticks = Arc::new(AtomicU64::new(0));
	let done = Arc::new(AtomicBool::new(false));
	let ticker = {
		let (ticks, done) = (Arc::clone(&ticks), Arc::clone(&done));
		tokio::spawn(async move {
			while !done.load(Ordering::Relaxed) {
				ticks.fetch_add(1, Ordering::Relaxed);
				tokio::task::yield_now().await;
			}
		})
	};
	let input = points(&[(0, "0"), (100, "100")]);
	// Ten million points on a microsecond grid: a few hundred milliseconds on one thread.
	let interpolator = Interpolator::new(Spline::Linear, Resolution::Microseconds).backend(Backend::Cpu);
	let out = interpolator.run_async(input, at(0), at(10)).await.expect("runs");
	assert_eq!(out.len(), 10_000_001);
	let after_bigdecimal = ticks.load(Ordering::Relaxed);
	assert!(after_bigdecimal > 100, "run_async blocked the executor: {after_bigdecimal} ticks");
	let out = interpolator.run_f64_async(vec![at(0), at(100)], vec![0.0, 100.0], at(0), at(10)).await.expect("runs");
	assert_eq!(out.values()[3_000_000], 3.0);
	done.store(true, Ordering::Relaxed);
	ticker.await.expect("ticker");
	let during_f64 = ticks.load(Ordering::Relaxed) - after_bigdecimal;
	assert!(during_f64 > 100, "run_f64_async blocked the executor: {during_f64} ticks");
}

/// `spawn` needs no feature and no particular runtime: here, on a single-threaded tokio
/// runtime, another task keeps running while it computes, and the result is `run`'s.
#[tokio::test(flavor = "current_thread")]
async fn spawned_interpolations_run_off_the_executor() {
	use std::sync::{
		Arc, atomic::{AtomicBool, AtomicU64, Ordering}
	};
	let ticks = Arc::new(AtomicU64::new(0));
	let done = Arc::new(AtomicBool::new(false));
	let ticker = {
		let (ticks, done) = (Arc::clone(&ticks), Arc::clone(&done));
		tokio::spawn(async move {
			while !done.load(Ordering::Relaxed) {
				ticks.fetch_add(1, Ordering::Relaxed);
				tokio::task::yield_now().await;
			}
		})
	};
	let input = points(&[(0, "0"), (100, "100")]);
	let interpolator = Interpolator::new(Spline::Linear, Resolution::Microseconds).backend(Backend::Cpu);
	let out = interpolator.spawn(input.clone(), at(0), at(10)).await.expect("runs");
	assert_eq!(out.len(), 10_000_001);
	assert_eq!(out.values()[5_000_000], dec("5"));
	let during = ticks.load(Ordering::Relaxed);
	assert!(during > 100, "spawn blocked the executor: {during} ticks");
	let out = interpolator.spawn_f64(vec![at(0), at(100)], vec![0.0, 100.0], at(0), at(10)).await.expect("runs");
	assert_eq!(out.values()[3_000_000], 3.0);
	done.store(true, Ordering::Relaxed);
	ticker.await.expect("ticker");
}

/// Every input value must be finite, even one a later duplicate replaces.
#[test]
fn bad_values_are_refused_wherever_they_are() {
	let ts = [at(0), at(1), at(1)];
	for vs in [[0.0, f64::NAN, 1.0], [0.0, 1.0, f64::INFINITY]] {
		assert_eq!(cpu(Spline::Linear).run_f64(&ts, &vs, at(0), at(1)), Err(Error::ValueOutOfRange { timestamp: at(1) }), "{vs:?}");
	}
}

/// Values at the very edge of f64 come back as themselves, not as an overflow.
#[test]
fn values_at_f64_max_survive() {
	let ts = [at(0), at(10), at(20)];
	let vs = [-1.515_246_496_309_115_1e296, f64::MAX, f64::MAX];
	let out = cpu(Spline::Linear).run_f64(&ts, &vs, at(0), at(20)).expect("finite inside the data");
	assert_eq!(out.values()[15], f64::MAX);
	assert!(out.values().iter().all(|v| v.is_finite()));
}

/// chrono can represent a leap second (`23:59:60.x`); splimes puts it on the POSIX scale,
/// where it is the same instant as `00:00:00.x` the next second, everywhere.
#[test]
fn leap_seconds_are_the_next_second() {
	let ts = |s: &str| DateTime::parse_from_rfc3339(s).expect("chrono parses it").to_utc();
	let input = vec![Point::new(ts("2016-12-31T23:59:58Z"), dec("1")), Point::new(ts("2016-12-31T23:59:59Z"), dec("2")), Point::new(ts("2016-12-31T23:59:60Z"), dec("3")), Point::new(ts("2017-01-01T00:00:00Z"), dec("4")), Point::new(ts("2017-01-01T00:00:01Z"), dec("5"))];
	// 23:59:60 and 00:00:00 are one instant; the later input, 4, wins.
	for spline in [Spline::Linear, Spline::Quadratic, Spline::Cubic] {
		let out = Interpolator::new(spline, Resolution::Milliseconds).backend(Backend::Cpu).run(&input, ts("2016-12-31T23:59:58Z"), ts("2017-01-01T00:00:01Z")).expect("no 0/0 from coincident knots");
		assert_eq!(out.len(), 3001, "{spline}");
		assert_eq!(out.values()[2000], dec("4"), "{spline}");
		assert_eq!(out.kinds()[2000], PointKind::Raw);
		assert_eq!(out.timestamps()[2000], ts("2017-01-01T00:00:00Z"));
	}
	// A grid that starts in a leap second starts at the folded instant, and never passes end.
	let out = cpu(Spline::Linear).run(&input, ts("2016-12-31T23:59:60Z"), ts("2017-01-01T00:00:01Z")).expect("runs");
	assert_eq!(out.timestamps(), &[ts("2017-01-01T00:00:00Z"), ts("2017-01-01T00:00:01Z")]);
	assert_eq!(out.values(), &[dec("4"), dec("5")]);
	let out = cpu(Spline::Linear).run(&input, ts("2016-12-31T23:59:59Z"), ts("2016-12-31T23:59:60.5Z")).expect("runs");
	assert_eq!(out.timestamps(), &[ts("2016-12-31T23:59:59Z"), ts("2017-01-01T00:00:00Z")]);
}

/// Every backend names the first grid point that overflowed, not whichever chunk failed
/// first.
#[test]
fn non_finite_results_name_the_first_point() {
	let steep = points(&[(0, "-1e308"), (1, "1e308")]);
	for backend in [Backend::Cpu, Backend::Parallel] {
		let err = Interpolator::new(Spline::Linear, Resolution::Milliseconds).backend(backend).run(&steep, at(0), at(200)).expect_err("overflows");
		// −1e308 + 2e308·t passes f64::MAX ≈ 1.798e308 at t ≈ 1.3989 s.
		assert_eq!(err, Error::NonFiniteResult { timestamp: at(0) + TimeDelta::milliseconds(1399) }, "{backend}");
	}
}

#[cfg(not(feature = "gpu"))]
#[test]
fn without_the_gpu_feature_gpu_is_unavailable() {
	let input = points(&[(0, "0"), (1, "1")]);
	assert!(matches!(Interpolator::new(Spline::Linear, Resolution::Seconds).backend(Backend::Gpu).run(&input, at(0), at(1)), Err(Error::GpuUnavailable(_))));
	assert!(splimes::prewarm_gpu().is_err());
	assert!(splimes::gpu_info().is_none());
}

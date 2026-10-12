//! Adversarial input on every backend: values mixing subnormals, ±`f64::MAX`, signed
//! zeros and ordinary numbers; duplicate instants, and knots at chrono's limits a
//! nanosecond apart; grids before, across and after the data; every method and bounds
//! factor; with and without `exact`. Whatever the input, no backend panics, and every
//! answer is either a typed error or a series that keeps the documented promises: finite
//! values on the grid, raw points exact, a constant series constant, `Cpu` and `Parallel`
//! bit-identical, and the same error on every `f64` backend.

mod common;

use chrono::{DateTime, TimeDelta, Utc};
use common::Rng;
use splimes::{Backend, Error, Interpolation, Interpolator, PointKind, Precision, Resolution, Spline};

fn value(r: &mut Rng) -> f64 {
	match r.below(10) {
		0 => f64::from_bits(r.below(8)),
		1 => -f64::from_bits(r.below(1 << 20)),
		2 => f64::MAX * (r.unit() * 2.0 - 1.0),
		3 => {
			if r.below(2) == 0 {
				f64::MAX
			} else {
				-f64::MAX
			}
		}
		4 => -0.0,
		5 => 1e300 * (r.unit() - 0.5),
		6 => f64::MIN_POSITIVE * f64::from(u32::try_from(r.below(4)).expect("small")),
		_ => (r.unit() - 0.5) * 100.0,
	}
}

fn small(r: &mut Rng, n: u64) -> i32 {
	i32::try_from(r.below(n)).expect("small")
}

/// The error's variant, which every `f64` backend must agree on (the point it names may
/// differ only in `f32`, where overflow is decided in single precision).
fn variant(result: &splimes::Result<Interpolation<f64>>) -> String {
	match result {
		Ok(_) => "Ok".to_owned(),
		Err(e) => format!("{e:?}").split(['{', '(', ' ']).next().unwrap_or_default().to_owned(),
	}
}

#[test]
fn adversarial_input_never_panics_and_keeps_the_promises() {
	let gpu = common::gpu_or_skip("adversarial_input_never_panics_and_keeps_the_promises (GPU part)");
	let mut runs = vec![(Backend::Cpu, Precision::F64), (Backend::Parallel, Precision::F64), (Backend::Auto, Precision::F64)];
	if let Some(info) = &gpu {
		if info.supports_f64 {
			runs.push((Backend::Gpu, Precision::F64));
		}
		runs.push((Backend::Gpu, Precision::F32));
	}
	let (min, max) = (DateTime::<Utc>::MIN_UTC, DateTime::<Utc>::MAX_UTC);
	let mut r = Rng::new(0xAD7E_25A1);
	let (mut ok, mut refused) = (0, 0);
	for case in 0..2_000 {
		let n = 1 + usize::try_from(r.below(12)).expect("small");
		let anchor = [min, max - TimeDelta::days(2), DateTime::<Utc>::UNIX_EPOCH, common::epoch()][usize::try_from(r.below(4)).expect("small")];
		let unit = [TimeDelta::nanoseconds(1), TimeDelta::milliseconds(1), TimeDelta::seconds(1), TimeDelta::hours(1)][usize::try_from(r.below(4)).expect("small")];
		let timestamps: Vec<DateTime<Utc>> = (0..n).map(|_| anchor.checked_add_signed(unit * small(&mut r, 40)).unwrap_or(anchor)).collect();
		let values: Vec<f64> = (0..n).map(|_| value(&mut r)).collect();
		let resolution = [Resolution::Nanoseconds, Resolution::Milliseconds, Resolution::Seconds, Resolution::Hours][usize::try_from(r.below(4)).expect("small")];
		let start = anchor.checked_add_signed(unit * (small(&mut r, 50) - 5)).unwrap_or(anchor);
		let end = start.checked_add_signed(unit * (small(&mut r, 60) - 3)).unwrap_or(start);
		let degree = 1 + usize::try_from(r.below(8)).expect("small");
		let spline = match r.below(5) {
			0 => Spline::Linear,
			1 => Spline::Quadratic,
			2 => Spline::Cubic,
			3 => Spline::Polynomial(degree, None),
			_ => Spline::Polynomial(degree, Some([0.0, 0.5, 1e300, f64::MAX][usize::try_from(r.below(4)).expect("small")])),
		};
		let exact = r.below(4) == 0;
		let context = || format!("case {case}: {spline}, exact {exact}, {resolution:?} {start}..{end}, timestamps {timestamps:?}, values {values:?}");

		let results: Vec<_> = runs.iter().map(|&(backend, precision)| Interpolator::new(spline, resolution).backend(backend).gpu_precision(precision).exact(exact).max_points(100_000).run_f64(&timestamps, &values, start, end)).collect();
		for (&(backend, precision), result) in runs.iter().zip(&results) {
			let Ok(out) = result else { continue };
			// A constant series is that constant everywhere, inside the data and out.
			let constant = values.iter().all(|&v| v == values[0]).then_some(values[0]);
			for ((&v, &kind), &at) in out.values().iter().zip(out.kinds()).zip(out.timestamps()) {
				assert!(v.is_finite() && at >= start && at <= end, "{backend} {precision}, {}", context());
				assert!(constant.is_none_or(|c| v == c), "{backend} {precision}: {v:e} at {at} in a constant series, {}", context());
				if kind == PointKind::Raw {
					let input = timestamps.iter().rposition(|&t| t == at).expect("a raw point is an input instant");
					assert_eq!(v.to_bits(), values[input].to_bits(), "{backend} {precision}: raw point at {at}, {}", context());
				}
			}
		}
		let (cpu, parallel) = (&results[0], &results[1]);
		match (cpu, parallel) {
			(Ok(a), Ok(b)) => assert!(a.values().iter().zip(b.values()).all(|(x, y)| x.to_bits() == y.to_bits()) && a.kinds() == b.kinds(), "Cpu and Parallel differ, {}", context()),
			_ => assert_eq!(cpu.as_ref().err(), parallel.as_ref().err(), "{}", context()),
		}
		for (&(backend, precision), result) in runs.iter().zip(&results) {
			if precision == Precision::F64 {
				assert_eq!(variant(result), variant(cpu), "{backend}: {result:?} vs Cpu's {cpu:?}, {}", context());
			} else if !matches!(result, Err(Error::NonFiniteResult { .. })) && !matches!(cpu, Err(Error::NonFiniteResult { .. })) {
				assert_eq!(variant(result), variant(cpu), "{backend} {precision}: {result:?} vs Cpu's {cpu:?}, {}", context());
			}
		}
		if cpu.is_ok() {
			ok += 1;
		} else {
			refused += 1;
		}
	}
	println!("{ok} series computed, {refused} refused, on {} backend(s)", runs.len());
	assert!(ok > 500 && refused > 500, "both outcomes are exercised: {ok} / {refused}");
}

/// Decimals in unusual forms — zero with huge exponents, values that round to zero or to
/// a subnormal, the edge of `f64`'s range, more digits than `f64` holds, trailing zeros —
/// go through `run` exactly as their nearest `f64`s go through `run_f64`, on every
/// backend; raw points return the caller's own decimal, digits and scale; and decimals
/// beyond `f64`'s range are refused, naming their instant.
#[test]
fn unusual_decimals_are_their_nearest_f64() {
	use std::str::FromStr;

	use bigdecimal::BigDecimal;
	use splimes::Point;

	const FORMS: [&str; 16] = ["0e400", "-0e-500", "0.000e400", "1e-400", "-2.4e-324", "4.9e-324", "2.2250738585072014e-308", "1.7976931348623157e308", "-1.7976931348623157e308", "123456789012345678901234567890.12345678901234567890", "1.00000000000000000000000000000001", "100.000", "-7E+2", "3.14159265358979323846264338327950288", "1e300", "-0.5"];
	const BEYOND: [&str; 3] = ["1.8e308", "-1e309", "1e400"];
	let gpu = common::gpu_or_skip("unusual_decimals_are_their_nearest_f64 (GPU part)");
	let mut runs = vec![(Backend::Cpu, Precision::F64), (Backend::Parallel, Precision::F64)];
	if let Some(info) = &gpu {
		if info.supports_f64 {
			runs.push((Backend::Gpu, Precision::F64));
		}
		runs.push((Backend::Gpu, Precision::F32));
	}
	let at = |s: i64| common::epoch() + TimeDelta::seconds(s);
	let mut r = Rng::new(0xDEC1_3A15);
	for case in 0..300 {
		let n = 1 + usize::try_from(r.below(6)).expect("small");
		let texts: Vec<&str> = (0..n).map(|_| FORMS[usize::try_from(r.below(16)).expect("small")]).collect();
		let decimals: Vec<BigDecimal> = texts.iter().map(|t| BigDecimal::from_str(t).expect("parses")).collect();
		let timestamps: Vec<_> = (0..n).map(|i| at(i64::try_from(i).expect("small") * 10)).collect();
		let nearest: Vec<f64> = texts.iter().map(|t| t.parse::<f64>().expect("parses")).collect();
		let points: Vec<Point> = timestamps.iter().zip(&decimals).map(|(&t, d)| Point::new(t, d.clone())).collect();
		let spline = [Spline::Linear, Spline::Cubic, Spline::Polynomial(3, Some(0.5))][usize::try_from(r.below(3)).expect("small")];
		for &(backend, precision) in &runs {
			let make = Interpolator::new(spline, Resolution::Seconds).backend(backend).gpu_precision(precision);
			let (decimal, float) = (make.run(&points, at(-5), at(10 * i64::try_from(n).expect("small") + 5)), make.run_f64(&timestamps, &nearest, at(-5), at(10 * i64::try_from(n).expect("small") + 5)));
			let context = format!("case {case}, {spline}, {backend} {precision}, {texts:?}");
			match (&decimal, &float) {
				(Ok(d), Ok(f)) => {
					assert_eq!(d.kinds(), f.kinds(), "{context}");
					for (i, ((dv, &fv), kind)) in d.values().iter().zip(f.values()).zip(d.kinds()).enumerate() {
						if *kind == PointKind::Raw {
							let input = timestamps.iter().position(|&t| t == d.timestamps()[i]).expect("raw");
							assert_eq!(dv.as_bigint_and_exponent(), decimals[input].as_bigint_and_exponent(), "{context}: raw point {i}");
						} else {
							assert_eq!(dv.to_string().parse::<f64>().expect("parses"), fv, "{context}: point {i}");
						}
					}
				}
				_ => assert_eq!(decimal.as_ref().err(), float.as_ref().err(), "{context}"),
			}
		}
	}
	for beyond in BEYOND {
		let points = [Point::new(at(0), BigDecimal::from(1)), Point::new(at(10), BigDecimal::from_str(beyond).expect("parses"))];
		for &(backend, precision) in &runs {
			let result = Interpolator::new(Spline::Linear, Resolution::Seconds).backend(backend).gpu_precision(precision).run(&points, at(0), at(10));
			assert_eq!(result.map(|r| r.len()), Err(Error::ValueOutOfRange { timestamp: at(10) }), "{beyond} on {backend} {precision}");
		}
	}
}

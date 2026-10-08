//! Shared test support: a deterministic generator, an exact reference implementation of
//! every method, and the GPU-availability gate.
#![allow(dead_code)]

use bigdecimal::BigDecimal;
use chrono::{DateTime, TimeDelta, Utc};
use splimes::{Point, Spline};

/// xorshift64*: deterministic, dependency-free, good enough for test data.
pub struct Rng(u64);

impl Rng {
	pub fn new(seed: u64) -> Self {
		Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
	}

	pub fn next_u64(&mut self) -> u64 {
		self.0 ^= self.0 >> 12;
		self.0 ^= self.0 << 25;
		self.0 ^= self.0 >> 27;
		self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
	}

	/// Uniform in `[0, 1)`.
	pub fn unit(&mut self) -> f64 {
		(self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
	}

	pub fn below(&mut self, n: u64) -> u64 {
		self.next_u64() % n
	}
}

pub fn epoch() -> DateTime<Utc> {
	DateTime::<Utc>::UNIX_EPOCH + TimeDelta::days(20_000)
}

/// `t` in nanoseconds since the epoch on the POSIX scale, the documented time model: a leap
/// second (`23:59:60.x`) is the same instant as `00:00:00.x` the next second.
pub fn posix(t: DateTime<Utc>) -> i128 {
	i128::from(t.timestamp()) * 1_000_000_000 + i128::from(t.timestamp_subsec_nanos())
}

pub fn nanos_between(a: DateTime<Utc>, b: DateTime<Utc>) -> i128 {
	posix(a) - posix(b)
}

/// The value the method defined on `Spline` takes at `t`, in exact decimal arithmetic on
/// the caller's own timestamps: no normalisation, no floating point beyond the documented
/// input conversion. Each value is first rounded to the nearest `f64` (exactly what the
/// library computes on), then used at its exact binary value; terms carry 60 digits.
///
/// Comparing against the unrounded decimals instead would charge the kernel for the
/// half-ulp input rounding amplified by the method's Lebesgue constant, which is the
/// same on every backend and is a property of computing in `f64`, not of the kernel.
pub fn exact(points: &[Point], spline: Spline, t: DateTime<Utc>) -> BigDecimal {
	exact_with_lebesgue(points, spline, t).0
}

/// [`exact`], and the window's Lebesgue function at `t`: `Λ(t) = Σ_j |L_j(t)|`, the sum of
/// the magnitudes of the Lagrange basis polynomials. It's how much the method amplifies a
/// perturbation of the input values, and so the yardstick for rounding error: 1 for
/// linear interpolation, about 1 for well-spread knots, growing with irregular spacing and
/// with distance outside the data. 1 where the method holds a value (Cubic outside).
pub fn exact_with_lebesgue(points: &[Point], spline: Spline, t: DateTime<Utc>) -> (BigDecimal, f64) {
	let knots: Vec<Point> = distinct(points).into_iter().map(|p| Point::new(p.timestamp, BigDecimal::try_from(to_f64(&p.value)).expect("finite"))).collect();
	let n = knots.len();
	let spline = spline.fallback_for(n);
	let x: Vec<BigDecimal> = knots.iter().map(|p| BigDecimal::from(nanos_between(p.timestamp, knots[0].timestamp))).collect();
	let target = BigDecimal::from(nanos_between(t, knots[0].timestamp));
	let below = target < x[0];
	let above = target > x[n - 1];
	if matches!(spline, Spline::Cubic) && (below || above) {
		return (if below { knots[0].value.clone() } else { knots[n - 1].value.clone() }, 1.0);
	}
	let m = spline.min_points().min(n);
	let p = x.iter().filter(|&xi| *xi <= target).count();
	let start = (p as isize - (m / 2) as isize).clamp(0, (n - m) as isize) as usize;
	let mut total = BigDecimal::from(0);
	let mut lebesgue = BigDecimal::from(0);
	for j in start..start + m {
		let mut basis = BigDecimal::from(1);
		for k in start..start + m {
			if k != j {
				// 60 digits is far beyond f64's 17 and keeps the products from growing unboundedly.
				basis = (basis * (&target - &x[k]) / (&x[j] - &x[k])).with_prec(60);
			}
		}
		total += (&knots[j].value * &basis).with_prec(60);
		lebesgue += basis.abs();
	}
	if let (Some(b), true) = (spline.bounds_factor(), below || above) {
		let (min, max) = value_range(&knots);
		let range = &max - &min;
		let b: BigDecimal = b.to_string().parse().expect("finite bounds factor");
		let lo = &min - &range * &b;
		let hi = &max + &range * &b;
		total = total.max(lo).min(hi);
	}
	(total, to_f64(&lebesgue).max(1.0))
}

/// Sorted by instant, last of each duplicate instant kept: the documented input contract.
pub fn distinct(points: &[Point]) -> Vec<Point> {
	let mut sorted = points.to_vec();
	sorted.sort_by_key(|p| posix(p.timestamp));
	let mut out: Vec<Point> = Vec::with_capacity(sorted.len());
	for p in sorted {
		match out.last_mut() {
			Some(last) if posix(last.timestamp) == posix(p.timestamp) => *last = p,
			_ => out.push(p),
		}
	}
	out
}

pub fn value_range(points: &[Point]) -> (BigDecimal, BigDecimal) {
	let min = points.iter().map(|p| &p.value).min().expect("non-empty").clone();
	let max = points.iter().map(|p| &p.value).max().expect("non-empty").clone();
	(min, max)
}

pub fn to_f64(d: &BigDecimal) -> f64 {
	d.to_string().parse().expect("decimal parses as f64")
}

/// Whether a GPU test should run. Without an adapter the test is skipped with a notice,
/// unless `SPLIMES_REQUIRE_GPU` is set, in which case a missing adapter fails it.
/// `SPLIMES_REQUIRE_GPU_F64` implies it and also fails an adapter without f64. CI sets it
/// on Linux (lavapipe has f64), so neither GPU nor GPU-f64 coverage can silently disappear.
pub fn gpu_or_skip(test: &str) -> Option<splimes::GpuInfo> {
	match splimes::prewarm_gpu() {
		Ok(info) => {
			assert!(info.supports_f64 || std::env::var_os("SPLIMES_REQUIRE_GPU_F64").is_none(), "{test}: SPLIMES_REQUIRE_GPU_F64 is set but {} has no f64 support", info.name);
			Some(info)
		}
		Err(e) if std::env::var_os("SPLIMES_REQUIRE_GPU").is_some() || std::env::var_os("SPLIMES_REQUIRE_GPU_F64").is_some() => panic!("{test}: SPLIMES_REQUIRE_GPU is set but the GPU is unavailable: {e}"),
		Err(e) => {
			eprintln!("{test}: skipped, {e}; set SPLIMES_REQUIRE_GPU=1 to make this a failure");
			None
		}
	}
}

/// The smallest executor: poll, park until woken, repeat. Enough to await an
/// `InterpolationFuture` without a runtime.
pub fn block_on<F: std::future::Future>(f: F) -> F::Output {
	use std::{
		sync::Arc, task::{Context, Poll, Wake}, thread::{self, Thread}
	};
	struct Unpark(Thread);
	impl Wake for Unpark {
		fn wake(self: Arc<Self>) {
			self.0.unpark();
		}
	}
	let waker = Arc::new(Unpark(thread::current())).into();
	let mut cx = Context::from_waker(&waker);
	let mut f = std::pin::pin!(f);
	loop {
		if let Poll::Ready(v) = f.as_mut().poll(&mut cx) {
			return v;
		}
		thread::park();
	}
}

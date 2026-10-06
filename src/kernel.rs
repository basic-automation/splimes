//! The interpolation kernel: the single definition of what every method computes. The
//! WGSL shaders in `gpu/shader.rs` are a line-by-line port of [`eval`].
//!
//! Times are exact integer nanoseconds from the first knot. Every time *difference* the
//! Lagrange formula needs is taken in integers, then converted and scaled once, so it
//! carries one rounding however long the series is. (Subtracting rounded absolute times
//! instead loses about one ulp of the series length in every difference: past a million
//! knots that alone exceeds the published bound.)

use rayon::prelude::*;

use crate::{
	MAX_POLYNOMIAL_DEGREE, Spline, prepare::Knots, time::{Grid, nanos_to_f64}, value::Value
};

/// The largest window: a degree-8 polynomial passes through 9 knots.
pub const MAX_WINDOW: usize = MAX_POLYNOMIAL_DEGREE + 1;

/// Beyond this many mean knot spacings from any knot in the window, the Lagrange products
/// are formed from scaled time differences and multiplied back at the end (see `eval`).
/// Inside the data the differences are a few spacings, so the arithmetic there is
/// unchanged; 256⁸ = 2⁶⁴ keeps even the `f32` kernel far from overflow below it.
pub const SCALE_FROM: f64 = 256.0;

/// A method reduced to its kernel parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Method {
	/// Window size: how many consecutive knots the local polynomial passes through.
	pub window: usize,
	/// Hold the first / last value outside the data (`Cubic`) instead of extending.
	pub hold: bool,
	/// Clamp extrapolated values to this normalised `[lo, hi]` (`Polynomial` with a
	/// bounds factor).
	pub bounds: Option<(f64, f64)>,
}

impl Method {
	/// The kernel for `spline` (already stepped down to fit) on these knots.
	pub fn new<V: Value>(spline: Spline, knots: &Knots<'_, V>) -> Self {
		let window = spline.min_points().min(knots.len());
		let bounds = spline.bounds_factor().map(|b| {
			let (lo, hi) = knots.y_range();
			let range = hi - lo;
			(lo - b * range, hi + b * range)
		});
		Self { window, hold: matches!(spline, Spline::Cubic), bounds }
	}
}

/// What the kernel reads: knot offsets in nanoseconds, normalised values, and the time
/// scale as `inv_h`, the reciprocal of the mean knot spacing in nanoseconds.
#[derive(Clone, Copy)]
pub struct Data<'k> {
	pub offsets: &'k [i128],
	pub y: &'k [f64],
	pub inv_h: f64,
}

/// `(a - b) / h`, with the subtraction exact: one rounding to `f64`, one multiplication.
/// The GPU kernels compute exactly this, so CPU and GPU `f64` agree to the last bit when
/// the difference is below 2⁵³ ns.
#[inline]
fn diff(a: i128, b: i128, inv_h: f64) -> f64 {
	nanos_to_f64(a - b) * inv_h
}

/// The Lagrange weights `y_j / Π_{k≠j} (u_j − u_k)` of the current window, prepared once
/// and reused for every grid point that shares the window. Their denominators come from
/// exact integer differences, so knots a nanosecond apart stay distinct.
#[derive(Clone, Copy)]
pub struct Window {
	start: usize,
	len: usize,
	weight: [f64; MAX_WINDOW],
}

impl Window {
	/// No window yet: the first `eval` prepares one.
	pub const fn new() -> Self {
		Self { start: usize::MAX, len: 0, weight: [0.0; MAX_WINDOW] }
	}

	fn prepare(&mut self, data: Data<'_>, start: usize, m: usize) {
		if self.start == start && self.len == m {
			return;
		}
		let (o, y) = (&data.offsets[start..start + m], &data.y[start..start + m]);
		for (j, (weight, &yj)) in self.weight.iter_mut().zip(y).enumerate() {
			let mut denominator = 1.0;
			for (k, &ok) in o.iter().enumerate() {
				if k != j {
					denominator *= diff(o[j], ok, data.inv_h);
				}
			}
			*weight = yj / denominator;
		}
		self.start = start;
		self.len = m;
	}
}

/// The value at time `t` (nanoseconds from the first knot), where `p` is the number of
/// knots at or before `t`. `window` caches the window's weights between calls; start
/// with [`Window::new`].
///
/// Every time difference `t − u_k` is taken exactly, from integers, for every point:
/// differencing two rounded times instead (say, both relative to the window's first knot)
/// loses about one ulp of the window's span, which with geometrically spaced knots is many
/// times the local spacing.
#[inline]
#[allow(clippy::many_single_char_names)] // The usual names in the numerics literature.
pub fn eval(data: Data<'_>, method: &Method, t: i128, p: usize, window: &mut Window) -> f64 {
	let Data { offsets, y, inv_h } = data;
	let n = offsets.len();
	let below = t < offsets[0];
	let outside = below || t > offsets[n - 1];
	if method.hold && outside {
		return if below { y[0] } else { y[n - 1] };
	}
	let m = method.window;
	// The window of m knots centred on t's segment, slid inward at the ends: one formula
	// for every method (see the table on `Spline`).
	let start = window_start(n, m, p);
	let o = &offsets[start..start + m];
	let value = match m {
		1 => y[start],
		2 => {
			let alpha = diff(t, o[0], inv_h) / diff(o[1], o[0], inv_h);
			y[start] + alpha * (y[start + 1] - y[start])
		}
		_ => {
			window.prepare(data, start, m);
			let mut dt = [0.0; MAX_WINDOW];
			let mut s: f64 = 1.0;
			for (d, &ok) in dt.iter_mut().zip(o) {
				*d = diff(t, ok, inv_h);
				s = s.max(d.abs());
			}
			// Far outside the data each term grows like dt^(m-1) and they alternate in
			// sign, so forming them directly overflows (inf - inf = NaN) long before the
			// value itself does. Summing with every dt divided by s, then multiplying the
			// sum by s^(m-1), only overflows when the true value does, and then to an
			// infinity of the right sign, which the bounds clamp handles correctly.
			let scaled = s > SCALE_FROM;
			if scaled {
				for d in &mut dt[..m] {
					*d /= s;
				}
			}
			let mut total = 0.0;
			for (j, &weight) in window.weight[..m].iter().enumerate() {
				let mut term = weight;
				for (k, &d) in dt[..m].iter().enumerate() {
					if k != j {
						term *= d;
					}
				}
				total += term;
			}
			if scaled {
				for _ in 1..m {
					total *= s;
				}
			}
			total
		}
	};
	match method.bounds {
		Some((lo, hi)) if outside => value.clamp(lo, hi),
		_ => value,
	}
}

/// Evaluates grid points `first..first + out.len()` into `out`, normalised.
///
/// Grid points ascend, so after one binary search for the first, the knot cursor only
/// walks forward: `O(n + m)` for the chunk instead of `O(m log n)`.
fn eval_chunk<V: Value>(knots: &Knots<'_, V>, grid: &Grid, method: &Method, first: usize, out: &mut [f64]) {
	let data = knots.data();
	let offsets = data.offsets;
	let mut window = Window::new();
	let mut p = None;
	for (slot, t) in out.iter_mut().zip(grid.offsets(knots.t0, first)) {
		let mut q = p.unwrap_or_else(|| offsets.partition_point(|&o| o <= t));
		while q < offsets.len() && offsets[q] <= t {
			q += 1;
		}
		p = Some(q);
		*slot = eval(data, method, t, q, &mut window);
	}
}

/// Grid points per rayon task: big enough to amortise the per-chunk binary search and
/// scheduling, small enough to balance across cores.
const PARALLEL_CHUNK: usize = 16 * 1024;

/// Every grid point, normalised, on the calling thread.
pub fn eval_serial<V: Value>(knots: &Knots<'_, V>, grid: &Grid, method: &Method, out: &mut [f64]) {
	eval_chunk(knots, grid, method, 0, out);
}

/// Every grid point, normalised, across rayon's pool.
pub fn eval_parallel<V: Value>(knots: &Knots<'_, V>, grid: &Grid, method: &Method, out: &mut [f64]) {
	out.par_chunks_mut(PARALLEL_CHUNK).enumerate().for_each(|(c, chunk)| eval_chunk(knots, grid, method, c * PARALLEL_CHUNK, chunk));
}

/// The first knot of the window for a point with `p` knots at or before it.
#[inline]
const fn window_start(n: usize, m: usize, p: usize) -> usize {
	let start = p.saturating_sub(m / 2);
	if start > n - m { n - m } else { start }
}

/// Knot gaps further than this factor from the mean spacing, either way, make a window
/// unsafe for the `f32` kernel: with every gap within `[h/1024, 1024h]`, a degree-8
/// window's products stay within 2^±80, far inside `f32`'s range.
const F32_GAP_RATIO: f64 = 1024.0;

/// For the `f32` kernel: which windows (indexed by first knot) it can't compute safely,
/// or `None` if it can compute them all.
pub fn f32_unsafe_windows<V: Value>(knots: &Knots<'_, V>, method: &Method) -> Option<Vec<bool>> {
	let m = method.window;
	let o = &knots.offsets;
	if m < 2 || o.len() < 2 {
		return None;
	}
	let (lo, hi) = (knots.h / F32_GAP_RATIO, knots.h * F32_GAP_RATIO);
	let bad: Vec<bool> = o.windows(2).map(|w| !(lo..=hi).contains(&nanos_to_f64(w[1] - w[0]))).collect();
	if !bad.contains(&true) {
		return None;
	}
	// A window starting at knot s spans gaps s..s+m-1; count bad gaps with a prefix sum.
	let mut prefix = Vec::with_capacity(bad.len() + 1);
	prefix.push(0_usize);
	for &b in &bad {
		prefix.push(prefix[prefix.len() - 1] + usize::from(b));
	}
	Some((0..=o.len() - m).map(|s| prefix[s + m - 1] > prefix[s]).collect())
}

/// Recomputes, in `f64` on the CPU, every grid point the GPU couldn't compute reliably:
/// any non-finite result, and with `unsafe_windows`, every point in a window the `f32`
/// kernel can't handle. Returns how many points it recomputed.
///
/// The `f32` kernel writes a non-finite value rather than clamping one, and windows that
/// mix dense and sparse knots can underflow it to a plausible finite 0; both get their
/// correct value here.
pub fn repair<V: Value>(knots: &Knots<'_, V>, grid: &Grid, method: &Method, out: &mut [f64], unsafe_windows: Option<&[bool]>) -> usize {
	let data = knots.data();
	let n = data.offsets.len();
	match unsafe_windows {
		None => out
			.par_iter_mut()
			.enumerate()
			.filter(|(_, v)| !v.is_finite())
			.map(|(k, v)| {
				let t = grid.offset_nanos(knots.t0, k);
				*v = eval(data, method, t, data.offsets.partition_point(|&o| o <= t), &mut Window::new());
			})
			.count(),
		Some(unsafe_windows) => out
			.par_chunks_mut(PARALLEL_CHUNK)
			.enumerate()
			.map(|(c, chunk)| {
				let mut window = Window::new();
				let mut p = None;
				let mut repaired = 0;
				for (v, t) in chunk.iter_mut().zip(grid.offsets(knots.t0, c * PARALLEL_CHUNK)) {
					let mut q = p.unwrap_or_else(|| data.offsets.partition_point(|&o| o <= t));
					while q < n && data.offsets[q] <= t {
						q += 1;
					}
					p = Some(q);
					if !v.is_finite() || unsafe_windows[window_start(n, method.window, q)] {
						*v = eval(data, method, t, q, &mut window);
						repaired += 1;
					}
				}
				repaired
			})
			.sum(),
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	const fn method(window: usize) -> Method {
		Method { window, hold: false, bounds: None }
	}

	/// Evaluates at `t` (nanoseconds), with times scaled by `h`.
	fn at(o: &[i128], y: &[f64], method: &Method, t: i128) -> f64 {
		eval(Data { offsets: o, y, inv_h: 0.1 }, method, t, o.partition_point(|&x| x <= t), &mut Window::new())
	}

	#[test]
	fn every_window_passes_through_the_knots() {
		let o = [0, 10, 25, 30, 42, 60];
		let y = [1.0, -2.0, 0.5, 4.0, 3.0, -1.0];
		// A one-knot window only occurs with one knot; see `one_knot_is_constant`.
		for window in 2..=o.len() {
			for (i, &t) in o.iter().enumerate() {
				let got = at(&o, &y, &method(window), t);
				assert!((got - y[i]).abs() < 1e-12, "window {window}, knot {i}: {got} vs {}", y[i]);
			}
		}
	}

	#[test]
	fn one_knot_is_constant() {
		for t in [-30, 0, 75] {
			assert_eq!(at(&[0], &[2.5], &method(1), t), 2.5);
		}
	}

	#[test]
	fn reproduces_polynomials_of_its_degree() {
		// A degree-d window reproduces any degree-d polynomial exactly, inside and out.
		let o: Vec<i128> = (0..8).map(|i| i * 9 + i * i / 2).collect();
		for degree in 1..=5_i32 {
			let f = |t: f64| (0..=degree).map(|k| f64::from(k + 1) * (t / 10.0).powi(k)).sum::<f64>() / 10.0;
			#[allow(clippy::cast_precision_loss)]
			let y: Vec<f64> = o.iter().map(|&t| f(t as f64)).collect();
			#[allow(clippy::cast_sign_loss)]
			let m = method(degree as usize + 1);
			for t in [-13, 2, 17, 33, 50, 79] {
				let got = at(&o, &y, &m, t);
				#[allow(clippy::cast_precision_loss)]
				let expected = f(t as f64);
				assert!((got - expected).abs() < 1e-9 * expected.abs().max(1.0), "degree {degree} at {t}: {got} vs {expected}");
			}
		}
	}

	#[test]
	fn linear_extends_the_edge_segments() {
		let o = [0, 10, 20];
		let y = [0.0, 1.0, 3.0];
		assert!((at(&o, &y, &method(2), -10) - -1.0).abs() < 1e-15);
		assert!((at(&o, &y, &method(2), 30) - 5.0).abs() < 1e-15);
		assert!((at(&o, &y, &method(2), 5) - 0.5).abs() < 1e-15);
	}

	#[test]
	fn hold_flattens_outside_only() {
		let o = [0, 10, 20, 30];
		let y = [0.0, 1.0, 8.0, 27.0];
		let cubic = Method { window: 4, hold: true, bounds: None };
		assert_eq!(at(&o, &y, &cubic, -50), 0.0);
		assert_eq!(at(&o, &y, &cubic, 90), 27.0);
		assert!((at(&o, &y, &cubic, 15) - 3.375).abs() < 1e-12);
	}

	#[test]
	fn bounds_clamp_extrapolation_only() {
		let o = [0, 10, 20];
		let y = [0.0, 1.0, 0.0];
		let bounded = Method { window: 3, hold: false, bounds: Some((-0.5, 1.5)) };
		// The parabola -u(u-2) reaches -8 at u = 4; clamped to -0.5.
		assert_eq!(at(&o, &y, &bounded, 40), -0.5);
		assert!((at(&o, &y, &bounded, 10) - 1.0).abs() < 1e-15);
	}

	#[test]
	fn quadratic_window_takes_one_knot_before_and_two_after() {
		// Between knots 1 and 2 the window must be knots 1, 2, 3; the parabola through
		// (1,0) (2,0) (3,1) is (u-1)(u-2)/2, which is -1/8 at 1.5. Knots 0..=2 would give 0.
		let o = [0, 10, 20, 30];
		let y = [0.0, 0.0, 0.0, 1.0];
		assert!((at(&o, &y, &method(3), 15) - -0.125).abs() < 1e-15);
	}

	#[test]
	fn far_extrapolation_does_not_cancel_infinities() {
		// Degree 8 through rising data, 10⁴¹ time units out (Lagrange is invariant under
		// rescaling time, so a small `h` stands in for a long way out). Formed directly, the
		// alternating terms pass 10³⁰⁰ and lose everything to cancellation or overflow; the
		// scaled form only grows as large as the value itself, about 10²⁹⁹, and keeps its sign.
		let o: Vec<i128> = (0..9).collect();
		let y: Vec<f64> = (0..9).map(|i| f64::from(i) / 4.0 - 1.0 + if i == 8 { 0.1 } else { 0.0 }).collect();
		let far: i128 = 10_i128.pow(38);
		let data = Data { offsets: &o, y: &y, inv_h: 1e3 };
		let free = Method { window: 9, hold: false, bounds: None };
		let v = eval(data, &free, far, 9, &mut Window::new());
		assert!(v.is_finite() && v > 1e290, "{v}");
		let bounded = Method { window: 9, hold: false, bounds: Some((-2.0, 2.0)) };
		assert_eq!(eval(data, &bounded, far, 9, &mut Window::new()), 2.0, "clamps to the upper bound");
		assert_eq!(eval(data, &bounded, -far, 0, &mut Window::new()), 2.0, "an even-degree polynomial rises on both sides");
		// A few hundred spacings out, where both forms are fine, they agree to rounding.
		let unit = Data { offsets: &o, y: &y, inv_h: 1.0 };
		let near = eval(unit, &free, 300, 9, &mut Window::new());
		let expected: f64 = (0..9_u32).map(|j| y[j as usize] * (0..9_u32).filter(|&k| k != j).map(|k| (300.0 - f64::from(k)) / (f64::from(j) - f64::from(k))).product::<f64>()).sum();
		assert!((near - expected).abs() <= 1e-9 * expected.abs(), "{near} vs {expected}");
	}

	#[test]
	fn differences_stay_exact_far_from_the_origin() {
		// Ten million one-second knots, then a point in the last gap: the local difference
		// is 0.25 s exactly, not 0.25 s give or take an ulp of 10⁷ s.
		let n: i128 = 10_000_000;
		let s: i128 = 1_000_000_000;
		let o = [(n - 2) * s, (n - 1) * s];
		let y = [0.0, 1.0];
		let got = eval(Data { offsets: &o, y: &y, inv_h: 1e-9 }, &method(2), (n - 2) * s + s / 4, 1, &mut Window::new());
		assert_eq!(got, 0.25);
		// Knots a nanosecond apart, a year in, stay distinct.
		let year: i128 = 365 * 86_400 * s;
		let o = [0, year, year + 1, 2 * year];
		let y = [0.0, 1.0, 1.0, 0.0];
		let got = eval(Data { offsets: &o, y: &y, inv_h: 3.0 / 2e16 }, &method(2), year / 2, 1, &mut Window::new());
		assert!((got - 0.5).abs() < 1e-15, "{got}");
	}
}

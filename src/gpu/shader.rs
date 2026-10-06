//! The WGSL kernels: a line-by-line port of `kernel::eval`.
//!
//! One template, two instantiations. Every time — knot or grid point — is a pair of
//! scalars `(hi, lo)` with `hi + lo` the time, written by the host from exact integers.
//! A difference is `(a.hi - b.hi) + (a.lo - b.lo)`: when `a` and `b` are near each other,
//! as every pair in a local window is, the `hi` difference is exact (Sterbenz), so the
//! result carries one rounding however far the times are from the origin. That is what
//! the CPU kernel gets by subtracting integer nanoseconds.
//!
//! - **f64**: times are nanoseconds since the first knot, split exactly; differences are
//!   multiplied by `1/h`, the reciprocal mean knot spacing, as on the CPU.
//! - **f32**: times are already normalised (divided by `h` on the host), split to about
//!   48 bits, and the multiplier is 1. Without the split, times past 2²⁴ ≈ 16.7 M spacings would
//!   lose whole units.

/// The shared body. The scalar `S` and `SUB` are supplied per precision.
const BODY: &str = r"
struct Params {
    n: u32,
    window: u32,
    hold: u32,
    bounded: u32,
    count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    inv_h: S,
    lo: S,
    hi: S,
    _pad3: S,
}

@group(0) @binding(0) var<storage, read> knot_t: array<vec2<S>>;
@group(0) @binding(1) var<storage, read> knot_y: array<S>;
@group(0) @binding(2) var<storage, read_write> out: array<S>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read> grid_t: array<vec2<S>>;

// (a - b) / h, with the subtraction done on hi/lo pairs: `kernel::diff`.
fn diff(a: vec2<S>, b: vec2<S>) -> S {
    return ((a.x - b.x) + (a.y - b.y)) * params.inv_h;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let t = grid_t[idx];
    let n = params.n;
    let zero = S(0.0);

    // p = number of knots at or before t.
    var lo = 0u;
    var hi = n;
    while (lo < hi) {
        let mid = (lo + hi) / 2u;
        if (diff(t, knot_t[mid]) >= zero) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    let p = lo;

    let below = diff(t, knot_t[0]) < zero;
    let outside = below || diff(t, knot_t[n - 1u]) > zero;
    if (params.hold != 0u && outside) {
        out[idx] = select(knot_y[n - 1u], knot_y[0], below);
        return;
    }

    let m = params.window;
    let start = u32(clamp(i32(p) - i32(m / 2u), 0, i32(n - m)));
    var v: S;
    if (m == 1u) {
        v = knot_y[start];
    } else if (m == 2u) {
        let alpha = diff(t, knot_t[start]) / diff(knot_t[start + 1u], knot_t[start]);
        v = knot_y[start] + alpha * (knot_y[start + 1u] - knot_y[start]);
    } else {
        // Exact differences for every point, Lagrange weights, scaled products far out:
        // see `kernel::eval` and `kernel::Window`.
        var dt: array<S, MAX_WINDOW>;
        var s = S(1.0);
        for (var k = 0u; k < m; k++) {
            dt[k] = diff(t, knot_t[start + k]);
            s = max(s, abs(dt[k]));
        }
        let scaled = s > S(SCALE_FROM);
        if (scaled) {
            for (var k = 0u; k < m; k++) {
                dt[k] = dt[k] / s;
            }
        }
        var total = zero;
        for (var j = 0u; j < m; j++) {
            var denominator = S(1.0);
            for (var k = 0u; k < m; k++) {
                if (k != j) {
                    denominator = denominator * diff(knot_t[start + j], knot_t[start + k]);
                }
            }
            var term = knot_y[start + j] / denominator;
            for (var k = 0u; k < m; k++) {
                if (k != j) {
                    term = term * dt[k];
                }
            }
            total = total + term;
        }
        if (scaled) {
            for (var k = 1u; k < m; k++) {
                total = total * s;
            }
        }
        v = total;
    }
    // A non-finite value is written as is, never clamped: clamp(NaN, lo, hi) may return
    // lo, a plausible wrong answer. The host recomputes such points in f64.
    if (params.bounded != 0u && outside && finite(v)) {
        v = clamp(v, params.lo, params.hi);
    }
    out[idx] = v;
}
";

/// The f64 kernel. Needs the `SHADER_F64` feature.
///
/// Its values are normalised to `[-1, 1]` and its products scaled, so it can't produce a
/// non-finite value; `finite` is constant `true`. (Testing an `f64`'s bits would need
/// `SHADER_INT64`, which not every f64-capable adapter has.)
pub fn f64_source() -> String {
	let header = "alias S = f64;\nfn finite(v: f64) -> bool { return true; }\n";
	format!("{header}{}", body())
}

/// The f32 kernel. `finite` tests the exponent bits, because WGSL lets the compiler assume
/// no NaN or infinity, so `v == v` may be folded to `true`.
pub fn f32_source() -> String {
	let header = "alias S = f32;\nfn finite(v: f32) -> bool { return (bitcast<u32>(v) & 0x7f800000u) != 0x7f800000u; }\n";
	format!("{header}{}", body())
}

fn body() -> String {
	BODY.replace("SCALE_FROM", &format!("{:.1}", crate::kernel::SCALE_FROM)).replace("MAX_WINDOW", &crate::kernel::MAX_WINDOW.to_string())
}

#[cfg(test)]
mod tests {
	use naga::valid::{Capabilities, ValidationFlags, Validator};

	/// Both kernels must parse and validate. The f64 kernel needs `Capabilities::FLOAT64`;
	/// the f32 kernel is the fallback for adapters without it and must validate without.
	///
	/// This runs on any machine, GPU or not, so a broken shader can't hide behind an
	/// f64-capable dev GPU and surface only on Apple, integrated or WARP adapters.
	#[test]
	fn shaders_parse_and_validate() {
		for (name, source, caps) in [("f64", super::f64_source(), Capabilities::FLOAT64), ("f32", super::f32_source(), Capabilities::empty())] {
			let module = naga::front::wgsl::parse_str(&source).unwrap_or_else(|e| panic!("{name}: parse error:\n{}", e.emit_to_string(&source)));
			Validator::new(ValidationFlags::all(), caps).validate(&module).unwrap_or_else(|e| panic!("{name}: validation error: {:?}", e.into_inner()));
		}
	}
}

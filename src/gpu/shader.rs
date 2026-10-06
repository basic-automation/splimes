//! The WGSL kernels: a line-by-line port of `kernel::eval`.
//!
//! One template, two instantiations (`S` = `f64` or `f32`). Every time — knot or grid
//! point — is nanoseconds since the first knot as a 96-bit two's-complement integer in
//! three `u32` words, written by the host; 96 bits cover chrono's whole range (about 2⁷³
//! ns) with room to subtract. A difference is taken in integer arithmetic with an
//! explicit borrow, converted from sign and magnitude (so no floating-point cancellation),
//! and scaled by `1/h`. That is exactly what the CPU kernel does with `i128`, and in `f64`
//! it gives the same, correctly rounded value. Being integer arithmetic, it also survives
//! compilers that reassociate floating point: Metal compiles shaders with fast math, and
//! an earlier hi/lo floating-point split lost its precision there.

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

@group(0) @binding(0) var<storage, read> knot_t: array<u32>;
@group(0) @binding(1) var<storage, read> knot_y: array<S>;
@group(0) @binding(2) var<storage, read_write> out: array<S>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read> grid_t: array<u32>;

fn knot_time(i: u32) -> vec3<u32> {
    return vec3<u32>(knot_t[3u * i], knot_t[3u * i + 1u], knot_t[3u * i + 2u]);
}

fn grid_time(i: u32) -> vec3<u32> {
    return vec3<u32>(grid_t[3u * i], grid_t[3u * i + 1u], grid_t[3u * i + 2u]);
}

// (a - b) / h for 96-bit nanosecond times stored as (low, middle, high) words:
// `kernel::diff`. Integer subtraction with borrow, then sign and magnitude, so there is no
// floating-point cancellation for a compiler to reassociate.
fn diff(a: vec3<u32>, b: vec3<u32>) -> S {
    let d0 = a.x - b.x;
    let borrow0 = select(0u, 1u, a.x < b.x);
    let d1 = a.y - b.y - borrow0;
    let borrow1 = select(0u, 1u, a.y < b.y || (a.y == b.y && borrow0 == 1u));
    let d2 = a.z - b.z - borrow1;
    let negative = (d2 & 0x80000000u) != 0u;
    var m0 = d0;
    var m1 = d1;
    var m2 = d2;
    if (negative) {
        m0 = ~d0 + 1u;
        let carry0 = select(0u, 1u, m0 == 0u);
        m1 = ~d1 + carry0;
        let carry1 = select(0u, 1u, carry0 == 1u && m1 == 0u);
        m2 = ~d2 + carry1;
    }
    let magnitude = S(m2) * S(18446744073709551616.0) + (S(m1) * S(4294967296.0) + S(m0));
    return select(magnitude, -magnitude, negative) * params.inv_h;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let t = grid_time(idx);
    let n = params.n;
    let zero = S(0.0);

    // p = number of knots at or before t.
    var lo = 0u;
    var hi = n;
    while (lo < hi) {
        let mid = (lo + hi) / 2u;
        if (diff(t, knot_time(mid)) >= zero) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    let p = lo;

    let below = diff(t, knot_time(0)) < zero;
    let outside = below || diff(t, knot_time(n - 1u)) > zero;
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
        let alpha = diff(t, knot_time(start)) / diff(knot_time(start + 1u), knot_time(start));
        v = knot_y[start] + alpha * (knot_y[start + 1u] - knot_y[start]);
    } else {
        // Exact differences for every point, Lagrange weights, scaled products far out:
        // see `kernel::eval` and `kernel::Window`.
        var dt: array<S, MAX_WINDOW>;
        var s = S(1.0);
        for (var k = 0u; k < m; k++) {
            dt[k] = diff(t, knot_time(start + k));
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
                    denominator = denominator * diff(knot_time(start + j), knot_time(start + k));
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

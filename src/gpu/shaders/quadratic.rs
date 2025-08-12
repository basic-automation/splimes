pub const QUADRATIC_INTERPOLATION_SHADER: &str = r"
const TIME_EPSILON: f32 = 1.0e-6;
const DENOMINATOR_EPSILON: f32 = 1.0e-10;

@group(0) @binding(0) var<storage, read> input_times: array<f32>;
@group(0) @binding(1) var<storage, read> input_values: array<f32>;
@group(0) @binding(2) var<storage, read> target_times: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_values: array<f32>;
@group(0) @binding(4) var<uniform> offset: u32;

fn quadratic_interpolate_general(target_time: f32) -> f32 {
    let input_count = arrayLength(&input_times);
    
    if (input_count < 3u) {
        return input_values[0];
    }

    // Normalize times relative to first input time
    let base_time = input_times[0];
    let norm_target_time = target_time - base_time;

    if (norm_target_time < (input_times[0] - base_time)) {
        // Backward extrapolation using first three points
        let t0 = input_times[0] - base_time;
        let t1 = input_times[1] - base_time;
        let t2 = input_times[2] - base_time;
        let v0 = input_values[0];
        let v1 = input_values[1];
        let v2 = input_values[2];

        let denom0 = (t0 - t1) * (t0 - t2);
        let denom1 = (t1 - t0) * (t1 - t2);
        let denom2 = (t2 - t0) * (t2 - t1);

        if (abs(denom0) < DENOMINATOR_EPSILON || abs(denom1) < DENOMINATOR_EPSILON || abs(denom2) < DENOMINATOR_EPSILON) {
            return v1;
        }

        let l0 = ((norm_target_time - t1) * (norm_target_time - t2)) / denom0;
        let l1 = ((norm_target_time - t0) * (norm_target_time - t2)) / denom1;
        let l2 = ((norm_target_time - t0) * (norm_target_time - t1)) / denom2;

        return v0 * l0 + v1 * l1 + v2 * l2;
    } else if (norm_target_time > (input_times[input_count - 1u] - base_time)) {
        // Forward extrapolation using last three points
        let t0 = input_times[input_count - 3u] - base_time;
        let t1 = input_times[input_count - 2u] - base_time;
        let t2 = input_times[input_count - 1u] - base_time;
        let v0 = input_values[input_count - 3u];
        let v1 = input_values[input_count - 2u];
        let v2 = input_values[input_count - 1u];

        let denom0 = (t0 - t1) * (t0 - t2);
        let denom1 = (t1 - t0) * (t1 - t2);
        let denom2 = (t2 - t0) * (t2 - t1);

        if (abs(denom0) < DENOMINATOR_EPSILON || abs(denom1) < DENOMINATOR_EPSILON || abs(denom2) < DENOMINATOR_EPSILON) {
            return v1;
        }

        let l0 = ((norm_target_time - t1) * (norm_target_time - t2)) / denom0;
        let l1 = ((norm_target_time - t0) * (norm_target_time - t2)) / denom1;
        let l2 = ((norm_target_time - t0) * (norm_target_time - t1)) / denom2;

        return v0 * l0 + v1 * l1 + v2 * l2;
    } else {
        // Interpolation
        var left = 0u;
        var right = input_count - 1u;
        while (left < right - 1u) {
            let mid = left + (right - left) / 2u;
            let norm_mid_time = input_times[mid] - base_time;
            if (norm_mid_time <= norm_target_time) {
                left = mid;
            } else {
                right = mid;
            }
        }

        var center_idx = left;
        if (norm_target_time > (input_times[left] - base_time)) {
            center_idx = min(left + 1u, input_count - 2u);
        }
        center_idx = max(center_idx, 1u);
        center_idx = min(center_idx, input_count - 2u);

        var i0: u32;
        var i1: u32;
        var i2: u32;
        if (center_idx == 0u) {
            i0 = 0u;
            i1 = 1u;
            i2 = 2u;
        } else if (center_idx >= input_count - 1u) {
            i0 = input_count - 3u;
            i1 = input_count - 2u;
            i2 = input_count - 1u;
        } else {
            i0 = center_idx - 1u;
            i1 = center_idx;
            i2 = center_idx + 1u;
        }

        let t0 = input_times[i0] - base_time;
        let t1 = input_times[i1] - base_time;
        let t2 = input_times[i2] - base_time;
        let v0 = input_values[i0];
        let v1 = input_values[i1];
        let v2 = input_values[i2];

        let denom0 = (t0 - t1) * (t0 - t2);
        let denom1 = (t1 - t0) * (t1 - t2);
        let denom2 = (t2 - t0) * (t2 - t1);

        if (abs(denom0) < DENOMINATOR_EPSILON || abs(denom1) < DENOMINATOR_EPSILON || abs(denom2) < DENOMINATOR_EPSILON) {
            return v1;
        }

        let l0 = ((norm_target_time - t1) * (norm_target_time - t2)) / denom0;
        let l1 = ((norm_target_time - t0) * (norm_target_time - t2)) / denom1;
        let l2 = ((norm_target_time - t0) * (norm_target_time - t1)) / denom2;

        return v0 * l0 + v1 * l1 + v2 * l2;
    }
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x + offset;
    let target_count = arrayLength(&target_times);
    
    if (index >= target_count) {
        return;
    }
    
    let input_count = arrayLength(&input_times);
    if (input_count < 3u) {
        output_values[index] = input_values[0];
        return;
    }

    let target_time = target_times[index];
    output_values[index] = quadratic_interpolate_general(target_time);
}
";

pub const QUADRATIC_INTERPOLATION_SHADER_F64: &str = r"
const TIME_EPSILON: f64 = f64(1.0e-6);
const DENOMINATOR_EPSILON: f64 = f64(1.0e-10);

@group(0) @binding(0) var<storage, read> input_times: array<f64>;
@group(0) @binding(1) var<storage, read> input_values: array<f64>;
@group(0) @binding(2) var<storage, read> target_times: array<f64>;
@group(0) @binding(3) var<storage, read_write> output_values: array<f64>;
@group(0) @binding(4) var<uniform> offset: u32;

fn quadratic_interpolate_general(target_time: f64) -> f64 {
    let input_count = arrayLength(&input_times);
    
    if (input_count < 3u) {
        return input_values[0];
    }

    // Normalize times relative to first input time
    let base_time = input_times[0];
    let norm_target_time = target_time - base_time;

    if (norm_target_time < (input_times[0] - base_time)) {
        // Backward extrapolation using first three points
        let t0 = input_times[0] - base_time;
        let t1 = input_times[1] - base_time;
        let t2 = input_times[2] - base_time;
        let v0 = input_values[0];
        let v1 = input_values[1];
        let v2 = input_values[2];

        let denom0 = (t0 - t1) * (t0 - t2);
        let denom1 = (t1 - t0) * (t1 - t2);
        let denom2 = (t2 - t0) * (t2 - t1);

        if (abs(denom0) < DENOMINATOR_EPSILON || abs(denom1) < DENOMINATOR_EPSILON || abs(denom2) < DENOMINATOR_EPSILON) {
            return v1;
        }

        let l0 = ((norm_target_time - t1) * (norm_target_time - t2)) / denom0;
        let l1 = ((norm_target_time - t0) * (norm_target_time - t2)) / denom1;
        let l2 = ((norm_target_time - t0) * (norm_target_time - t1)) / denom2;

        return v0 * l0 + v1 * l1 + v2 * l2;
    } else if (norm_target_time > (input_times[input_count - 1u] - base_time)) {
        // Forward extrapolation using last three points
        let t0 = input_times[input_count - 3u] - base_time;
        let t1 = input_times[input_count - 2u] - base_time;
        let t2 = input_times[input_count - 1u] - base_time;
        let v0 = input_values[input_count - 3u];
        let v1 = input_values[input_count - 2u];
        let v2 = input_values[input_count - 1u];

        let denom0 = (t0 - t1) * (t0 - t2);
        let denom1 = (t1 - t0) * (t1 - t2);
        let denom2 = (t2 - t0) * (t2 - t1);

        if (abs(denom0) < DENOMINATOR_EPSILON || abs(denom1) < DENOMINATOR_EPSILON || abs(denom2) < DENOMINATOR_EPSILON) {
            return v1;
        }

        let l0 = ((norm_target_time - t1) * (norm_target_time - t2)) / denom0;
        let l1 = ((norm_target_time - t0) * (norm_target_time - t2)) / denom1;
        let l2 = ((norm_target_time - t0) * (norm_target_time - t1)) / denom2;

        return v0 * l0 + v1 * l1 + v2 * l2;
    } else {
        // Interpolation
        var left = 0u;
        var right = input_count - 1u;

        while (left < right - 1u) {
            let mid = left + (right - left) / 2u;
            if (norm_target_time < (input_times[mid] - base_time)) {
                right = mid;
            } else {
                left = mid;
            }
        }

        let center_idx = left;

        let i1 = max(1u, min(center_idx, input_count - 2u));
        let i0 = i1 - 1u;
        let i2 = i1 + 1u;

        let t0 = input_times[i0] - base_time;
        let t1 = input_times[i1] - base_time;
        let t2 = input_times[i2] - base_time;
        let v0 = input_values[i0];
        let v1 = input_values[i1];
        let v2 = input_values[i2];

        let denom0 = (t0 - t1) * (t0 - t2);
        let denom1 = (t1 - t0) * (t1 - t2);
        let denom2 = (t2 - t0) * (t2 - t1);

        if (abs(denom0) < DENOMINATOR_EPSILON || abs(denom1) < DENOMINATOR_EPSILON || abs(denom2) < DENOMINATOR_EPSILON) {
            return v1;
        }

        let l0 = ((norm_target_time - t1) * (norm_target_time - t2)) / denom0;
        let l1 = ((norm_target_time - t0) * (norm_target_time - t2)) / denom1;
        let l2 = ((norm_target_time - t0) * (norm_target_time - t1)) / denom2;

        return v0 * l0 + v1 * l1 + v2 * l2;
    }
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x + offset;
    let target_count = arrayLength(&target_times);
    
    if (index >= target_count) {
        return;
    }
    
    let input_count = arrayLength(&input_times);
    if (input_count < 3u) {
        output_values[index] = input_values[0];
        return;
    }

    let target_time = target_times[index];
    output_values[index] = quadratic_interpolate_general(target_time);
}
";

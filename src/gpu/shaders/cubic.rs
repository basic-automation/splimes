pub const CUBIC_INTERPOLATION_SHADER: &str = r"
@group(0) @binding(0) var<storage, read> input_times: array<f32>;
@group(0) @binding(1) var<storage, read> input_values: array<f32>;
@group(0) @binding(2) var<storage, read> target_times: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_values: array<f32>;
@group(0) @binding(4) var<uniform> offset: u32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x + offset;
    let target_count = arrayLength(&target_times);
    
    if (index >= target_count) {
        return;
    }
    
    let target_time = target_times[index];
    let input_count = arrayLength(&input_times);
    
    if (input_count < 4u) {
        output_values[index] = 0.0;
        return;
    }
    
    if (target_time <= input_times[0]) {
        output_values[index] = input_values[0];
        return;
    }
    
    if (target_time >= input_times[input_count - 1u]) {
        output_values[index] = input_values[input_count - 1u];
        return;
    }
    
    let segment_idx = find_segment_binary(target_time);
    
    let i1 = max(1u, min(segment_idx, input_count - 2u));
    let i0 = i1 - 1u;
    let i2 = i1 + 1u;
    let i3 = i1 + 2u;
    
    var final_i0 = i0;
    var final_i1 = i1;
    var final_i2 = i2;
    var final_i3 = i3;
    
    if (i3 >= input_count) {
        final_i0 = input_count - 4u;
        final_i1 = input_count - 3u;
        final_i2 = input_count - 2u;
        final_i3 = input_count - 1u;
    }
    
    let t0 = input_times[final_i0];
    let t1 = input_times[final_i1];
    let t2 = input_times[final_i2];
    let t3 = input_times[final_i3];
    
    let v0 = input_values[final_i0];
    let v1 = input_values[final_i1];
    let v2 = input_values[final_i2];
    let v3 = input_values[final_i3];
    
    output_values[index] = cubic_interpolate_lagrange(target_time, t0, t1, t2, t3, v0, v1, v2, v3);
}

fn find_segment_binary(target_time: f32) -> u32 {
    let input_count = arrayLength(&input_times);
    var left = 0u;
    var right = input_count - 1u;
    
    while (left < right - 1u) {
        let mid = left + (right - left) / 2u;
        if (target_time < input_times[mid]) {
            right = mid;
        } else {
            left = mid;
        }
    }
    
    return left;
}

fn cubic_interpolate_lagrange(t: f32, t0: f32, t1: f32, t2: f32, t3: f32, v0: f32, v1: f32, v2: f32, v3: f32) -> f32 {
    let denom0 = (t0 - t1) * (t0 - t2) * (t0 - t3);
    let denom1 = (t1 - t0) * (t1 - t2) * (t1 - t3);
    let denom2 = (t2 - t0) * (t2 - t1) * (t2 - t3);
    let denom3 = (t3 - t0) * (t3 - t1) * (t3 - t2);
    
    if (abs(denom0) < 1e-6 || abs(denom1) < 1e-6 || abs(denom2) < 1e-6 || abs(denom3) < 1e-6) {
        return v1; // Consistent fallback with CPU
    }
    
    let l0 = ((t - t1) * (t - t2) * (t - t3)) / denom0;
    let l1 = ((t - t0) * (t - t2) * (t - t3)) / denom1;
    let l2 = ((t - t0) * (t - t1) * (t - t3)) / denom2;
    let l3 = ((t - t0) * (t - t1) * (t - t2)) / denom3;
    
    return l0 * v0 + l1 * v1 + l2 * v2 + l3 * v3;
}
";

pub const CUBIC_INTERPOLATION_SHADER_F64: &str = r"
@group(0) @binding(0) var<storage, read> input_times: array<f64>;
@group(0) @binding(1) var<storage, read> input_values: array<f64>;
@group(0) @binding(2) var<storage, read> target_times: array<f64>;
@group(0) @binding(3) var<storage, read_write> output_values: array<f64>;
@group(0) @binding(4) var<uniform> offset: u32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x + offset;
    let target_count = arrayLength(&target_times);
    
    if (index >= target_count) {
        return;
    }
    
    let target_time = target_times[index];
    let input_count = arrayLength(&input_times);
    
    if (input_count < 4u) {
        output_values[index] = f64(0.0);
        return;
    }
    
    if (target_time <= input_times[0]) {
        output_values[index] = input_values[0];
        return;
    }
    
    if (target_time >= input_times[input_count - 1u]) {
        output_values[index] = input_values[input_count - 1u];
        return;
    }
    
    let segment_idx = find_segment_binary(target_time);
    
    let i1 = max(1u, min(segment_idx, input_count - 2u));
    let i0 = i1 - 1u;
    let i2 = i1 + 1u;
    let i3 = i1 + 2u;
    
    var final_i0 = i0;
    var final_i1 = i1;
    var final_i2 = i2;
    var final_i3 = i3;
    
    if (i3 >= input_count) {
        final_i0 = input_count - 4u;
        final_i1 = input_count - 3u;
        final_i2 = input_count - 2u;
        final_i3 = input_count - 1u;
    }
    
    let t0 = input_times[final_i0];
    let t1 = input_times[final_i1];
    let t2 = input_times[final_i2];
    let t3 = input_times[final_i3];
    
    let v0 = input_values[final_i0];
    let v1 = input_values[final_i1];
    let v2 = input_values[final_i2];
    let v3 = input_values[final_i3];
    
    output_values[index] = cubic_interpolate_lagrange(target_time, t0, t1, t2, t3, v0, v1, v2, v3);
}

fn find_segment_binary(target_time: f64) -> u32 {
    let input_count = arrayLength(&input_times);
    var left = 0u;
    var right = input_count - 1u;
    
    while (left < right - 1u) {
        let mid = left + (right - left) / 2u;
        if (target_time < input_times[mid]) {
            right = mid;
        } else {
            left = mid;
        }
    }
    
    return left;
}

fn cubic_interpolate_lagrange(t: f64, t0: f64, t1: f64, t2: f64, t3: f64, v0: f64, v1: f64, v2: f64, v3: f64) -> f64 {
    let denom0 = (t0 - t1) * (t0 - t2) * (t0 - t3);
    let denom1 = (t1 - t0) * (t1 - t2) * (t1 - t3);
    let denom2 = (t2 - t0) * (t2 - t1) * (t2 - t3);
    let denom3 = (t3 - t0) * (t3 - t1) * (t3 - t2);
    
    if (abs(denom0) < f64(1e-6) || abs(denom1) < f64(1e-6) || abs(denom2) < f64(1e-6) || abs(denom3) < f64(1e-6)) {
        return v1; // Consistent fallback with CPU
    }
    
    let l0 = ((t - t1) * (t - t2) * (t - t3)) / denom0;
    let l1 = ((t - t0) * (t - t2) * (t - t3)) / denom1;
    let l2 = ((t - t0) * (t - t1) * (t - t3)) / denom2;
    let l3 = ((t - t0) * (t - t1) * (t - t2)) / denom3;
    
    return l0 * v0 + l1 * v1 + l2 * v2 + l3 * v3;
}
";

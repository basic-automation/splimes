/// GPU compute shader for linear interpolation with f64 precision
pub const LINEAR_INTERPOLATION_SHADER_F64: &str = r"
@group(0) @binding(0) var<storage, read> input_times: array<f64>;
@group(0) @binding(1) var<storage, read> input_values: array<f64>;
@group(0) @binding(2) var<storage, read> target_times: array<f64>;
@group(0) @binding(3) var<storage, read_write> output_values: array<f64>;

fn is_near_zero(val: f64) -> bool {
    return abs(val) < f64(1e-12);
}

fn extrapolate(t: f64, t0: f64, v0: f64, t1: f64, v1: f64) -> f64 {
    let dt = t1 - t0;
    if (is_near_zero(dt)) {
        return v0;
    }
    let slope = (v1 - v0) / dt;
    return v0 + slope * (t - t0);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let target_count = arrayLength(&target_times);
    
    if (index >= target_count) {
        return;
    }
    
    let target_time = target_times[index];
    let input_count = arrayLength(&input_times);
    
    if (input_count == 0u) {
        output_values[index] = f64(0.0);
        return;
    }
    if (input_count == 1u) {
        output_values[index] = input_values[0];
        return;
    }
    
    if (target_time < input_times[0]) {
        output_values[index] = extrapolate(target_time, input_times[0], input_values[0], input_times[1], input_values[1]);
        return;
    }
    
    if (target_time > input_times[input_count - 1u]) {
        output_values[index] = extrapolate(target_time, input_times[input_count - 2u], input_values[input_count - 2u], input_times[input_count - 1u], input_values[input_count - 1u]);
        return;
    }
    
    var left = 0u;
    var right = input_count - 1u;
    
    while (right - left > 1u) {
        let mid = (left + right) / 2u;
        if (target_time < input_times[mid]) {
            right = mid;
        } else {
            left = mid;
        }
    }
    
    let left_idx = left;
    let right_idx = right;
    
    if (abs(target_time - input_times[left_idx]) < f64(1e-12)) {
        output_values[index] = input_values[left_idx];
        return;
    }
    if (abs(target_time - input_times[right_idx]) < f64(1e-12)) {
        output_values[index] = input_values[right_idx];
        return;
    }
    
    let t0 = input_times[left_idx];
    let t1 = input_times[right_idx];
    let v0 = input_values[left_idx];
    let v1 = input_values[right_idx];
    
    let dt = t1 - t0;
    if (is_near_zero(dt)) {
        output_values[index] = v0;
        return;
    }
    
    let alpha = (target_time - t0) / dt;
    output_values[index] = v0 + alpha * (v1 - v0);
}
";

/// GPU compute shader for linear interpolation with f32 precision (fallback)
pub const LINEAR_INTERPOLATION_SHADER_F32: &str = r"
@group(0) @binding(0) var<storage, read> input_times: array<f32>;
@group(0) @binding(1) var<storage, read> input_values: array<f32>;
@group(0) @binding(2) var<storage, read> target_times: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_values: array<f32>;

fn is_near_zero(val: f32) -> bool {
    return abs(val) < 1e-8;
}

fn extrapolate(t: f32, t0: f32, v0: f32, t1: f32, v1: f32) -> f32 {
    let dt = t1 - t0;
    if (is_near_zero(dt)) {
        return v0;
    }
    let slope = (v1 - v0) / dt;
    let result = v0 + slope * (t - t0);
    return clamp(result, -1e6, 1e6); // Prevent overflow
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let target_count = arrayLength(&target_times);
    
    if (index >= target_count) {
        return;
    }
    
    let target_time = target_times[index];
    let input_count = arrayLength(&input_times);
    
    if (input_count == 0u) {
        output_values[index] = 0.0;
        return;
    }
    if (input_count == 1u) {
        output_values[index] = input_values[0];
        return;
    }
    
    if (target_time < input_times[0]) {
        output_values[index] = extrapolate(target_time, input_times[0], input_values[0], input_times[1], input_values[1]);
        return;
    }
    
    if (target_time > input_times[input_count - 1u]) {
        output_values[index] = extrapolate(target_time, input_times[input_count - 2u], input_values[input_count - 2u], input_times[input_count - 1u], input_values[input_count - 1u]);
        return;
    }
    
    var left = 0u;
    var right = input_count - 1u;
    
    while (right - left > 1u) {
        let mid = (left + right) / 2u;
        if (target_time < input_times[mid]) {
            right = mid;
        } else {
            left = mid;
        }
    }
    
    let left_idx = left;
    let right_idx = right;
    
    if (abs(target_time - input_times[left_idx]) < 1e-8) {
        output_values[index] = input_values[left_idx];
        return;
    }
    if (abs(target_time - input_times[right_idx]) < 1e-8) {
        output_values[index] = input_values[right_idx];
        return;
    }
    
    let t0 = input_times[left_idx];
    let t1 = input_times[right_idx];
    let v0 = input_values[left_idx];
    let v1 = input_values[right_idx];
    
    let dt = t1 - t0;
    if (is_near_zero(dt)) {
        output_values[index] = (v0 + v1) / 2.0;
        return;
    }
    
    let alpha = (target_time - t0) / dt;
    let result = v0 + alpha * (v1 - v0);
    output_values[index] = clamp(result, -1e6, 1e6); // Prevent overflow
}
";

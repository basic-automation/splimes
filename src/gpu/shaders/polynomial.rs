pub const POLYNOMIAL_INTERPOLATION_SHADER: &str = r"
@group(0) @binding(0) var<storage, read> input_times: array<f32>;
@group(0) @binding(1) var<storage, read> input_values: array<f32>;
@group(0) @binding(2) var<storage, read> target_times: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_values: array<f32>;
@group(0) @binding(4) var<uniform> config: PolynomialConfig;

struct PolynomialConfig {
    offset: u32,
    max_degree: u32,
    bounds_factor_bits: u32,
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x + config.offset;
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
    
    for (var i = 0u; i < input_count; i++) {
        if (abs(input_times[i] - target_time) < 1e-7) {
            output_values[index] = input_values[i];
            return;
        }
    }
    
    let bounds_factor = bitcast<f32>(config.bounds_factor_bits);
    let is_bounded = !(bounds_factor != bounds_factor); // Explicitly check for non-NaN
    
    let first_time = input_times[0];
    let last_time = input_times[input_count - 1u];
    let is_extrapolating = target_time < first_time || target_time > last_time;
    
    let safe_degree = determine_safe_degree(config.max_degree, input_count);
    var result = polynomial_interpolate(target_time, safe_degree);
    
    if (is_extrapolating && is_bounded) {
        var min_value = input_values[0];
        var max_value = input_values[0];
        
        for (var i = 1u; i < input_count; i++) {
            min_value = min(min_value, input_values[i]);
            max_value = max(max_value, input_values[i]);
        }
        
        let range = max_value - min_value;
        let lower_bound = min_value - range * bounds_factor;
        let upper_bound = max_value + range * bounds_factor;
        
        result = clamp(result, lower_bound, upper_bound);
    }
    
    output_values[index] = result;
}

fn determine_safe_degree(requested_degree: u32, available_points: u32) -> u32 {
    if (available_points <= 1u) {
        return 0u;
    }
    
    let max_degree_by_points = available_points - 1u;
    let stability_cap = 8u;
    return min(min(requested_degree, max_degree_by_points), stability_cap);
}

fn polynomial_interpolate(target_time: f32, degree: u32) -> f32 {
    let input_count = arrayLength(&input_times);
    let num_points = min(degree + 1u, input_count);
    
    if (num_points == 0u) {
        return 0.0;
    }
    
    if (num_points == 1u) {
        return input_values[0];
    }
    
    let window_start = select_window(target_time, num_points);
    return lagrange_interpolate(target_time, window_start, num_points);
}

fn select_window(target_time: f32, num_points: u32) -> u32 {
    let input_count = arrayLength(&input_times);
    
    if (num_points >= input_count) {
        return 0u;
    }
    
    var insert_pos = 0u;
    for (var i = 0u; i < input_count; i++) {
        if (input_times[i] <= target_time) {
            insert_pos = i + 1u;
        } else {
            break;
        }
    }
    
    let half_window = num_points / 2u;
    var start_idx = insert_pos - min(insert_pos, half_window);
    
    if (start_idx + num_points > input_count) {
        start_idx = input_count - num_points;
    }
    
    return start_idx;
}

fn lagrange_interpolate(target_time: f32, start_idx: u32, num_points: u32) -> f32 {
    var result = 0.0;
    var has_near_zero_denominator = false;
    
    for (var j = 0u; j < num_points; j++) {
        let j_idx = start_idx + j;
        let tj = input_times[j_idx];
        let yj = input_values[j_idx];
        
        var basis = 1.0;
        
        for (var k = 0u; k < num_points; k++) {
            if (k != j) {
                let k_idx = start_idx + k;
                let tk = input_times[k_idx];
                let denominator = tj - tk;
                
                if (abs(denominator) < 1e-12) {
                    has_near_zero_denominator = true;
                    break;
                }
                
                basis *= (target_time - tk) / denominator;
            }
        }
        
        if (!has_near_zero_denominator) {
            result += yj * basis;
        }
    }
    
    if (has_near_zero_denominator && num_points >= 2u && config.max_degree == 1u) {
        let t1 = input_times[start_idx];
        let t2 = input_times[start_idx + 1u];
        let v1 = input_values[start_idx];
        let v2 = input_values[start_idx + 1u];
        let t_norm = (target_time - t1) / max(t2 - t1, 1e-12);
        result = v1 + t_norm * (v2 - v1);
    }
    
    return result;
}
";

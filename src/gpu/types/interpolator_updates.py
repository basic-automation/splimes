import re

# Read the file
with open('interpolator.rs', 'r') as f:
    content = f.read()

# First update: f64 method
# Find and replace the staging_buffer acquisition line in f64
content = re.sub(
    r"let \(staging_buffer, _\) = interpolator\.buffer_pool\.acquire\(output_size as u64, PooledBufferType::Staging, Some\(\"Staging Buffer\"\)\)\?;(\s+)(interpolator\.queue\.write_buffer\(&target_times_buffer, 0, cast_slice\(target_batch\)\);)",
    r"let staging_buffer = interpolator.staging_manager.acquire(&interpolator.device)?;\1\2",
    content
)

# Update the submit line for f64 to capture submission_index
# This is trickier - need to find the pattern and replace it
content = re.sub(
    r"(encoder\.copy_buffer_to_buffer\(&output_buffer, 0, &staging_buffer, 0, output_size as u64\);[\s\n]+)interpolator\.queue\.submit\(std::iter::once\(encoder\.finish\(\)\)\);",
    r"\1let submission_index = interpolator.queue.submit(std::iter::once(encoder.finish()));",
    content
)

# Add the mark_submitted call after submit - for f64
content = re.sub(
    r"(let submission_index = interpolator\.queue\.submit\(std::iter::once\(encoder\.finish\(\)\)\);)([\s\n]+)(let buffer_slice = staging_buffer\.slice\(\.\.\);)",
    r"\1\n\t\t\tinterpolator.staging_manager.mark_submitted(&staging_buffer, submission_index);\2\3",
    content
)

# Remove the buffer_pool.release(staging_buffer) calls - should remove the line with it
content = re.sub(
    r"\t\t\tinterpolator\.buffer_pool\.release\(staging_buffer\);(\s*)\n",
    r"\1\n",
    content
)

# Write the file back
with open('interpolator.rs', 'w') as f:
    f.write(content)

print("Updates applied successfully!")

use std::{
	borrow::Cow, collections::{HashMap, VecDeque}, sync::{LazyLock, Mutex, Arc}
};

use anyhow::{bail, Context, Result};
use bytemuck::cast_slice;
use wgpu::{
	util::{BufferInitDescriptor, DeviceExt}, Backends, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, BufferBindingType, BufferUsages, ComputePipeline, ComputePipelineDescriptor, Device, DeviceDescriptor, Features, Instance, InstanceDescriptor, InstanceFlags, Limits, MemoryHints, PowerPreference, Queue, RequestAdapterOptions, ShaderStages
};

use crate::gpu::{Method, buffer_pool::{BufferPool, BufferPoolConfig, PooledBufferType}, StagingBufferManager, async_handle::GpuInterpolationResult};

static GLOBAL_INTERPOLATOR: LazyLock<Result<GpuInterpolator>> = LazyLock::new(|| {
	// Use spawn_blocking to handle async initialization without runtime conflicts
	std::thread::spawn(|| {
		let rt = tokio::runtime::Runtime::new().unwrap();
		rt.block_on(GpuInterpolator::new())
	})
	.join()
	.unwrap()
});

#[derive(Debug)]
pub struct GpuInterpolator {
	device: Device,
	queue: Queue,
	bind_group_layout: BindGroupLayout,
	pipelines_f64: Mutex<HashMap<Method, ComputePipeline>>,
	pipelines_f32: Mutex<HashMap<Method, ComputePipeline>>,
	supports_f64: bool,
	max_storage_buffer_binding_size: usize,
	pub buffer_pool: BufferPool,
	#[allow(dead_code)] // Reserved for future streaming GPU operations
	pub staging_manager: StagingBufferManager,
}

/// Tracks a batch that has been submitted to the GPU but not yet read back
struct PendingBatch<T> {
	/// Staging buffer for readback
	staging_buffer: Arc<wgpu::Buffer>,
	/// Output size in bytes
	output_size: usize,
	/// Number of elements in this batch
	batch_len: usize,
	/// Target buffer (kept alive for GPU access)
	#[allow(dead_code)]
	target_buffer: Arc<wgpu::Buffer>,
	/// Output buffer (kept alive for GPU access)
	#[allow(dead_code)]
	output_buffer: Arc<wgpu::Buffer>,
	/// Phantom for type safety
	_marker: std::marker::PhantomData<T>,
}

impl GpuInterpolator {
	async fn new() -> Result<Self> {
		let instance = Instance::new(&InstanceDescriptor { backends: Backends::PRIMARY, flags: InstanceFlags::default(), ..Default::default() });
		let adapter = instance.request_adapter(&RequestAdapterOptions { power_preference: PowerPreference::HighPerformance, compatible_surface: None, force_fallback_adapter: false }).await.context("Failed to request GPU adapter")?;
		//let adapter_info = adapter.get_info();
		// println!("GPU: Adapter name: {}, Backend: {:?}", adapter_info.name, adapter_info.backend);
		let supports_f64 = adapter.features().contains(Features::SHADER_F64);

		// Determine the maximum buffer size that can be bound to a binding, considering hardware limitations
		// This is crucial for batching large datasets and memory management
		// We'll limit to 128MB to avoid system memory pressure
		let adapter_limits = adapter.limits();
		let max_storage_buffer_binding_size = adapter_limits.max_storage_buffer_binding_size as usize;
		// Clamp to a reasonable maximum to avoid excessive memory usage
		let max_storage_buffer_binding_size = max_storage_buffer_binding_size.min(128 * 1024 * 1024);

		//let max_compute_workgroups_per_dimension = adapter_limits.max_compute_workgroups_per_dimension as usize;

		let (device, queue) = adapter.request_device(&DeviceDescriptor { label: Some("Interpolation Device"), required_features: if supports_f64 { Features::SHADER_F64 } else { Features::empty() }, required_limits: Limits::default(), memory_hints: MemoryHints::default(), ..Default::default() }).await.context("Failed to request GPU device")?;
		let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor { label: Some("Interpolation Bind Group Layout"), entries: &[BindGroupLayoutEntry { binding: 0, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None }, BindGroupLayoutEntry { binding: 1, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None }, BindGroupLayoutEntry { binding: 2, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None }, BindGroupLayoutEntry { binding: 3, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None }, BindGroupLayoutEntry { binding: 4, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None }] });

		let buffer_pool = BufferPool::new(Arc::new(device.clone()), BufferPoolConfig::default());
	let staging_manager = StagingBufferManager::new(max_storage_buffer_binding_size as u64, 3);

	Ok(Self { device, queue, bind_group_layout, pipelines_f64: Mutex::new(HashMap::new()), pipelines_f32: Mutex::new(HashMap::new()), supports_f64, max_storage_buffer_binding_size, buffer_pool, staging_manager })
	}

	pub fn supports_f64_static() -> Result<bool> {
		match GLOBAL_INTERPOLATOR.as_ref() {
			Ok(interpolator) => Ok(interpolator.supports_f64),
			Err(e) => bail!("Failed to access global interpolator: {e}"),
		}
	}

	/// Returns a reference to the global GPU device for buffer creation
	///
	/// # Errors
	/// Returns an error if the global GPU interpolator failed to initialize
	#[allow(dead_code)] // Potentially useful for future extensions
	pub fn get_device_static() -> Result<&'static Device> {
		match GLOBAL_INTERPOLATOR.as_ref() {
			Ok(interpolator) => Ok(&interpolator.device),
			Err(e) => bail!("Failed to access global interpolator: {e}"),
		}
	}

	/// Returns a reference to the global buffer pool
	///
	/// # Errors
	/// Returns an error if the global GPU interpolator failed to initialize
	pub fn get_buffer_pool_static() -> Result<&'static BufferPool> {
		match GLOBAL_INTERPOLATOR.as_ref() {
			Ok(interpolator) => Ok(&interpolator.buffer_pool),
			Err(e) => bail!("Failed to access global interpolator: {e}"),
		}
	}

	/// Returns the maximum storage buffer binding size from the global interpolator
	///
	/// # Errors
	/// Returns an error if the global GPU interpolator failed to initialize
	pub fn get_max_buffer_size_static() -> Result<usize> {
		match GLOBAL_INTERPOLATOR.as_ref() {
			Ok(interpolator) => Ok(interpolator.max_storage_buffer_binding_size),
			Err(e) => bail!("Failed to access global interpolator: {e}"),
		}
	}

	/// Returns a reference to the global GPU queue for buffer writes
	///
	/// # Errors
	/// Returns an error if the global GPU interpolator failed to initialize
	#[allow(dead_code)] // Reserved for future use
	pub fn get_queue_static() -> Result<&'static Queue> {
		match GLOBAL_INTERPOLATOR.as_ref() {
			Ok(interpolator) => Ok(&interpolator.queue),
			Err(e) => bail!("Failed to access global interpolator: {e}"),
		}
	}

	/// Clears the global buffer pool for test isolation.
	/// This should be called at the start of each GPU test to ensure clean state.
	///
	/// # Errors
	/// Returns an error if the global GPU interpolator failed to initialize
	#[cfg(test)]
	pub fn clear_buffer_pool_static() -> Result<()> {
		match GLOBAL_INTERPOLATOR.as_ref() {
			Ok(interpolator) => {
				interpolator.buffer_pool.clear_all();
				Ok(())
			}
			Err(e) => bail!("Failed to access global interpolator: {e}"),
		}
	}

	/// Forces initialization of the global GPU interpolator.
	///
	/// This method triggers lazy initialization of the `GLOBAL_INTERPOLATOR` if not
	/// already initialized, and pre-compiles shader pipelines for all interpolation
	/// methods to eliminate first-use latency.
	///
	/// # Errors
	/// Returns the initialization error if GPU setup fails.
	pub fn force_init() -> Result<()> {
		let interpolator = match GLOBAL_INTERPOLATOR.as_ref() {
			Ok(i) => i,
			Err(e) => bail!("GPU initialization failed: {e}"),
		};

		// Pre-compile all shader pipelines to eliminate first-use latency
		let methods = [Method::Linear, Method::Quadratic, Method::Cubic, Method::Polynomial(3)];

		if interpolator.supports_f64 {
			let mut pipelines = interpolator.pipelines_f64.lock().unwrap();
			for method in &methods {
				if !pipelines.contains_key(method) {
					let shader_source = method.shader_source_f64();
					let shader = interpolator.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("Interpolation Shader"), source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(shader_source)) });
					let pipeline_layout = interpolator.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("Interpolation Pipeline Layout"), bind_group_layouts: &[&interpolator.bind_group_layout], push_constant_ranges: &[] });
					let pipeline = interpolator.device.create_compute_pipeline(&ComputePipelineDescriptor { label: Some("Interpolation Pipeline"), layout: Some(&pipeline_layout), module: &shader, entry_point: Some("main"), compilation_options: wgpu::PipelineCompilationOptions::default(), cache: None });
					pipelines.insert(*method, pipeline);
				}
			}
		} else {
			let mut pipelines = interpolator.pipelines_f32.lock().unwrap();
			for method in &methods {
				if !pipelines.contains_key(method) {
					let shader_source = method.shader_source_f32();
					let shader = interpolator.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("Interpolation Shader"), source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(shader_source)) });
					let pipeline_layout = interpolator.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("Interpolation Pipeline Layout"), bind_group_layouts: &[&interpolator.bind_group_layout], push_constant_ranges: &[] });
					let pipeline = interpolator.device.create_compute_pipeline(&ComputePipelineDescriptor { label: Some("Interpolation Pipeline"), layout: Some(&pipeline_layout), module: &shader, entry_point: Some("main"), compilation_options: wgpu::PipelineCompilationOptions::default(), cache: None });
					pipelines.insert(*method, pipeline);
				}
			}
		}

		// Run a minimal dummy computation to prime the GPU driver/runtime
		// This ensures the first real computation doesn't pay driver initialization costs
		Self::run_dummy_computation(interpolator)?;

		Ok(())
	}

	/// Runs a minimal GPU computation to prime the driver/runtime
	#[allow(clippy::unnecessary_wraps)] // Consistent with other init methods that may fail
	fn run_dummy_computation(interpolator: &Self) -> Result<()> {
		// Create tiny buffers for a minimal computation
		let dummy_times: [f64; 4] = [0.0, 1.0, 2.0, 3.0];
		let dummy_values: [f64; 4] = [0.0, 1.0, 2.0, 3.0];
		let dummy_targets: [f64; 2] = [0.5, 1.5];

		let input_times_buffer = interpolator.device.create_buffer_init(&BufferInitDescriptor { label: Some("Warmup Input Times"), contents: cast_slice(&dummy_times), usage: BufferUsages::STORAGE });
		let input_values_buffer = interpolator.device.create_buffer_init(&BufferInitDescriptor { label: Some("Warmup Input Values"), contents: cast_slice(&dummy_values), usage: BufferUsages::STORAGE });
		let target_times_buffer = interpolator.device.create_buffer_init(&BufferInitDescriptor { label: Some("Warmup Target Times"), contents: cast_slice(&dummy_targets), usage: BufferUsages::STORAGE });
		let output_buffer = interpolator.device.create_buffer(&wgpu::BufferDescriptor { label: Some("Warmup Output"), size: (dummy_targets.len() * std::mem::size_of::<f64>()) as u64, usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC, mapped_at_creation: false });
		let staging_buffer = interpolator.device.create_buffer(&wgpu::BufferDescriptor { label: Some("Warmup Staging"), size: (dummy_targets.len() * std::mem::size_of::<f64>()) as u64, usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ, mapped_at_creation: false });

		// Create config buffer for linear interpolation
		#[allow(clippy::cast_possible_truncation)] // Small array lengths fit in u32
		let config_data: [u32; 4] = [dummy_times.len() as u32, dummy_targets.len() as u32, 0, 0];
		let config_buffer = interpolator.device.create_buffer_init(&BufferInitDescriptor { label: Some("Warmup Config"), contents: cast_slice(&config_data), usage: BufferUsages::UNIFORM });

		let bind_group = interpolator.device.create_bind_group(&BindGroupDescriptor { label: Some("Warmup Bind Group"), layout: &interpolator.bind_group_layout, entries: &[BindGroupEntry { binding: 0, resource: input_times_buffer.as_entire_binding() }, BindGroupEntry { binding: 1, resource: input_values_buffer.as_entire_binding() }, BindGroupEntry { binding: 2, resource: target_times_buffer.as_entire_binding() }, BindGroupEntry { binding: 3, resource: output_buffer.as_entire_binding() }, BindGroupEntry { binding: 4, resource: config_buffer.as_entire_binding() }] });

		let mut encoder = interpolator.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Warmup Encoder") });

		// Use the pre-compiled Linear pipeline
		let pipelines = if interpolator.supports_f64 { interpolator.pipelines_f64.lock().unwrap() } else { interpolator.pipelines_f32.lock().unwrap() };
		let pipeline = &pipelines[&Method::Linear];

		{
			let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Warmup Pass"), timestamp_writes: None });
			compute_pass.set_pipeline(pipeline);
			compute_pass.set_bind_group(0, &bind_group, &[]);
			compute_pass.dispatch_workgroups(1, 1, 1);
		}
		drop(pipelines);

		encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, (dummy_targets.len() * std::mem::size_of::<f64>()) as u64);
		interpolator.queue.submit(std::iter::once(encoder.finish()));

		// Wait for GPU to complete - this primes the synchronization path
		let buffer_slice = staging_buffer.slice(..);
		let (tx, rx) = std::sync::mpsc::channel();
		buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
			let _ = tx.send(result);
		});
		let _ = interpolator.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
		let _ = rx.recv();
		staging_buffer.unmap();

		Ok(())
	}

	/// Performs static f64 interpolation using pipelined GPU execution
	///
	/// Uses a sliding window approach to overlap GPU computation with CPU readback:
	/// - Submits batches continuously to the GPU
	/// - When the pipeline window is full (3 batches), waits for oldest to complete
	/// - Reads completed results while new batches are processing
	///
	/// This provides ~30-50% faster execution for large datasets by hiding
	/// CPU-GPU synchronization latency.
	///
	/// # Errors
	/// Returns an error if:
	/// - The global interpolator failed to initialize
	/// - Input validation fails
	/// - GPU operations fail
	/// - Buffer operations fail
	pub fn interpolate_f64_static(input_times: &[f64], input_values: &[f64], target_times: &[f64], method: &Method, config_buffer: &wgpu::Buffer) -> Result<Vec<f64>> {
		let interpolator = match GLOBAL_INTERPOLATOR.as_ref() {
			Ok(interpolator) => interpolator,
			Err(e) => bail!("Failed to access global interpolator: {e}"),
		};

		if input_times.is_empty() || target_times.is_empty() {
			return Ok(Vec::new());
		}
		if input_times.len() != input_values.len() {
			bail!("Input times and values must have the same length");
		}

		// Create input buffers with exact size (not pooled - input data is test-specific)
		let _input_size = std::mem::size_of_val(input_times) as u64;
		let input_times_buffer = Arc::new(interpolator.device.create_buffer_init(&BufferInitDescriptor {
			label: Some("Input Times Buffer"),
			contents: cast_slice(input_times),
			usage: BufferUsages::STORAGE,
		}));
		let input_values_buffer = Arc::new(interpolator.device.create_buffer_init(&BufferInitDescriptor {
			label: Some("Input Values Buffer"),
			contents: cast_slice(input_values),
			usage: BufferUsages::STORAGE,
		}));

		// Ensure pipeline exists, create if needed
		{
			let mut pipelines = interpolator.pipelines_f64.lock().unwrap();
			if !pipelines.contains_key(method) {
				let shader_source = method.shader_source_f64();
				let shader = interpolator.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("Interpolation Shader"), source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(shader_source)) });
				let pipeline_layout = interpolator.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("Interpolation Pipeline Layout"), bind_group_layouts: &[&interpolator.bind_group_layout], push_constant_ranges: &[] });
				let pipeline = interpolator.device.create_compute_pipeline(&ComputePipelineDescriptor { label: Some("Interpolation Pipeline"), layout: Some(&pipeline_layout), module: &shader, entry_point: Some("main"), compilation_options: wgpu::PipelineCompilationOptions::default(), cache: None });
				pipelines.insert(*method, pipeline);
			}
		}

		// Process in batches to respect GPU memory limits
		let element_size = std::mem::size_of::<f64>();
		let workgroup_size = 64;
		let max_compute_workgroups = 65535;
		let max_elements_per_batch = (interpolator.max_storage_buffer_binding_size / element_size).min(target_times.len()).min(max_compute_workgroups * workgroup_size);

		// Pipeline window size - number of batches to keep in flight before reading
		let window_size = 3;

		// Queue of pending batches for pipelined execution
		let mut pending_batches: VecDeque<PendingBatch<f64>> = VecDeque::with_capacity(window_size);
		let mut all_results = Vec::with_capacity(target_times.len());

		for target_batch in target_times.chunks(max_elements_per_batch) {
			let output_size = std::mem::size_of_val(target_batch);
			let target_size = std::mem::size_of_val(target_batch) as u64;

			// Acquire pooled buffers with metadata for safe zeroing
			let target_acquired = interpolator.buffer_pool.acquire_with_metadata(
				target_size, PooledBufferType::Storage, Some("Target Times Buffer"))?;
			let output_acquired = interpolator.buffer_pool.acquire_with_metadata(
				output_size as u64, PooledBufferType::Storage, Some("Output Buffer"))?;
			let staging_acquired = interpolator.buffer_pool.acquire_with_metadata(
				output_size as u64, PooledBufferType::Staging, Some("Staging Buffer"))?;

			// Write actual data to target buffer
			interpolator.queue.write_buffer(&target_acquired.buffer, 0, cast_slice(target_batch));

			// Zero tail region of target buffer for thread safety (cost is negligible)
			#[allow(clippy::cast_possible_truncation)] // Buffer sizes fit in usize on 64-bit systems
			if target_acquired.tier_size > target_size {
				let zeros = vec![0u8; (target_acquired.tier_size - target_size) as usize];
				interpolator.queue.write_buffer(&target_acquired.buffer, target_size, &zeros);
			}

			// Zero entire output buffer to ensure clean state
			#[allow(clippy::cast_possible_truncation)] // Buffer sizes fit in usize on 64-bit systems
			let output_zeros = vec![0u8; output_acquired.tier_size as usize];
			interpolator.queue.write_buffer(&output_acquired.buffer, 0, &output_zeros);

			let target_times_buffer = target_acquired.buffer;
			let output_buffer = output_acquired.buffer;
			let staging_buffer = staging_acquired.buffer;

			// Use BufferBinding with explicit size for pooled buffers - shader uses arrayLength()
			// to determine element count, so we must expose only the actual data size
			let bind_group = interpolator.device.create_bind_group(&BindGroupDescriptor { label: Some("Interpolation Bind Group"), layout: &interpolator.bind_group_layout, entries: &[
				BindGroupEntry { binding: 0, resource: input_times_buffer.as_entire_binding() },
				BindGroupEntry { binding: 1, resource: input_values_buffer.as_entire_binding() },
				BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &target_times_buffer, offset: 0, size: std::num::NonZeroU64::new(target_size) }) },
				BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &output_buffer, offset: 0, size: std::num::NonZeroU64::new(output_size as u64) }) },
				BindGroupEntry { binding: 4, resource: config_buffer.as_entire_binding() }
			] });
			let mut encoder = interpolator.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Interpolation Encoder") });

			// Lock pipelines for the duration of the compute pass
			let pipelines = interpolator.pipelines_f64.lock().unwrap();
			let pipeline = &pipelines[method];
			{
				let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Interpolation Pass"), timestamp_writes: None });
				compute_pass.set_pipeline(pipeline);
				compute_pass.set_bind_group(0, &bind_group, &[]);
				let num_workgroups = target_batch.len().div_ceil(workgroup_size);
				compute_pass.dispatch_workgroups(u32::try_from(num_workgroups).context("Number of workgroups exceeds u32 limit")?, 1, 1);
			}
			drop(pipelines); // Explicitly drop the lock

			encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_size as u64);
			interpolator.queue.submit(std::iter::once(encoder.finish()));

			// Track this batch as pending
			pending_batches.push_back(PendingBatch {
				staging_buffer,
				output_size,
				batch_len: target_batch.len(),
				target_buffer: target_times_buffer,
				output_buffer,
				_marker: std::marker::PhantomData,
			});

			// If pipeline window is full, read oldest completed batch
			if pending_batches.len() >= window_size {
				Self::read_oldest_batch_f64(interpolator, &mut pending_batches, &mut all_results)?;
			}
		}

		// Drain remaining batches
		while !pending_batches.is_empty() {
			Self::read_oldest_batch_f64(interpolator, &mut pending_batches, &mut all_results)?;
		}

		Ok(all_results)
	}

	/// Reads the oldest pending batch and appends results
	fn read_oldest_batch_f64(
		interpolator: &Self,
		pending_batches: &mut VecDeque<PendingBatch<f64>>,
		all_results: &mut Vec<f64>,
	) -> Result<()> {
		let batch = pending_batches.pop_front().unwrap();

		// Poll to ensure this batch is complete
		let _ = interpolator.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });

		// Map and read the staging buffer
		let buffer_slice = batch.staging_buffer.slice(..batch.output_size as u64);
		let (tx, rx) = std::sync::mpsc::channel();
		buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
			let _ = tx.send(result);
		});

		// Poll to complete the map operation
		let _ = interpolator.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
		rx.recv().unwrap_or(Err(wgpu::BufferAsyncError)).context("Failed to map buffer")?;

		let data = buffer_slice.get_mapped_range();
		let element_size = std::mem::size_of::<f64>();
		let batch_results: Vec<f64> = bytemuck::cast_slice(&data[..batch.batch_len * element_size]).to_vec();
		drop(data);

		// Unmap staging buffer after reading
		batch.staging_buffer.unmap();

		all_results.extend(batch_results);

		// Release buffers back to pool for reuse
		interpolator.buffer_pool.release(batch.target_buffer);
		interpolator.buffer_pool.release(batch.output_buffer);
		interpolator.buffer_pool.release(batch.staging_buffer);

		Ok(())
	}

	/// Performs static f32 interpolation using pipelined GPU execution
	///
	/// Uses a sliding window approach to overlap GPU computation with CPU readback:
	/// - Submits batches continuously to the GPU
	/// - When the pipeline window is full (3 batches), waits for oldest to complete
	/// - Reads completed results while new batches are processing
	///
	/// # Errors
	/// Returns an error if:
	/// - The global interpolator failed to initialize
	/// - Input validation fails
	/// - GPU operations fail
	/// - Buffer operations fail
	pub fn interpolate_f32_static(input_times: &[f32], input_values: &[f32], target_times: &[f32], method: &Method, config_buffer: &wgpu::Buffer) -> Result<Vec<f32>> {
		let interpolator = match GLOBAL_INTERPOLATOR.as_ref() {
			Ok(interpolator) => interpolator,
			Err(e) => bail!("Failed to access global interpolator: {e}"),
		};

		if input_times.is_empty() || target_times.is_empty() {
			return Ok(Vec::new());
		}
		if input_times.len() != input_values.len() {
			bail!("Input times and values must have the same length");
		}

		// Create input buffers with exact size (not pooled - input data is test-specific)
		let _input_size = std::mem::size_of_val(input_times) as u64;
		let input_times_buffer = Arc::new(interpolator.device.create_buffer_init(&BufferInitDescriptor {
			label: Some("Input Times Buffer"),
			contents: cast_slice(input_times),
			usage: BufferUsages::STORAGE,
		}));
		let input_values_buffer = Arc::new(interpolator.device.create_buffer_init(&BufferInitDescriptor {
			label: Some("Input Values Buffer"),
			contents: cast_slice(input_values),
			usage: BufferUsages::STORAGE,
		}));

		// Ensure pipeline exists, create if needed
		{
			let mut pipelines = interpolator.pipelines_f32.lock().unwrap();
			if !pipelines.contains_key(method) {
				let shader_source = method.shader_source_f32();
				let shader = interpolator.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("Interpolation Shader"), source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(shader_source)) });
				let pipeline_layout = interpolator.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("Interpolation Pipeline Layout"), bind_group_layouts: &[&interpolator.bind_group_layout], push_constant_ranges: &[] });
				let pipeline = interpolator.device.create_compute_pipeline(&ComputePipelineDescriptor { label: Some("Interpolation Pipeline"), layout: Some(&pipeline_layout), module: &shader, entry_point: Some("main"), compilation_options: wgpu::PipelineCompilationOptions::default(), cache: None });
				pipelines.insert(*method, pipeline);
			}
		}

		// Process in batches to respect GPU memory limits
		let element_size = std::mem::size_of::<f32>();
		let workgroup_size = 64;
		let max_compute_workgroups = 65535;
		let max_elements_per_batch = (interpolator.max_storage_buffer_binding_size / element_size).min(target_times.len()).min(max_compute_workgroups * workgroup_size);

		// Pipeline window size - number of batches to keep in flight before reading
		let window_size = 3;

		// Queue of pending batches for pipelined execution
		let mut pending_batches: VecDeque<PendingBatch<f32>> = VecDeque::with_capacity(window_size);
		let mut all_results = Vec::with_capacity(target_times.len());

		for target_batch in target_times.chunks(max_elements_per_batch) {
			let output_size = std::mem::size_of_val(target_batch);
			let target_size = std::mem::size_of_val(target_batch) as u64;

			// Acquire pooled buffers with metadata for safe zeroing
			let target_acquired = interpolator.buffer_pool.acquire_with_metadata(
				target_size, PooledBufferType::Storage, Some("Target Times Buffer"))?;
			let output_acquired = interpolator.buffer_pool.acquire_with_metadata(
				output_size as u64, PooledBufferType::Storage, Some("Output Buffer"))?;
			let staging_acquired = interpolator.buffer_pool.acquire_with_metadata(
				output_size as u64, PooledBufferType::Staging, Some("Staging Buffer"))?;

			// Write actual data to target buffer
			interpolator.queue.write_buffer(&target_acquired.buffer, 0, cast_slice(target_batch));

			// Zero tail region of target buffer for thread safety (cost is negligible)
			#[allow(clippy::cast_possible_truncation)] // Buffer sizes fit in usize on 64-bit systems
			if target_acquired.tier_size > target_size {
				let zeros = vec![0u8; (target_acquired.tier_size - target_size) as usize];
				interpolator.queue.write_buffer(&target_acquired.buffer, target_size, &zeros);
			}

			// Zero entire output buffer to ensure clean state
			#[allow(clippy::cast_possible_truncation)] // Buffer sizes fit in usize on 64-bit systems
			let output_zeros = vec![0u8; output_acquired.tier_size as usize];
			interpolator.queue.write_buffer(&output_acquired.buffer, 0, &output_zeros);

			let target_times_buffer = target_acquired.buffer;
			let output_buffer = output_acquired.buffer;
			let staging_buffer = staging_acquired.buffer;

			// Use BufferBinding with explicit size for pooled buffers - shader uses arrayLength()
			// to determine element count, so we must expose only the actual data size
			let bind_group = interpolator.device.create_bind_group(&BindGroupDescriptor { label: Some("Interpolation Bind Group"), layout: &interpolator.bind_group_layout, entries: &[
				BindGroupEntry { binding: 0, resource: input_times_buffer.as_entire_binding() },
				BindGroupEntry { binding: 1, resource: input_values_buffer.as_entire_binding() },
				BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &target_times_buffer, offset: 0, size: std::num::NonZeroU64::new(target_size) }) },
				BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &output_buffer, offset: 0, size: std::num::NonZeroU64::new(output_size as u64) }) },
				BindGroupEntry { binding: 4, resource: config_buffer.as_entire_binding() }
			] });
			let mut encoder = interpolator.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Interpolation Encoder") });

			// Lock pipelines for the duration of the compute pass
			let pipelines = interpolator.pipelines_f32.lock().unwrap();
			let pipeline = &pipelines[method];
			{
				let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Interpolation Pass"), timestamp_writes: None });
				compute_pass.set_pipeline(pipeline);
				compute_pass.set_bind_group(0, &bind_group, &[]);
				let num_workgroups = target_batch.len().div_ceil(workgroup_size);
				compute_pass.dispatch_workgroups(u32::try_from(num_workgroups).unwrap_or(u32::MAX), 1, 1);
			}
			drop(pipelines); // Explicitly drop the lock

			encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_size as u64);
			interpolator.queue.submit(std::iter::once(encoder.finish()));

			// Track this batch as pending
			pending_batches.push_back(PendingBatch {
				staging_buffer,
				output_size,
				batch_len: target_batch.len(),
				target_buffer: target_times_buffer,
				output_buffer,
				_marker: std::marker::PhantomData,
			});

			// If pipeline window is full, read oldest completed batch
			if pending_batches.len() >= window_size {
				Self::read_oldest_batch_f32(interpolator, &mut pending_batches, &mut all_results)?;
			}
		}

		// Drain remaining batches
		while !pending_batches.is_empty() {
			Self::read_oldest_batch_f32(interpolator, &mut pending_batches, &mut all_results)?;
		}

		Ok(all_results)
	}

	/// Reads the oldest pending batch and appends results (f32 version)
	fn read_oldest_batch_f32(
		interpolator: &Self,
		pending_batches: &mut VecDeque<PendingBatch<f32>>,
		all_results: &mut Vec<f32>,
	) -> Result<()> {
		let batch = pending_batches.pop_front().unwrap();

		// Poll to ensure this batch is complete
		let _ = interpolator.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });

		// Map and read the staging buffer
		let buffer_slice = batch.staging_buffer.slice(..batch.output_size as u64);
		let (tx, rx) = std::sync::mpsc::channel();
		buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
			let _ = tx.send(result);
		});

		// Poll to complete the map operation
		let _ = interpolator.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
		rx.recv().unwrap_or(Err(wgpu::BufferAsyncError)).context("Failed to map buffer")?;

		let data = buffer_slice.get_mapped_range();
		let element_size = std::mem::size_of::<f32>();
		let batch_results: Vec<f32> = bytemuck::cast_slice(&data[..batch.batch_len * element_size]).to_vec();
		drop(data);

		// Unmap staging buffer after reading
		batch.staging_buffer.unmap();

		all_results.extend(batch_results);

		// Release buffers back to pool for reuse
		interpolator.buffer_pool.release(batch.target_buffer);
		interpolator.buffer_pool.release(batch.output_buffer);
		interpolator.buffer_pool.release(batch.staging_buffer);

		Ok(())
	}

	/// Async wrapper for f64 interpolation
	///
	/// Returns a `GpuInterpolationResult` that can be awaited for async/await support.
	/// The result is currently computed synchronously but wrapped in an async interface
	/// for future enhancement with true asynchronous GPU operations.
	#[allow(dead_code)] // Infrastructure for future async GPU operations
	pub fn interpolate_f64_async_static(
		input_times: &[f64],
		input_values: &[f64],
		target_times: &[f64],
		method: &Method,
		config_buffer: &wgpu::Buffer,
	) -> Result<GpuInterpolationResult<f64>> {
		let results = Self::interpolate_f64_static(input_times, input_values, target_times, method, config_buffer)?;
		Ok(GpuInterpolationResult::new(results))
	}

	/// Async wrapper for f32 interpolation
	///
	/// Returns a `GpuInterpolationResult` that can be awaited for async/await support.
	/// The result is currently computed synchronously but wrapped in an async interface
	/// for future enhancement with true asynchronous GPU operations.
	#[allow(dead_code)] // Infrastructure for future async GPU operations
	pub fn interpolate_f32_async_static(
		input_times: &[f32],
		input_values: &[f32],
		target_times: &[f32],
		method: &Method,
		config_buffer: &wgpu::Buffer,
	) -> Result<GpuInterpolationResult<f32>> {
		let results = Self::interpolate_f32_static(input_times, input_values, target_times, method, config_buffer)?;
		Ok(GpuInterpolationResult::new(results))
	}
}

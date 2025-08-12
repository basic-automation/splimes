use std::{
	borrow::Cow, collections::HashMap, sync::{LazyLock, Mutex}
};

use anyhow::{bail, Context, Result};
use bytemuck::cast_slice;
use wgpu::{
	util::{BufferInitDescriptor, DeviceExt}, Backends, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, BufferBindingType, BufferUsages, ComputePipeline, ComputePipelineDescriptor, Device, DeviceDescriptor, Dx12Compiler, Features, Gles3MinorVersion, Instance, InstanceDescriptor, InstanceFlags, Limits, PowerPreference, Queue, RequestAdapterOptions, ShaderStages
};

use crate::gpu::Method;

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
}

impl GpuInterpolator {
	async fn new() -> Result<Self> {
		let instance = Instance::new(InstanceDescriptor { backends: Backends::PRIMARY, flags: InstanceFlags::default(), dx12_shader_compiler: Dx12Compiler::default(), gles_minor_version: Gles3MinorVersion::default() });
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

		let (device, queue) = adapter.request_device(&DeviceDescriptor { label: Some("Interpolation Device"), required_features: if supports_f64 { Features::SHADER_F64 } else { Features::empty() }, required_limits: Limits::default() }, None).await.context("Failed to request GPU device")?;
		let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor { label: Some("Interpolation Bind Group Layout"), entries: &[BindGroupLayoutEntry { binding: 0, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None }, BindGroupLayoutEntry { binding: 1, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None }, BindGroupLayoutEntry { binding: 2, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None }, BindGroupLayoutEntry { binding: 3, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None }, BindGroupLayoutEntry { binding: 4, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None }] });

		Ok(Self { device, queue, bind_group_layout, pipelines_f64: Mutex::new(HashMap::new()), pipelines_f32: Mutex::new(HashMap::new()), supports_f64, max_storage_buffer_binding_size })
	}

	pub fn supports_f64_static() -> Result<bool> {
		match GLOBAL_INTERPOLATOR.as_ref() {
			Ok(interpolator) => Ok(interpolator.supports_f64),
			Err(e) => bail!("Failed to access global interpolator: {}", e),
		}
	}

	/// Returns a reference to the global GPU device for buffer creation
	///
	/// # Errors
	/// Returns an error if the global GPU interpolator failed to initialize
	pub fn get_device_static() -> Result<&'static Device> {
		match GLOBAL_INTERPOLATOR.as_ref() {
			Ok(interpolator) => Ok(&interpolator.device),
			Err(e) => bail!("Failed to access global interpolator: {}", e),
		}
	}

	/// Performs static f64 interpolation using the global GPU interpolator instance
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
			Err(e) => bail!("Failed to access global interpolator: {}", e),
		};

		if input_times.is_empty() || target_times.is_empty() {
			return Ok(Vec::new());
		}
		if input_times.len() != input_values.len() {
			bail!("Input times and values must have the same length");
		}
		let input_times_buffer = interpolator.device.create_buffer_init(&BufferInitDescriptor { label: Some("Input Times Buffer"), contents: cast_slice(input_times), usage: BufferUsages::STORAGE });
		let input_values_buffer = interpolator.device.create_buffer_init(&BufferInitDescriptor { label: Some("Input Values Buffer"), contents: cast_slice(input_values), usage: BufferUsages::STORAGE });

		// Ensure pipeline exists, create if needed
		{
			let mut pipelines = interpolator.pipelines_f64.lock().unwrap();
			if !pipelines.contains_key(method) {
				let shader_source = method.shader_source_f64();
				let shader = interpolator.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("Interpolation Shader"), source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(shader_source)) });
				let pipeline_layout = interpolator.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("Interpolation Pipeline Layout"), bind_group_layouts: &[&interpolator.bind_group_layout], push_constant_ranges: &[] });
				let pipeline = interpolator.device.create_compute_pipeline(&ComputePipelineDescriptor { label: Some("Interpolation Pipeline"), layout: Some(&pipeline_layout), module: &shader, entry_point: "main" });
				pipelines.insert(*method, pipeline);
			}
		}

		// Process in batches to respect GPU memory limits
		let element_size = std::mem::size_of::<f64>();
		let workgroup_size = 64;
		let max_compute_workgroups = 65535;
		let max_elements_per_batch = (interpolator.max_storage_buffer_binding_size / element_size).min(target_times.len()).min(max_compute_workgroups * workgroup_size);
		let mut all_results = Vec::with_capacity(target_times.len());
		for target_batch in target_times.chunks(max_elements_per_batch) {
			let output_size = std::mem::size_of_val(target_batch);
			let target_times_buffer = interpolator.device.create_buffer_init(&BufferInitDescriptor { label: Some("Target Times Buffer"), contents: cast_slice(target_batch), usage: BufferUsages::STORAGE });
			let output_buffer = interpolator.device.create_buffer(&wgpu::BufferDescriptor { label: Some("Output Buffer"), size: output_size as u64, usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC, mapped_at_creation: false });
			let staging_buffer = interpolator.device.create_buffer(&wgpu::BufferDescriptor { label: Some("Staging Buffer"), size: output_size as u64, usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ, mapped_at_creation: false });
			let bind_group = interpolator.device.create_bind_group(&BindGroupDescriptor { label: Some("Interpolation Bind Group"), layout: &interpolator.bind_group_layout, entries: &[BindGroupEntry { binding: 0, resource: input_times_buffer.as_entire_binding() }, BindGroupEntry { binding: 1, resource: input_values_buffer.as_entire_binding() }, BindGroupEntry { binding: 2, resource: target_times_buffer.as_entire_binding() }, BindGroupEntry { binding: 3, resource: output_buffer.as_entire_binding() }, BindGroupEntry { binding: 4, resource: config_buffer.as_entire_binding() }] });
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
			let buffer_slice = staging_buffer.slice(..);
			let (tx, rx) = std::sync::mpsc::channel();
			buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
				tx.send(result).unwrap();
			});
			interpolator.device.poll(wgpu::Maintain::Wait);
			rx.recv().unwrap().context("Failed to map buffer")?;
			let data = buffer_slice.get_mapped_range();
			let batch_results: Vec<f64> = bytemuck::cast_slice(&data).to_vec();
			drop(data);
			staging_buffer.unmap();
			all_results.extend(batch_results);
		}
		Ok(all_results)
	}

	/// Performs static f32 interpolation using the global GPU interpolator instance
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
			Err(e) => bail!("Failed to access global interpolator: {}", e),
		};

		if input_times.is_empty() || target_times.is_empty() {
			return Ok(Vec::new());
		}
		if input_times.len() != input_values.len() {
			bail!("Input times and values must have the same length");
		}

		let input_times_buffer = interpolator.device.create_buffer_init(&BufferInitDescriptor { label: Some("Input Times Buffer"), contents: cast_slice(input_times), usage: BufferUsages::STORAGE });
		let input_values_buffer = interpolator.device.create_buffer_init(&BufferInitDescriptor { label: Some("Input Values Buffer"), contents: cast_slice(input_values), usage: BufferUsages::STORAGE });

		// Ensure pipeline exists, create if needed
		{
			let mut pipelines = interpolator.pipelines_f32.lock().unwrap();
			if !pipelines.contains_key(method) {
				let shader_source = method.shader_source_f32();
				let shader = interpolator.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("Interpolation Shader"), source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(shader_source)) });
				let pipeline_layout = interpolator.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("Interpolation Pipeline Layout"), bind_group_layouts: &[&interpolator.bind_group_layout], push_constant_ranges: &[] });
				let pipeline = interpolator.device.create_compute_pipeline(&ComputePipelineDescriptor { label: Some("Interpolation Pipeline"), layout: Some(&pipeline_layout), module: &shader, entry_point: "main" });
				pipelines.insert(*method, pipeline);
			}
		}

		// Process in batches to respect GPU memory limits
		let element_size = std::mem::size_of::<f32>();
		let workgroup_size = 64;
		let max_compute_workgroups = 65535;
		let max_elements_per_batch = (interpolator.max_storage_buffer_binding_size / element_size).min(target_times.len()).min(max_compute_workgroups * workgroup_size);
		let mut all_results = Vec::with_capacity(target_times.len());

		for target_batch in target_times.chunks(max_elements_per_batch) {
			let output_size = std::mem::size_of_val(target_batch);
			let target_times_buffer = interpolator.device.create_buffer_init(&BufferInitDescriptor { label: Some("Target Times Buffer"), contents: cast_slice(target_batch), usage: BufferUsages::STORAGE });
			let output_buffer = interpolator.device.create_buffer(&wgpu::BufferDescriptor { label: Some("Output Buffer"), size: output_size as u64, usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC, mapped_at_creation: false });
			let staging_buffer = interpolator.device.create_buffer(&wgpu::BufferDescriptor { label: Some("Staging Buffer"), size: output_size as u64, usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ, mapped_at_creation: false });
			let bind_group = interpolator.device.create_bind_group(&BindGroupDescriptor { label: Some("Interpolation Bind Group"), layout: &interpolator.bind_group_layout, entries: &[BindGroupEntry { binding: 0, resource: input_times_buffer.as_entire_binding() }, BindGroupEntry { binding: 1, resource: input_values_buffer.as_entire_binding() }, BindGroupEntry { binding: 2, resource: target_times_buffer.as_entire_binding() }, BindGroupEntry { binding: 3, resource: output_buffer.as_entire_binding() }, BindGroupEntry { binding: 4, resource: config_buffer.as_entire_binding() }] });
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
			let buffer_slice = staging_buffer.slice(..);
			let (tx, rx) = std::sync::mpsc::channel();
			buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
				tx.send(result).unwrap();
			});
			interpolator.device.poll(wgpu::Maintain::Wait);
			rx.recv().unwrap().context("Failed to map buffer")?;
			let data = buffer_slice.get_mapped_range();
			let batch_results: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
			drop(data);
			staging_buffer.unmap();
			all_results.extend(batch_results);
		}
		Ok(all_results)
	}
}

//! The process-wide device: opened once, on first use or by `prewarm_gpu`.

use std::{
	borrow::Cow, sync::{
		Arc, Mutex, OnceLock, PoisonError, atomic::{AtomicBool, Ordering}
	}
};

use wgpu::{BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, BufferBindingType, ComputePipeline, ComputePipelineDescriptor, DeviceDescriptor, Features, Instance, InstanceDescriptor, Limits, PipelineLayoutDescriptor, PowerPreference, RequestAdapterOptions, ShaderModuleDescriptor, ShaderSource, ShaderStages};

use super::{
	GpuConfig, GpuInfo, GpuPoolStats, run::{self, Pool}, shader
};
use crate::{Error, Precision, Result};

pub struct Context {
	pub device: wgpu::Device,
	pub queue: wgpu::Queue,
	pub info: GpuInfo,
	pub config: GpuConfig,
	pub f64: Option<Kernel>,
	pub f32: Kernel,
	/// The adapter's limit on one storage-buffer binding, in bytes.
	pub max_binding_bytes: u64,
	/// Set by the device-lost callback; every later call fails fast instead of hanging.
	pub lost: Arc<AtomicBool>,
	pub pool: Mutex<Pool>,
}

pub struct Kernel {
	pub layout: BindGroupLayout,
	pub pipeline: ComputePipeline,
}

impl Context {
	pub fn kernel(&self, precision: Precision) -> Result<&Kernel> {
		if self.lost.load(Ordering::Acquire) {
			return Err(Error::GpuUnavailable("the GPU device was lost".to_owned()));
		}
		match precision {
			Precision::F64 => self.f64.as_ref().ok_or_else(|| Error::GpuUnavailable(format!("{} has no f64 shader support; ask for Precision::F32 to run in single precision", self.info.name))),
			Precision::F32 => Ok(&self.f32),
		}
	}

	pub fn pool_stats(&self) -> GpuPoolStats {
		self.pool.lock().unwrap_or_else(PoisonError::into_inner).stats()
	}
}

static CONTEXT: OnceLock<Result<Context, String>> = OnceLock::new();
static STARTED: AtomicBool = AtomicBool::new(false);

/// Whether anything has begun opening the device.
pub fn started() -> bool {
	STARTED.load(Ordering::Acquire)
}

/// The device, opening it first if need be. Concurrent first callers wait for one open.
pub fn get() -> Result<&'static Context> {
	STARTED.store(true, Ordering::Release);
	CONTEXT.get_or_init(open).as_ref().map_err(|reason| Error::GpuUnavailable(reason.clone()))
}

/// The device if it is already open; never blocks.
pub fn try_get() -> Option<&'static Context> {
	CONTEXT.get().and_then(|c| c.as_ref().ok())
}

/// Whether the device is open and has a kernel for `precision`. Never blocks and never
/// opens the device.
///
/// splimes deliberately never opens the device on a thread of its own: a process that
/// exits while a driver is still initialising on another thread can crash in the
/// driver's exit handlers.
pub fn ready(precision: Precision) -> bool {
	try_get().is_some_and(|ctx| ctx.kernel(precision).is_ok())
}

fn open() -> Result<Context, String> {
	let config = super::fix_config();
	// wgpu's environment variables apply: `WGPU_BACKEND` limits the graphics APIs tried,
	// and the debugging and validation flags work as in any wgpu program.
	let instance = Instance::new(InstanceDescriptor { flags: wgpu::InstanceFlags::from_build_config(), ..InstanceDescriptor::new_without_display_handle() }.with_env());
	let adapter = match std::env::var(ADAPTER_NAME_VAR) {
		Ok(wanted) if !wanted.trim().is_empty() => {
			let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
			let names: Vec<String> = adapters.iter().map(|a| a.get_info().name).collect();
			// wgpu's own helper for this variable panics when nothing matches; say so instead.
			adapters.into_iter().find(|a| name_matches(&a.get_info().name, &wanted)).ok_or_else(|| format!("{ADAPTER_NAME_VAR}={wanted:?} matches no adapter (found: {})", if names.is_empty() { "none".to_owned() } else { names.join(", ") }))?
		}
		_ => {
			let power_preference = if config.low_power { PowerPreference::LowPower } else { PowerPreference::HighPerformance };
			pollster::block_on(instance.request_adapter(&RequestAdapterOptions { power_preference, compatible_surface: None, force_fallback_adapter: false, apply_limit_buckets: false })).map_err(|e| format!("no GPU adapter: {e}"))?
		}
	};
	let adapter_info = adapter.get_info();
	let supports_f64 = adapter.features().contains(Features::SHADER_F64);
	let adapter_limits = adapter.limits();

	// The defaults are what every adapter wgpu supports can meet; raise only the two buffer
	// limits, to what this adapter allows, so large series fit in fewer dispatches.
	let limits = Limits { max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size, max_buffer_size: adapter_limits.max_buffer_size, ..Limits::default() };
	let (device, queue) = pollster::block_on(adapter.request_device(&DeviceDescriptor { label: Some("splimes"), required_features: if supports_f64 { Features::SHADER_F64 } else { Features::empty() }, required_limits: limits, ..Default::default() })).map_err(|e| format!("could not open {}: {e}", adapter_info.name))?;

	// Without a handler, wgpu panics on any error not caught by an error scope. Every call
	// runs inside scopes (see `run::eval`), so this only sees stragglers; log them.
	device.on_uncaptured_error(Arc::new(|e| log::error!("splimes: uncaptured GPU error: {e}")));
	let lost = Arc::new(AtomicBool::new(false));
	let flag = Arc::clone(&lost);
	device.set_device_lost_callback(move |reason, message| {
		flag.store(true, Ordering::Release);
		log::error!("splimes: GPU device lost ({reason:?}): {message}");
	});

	// Each kernel compiles inside its own error scopes, for every error class: shader
	// generation and driver pipeline failures are Internal errors, not Validation ones.
	// An adapter that advertises f64 but can't compile the f64 kernel still gets f32.
	let f32 = run::captured(&device, || Ok(kernel(&device, &shader::f32_source()))).map_err(|e| format!("kernel compilation failed on {}: {e}", adapter_info.name))?;
	let f64 = if supports_f64 { run::captured(&device, || Ok(kernel(&device, &shader::f64_source()))).inspect_err(|e| log::warn!("splimes: {} advertises f64 but its f64 kernel failed to compile, so only f32 is available: {e}", adapter_info.name)).ok() } else { None };
	let supports_f64 = f64.is_some();

	let info = GpuInfo {
		name: adapter_info.name,
		api: adapter_info.backend.to_str().to_owned(),
		device_type: match adapter_info.device_type {
			wgpu::DeviceType::DiscreteGpu => "discrete",
			wgpu::DeviceType::IntegratedGpu => "integrated",
			wgpu::DeviceType::VirtualGpu => "virtual",
			wgpu::DeviceType::Cpu => "cpu",
			wgpu::DeviceType::Other => "other",
		}
		.to_owned(),
		driver: format!("{} {}", adapter_info.driver, adapter_info.driver_info).trim().to_owned(),
		supports_f64,
	};
	log::info!("splimes: GPU ready: {} ({}, {}), f64 {}", info.name, info.api, info.device_type, if supports_f64 { "supported" } else { "unsupported" });
	let pool = Mutex::new(Pool::new(config.max_pool_bytes));
	Ok(Context { device, queue, info, config, f64, f32, max_binding_bytes: adapter_limits.max_storage_buffer_binding_size, lost, pool })
}

/// Names the adapter to use, as in wgpu's examples: the first adapter whose name contains
/// the value, ignoring case. Unset or blank, splimes asks wgpu for one by power preference.
const ADAPTER_NAME_VAR: &str = "WGPU_ADAPTER_NAME";

/// Whether `WGPU_ADAPTER_NAME=wanted` selects the adapter called `name`.
fn name_matches(name: &str, wanted: &str) -> bool {
	name.to_lowercase().contains(&wanted.trim().to_lowercase())
}

/// Compiles one kernel: knot times, knot values, output, params, grid times.
fn kernel(device: &wgpu::Device, source: &str) -> Kernel {
	let storage = |binding, read_only| BindGroupLayoutEntry { binding, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None }, count: None };
	let entries = [storage(0, true), storage(1, true), storage(2, false), BindGroupLayoutEntry { binding: 3, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None }, storage(4, true)];
	let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor { label: Some("splimes"), entries: &entries });
	let module = device.create_shader_module(ShaderModuleDescriptor { label: Some("splimes"), source: ShaderSource::Wgsl(Cow::Borrowed(source)) });
	let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor { label: Some("splimes"), bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
	let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor { label: Some("splimes"), layout: Some(&pipeline_layout), module: &module, entry_point: Some("main"), compilation_options: wgpu::PipelineCompilationOptions::default(), cache: None });
	Kernel { layout, pipeline }
}

#[cfg(test)]
mod tests {
	use super::name_matches;

	#[test]
	fn adapter_names_match_case_insensitively_by_substring() {
		assert!(name_matches("NVIDIA GeForce RTX 4070 Ti SUPER", "nvidia"));
		assert!(name_matches("NVIDIA GeForce RTX 4070 Ti SUPER", " RTX 4070 "));
		assert!(name_matches("llvmpipe (LLVM 20.1.2, 256 bits)", "LLVMpipe"));
		assert!(!name_matches("llvmpipe (LLVM 20.1.2, 256 bits)", "nvidia"));
		assert!(!name_matches("AMD Radeon Graphics", "Radeon RX"));
	}
}

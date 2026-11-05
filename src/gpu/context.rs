//! The process-wide devices: opened once, on first use or by `prewarm_gpu`.

use std::{
	borrow::Cow, sync::{
		Arc, Mutex, OnceLock, PoisonError, atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering}
	}
};

use wgpu::{Adapter, BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, BufferBindingType, ComputePipeline, ComputePipelineDescriptor, DeviceDescriptor, Features, Instance, InstanceDescriptor, Limits, PipelineLayoutDescriptor, PowerPreference, RequestAdapterOptions, ShaderModuleDescriptor, ShaderSource, ShaderStages};

use super::{
	AdapterSelector, GpuAdapterInfo, GpuConfig, GpuDeviceStats, GpuDevices, GpuInfo, GpuMemoryReport, GpuPoolStats, run::{self, Pool}, shader
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
	/// Calls (or shares of a split call) running on this device now.
	pub in_flight: AtomicUsize,
	/// Calls (or shares of a split call) run on this device.
	pub calls: AtomicU64,
	/// Chunks dispatched on this device.
	pub chunks: AtomicU64,
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

	pub fn is_lost(&self) -> bool {
		self.lost.load(Ordering::Acquire)
	}

	/// Grid points per dispatch on this device.
	pub fn chunk_len(&self) -> usize {
		// Bounded by MAX_CHUNK, which fits usize.
		usize::try_from(u64::from(self.config.chunk_points.max(1)).min(run::MAX_CHUNK).min(self.max_binding_bytes / run::TIME_BYTES)).unwrap_or(usize::MAX)
	}

	/// Whether this device can take a call of `knots` knots in `precision`: not lost, has the
	/// kernel, and the knot times fit one binding.
	pub fn can_run(&self, precision: Precision, knots: usize) -> bool {
		self.kernel(precision).is_ok() && (knots as u64).saturating_mul(run::TIME_BYTES) <= self.max_binding_bytes
	}

	pub fn stats(&self) -> GpuDeviceStats {
		GpuDeviceStats { info: self.info.clone(), pool: self.pool_stats(), calls: self.calls.load(Ordering::Relaxed), chunks: self.chunks.load(Ordering::Relaxed), in_flight: self.in_flight.load(Ordering::Relaxed), lost: self.is_lost() }
	}

	pub fn memory_report(&self) -> Option<GpuMemoryReport> {
		if self.lost.load(Ordering::Acquire) {
			return None;
		}
		// Frees what completed submissions no longer need; never waits. A failure here
		// (e.g. a lost device) leaves the report as it is.
		let _ = self.device.poll(wgpu::PollType::Poll);
		self.device.generate_allocator_report().map(|r| GpuMemoryReport { allocated_bytes: r.total_allocated_bytes, reserved_bytes: r.total_reserved_bytes, allocations: r.allocations.len(), blocks: r.blocks.len() })
	}
}

/// Every open device; the first is the primary, which `gpu_info` reports.
pub struct Devices {
	pub list: Vec<Context>,
	/// Rotates the starting device for ties in the least-loaded choice.
	pub next: AtomicUsize,
}

impl Devices {
	pub fn primary(&self) -> &Context {
		&self.list[0]
	}

	/// Opens the devices `config` asks for.
	pub fn open(config: &GpuConfig) -> Result<Self, String> {
		let instance = instance();
		let list = match &config.devices {
			GpuDevices::One => vec![open_adapter(&select_one(&instance, config)?, config)?],
			GpuDevices::List(selectors) => {
				if selectors.is_empty() {
					return Err("GpuDevices::List names no adapter".to_owned());
				}
				let adapters = enumerate(&instance);
				selectors.iter().map(|s| find(&adapters, s).and_then(|a| open_adapter(a, config))).collect::<Result<Vec<_>, _>>()?
			}
			GpuDevices::All => {
				let adapters = all_devices(enumerate(&instance), config.low_power);
				let found = names(&adapters);
				let opened: Vec<Context> = adapters.iter().filter_map(|a| open_adapter(a, config).inspect_err(|e| log::warn!("splimes: skipping an adapter: {e}")).ok()).collect();
				if opened.is_empty() {
					return Err(format!("no GPU adapter opened (found: {found})"));
				}
				opened
			}
		};
		Ok(Self { list, next: AtomicUsize::new(0) })
	}
}

static DEVICES: OnceLock<Result<Devices, String>> = OnceLock::new();
static STARTED: AtomicBool = AtomicBool::new(false);

/// Whether anything has begun opening the device.
pub fn started() -> bool {
	STARTED.load(Ordering::Acquire)
}

/// The devices, opening them first if need be. Concurrent first callers wait for one open.
pub fn get() -> Result<&'static Devices> {
	STARTED.store(true, Ordering::Release);
	DEVICES.get_or_init(|| Devices::open(&super::fix_config())).as_ref().map_err(|reason| Error::GpuUnavailable(reason.clone()))
}

/// The devices if they are already open; never blocks.
pub fn try_get() -> Option<&'static Devices> {
	DEVICES.get().and_then(|c| c.as_ref().ok())
}

/// Whether a device is open and has a kernel for `precision`. Never blocks and never
/// opens the device.
///
/// splimes deliberately never opens the device on a thread of its own: a process that
/// exits while a driver is still initialising on another thread can crash in the
/// driver's exit handlers.
pub fn ready(precision: Precision) -> bool {
	try_get().is_some_and(|devices| devices.list.iter().any(|ctx| ctx.kernel(precision).is_ok()))
}

/// The primary device the process-wide configuration selects, opened on its own: for
/// tests that need a device nothing else shares.
#[cfg(test)]
pub fn open() -> Result<Context, String> {
	let config = super::fix_config();
	open_adapter(&select_one(&instance(), &config)?, &config)
}

fn instance() -> Instance {
	// wgpu's environment variables apply: `WGPU_BACKEND` limits the graphics APIs tried,
	// and the debugging and validation flags work as in any wgpu program.
	Instance::new(InstanceDescriptor { flags: wgpu::InstanceFlags::from_build_config(), ..InstanceDescriptor::new_without_display_handle() }.with_env())
}

/// Every adapter wgpu offers, in its order; `GpuAdapterInfo::index` counts in it.
fn enumerate(instance: &Instance) -> Vec<Adapter> {
	pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()))
}

/// The adapters wgpu offers, without opening any.
pub fn adapters() -> Vec<GpuAdapterInfo> {
	enumerate(&instance())
		.iter()
		.enumerate()
		.map(|(index, a)| {
			let i = a.get_info();
			GpuAdapterInfo { index, api: i.backend.to_str().to_owned(), device_type: device_type(i.device_type).to_owned(), driver: driver(&i), vendor: i.vendor, device: i.device, advertises_f64: a.features().contains(Features::SHADER_F64), name: i.name }
		})
		.collect()
}

fn names(adapters: &[Adapter]) -> String {
	if adapters.is_empty() { "none".to_owned() } else { adapters.iter().map(|a| format!("{} ({})", a.get_info().name, a.get_info().backend.to_str())).collect::<Vec<_>>().join(", ") }
}

/// The adapter `selector` picks out of `adapters`.
fn find<'a>(adapters: &'a [Adapter], selector: &AdapterSelector) -> Result<&'a Adapter, String> {
	let found = match selector {
		AdapterSelector::Index(i) => adapters.get(*i),
		AdapterSelector::Name(name) => adapters.iter().find(|a| name_matches(&a.get_info().name, name)),
		AdapterSelector::Api(api) => adapters.iter().find(|a| a.get_info().backend.to_str().eq_ignore_ascii_case(api.trim())),
	};
	found.ok_or_else(|| format!("{selector:?} matches no adapter (found: {})", names(adapters)))
}

/// The single adapter of `GpuDevices::One`.
fn select_one(instance: &Instance, config: &GpuConfig) -> Result<Adapter, String> {
	if let Some(selector) = &config.adapter {
		return find(&enumerate(instance), selector).cloned();
	}
	match std::env::var(ADAPTER_NAME_VAR) {
		Ok(wanted) if !wanted.trim().is_empty() => {
			let adapters = enumerate(instance);
			// wgpu's own helper for this variable panics when nothing matches; say so instead.
			adapters.iter().find(|a| name_matches(&a.get_info().name, &wanted)).cloned().ok_or_else(|| format!("{ADAPTER_NAME_VAR}={wanted:?} matches no adapter (found: {})", names(&adapters)))
		}
		_ => {
			let power_preference = if config.low_power { PowerPreference::LowPower } else { PowerPreference::HighPerformance };
			pollster::block_on(instance.request_adapter(&RequestAdapterOptions { power_preference, compatible_surface: None, force_fallback_adapter: false, apply_limit_buckets: false })).map_err(|e| format!("no GPU adapter: {e}"))
		}
	}
}

/// `GpuDevices::All`: one adapter per physical device, software rasterisers only if there
/// is nothing else, discrete first (integrated first for `low_power`).
fn all_devices(adapters: Vec<Adapter>, low_power: bool) -> Vec<Adapter> {
	let mut seen = std::collections::HashSet::new();
	let mut kept: Vec<Adapter> = adapters
		.into_iter()
		.filter(|a| {
			let i = a.get_info();
			// Without PCI IDs, the name and type stand in for the device.
			seen.insert((i.vendor, i.device, i.name, device_type(i.device_type)))
		})
		.collect();
	if kept.iter().any(|a| a.get_info().device_type != wgpu::DeviceType::Cpu) {
		kept.retain(|a| a.get_info().device_type != wgpu::DeviceType::Cpu);
	}
	// Stable, so wgpu's order breaks ties.
	kept.sort_by_key(|a| match (a.get_info().device_type, low_power) {
		(wgpu::DeviceType::DiscreteGpu, false) | (wgpu::DeviceType::IntegratedGpu, true) => 0,
		(wgpu::DeviceType::IntegratedGpu | wgpu::DeviceType::DiscreteGpu, _) => 1,
		(wgpu::DeviceType::VirtualGpu, _) => 2,
		(wgpu::DeviceType::Other, _) => 3,
		(wgpu::DeviceType::Cpu, _) => 4,
	});
	kept
}

const fn device_type(t: wgpu::DeviceType) -> &'static str {
	match t {
		wgpu::DeviceType::DiscreteGpu => "discrete",
		wgpu::DeviceType::IntegratedGpu => "integrated",
		wgpu::DeviceType::VirtualGpu => "virtual",
		wgpu::DeviceType::Cpu => "cpu",
		wgpu::DeviceType::Other => "other",
	}
}

fn driver(info: &wgpu::AdapterInfo) -> String {
	format!("{} {}", info.driver, info.driver_info).trim().to_owned()
}

/// Opens a device on `adapter` and compiles the kernels on it.
fn open_adapter(adapter: &Adapter, config: &GpuConfig) -> Result<Context, String> {
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
	let lost_name = adapter_info.name.clone();
	device.set_device_lost_callback(move |reason, message| {
		flag.store(true, Ordering::Release);
		log::error!("splimes: GPU device lost on {lost_name} ({reason:?}): {message}");
	});

	// Each kernel compiles inside its own error scopes, for every error class: shader
	// generation and driver pipeline failures are Internal errors, not Validation ones.
	// An adapter that advertises f64 but can't compile the f64 kernel still gets f32.
	let f32 = run::captured(&device, || Ok(kernel(&device, &shader::f32_source()))).map_err(|e| format!("kernel compilation failed on {}: {e}", adapter_info.name))?;
	let f64 = if supports_f64 { run::captured(&device, || Ok(kernel(&device, &shader::f64_source()))).inspect_err(|e| log::warn!("splimes: {} advertises f64 but its f64 kernel failed to compile, so only f32 is available: {e}", adapter_info.name)).ok() } else { None };
	let supports_f64 = f64.is_some();

	let info = GpuInfo { driver: driver(&adapter_info), name: adapter_info.name, api: adapter_info.backend.to_str().to_owned(), device_type: device_type(adapter_info.device_type).to_owned(), supports_f64 };
	log::info!("splimes: GPU ready: {} ({}, {}), f64 {}", info.name, info.api, info.device_type, if supports_f64 { "supported" } else { "unsupported" });
	let pool = Mutex::new(Pool::new(config.max_pool_bytes));
	Ok(Context { device, queue, info, config: config.clone(), f64, f32, max_binding_bytes: adapter_limits.max_storage_buffer_binding_size, lost, pool, in_flight: AtomicUsize::new(0), calls: AtomicU64::new(0), chunks: AtomicU64::new(0) })
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

	/// The device is opened with the adapter's own buffer limits, not wgpu's defaults
	/// (128 MiB bindings, 256 MiB buffers): `max_binding_bytes` sizes every chunk and
	/// bounds the knots a call accepts, so it must be what the device really allows, and
	/// a buffer must be able to hold a whole binding.
	#[test]
	fn the_device_has_the_limits_splimes_plans_for() {
		let Ok(devices) = super::get() else {
			eprintln!("the_device_has_the_limits_splimes_plans_for: skipped, no GPU");
			return;
		};
		let ctx = devices.primary();
		let limits = ctx.device.limits();
		eprintln!("{}: storage binding {} bytes, buffer {} bytes", ctx.info.name, limits.max_storage_buffer_binding_size, limits.max_buffer_size);
		assert_eq!(limits.max_storage_buffer_binding_size, ctx.max_binding_bytes);
		assert!(limits.max_buffer_size >= ctx.max_binding_bytes, "{limits:?}");
	}

	#[test]
	fn adapter_names_match_case_insensitively_by_substring() {
		assert!(name_matches("NVIDIA GeForce RTX 4070 Ti SUPER", "nvidia"));
		assert!(name_matches("NVIDIA GeForce RTX 4070 Ti SUPER", " RTX 4070 "));
		assert!(name_matches("llvmpipe (LLVM 20.1.2, 256 bits)", "LLVMpipe"));
		assert!(!name_matches("llvmpipe (LLVM 20.1.2, 256 bits)", "nvidia"));
		assert!(!name_matches("AMD Radeon Graphics", "Radeon RX"));
	}
}

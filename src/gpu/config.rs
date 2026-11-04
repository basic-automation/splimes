/// Process-wide GPU settings, applied once when the GPU starts.
///
/// Pass to [`configure_gpu`](crate::configure_gpu) or
/// [`prewarm_gpu_with_config`](crate::prewarm_gpu_with_config) **before** anything else
/// touches the GPU. Both fail with [`Error::GpuAlreadyConfigured`](crate::Error::GpuAlreadyConfigured)
/// afterwards, because the device and its buffers already exist.
///
/// `#[non_exhaustive]`: start from [`GpuConfig::default`] or a preset and use the setters.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct GpuConfig {
	/// Most bytes of idle device buffers kept for reuse between calls (default 512 MiB).
	/// Buffers beyond it are freed when a call finishes.
	pub max_pool_bytes: u64,
	/// Grid points per GPU dispatch (default 4 Mi). A call holds buffers for at most two
	/// chunks at a time, so this bounds a call's GPU memory: per chunk point, up to about
	/// 80 bytes in `f64` (64 in `f32`) — output, read-back and grid-time buffers, plus
	/// wgpu's temporary upload copy — with the chunk rounded up to a power of two, plus
	/// the inputs. Larger chunks mean fewer round trips. Clamped to the adapter's
	/// storage-buffer binding limit.
	pub chunk_points: u32,
	/// Prefer a low-power adapter (an integrated GPU) over a high-performance one when the
	/// system has both (default `false`). Ignored when the `WGPU_ADAPTER_NAME` environment
	/// variable names the adapter (see the crate docs' "Backends") or when
	/// [`adapter`](Self::adapter) does. Under [`GpuDevices::All`] it orders the devices:
	/// integrated before discrete.
	pub low_power: bool,
	/// The adapter to open, from [`gpu_adapters`](crate::gpu_adapters) (default `None`: the
	/// one `WGPU_ADAPTER_NAME` names, else the one wgpu prefers for
	/// [`low_power`](Self::low_power)). Set, it takes precedence over `WGPU_ADAPTER_NAME`; a
	/// selector that matches no adapter makes the GPU unavailable, with the adapters found
	/// in the error. Under [`GpuDevices::One`] only; [`GpuDevices::List`] names its own.
	pub adapter: Option<AdapterSelector>,
	/// How many devices to open and spread work over (default [`GpuDevices::One`]).
	pub devices: GpuDevices,
	/// With more than one device, the fewest grid points a call needs before its chunks
	/// are split across the devices (default 0: any call of more than one
	/// [`chunk_points`](Self::chunk_points) chunk). Smaller calls run whole on one device,
	/// the one with the fewest calls in flight. Ignored with one device.
	pub split_min_points: u64,
}

/// Selects one adapter from [`gpu_adapters`](crate::gpu_adapters).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum AdapterSelector {
	/// The adapter at this [`GpuAdapterInfo::index`].
	Index(usize),
	/// The first adapter whose name contains this, ignoring case, as `WGPU_ADAPTER_NAME`
	/// matches.
	Name(String),
	/// The first adapter on this graphics API (`"vulkan"`, `"metal"`, `"dx12"`, `"gl"`),
	/// ignoring case.
	Api(String),
}

/// Which devices splimes opens. Every device gets its own buffer pool (each bounded by
/// [`GpuConfig::max_pool_bytes`]), its own error scopes and its own device-lost handling.
///
/// With more than one device, each call goes whole to the device with the fewest calls in
/// flight (ties in turn), unless it has at least [`GpuConfig::split_min_points`] grid
/// points and more than one chunk, when its chunks are shared out as the devices ask for
/// them, so a faster device takes more. A device that is lost is skipped from then on: the
/// call it was running is retried once on the others, and later calls never use it.
/// Every grid point is computed independently, so on identical devices the results match
/// one device's bit for bit; on different hardware each point is the result of the device
/// that computed it.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum GpuDevices {
	/// One device (the default), chosen by [`GpuConfig::adapter`], `WGPU_ADAPTER_NAME` or
	/// [`GpuConfig::low_power`].
	#[default]
	One,
	/// Every adapter that opens and compiles the kernels, one per physical device (an
	/// adapter wgpu offers on two APIs counts once, on the first API it lists), leaving out
	/// software rasterisers (`device_type == "cpu"`) unless there is nothing else. Adapters
	/// that fail to open are logged and skipped; the GPU is unavailable only if none opens.
	/// The first device, which [`gpu_info`](crate::gpu_info) reports, is discrete before
	/// integrated (the reverse with [`GpuConfig::low_power`]).
	All,
	/// These adapters, in this order; each must match and open, or the GPU is unavailable.
	/// The same adapter named twice opens two devices on it.
	List(Vec<AdapterSelector>),
}

impl Default for GpuConfig {
	fn default() -> Self {
		Self::DEFAULT
	}
}

impl GpuConfig {
	/// The defaults: 512 MiB pool, 4 Mi points per dispatch, high-performance adapter.
	pub const DEFAULT: Self = Self { max_pool_bytes: 512 << 20, chunk_points: 4 << 20, low_power: false, adapter: None, devices: GpuDevices::One, split_min_points: 0 };

	/// Small buffers and pool, for memory-constrained machines: 128 MiB pool, 1 Mi points
	/// per dispatch.
	#[must_use]
	pub const fn low_memory() -> Self {
		Self { max_pool_bytes: 128 << 20, chunk_points: 1 << 20, ..Self::DEFAULT }
	}

	/// The smallest footprint: 32 MiB pool, 256 Ki points per dispatch.
	#[must_use]
	pub const fn minimal() -> Self {
		Self { max_pool_bytes: 32 << 20, chunk_points: 256 << 10, ..Self::DEFAULT }
	}

	/// Large buffers and pool, for big jobs on machines with memory to spare: 16 Mi points
	/// per dispatch, and a pool (1 GiB + 1 MiB) that keeps both of a call's buffer sets
	/// for the next call.
	#[must_use]
	pub const fn high_performance() -> Self {
		Self { max_pool_bytes: (1 << 30) + (1 << 20), chunk_points: 16 << 20, ..Self::DEFAULT }
	}

	/// Sets [`max_pool_bytes`](Self::max_pool_bytes).
	#[must_use]
	pub const fn with_max_pool_bytes(mut self, bytes: u64) -> Self {
		self.max_pool_bytes = bytes;
		self
	}

	/// Sets [`chunk_points`](Self::chunk_points). Zero is treated as one.
	#[must_use]
	pub const fn with_chunk_points(mut self, points: u32) -> Self {
		self.chunk_points = points;
		self
	}

	/// Sets [`low_power`](Self::low_power).
	#[must_use]
	pub const fn with_low_power(mut self, low_power: bool) -> Self {
		self.low_power = low_power;
		self
	}

	/// Sets [`adapter`](Self::adapter).
	#[must_use]
	pub fn with_adapter(mut self, adapter: AdapterSelector) -> Self {
		self.adapter = Some(adapter);
		self
	}

	/// Sets [`devices`](Self::devices).
	#[must_use]
	pub fn with_devices(mut self, devices: GpuDevices) -> Self {
		self.devices = devices;
		self
	}

	/// Sets [`split_min_points`](Self::split_min_points).
	#[must_use]
	pub const fn with_split_min_points(mut self, points: u64) -> Self {
		self.split_min_points = points;
		self
	}
}

/// An adapter wgpu offers, as [`gpu_adapters`](crate::gpu_adapters) lists it.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct GpuAdapterInfo {
	/// Position in the listing, for [`AdapterSelector::Index`].
	pub index: usize,
	/// Adapter name, as in [`GpuInfo::name`].
	pub name: String,
	/// Graphics API, as in [`GpuInfo::api`].
	pub api: String,
	/// Device type, as in [`GpuInfo::device_type`].
	pub device_type: String,
	/// Driver name and version; may be empty.
	pub driver: String,
	/// PCI vendor ID, or 0 if the backend doesn't report one.
	pub vendor: u32,
	/// PCI device ID, or 0 if the backend doesn't report one.
	pub device: u32,
	/// Whether the adapter advertises `f64` shaders. Unlike [`GpuInfo::supports_f64`], the
	/// kernel hasn't been compiled yet, so a driver could still fail it.
	pub advertises_f64: bool,
}

/// One open device and its counters, from [`gpu_devices`](crate::gpu_devices).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct GpuDeviceStats {
	/// The device's adapter.
	pub info: GpuInfo,
	/// Its buffer pool.
	pub pool: GpuPoolStats,
	/// Calls, or shares of a split call, it has run (successfully or not).
	pub calls: u64,
	/// Chunks it has dispatched.
	pub chunks: u64,
	/// Calls, or shares of one, running on it now.
	pub in_flight: usize,
	/// Whether the device was lost; it gets no more work.
	pub lost: bool,
}

/// The GPU splimes is using.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct GpuInfo {
	/// Adapter name, e.g. `"NVIDIA GeForce RTX 4070 Ti SUPER"` or `"llvmpipe (LLVM 19.1.7, 256 bits)"`.
	pub name: String,
	/// Graphics API: `"vulkan"`, `"metal"`, `"dx12"`, …
	pub api: String,
	/// `"discrete"`, `"integrated"`, `"virtual"`, `"cpu"` (a software rasteriser) or `"other"`.
	pub device_type: String,
	/// Driver name and version, as reported by the driver; may be empty.
	pub driver: String,
	/// Whether the adapter runs `f64` shaders, i.e. whether [`Precision::F64`](crate::Precision::F64)
	/// is available on the GPU.
	pub supports_f64: bool,
}

/// Counters for the GPU buffer pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct GpuPoolStats {
	/// Idle buffer sets currently held for reuse.
	pub idle_sets: usize,
	/// Bytes held by idle buffer sets.
	pub idle_bytes: u64,
	/// Buffer sets created since the GPU started.
	pub created: u64,
	/// Times a call reused an idle buffer set instead of creating one.
	pub reused: u64,
	/// Buffer sets freed because the pool was over [`GpuConfig::max_pool_bytes`].
	pub evicted: u64,
}

#[cfg(test)]
mod tests {
	use super::*;

	const MIB: u64 = 1 << 20;

	/// Each preset is what its docs say, so a caller choosing one by its description gets it.
	#[test]
	fn presets_are_as_documented() {
		let preset = |c: GpuConfig| (c.max_pool_bytes, c.chunk_points, c.low_power);
		assert_eq!(GpuConfig::default(), GpuConfig::DEFAULT);
		assert_eq!(preset(GpuConfig::DEFAULT), (512 * MIB, 4 << 20, false));
		assert_eq!(preset(GpuConfig::low_memory()), (128 * MIB, 1 << 20, false));
		assert_eq!(preset(GpuConfig::minimal()), (32 * MIB, 256 << 10, false));
		assert_eq!(preset(GpuConfig::high_performance()), (1024 * MIB + MIB, 16 << 20, false));
		// The setters change one field each.
		let set = GpuConfig::minimal().with_max_pool_bytes(7).with_chunk_points(9).with_low_power(true);
		assert_eq!(preset(set), (7, 9, true));
		// Every preset opens one device, chosen as before.
		for c in [GpuConfig::DEFAULT, GpuConfig::low_memory(), GpuConfig::minimal(), GpuConfig::high_performance()] {
			assert_eq!((c.adapter, c.devices, c.split_min_points), (None, GpuDevices::One, 0));
		}
		let set = GpuConfig::default().with_adapter(AdapterSelector::Index(1)).with_devices(GpuDevices::All).with_split_min_points(5);
		assert_eq!((set.adapter, set.devices, set.split_min_points), (Some(AdapterSelector::Index(1)), GpuDevices::All, 5));
	}
}

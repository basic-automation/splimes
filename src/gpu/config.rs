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
	/// variable names the adapter (see the crate docs' "Backends").
	pub low_power: bool,
}

impl Default for GpuConfig {
	fn default() -> Self {
		Self::DEFAULT
	}
}

impl GpuConfig {
	/// The defaults: 512 MiB pool, 4 Mi points per dispatch, high-performance adapter.
	pub const DEFAULT: Self = Self { max_pool_bytes: 512 << 20, chunk_points: 4 << 20, low_power: false };

	/// Small buffers and pool, for memory-constrained machines: 128 MiB pool, 1 Mi points
	/// per dispatch.
	#[must_use]
	pub const fn low_memory() -> Self {
		Self { max_pool_bytes: 128 << 20, chunk_points: 1 << 20, low_power: false }
	}

	/// The smallest footprint: 32 MiB pool, 256 Ki points per dispatch.
	#[must_use]
	pub const fn minimal() -> Self {
		Self { max_pool_bytes: 32 << 20, chunk_points: 256 << 10, low_power: false }
	}

	/// Large buffers and pool, for big jobs on machines with memory to spare: 16 Mi points
	/// per dispatch, and a pool (1 GiB + 1 MiB) that keeps both of a call's buffer sets
	/// for the next call.
	#[must_use]
	pub const fn high_performance() -> Self {
		Self { max_pool_bytes: (1 << 30) + (1 << 20), chunk_points: 16 << 20, low_power: false }
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

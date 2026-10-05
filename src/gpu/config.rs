use crate::gpu::buffer_pool::BufferPoolConfig;

/// Global GPU configuration for interpolation operations
#[derive(Debug, Clone)]
pub struct GpuConfig {
	/// Configuration for the buffer pool
	pub buffer_pool: BufferPoolConfig,
	/// Number of staging buffers to maintain (default: 3)
	pub num_staging_buffers: usize,
	/// Maximum command batch size (for future optimization)
	pub max_command_batch_size: usize,
}

impl Default for GpuConfig {
	fn default() -> Self {
		Self::default_config()
	}
}

impl GpuConfig {
	/// Default configuration for balanced performance and memory usage
	const fn default_config() -> Self {
		Self {
			buffer_pool: BufferPoolConfig {
				max_pool_memory: 512 * 1024 * 1024, // 512MB
				eviction_timeout_secs: 30,
			},
			num_staging_buffers: 3,
			max_command_batch_size: 16,
		}
	}

	/// Conservative configuration for low-memory systems
	///
	/// Uses smaller pool, fewer buffers, and smaller batches
	#[must_use]
	pub const fn low_memory() -> Self {
		Self {
			buffer_pool: BufferPoolConfig {
				max_pool_memory: 128 * 1024 * 1024, // 128MB
				eviction_timeout_secs: 20,
			},
			num_staging_buffers: 2,
			max_command_batch_size: 8,
		}
	}

	/// High-performance configuration for systems with abundant memory
	///
	/// Uses larger pool, more buffers, and larger batches
	#[must_use]
	pub const fn high_performance() -> Self {
		Self {
			buffer_pool: BufferPoolConfig {
				max_pool_memory: 1024 * 1024 * 1024, // 1GB
				eviction_timeout_secs: 45,
			},
			num_staging_buffers: 4,
			max_command_batch_size: 32,
		}
	}

	/// Minimal configuration for extremely constrained environments
	#[must_use]
	pub const fn minimal() -> Self {
		Self {
			buffer_pool: BufferPoolConfig {
				max_pool_memory: 64 * 1024 * 1024, // 64MB
				eviction_timeout_secs: 15,
			},
			num_staging_buffers: 1,
			max_command_batch_size: 4,
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_default_config() {
		let config = GpuConfig::default();
		assert_eq!(config.buffer_pool.max_pool_memory, 512 * 1024 * 1024);
		assert_eq!(config.num_staging_buffers, 3);
		assert_eq!(config.max_command_batch_size, 16);
	}

	#[test]
	fn test_low_memory_config() {
		let config = GpuConfig::low_memory();
		assert_eq!(config.buffer_pool.max_pool_memory, 128 * 1024 * 1024);
		assert_eq!(config.num_staging_buffers, 2);
		assert_eq!(config.max_command_batch_size, 8);
	}

	#[test]
	fn test_high_performance_config() {
		let config = GpuConfig::high_performance();
		assert_eq!(config.buffer_pool.max_pool_memory, 1024 * 1024 * 1024);
		assert_eq!(config.num_staging_buffers, 4);
		assert_eq!(config.max_command_batch_size, 32);
	}

	#[test]
	fn test_minimal_config() {
		let config = GpuConfig::minimal();
		assert_eq!(config.buffer_pool.max_pool_memory, 64 * 1024 * 1024);
		assert_eq!(config.num_staging_buffers, 1);
	}

	#[test]
	fn test_configs_are_cloneable() {
		let config = GpuConfig::default();
		let cloned = config.clone();
		assert_eq!(cloned.num_staging_buffers, config.num_staging_buffers);
	}
}

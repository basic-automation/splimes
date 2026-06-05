use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use anyhow::{bail, Result};
use wgpu::{Device, Buffer, BufferUsages};

/// Size-tiered bucket strategy for efficient memory management
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum BufferSizeTier {
    Tiny,      // 4KB - for config buffers
    Small,     // 64KB - for small batches
    Medium,    // 1MB - for medium batches
    Large,     // 16MB - for large batches
    XLarge,    // 128MB - for very large batches
}

impl BufferSizeTier {
    /// Get the size in bytes for this tier
    pub const fn size_bytes(self) -> u64 {
        match self {
            Self::Tiny => 4 * 1024,
            Self::Small => 64 * 1024,
            Self::Medium => 1024 * 1024,
            Self::Large => 16 * 1024 * 1024,
            Self::XLarge => 128 * 1024 * 1024,
        }
    }

    /// Find the appropriate tier for a given size
    pub const fn for_size(size: u64) -> Self {
        match size {
            0..=4096 => Self::Tiny,
            4097..=65536 => Self::Small,
            65_537..=1_048_576 => Self::Medium,
            1_048_577..=16_777_216 => Self::Large,
            _ => Self::XLarge,
        }
    }
}

/// Buffer usage type for pool segregation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PooledBufferType {
    Storage,     // Input/output data buffers (STORAGE | COPY_SRC/COPY_DST)
    Staging,     // CPU readback buffers (COPY_DST | MAP_READ)
    Uniform,     // Config buffers (UNIFORM)
}

/// Configuration for buffer pool behavior
#[derive(Debug, Clone)]
pub struct BufferPoolConfig {
    /// Maximum total memory to keep in pool (default: 512MB)
    pub max_pool_memory: u64,
    /// Eviction timeout in seconds (default: 30s)
    pub eviction_timeout_secs: u64,
}

impl Default for BufferPoolConfig {
    fn default() -> Self {
        Self {
            max_pool_memory: 512 * 1024 * 1024,
            eviction_timeout_secs: 30,
        }
    }
}

/// Pooled buffer wrapper with metadata
struct PooledBuffer {
    buffer: Arc<Buffer>,
    size: u64,
    last_used_timestamp: u64,
}

impl PooledBuffer {
    fn new(buffer: Arc<Buffer>, size: u64) -> Self {
        Self {
            buffer,
            size,
            last_used_timestamp: current_timestamp(),
        }
    }
}

/// Statistics about buffer pool usage
#[derive(Debug, Clone, Default)]
pub struct BufferPoolStats {
    /// Total number of buffers in pool
    pub total_buffers: usize,
    /// Total memory allocated to pool
    pub total_allocated_bytes: u64,
    /// Number of allocations since creation
    pub total_allocations: u64,
    /// Number of reuses from pool
    pub total_reuses: u64,
}

/// Result of acquiring a buffer from the pool with metadata for safe zeroing
#[derive(Debug)]
pub struct AcquiredBuffer {
    /// The acquired buffer
    pub buffer: Arc<Buffer>,
    /// True if buffer was reused from pool (not freshly allocated)
    #[allow(dead_code)] // Available for callers to optimize zeroing
    pub is_reused: bool,
    /// Actual buffer size (may be larger than requested due to tiering)
    pub tier_size: u64,
}

/// Thread-safe buffer pool with LRU eviction
pub struct BufferPool {
    device: Arc<Device>,
    config: BufferPoolConfig,
    /// Pool organized by (`BufferType`, `SizeTier`) -> Vec<PooledBuffer>
    pools: Mutex<HashMap<(PooledBufferType, BufferSizeTier), VecDeque<PooledBuffer>>>,
    total_allocated: Arc<AtomicU64>,
    total_allocations: Arc<AtomicU64>,
    total_reuses: Arc<AtomicU64>,
}

impl std::fmt::Debug for BufferPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BufferPool")
            .field("config", &self.config)
            .field("total_allocated", &self.total_allocated.load(std::sync::atomic::Ordering::Relaxed))
            .field("total_allocations", &self.total_allocations.load(std::sync::atomic::Ordering::Relaxed))
            .field("total_reuses", &self.total_reuses.load(std::sync::atomic::Ordering::Relaxed))
            .finish_non_exhaustive()
    }
}

impl BufferPool {
    /// Create a new buffer pool
    pub fn new(device: Arc<Device>, config: BufferPoolConfig) -> Self {
        Self {
            device,
            config,
            pools: Mutex::new(HashMap::new()),
            total_allocated: Arc::new(AtomicU64::new(0)),
            total_allocations: Arc::new(AtomicU64::new(0)),
            total_reuses: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Acquire a buffer from pool or create new one
    /// Returns (Buffer, `is_new`) - `is_new` indicates if buffer was newly allocated
    #[allow(clippy::significant_drop_tightening)] // Lock held intentionally during entire pool operation
    pub fn acquire(
        &self,
        size: u64,
        buffer_type: PooledBufferType,
        label: Option<&str>,
    ) -> Result<(Arc<Buffer>, bool)> {
        if size == 0 {
            bail!("Cannot acquire buffer of size 0");
        }

        let tier = BufferSizeTier::for_size(size);
        let tier_size = tier.size_bytes();

        // Try to evict old buffers if pool is getting too full
        self.evict_if_needed()?;

        let mut pools = self.pools.lock().unwrap();
        let key = (buffer_type, tier);

        // Try to reuse existing buffer from pool
        if let Some(deque) = pools.get_mut(&key)
            && let Some(mut pooled) = deque.pop_front() {
                pooled.last_used_timestamp = current_timestamp();
                let buffer = pooled.buffer;
                self.total_reuses.fetch_add(1, Ordering::Relaxed);
                return Ok((buffer, false));
            }

        // Create new buffer
        let buffer_usage = match buffer_type {
            PooledBufferType::Storage => BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            PooledBufferType::Staging => BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            PooledBufferType::Uniform => BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        };

        let buffer = Arc::new(
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label,
                size: tier_size,
                usage: buffer_usage,
                mapped_at_creation: false,
            })
        );

        let current_total = self.total_allocated.fetch_add(tier_size, Ordering::Relaxed);
        if current_total + tier_size > self.config.max_pool_memory {
            bail!(
                "Buffer pool memory limit exceeded: {} + {} > {}",
                current_total,
                tier_size,
                self.config.max_pool_memory
            );
        }

        self.total_allocations.fetch_add(1, Ordering::Relaxed);

        Ok((buffer, true))
    }

    /// Acquire a buffer from pool with metadata for safe zeroing
    /// Returns an `AcquiredBuffer` containing the buffer and info about its actual size
    #[allow(clippy::significant_drop_tightening)] // Lock held intentionally during entire pool operation
    pub fn acquire_with_metadata(
        &self,
        size: u64,
        buffer_type: PooledBufferType,
        label: Option<&str>,
    ) -> Result<AcquiredBuffer> {
        let tier_size = BufferSizeTier::for_size(size).size_bytes();
        let (buffer, is_new) = self.acquire(size, buffer_type, label)?;
        Ok(AcquiredBuffer {
            buffer,
            is_reused: !is_new,
            tier_size,
        })
    }

    /// Release buffer back to pool for reuse
    #[allow(clippy::significant_drop_tightening)] // Lock held intentionally during pool operation
    pub fn release(&self, buffer: Arc<Buffer>) {
        // Only pool if we have exactly one Arc reference (us)
        if Arc::strong_count(&buffer) != 1 {
            return; // Other references exist, let it drop naturally
        }

        let buffer_size = buffer.size();
        let tier = BufferSizeTier::for_size(buffer_size);

        // Try to determine buffer type from usage flags
        // This is a heuristic since we don't track it explicitly
        let buffer_type = if buffer.usage().contains(BufferUsages::UNIFORM) {
            PooledBufferType::Uniform
        } else if buffer.usage().contains(BufferUsages::MAP_READ) {
            PooledBufferType::Staging
        } else {
            PooledBufferType::Storage
        };

        let mut pools = self.pools.lock().unwrap();
        let key = (buffer_type, tier);

        let deque = pools.entry(key).or_default();
        deque.push_back(PooledBuffer::new(buffer, buffer_size));
    }

    /// Get pool statistics
    #[allow(clippy::significant_drop_tightening)] // Lock held intentionally during stats collection
    pub fn stats(&self) -> BufferPoolStats {
        let pools = self.pools.lock().unwrap();
        let total_buffers: usize = pools.values().map(std::collections::VecDeque::len).sum();
        let total_allocated = self.total_allocated.load(Ordering::Relaxed);

        BufferPoolStats {
            total_buffers,
            total_allocated_bytes: total_allocated,
            total_allocations: self.total_allocations.load(Ordering::Relaxed),
            total_reuses: self.total_reuses.load(Ordering::Relaxed),
        }
    }

    /// Evict old buffers if total memory exceeds threshold
    #[allow(clippy::unnecessary_wraps, clippy::significant_drop_tightening)] // API consistency and lock held intentionally
    fn evict_if_needed(&self) -> Result<()> {
        let current_total = self.total_allocated.load(Ordering::Relaxed);
        let target_memory = (self.config.max_pool_memory * 90) / 100; // 90% threshold

        if current_total <= target_memory {
            return Ok(());
        }

        let mut pools = self.pools.lock().unwrap();
        let current_time = current_timestamp();
        let timeout_secs = self.config.eviction_timeout_secs;

        let mut evicted = 0u64;
        for deque in pools.values_mut() {
            deque.retain(|pooled| {
                let age_secs = (current_time - pooled.last_used_timestamp) / 1000;
                if age_secs > timeout_secs {
                    evicted += pooled.size;
                    false
                } else {
                    true
                }
            });
        }

        self.total_allocated.fetch_sub(evicted, Ordering::Relaxed);

        Ok(())
    }

    /// Force clear all pooled buffers (for testing/cleanup)
    #[allow(dead_code, clippy::significant_drop_tightening)] // Reserved for future use and lock held intentionally
    pub fn clear(&self) {
        let mut pools = self.pools.lock().unwrap();
        let total_size: u64 = pools
            .values()
            .flat_map(|deque| deque.iter().map(|b| b.size))
            .sum();
        pools.clear();
        self.total_allocated.fetch_sub(total_size, Ordering::Relaxed);
    }

    /// Clear all pooled buffers and reset all counters (for test isolation)
    #[allow(dead_code, clippy::significant_drop_tightening)] // Used in test isolation, lock held intentionally
    pub fn clear_all(&self) {
        let mut pools = self.pools.lock().unwrap();
        pools.clear();
        self.total_allocated.store(0, Ordering::Relaxed);
        self.total_allocations.store(0, Ordering::Relaxed);
        self.total_reuses.store(0, Ordering::Relaxed);
    }
}

/// Get current time in milliseconds since `UNIX_EPOCH`
#[allow(clippy::cast_possible_truncation)] // Timestamp in ms fits in u64 for thousands of years
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_size_tier_selection() {
        assert_eq!(BufferSizeTier::for_size(1024), BufferSizeTier::Tiny);
        assert_eq!(BufferSizeTier::for_size(4096), BufferSizeTier::Tiny);
        assert_eq!(BufferSizeTier::for_size(8192), BufferSizeTier::Small);
        assert_eq!(BufferSizeTier::for_size(65536), BufferSizeTier::Small);
        assert_eq!(BufferSizeTier::for_size(100_000), BufferSizeTier::Medium);
        assert_eq!(BufferSizeTier::for_size(1_048_576), BufferSizeTier::Medium);
        assert_eq!(BufferSizeTier::for_size(10_000_000), BufferSizeTier::Large);
        assert_eq!(BufferSizeTier::for_size(17_000_000), BufferSizeTier::XLarge);
    }

    #[test]
    fn test_buffer_pool_config_default() {
        let config = BufferPoolConfig::default();
        assert_eq!(config.max_pool_memory, 512 * 1024 * 1024);
        assert_eq!(config.eviction_timeout_secs, 30);
    }
}

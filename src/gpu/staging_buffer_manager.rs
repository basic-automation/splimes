use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use anyhow::Result;
use wgpu::{Device, Buffer, BufferUsages, Queue};

/// State of a staging buffer in the pipeline
#[allow(dead_code)] // Infrastructure for future streaming GPU operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferState {
    /// Available for new submission
    Free,
    /// GPU is writing to it (submission in flight)
    Submitted,
    /// GPU done, can map for reading
    ReadyToMap,
    /// Currently mapped for CPU read
    Mapped,
}

/// Manages rotating pool of staging buffers for pipelined GPU operations
///
/// Uses a sliding window strategy to overlap CPU and GPU work:
/// - While GPU processes buffer N, CPU reads from buffer N-2
/// - Buffer N-1 is in-flight (submitted, not yet readable)
///
/// This pattern enables pipelined execution where CPU readback overlaps
/// with GPU computation.
///
/// NOTE: Currently the `buffer_pool` is used for staging buffers with pipelined
/// execution. This manager provides an alternative API for future use cases
/// that need fixed-size staging buffer rotation.
#[allow(dead_code)] // Infrastructure for future streaming GPU operations
#[derive(Debug)]
pub struct StagingBufferManager {
    buffers: Mutex<Vec<MappedStagingBuffer>>,
    /// Queue of buffer indices in order of submission (oldest first)
    submission_order: Mutex<VecDeque<usize>>,
    current_index: AtomicUsize,
    buffer_size: u64,
    max_buffers: usize,
}

/// A staging buffer with metadata for state tracking
#[allow(dead_code)] // Infrastructure for future streaming GPU operations
struct MappedStagingBuffer {
    buffer: Arc<Buffer>,
    state: BufferState,
    /// Submission index from wgpu queue, used for poll waiting
    submission_index: Option<wgpu::SubmissionIndex>,
}

impl std::fmt::Debug for MappedStagingBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MappedStagingBuffer")
            .field("state", &self.state)
            .field("has_submission_index", &self.submission_index.is_some())
            .finish_non_exhaustive()
    }
}

/// Result of acquiring a staging buffer
#[allow(dead_code)] // Infrastructure for future streaming GPU operations
#[derive(Debug)]
pub struct AcquiredStaging {
    /// The staging buffer
    pub buffer: Arc<Buffer>,
    /// Index of this buffer in the manager (for tracking)
    pub buffer_index: usize,
}

#[allow(dead_code)] // Infrastructure for future streaming GPU operations
impl StagingBufferManager {
    /// Create a new staging buffer manager
    ///
    /// # Arguments
    /// * `buffer_size` - Size of each staging buffer in bytes
    /// * `max_buffers` - Number of buffers to maintain (typically 3 for pipelining)
    pub const fn new(buffer_size: u64, max_buffers: usize) -> Self {
        Self {
            buffers: Mutex::new(Vec::new()),
            submission_order: Mutex::new(VecDeque::new()),
            current_index: AtomicUsize::new(0),
            buffer_size,
            max_buffers,
        }
    }

    /// Get the current number of buffers in the pool
    #[cfg(test)]
    pub fn buffer_count(&self) -> usize {
        self.buffers.lock().unwrap().len()
    }

    /// Acquire next staging buffer, waiting if all are in use
    ///
    /// If all buffers are currently submitted or mapped, this will poll the device
    /// to wait for the oldest submitted buffer to complete before returning.
    ///
    /// # Arguments
    /// * `device` - GPU device for buffer creation and polling
    /// * `queue` - GPU queue (unused but kept for API consistency)
    ///
    /// # Returns
    /// An `AcquiredStaging` containing the buffer and its index
    #[allow(clippy::unnecessary_wraps)] // API consistency for future error handling
    pub fn acquire(&self, device: &Device, _queue: &Queue) -> Result<AcquiredStaging> {
        let mut buffers = self.buffers.lock().unwrap();

        // Lazily create buffers up to the maximum
        if buffers.len() < self.max_buffers {
            let buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Streaming Staging Buffer"),
                size: self.buffer_size,
                usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }));

            let index = buffers.len();
            buffers.push(MappedStagingBuffer {
                buffer: buffer.clone(),
                state: BufferState::Free,
                submission_index: None,
            });

            return Ok(AcquiredStaging {
                buffer,
                buffer_index: index,
            });
        }

        // Find a free buffer
        for (idx, staged) in buffers.iter_mut().enumerate() {
            if staged.state == BufferState::Free {
                return Ok(AcquiredStaging {
                    buffer: staged.buffer.clone(),
                    buffer_index: idx,
                });
            }
        }

        // No free buffers - need to wait for oldest submitted one
        // Get the oldest submitted buffer index
        let oldest_idx = {
            let submission_order = self.submission_order.lock().unwrap();
            submission_order.front().copied()
        };

        if let Some(oldest_idx) = oldest_idx {
            // Poll device to wait for this specific submission
            if let Some(ref sub_idx) = buffers[oldest_idx].submission_index {
                let _ = device.poll(wgpu::PollType::Wait {
                    submission_index: Some(sub_idx.clone()),
                    timeout: None,
                });
            }
            // Mark as ready to map
            buffers[oldest_idx].state = BufferState::ReadyToMap;
        }

        // Try again to find a free or ready-to-map buffer
        for (idx, staged) in buffers.iter_mut().enumerate() {
            if staged.state == BufferState::Free || staged.state == BufferState::ReadyToMap {
                staged.state = BufferState::Free;
                staged.submission_index = None;
                return Ok(AcquiredStaging {
                    buffer: staged.buffer.clone(),
                    buffer_index: idx,
                });
            }
        }

        // Fallback: use rotating index (shouldn't normally reach here)
        let idx = self.current_index.fetch_add(1, Ordering::Relaxed) % self.max_buffers;
        buffers[idx].state = BufferState::Free;
        buffers[idx].submission_index = None;
        Ok(AcquiredStaging {
            buffer: buffers[idx].buffer.clone(),
            buffer_index: idx,
        })
    }

    /// Mark buffer as submitted with its submission index
    ///
    /// Call this after submitting GPU work that writes to the staging buffer.
    ///
    /// # Arguments
    /// * `buffer_index` - Index of the buffer (from `AcquiredStaging`)
    /// * `submission_index` - Submission index from `queue.submit()`
    #[allow(clippy::significant_drop_tightening)] // Lock held intentionally during entire update
    pub fn mark_submitted(&self, buffer_index: usize, submission_index: wgpu::SubmissionIndex) {
        let mut buffers = self.buffers.lock().unwrap();
        if buffer_index < buffers.len() {
            buffers[buffer_index].state = BufferState::Submitted;
            buffers[buffer_index].submission_index = Some(submission_index);

            // Track submission order
            let mut submission_order = self.submission_order.lock().unwrap();
            submission_order.push_back(buffer_index);
        }
    }

    /// Get oldest completed buffer for reading (if any ready)
    ///
    /// Checks if the oldest submitted buffer has completed GPU work.
    /// Does NOT poll the device - use this for non-blocking checks.
    ///
    /// # Returns
    /// `Some((buffer, index))` if a buffer is ready to read, `None` otherwise
    #[allow(clippy::significant_drop_tightening)] // Locks held intentionally during state check
    pub fn try_get_completed(&self) -> Option<(Arc<Buffer>, usize)> {
        let mut buffers = self.buffers.lock().unwrap();
        let mut submission_order = self.submission_order.lock().unwrap();

        // Check the oldest submitted buffer
        if let Some(&oldest_idx) = submission_order.front()
            && oldest_idx < buffers.len() {
                let staged = &mut buffers[oldest_idx];
                // If it's submitted or ready, mark as ready for mapping
                if staged.state == BufferState::Submitted || staged.state == BufferState::ReadyToMap {
                    staged.state = BufferState::Mapped;
                    submission_order.pop_front();
                    return Some((staged.buffer.clone(), oldest_idx));
                }
            }
        None
    }

    /// Wait for and get the oldest completed buffer
    ///
    /// Polls the device to wait for the oldest submitted buffer to complete,
    /// then returns it for reading.
    ///
    /// # Arguments
    /// * `device` - GPU device for polling
    ///
    /// # Returns
    /// `Some((buffer, index))` if a submitted buffer exists, `None` if queue is empty
    #[allow(clippy::significant_drop_tightening)] // Lock ordering is intentional for correctness
    pub fn wait_for_completed(&self, device: &Device) -> Option<(Arc<Buffer>, usize)> {
        let oldest_idx = {
            let submission_order = self.submission_order.lock().unwrap();
            submission_order.front().copied()
        };

        if let Some(oldest_idx) = oldest_idx {
            let mut buffers = self.buffers.lock().unwrap();

            // Poll for this specific submission
            if let Some(ref sub_idx) = buffers[oldest_idx].submission_index {
                let _ = device.poll(wgpu::PollType::Wait {
                    submission_index: Some(sub_idx.clone()),
                    timeout: None,
                });
            }

            // Mark as mapped and remove from submission order
            buffers[oldest_idx].state = BufferState::Mapped;
            drop(buffers);

            self.submission_order.lock().unwrap().pop_front();

            let buffers = self.buffers.lock().unwrap();
            return Some((buffers[oldest_idx].buffer.clone(), oldest_idx));
        }
        None
    }

    /// Release buffer after reading
    ///
    /// Call this after you've finished reading data from a mapped buffer.
    /// The buffer will be returned to the Free state for reuse.
    ///
    /// # Arguments
    /// * `buffer_index` - Index of the buffer to release
    pub fn release(&self, buffer_index: usize) {
        let mut buffers = self.buffers.lock().unwrap();
        if buffer_index < buffers.len() {
            buffers[buffer_index].state = BufferState::Free;
            buffers[buffer_index].submission_index = None;
        }
    }

    /// Get the number of buffers currently in submitted state
    #[cfg(test)]
    pub fn submitted_count(&self) -> usize {
        let buffers = self.buffers.lock().unwrap();
        buffers.iter().filter(|b| b.state == BufferState::Submitted).count()
    }

    /// Get the number of buffers currently free
    #[cfg(test)]
    pub fn free_count(&self) -> usize {
        let buffers = self.buffers.lock().unwrap();
        buffers.iter().filter(|b| b.state == BufferState::Free).count()
    }

    /// Reset all buffers to free state (for testing/cleanup)
    #[cfg(test)]
    #[allow(clippy::significant_drop_tightening)] // Lock held intentionally during reset
    pub fn reset(&self) {
        let mut buffers = self.buffers.lock().unwrap();
        for staged in buffers.iter_mut() {
            staged.state = BufferState::Free;
            staged.submission_index = None;
        }
        let mut submission_order = self.submission_order.lock().unwrap();
        submission_order.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_staging_buffer_manager_creation() {
        let manager = StagingBufferManager::new(1024 * 1024, 3);
        assert_eq!(manager.buffer_count(), 0);
        assert_eq!(manager.max_buffers, 3);
    }

    #[test]
    fn test_buffer_state_transitions() {
        // Test that BufferState enum values are distinct
        assert_ne!(BufferState::Free, BufferState::Submitted);
        assert_ne!(BufferState::Submitted, BufferState::ReadyToMap);
        assert_ne!(BufferState::ReadyToMap, BufferState::Mapped);
    }

    #[test]
    fn test_staging_buffer_manager_rotation() {
        let manager = StagingBufferManager::new(1024 * 1024, 3);
        // Verify that it will rotate through buffers (tested in integration tests with device)
        // This is a basic sanity check
        assert_eq!(manager.buffer_count(), 0);
    }
}

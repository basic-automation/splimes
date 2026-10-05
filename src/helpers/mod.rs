pub mod batch;
pub mod estimate_output_points;
pub mod generate_target_times;
pub mod should_use_gpu;

pub use batch::{InterpolationState, batch};
pub use estimate_output_points::estimate_output_points;
pub use generate_target_times::{TargetTimesIterator, generate_target_times};
pub use should_use_gpu::{InterpolationStrategy, should_use_gpu};

/// A [`sysinfo::System`] that has loaded **only** memory figures.
///
/// The interpolation paths need total/available memory to size batches and decide when
/// to spill to disk. `System::new_all()` also enumerates every process, disk, network
/// interface and CPU on the machine: measured at ~445 ms per call versus ~38 µs for this
/// (sysinfo 0.39, Linux, a busy 16-core box). It ran on every interpolation call.
pub(crate) fn memory_only_system() -> sysinfo::System {
	sysinfo::System::new_with_specifics(sysinfo::RefreshKind::nothing().with_memory(sysinfo::MemoryRefreshKind::everything()))
}

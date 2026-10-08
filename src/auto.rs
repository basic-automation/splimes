//! How [`Backend::Auto`] picks a backend.

use std::sync::{PoisonError, RwLock};

use crate::{Backend, Precision};

/// The output sizes at which [`Backend::Auto`] moves to a faster backend.
///
/// A call with `m` grid points runs on [`Backend::Gpu`] if `m` reaches the GPU threshold
/// for its [`Precision`] and the GPU is already running (started by
/// [`prewarm_gpu`](crate::prewarm_gpu), [`calibrate`] or an explicit `Backend::Gpu` call)
/// and can run that precision; otherwise on [`Backend::Parallel`] if
/// `m >= parallel_min_points`; otherwise on [`Backend::Cpu`]. Only the grid size matters:
/// the kernel's cost per point is logarithmic in the number of inputs, so inputs barely
/// move the crossovers. (Preparing the input is separate: inputs of 16 Ki points or more
/// are prepared on rayon's pool whatever the grid size and these thresholds.)
///
/// The defaults come from the measurements in `BENCHMARKS.md`: rayon pays off from about
/// 64 Ki points, and the GPU is off, because on a 16-core desktop it only matched the
/// rayon backend except on the heaviest kernels. Building the output (timestamps, values,
/// provenance) costs the same whichever backend computed the values, and dominates.
/// Machines with fewer cores or a faster GPU differ; [`calibrate`] measures this machine
/// and sets the thresholds to match.
///
/// [`calibrate`]: crate::calibrate
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct AutoThresholds {
	/// Grid points from which `Auto` computes on the rayon pool.
	pub parallel_min_points: usize,
	/// Grid points from which `Auto` uses the GPU for [`Precision::F64`] calls.
	/// `usize::MAX` means never.
	pub gpu_min_points: usize,
	/// Grid points from which `Auto` uses the GPU for [`Precision::F32`] calls.
	/// `usize::MAX` means never.
	pub gpu_f32_min_points: usize,
}

impl AutoThresholds {
	/// The built-in defaults: rayon from 64 Ki points, never the GPU.
	pub const DEFAULT: Self = Self { parallel_min_points: 65_536, gpu_min_points: usize::MAX, gpu_f32_min_points: usize::MAX };

	/// Thresholds with the given crossovers, the GPU one applying to both precisions.
	#[must_use]
	pub const fn new(parallel_min_points: usize, gpu_min_points: usize) -> Self {
		Self { parallel_min_points, gpu_min_points, gpu_f32_min_points: gpu_min_points }
	}

	/// The same thresholds with a different GPU threshold for [`Precision::F32`] calls.
	#[must_use]
	pub const fn with_gpu_f32(mut self, gpu_f32_min_points: usize) -> Self {
		self.gpu_f32_min_points = gpu_f32_min_points;
		self
	}

	/// The GPU threshold for calls in `precision`.
	#[must_use]
	pub const fn gpu_min_points_for(&self, precision: Precision) -> usize {
		match precision {
			Precision::F64 => self.gpu_min_points,
			Precision::F32 => self.gpu_f32_min_points,
		}
	}
}

impl Default for AutoThresholds {
	fn default() -> Self {
		Self::DEFAULT
	}
}

/// One lock for all the thresholds, so a reader never sees a mix of two settings.
static THRESHOLDS: RwLock<AutoThresholds> = RwLock::new(AutoThresholds::DEFAULT);

/// The thresholds [`Backend::Auto`] is using.
#[must_use]
pub fn auto_thresholds() -> AutoThresholds {
	*THRESHOLDS.read().unwrap_or_else(PoisonError::into_inner)
}

/// Replaces the thresholds [`Backend::Auto`] uses, process-wide.
pub fn set_auto_thresholds(thresholds: AutoThresholds) {
	*THRESHOLDS.write().unwrap_or_else(PoisonError::into_inner) = thresholds;
}

#[cfg(test)]
thread_local! {
	/// Makes `Auto` pick the GPU on this thread regardless of size, to test fallback.
	pub static FORCE_GPU: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The backend `Auto` runs a call with `points` grid points on.
pub fn choose(points: usize, precision: Precision) -> Backend {
	#[cfg(test)]
	if FORCE_GPU.get() {
		return Backend::Gpu;
	}
	let t = auto_thresholds();
	if points >= t.gpu_min_points_for(precision) && crate::gpu::ready(precision) {
		Backend::Gpu
	} else if points >= t.parallel_min_points {
		Backend::Parallel
	} else {
		Backend::Cpu
	}
}

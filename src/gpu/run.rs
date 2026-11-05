//! Running the kernel: upload the knots, dispatch the grid in chunks, read back.

use std::{
	collections::VecDeque, sync::{PoisonError, atomic::Ordering, mpsc}, time::Duration
};

use rayon::prelude::*;
use wgpu::{BindGroupDescriptor, BindGroupEntry, Buffer, BufferDescriptor, BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor, ErrorFilter, MapMode, PollType};

use super::{GpuPoolStats, context::Context};
use crate::{Error, Precision, Result, kernel::Method, prepare::Knots, time::Grid, value::Value};

const WORKGROUP: u32 = 256;
/// Dispatches have at most 65,535 workgroups per dimension.
const MAX_CHUNK: u64 = 65_535 * WORKGROUP as u64;
/// Chunks in flight at once: one computing while the previous one is read back.
const IN_FLIGHT: usize = 2;
/// How long to wait for one chunk before declaring the GPU hung.
const CHUNK_TIMEOUT: Duration = Duration::from_secs(60);

pub fn eval<V: Value>(ctx: &Context, knots: &Knots<'_, V>, grid: &Grid, method: &Method, precision: Precision, out: &mut [f64]) -> Result<()> {
	#[cfg(test)]
	if tests::FAIL.get() {
		return Err(Error::Gpu("injected failure".to_owned()));
	}
	let kernel = ctx.kernel(precision)?;
	// Buffer sets go back to the pool only if the device reported no error for this call:
	// a set with a buffer that failed to allocate would otherwise fail every later call.
	let finished = captured(&ctx.device, || dispatch(ctx, kernel, knots, grid, method, precision, out))?;
	release(ctx, finished);
	Ok(())
}

/// Runs `f` inside error scopes for every error class and returns the first error the
/// device reported, if any, in preference to `f`'s own result.
///
/// Error scopes are per thread and catch everything `f` does on the device. They are what
/// turns a validation, out-of-memory or internal error into an `Err` instead of wgpu's
/// default, a panic.
pub fn captured<T>(device: &wgpu::Device, f: impl FnOnce() -> Result<T>) -> Result<T> {
	let scopes = [ErrorFilter::Internal, ErrorFilter::OutOfMemory, ErrorFilter::Validation].map(|filter| device.push_error_scope(filter));
	let result = f();
	// Scopes pop innermost first. Report the most fundamental error: an allocation that
	// failed makes every later use of that buffer a validation error too.
	let mut errors: Vec<wgpu::Error> = scopes.into_iter().rev().filter_map(|scope| pollster::block_on(scope.pop())).collect();
	errors.sort_by_key(|e| match e {
		wgpu::Error::OutOfMemory { .. } => 0,
		wgpu::Error::Internal { .. } => 1,
		wgpu::Error::Validation { .. } => 2,
	});
	errors.into_iter().next().map_or(result, |e| Err(Error::Gpu(e.to_string())))
}

/// Times as the kernels store them: 96-bit two's-complement nanoseconds since the first
/// knot, as `[low, middle, high]` words. Any two instants chrono can represent are less
/// than 2⁷⁴ ns apart, so every offset and every difference fits.
fn times(offsets: impl Iterator<Item = i128>, out: &mut Vec<u8>) {
	for o in offsets {
		#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // Two's-complement words.
		let words = [o as u32, (o >> 32) as u32, (o >> 64) as u32];
		out.extend_from_slice(bytemuck::cast_slice(&words));
	}
}

/// Bytes of grid times each host task writes into staging memory: 16 Ki points of 12
/// bytes.
const STAGE_BLOCK: usize = 12 * 16 * 1024;

fn dispatch<V: Value>(ctx: &Context, kernel: &super::context::Kernel, knots: &Knots<'_, V>, grid: &Grid, method: &Method, precision: Precision, out: &mut [f64]) -> Result<Vec<Slot>> {
	let elem = precision_bytes(precision);
	// A time is three u32 words, in either precision.
	let time_bytes: u64 = 12;
	let pair_len: usize = 12;
	let n = knots.len();
	if n as u64 * time_bytes > ctx.max_binding_bytes {
		return Err(Error::Gpu(format!("{n} knots exceed {}'s storage-binding limit of {} bytes", ctx.info.name, ctx.max_binding_bytes)));
	}
	let n_u32 = u32::try_from(n).map_err(|_| Error::Gpu(format!("{n} knots exceed the GPU kernel's u32 indexing")))?;

	let mut knot_u = Vec::with_capacity(n * pair_len);
	times(knots.offsets.iter().copied(), &mut knot_u);
	let knot_y: Vec<u8> = match precision {
		Precision::F64 => bytemuck::cast_slice(&knots.y).to_vec(),
		#[allow(clippy::cast_possible_truncation)] // Single precision is what was asked for.
		Precision::F32 => bytemuck::cast_slice(&knots.y.iter().map(|&v| v as f32).collect::<Vec<f32>>()).to_vec(),
	};
	let knot_u = upload(ctx, "splimes knot times", &knot_u);
	let knot_y = upload(ctx, "splimes knot values", &knot_y);

	#[allow(clippy::cast_possible_truncation)] // Bounded by MAX_CHUNK.
	let chunk = u64::from(ctx.config.chunk_points.max(1)).min(MAX_CHUNK).min(ctx.max_binding_bytes / time_bytes) as usize;
	// Times are nanoseconds in both kernels, scaled by 1/h.
	let inv_h = 1.0 / knots.h;
	let (lo, hi) = method.bounds.unwrap_or((0.0, 0.0));
	#[allow(clippy::cast_possible_truncation)] // `len <= MAX_CHUNK`, which fits u32.
	let header = |len: usize| [n_u32, method.window as u32, u32::from(method.hold), u32::from(method.bounds.is_some()), len as u32, 0, 0, 0];

	let mut pending: VecDeque<(Slot, usize, usize, wgpu::SubmissionIndex)> = VecDeque::with_capacity(IN_FLIGHT);
	// Buffer sets whose chunk has been read back, ready for the next chunk: a call holds at
	// most `IN_FLIGHT` sets however long its grid.
	let mut finished: Vec<Slot> = Vec::with_capacity(IN_FLIGHT);
	let mut first = 0;
	while first < grid.len {
		let len = chunk.min(grid.len - first);
		let slot = finished.iter().position(|s| s.capacity >= len).map_or_else(|| acquire(ctx, precision, len), |i| finished.swap_remove(i));

		let mut params = bytemuck::cast_slice(&header(len)).to_vec();
		match precision {
			Precision::F64 => params.extend_from_slice(bytemuck::cast_slice(&[inv_h, lo, hi, 0.0])),
			#[allow(clippy::cast_possible_truncation)]
			Precision::F32 => params.extend_from_slice(bytemuck::cast_slice(&[inv_h as f32, lo as f32, hi as f32, 0.0])),
		}
		ctx.queue.write_buffer(&slot.params, 0, &params);

		// Grid times, exact on the host, written in parallel straight into the queue's
		// staging memory: no intermediate buffer, no serial copy. (Fixed-size blocks, because
		// wgpu only lets a sized write-only view cross threads.)
		let size = wgpu::BufferSize::new(len as u64 * time_bytes).ok_or_else(|| Error::Gpu("empty chunk".to_owned()))?;
		let mut staging = ctx.queue.write_buffer_with(&slot.grid, 0, size).ok_or_else(|| Error::Gpu("could not stage the grid times".to_owned()))?;
		let (blocks, tail) = staging.slice(..).into_chunks::<STAGE_BLOCK>();
		let per_block = STAGE_BLOCK / pair_len;
		let fill = |from: usize, mut part: wgpu::WriteOnly<'_, [u8]>| {
			let mut bytes = Vec::with_capacity(part.len());
			times(grid.offsets(knots.t0, from).take(part.len() / pair_len), &mut bytes);
			part.copy_from_slice(&bytes);
		};
		let blocks: Vec<wgpu::WriteOnly<'_, [u8; STAGE_BLOCK]>> = blocks.into_iter().collect();
		let whole = blocks.len();
		blocks.into_par_iter().enumerate().for_each(|(c, block)| fill(first + c * per_block, block.into()));
		if !tail.is_empty() {
			fill(first + whole * per_block, tail);
		}
		drop(staging);

		let bytes = len as u64 * elem;
		let entries = [BindGroupEntry { binding: 0, resource: knot_u.as_entire_binding() }, BindGroupEntry { binding: 1, resource: knot_y.as_entire_binding() }, sized(2, &slot.out, bytes), BindGroupEntry { binding: 3, resource: slot.params.as_entire_binding() }, sized(4, &slot.grid, len as u64 * time_bytes)];
		let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor { label: Some("splimes"), layout: &kernel.layout, entries: &entries });
		let mut encoder = ctx.device.create_command_encoder(&CommandEncoderDescriptor { label: Some("splimes") });
		{
			let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor { label: Some("splimes"), timestamp_writes: None });
			pass.set_pipeline(&kernel.pipeline);
			pass.set_bind_group(0, &bind_group, &[]);
			#[allow(clippy::cast_possible_truncation)]
			pass.dispatch_workgroups((len as u32).div_ceil(WORKGROUP), 1, 1);
		}
		encoder.copy_buffer_to_buffer(&slot.out, 0, &slot.staging, 0, bytes);
		let submission = ctx.queue.submit(Some(encoder.finish()));
		pending.push_back((slot, first, len, submission));
		if pending.len() >= IN_FLIGHT
			&& let Some(job) = pending.pop_front()
		{
			finished.push(read_back(ctx, job, precision, out)?);
		}
		first += len;
	}
	while let Some(job) = pending.pop_front() {
		finished.push(read_back(ctx, job, precision, out)?);
	}
	Ok(finished)
}

/// A storage buffer holding `bytes`, created unmapped and filled through the queue, so an
/// allocation failure is reported to the error scopes like any other. (`create_buffer_init`
/// maps at creation and panics if the allocation failed.) `bytes` is a whole number of
/// 4-byte scalars, as copies require.
fn upload(ctx: &Context, label: &'static str, bytes: &[u8]) -> Buffer {
	let buffer = ctx.device.create_buffer(&BufferDescriptor { label: Some(label), size: bytes.len() as u64, usage: BufferUsages::STORAGE | BufferUsages::COPY_DST, mapped_at_creation: false });
	ctx.queue.write_buffer(&buffer, 0, bytes);
	buffer
}

/// A binding of the first `size` bytes of a pooled buffer, which may be larger: the
/// kernel sizes its loops by `params`, but out-of-range writes must still be impossible.
const fn sized(binding: u32, buffer: &Buffer, size: u64) -> BindGroupEntry<'_> {
	BindGroupEntry { binding, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer, offset: 0, size: std::num::NonZeroU64::new(size) }) }
}

/// Waits for one chunk, copies its results into `out`, and hands back its buffer set.
fn read_back(ctx: &Context, (slot, first, len, submission): (Slot, usize, usize, wgpu::SubmissionIndex), precision: Precision, out: &mut [f64]) -> Result<Slot> {
	let bytes = len as u64 * precision_bytes(precision);
	let slice = slot.staging.slice(..bytes);
	let (tx, rx) = mpsc::channel();
	slice.map_async(MapMode::Read, move |r| {
		let _ = tx.send(r);
	});
	#[cfg(test)]
	match tests::LOSE_BEFORE_POLL.get() {
		Some(tests::Loss::Destroyed) => ctx.device.destroy(),
		Some(tests::Loss::Signalled) => ctx.lost.store(true, Ordering::Release),
		None => {}
	}
	ctx.device.poll(PollType::Wait { submission_index: Some(submission), timeout: Some(CHUNK_TIMEOUT) }).map_err(|e| Error::Gpu(format!("waiting for the GPU: {e}")))?;
	rx.recv().map_err(|_| Error::Gpu("buffer mapping was abandoned".to_owned()))?.map_err(|e| Error::Gpu(format!("mapping the result buffer: {e}")))?;
	// wgpu 30 can report a mapping as successful although the device was lost before it
	// was processed (gfx-rs/wgpu#10301), and the buffer then holds whatever it held before:
	// for a pooled buffer, an earlier call's results. Never return those as this call's.
	if ctx.lost.load(Ordering::Acquire) {
		return Err(Error::GpuUnavailable("the GPU device was lost during the call".to_owned()));
	}
	{
		let view = slice.get_mapped_range().map_err(|e| Error::Gpu(format!("reading the result buffer: {e}")))?;
		let dest = &mut out[first..first + len];
		// Mapped memory may be uncached or write-combined; copy it out once, in parallel.
		match precision {
			Precision::F64 => dest.par_chunks_mut(64 * 1024).zip(bytemuck::cast_slice::<u8, f64>(&view).par_chunks(64 * 1024)).for_each(|(d, s)| d.copy_from_slice(s)),
			Precision::F32 => dest.par_chunks_mut(64 * 1024).zip(bytemuck::cast_slice::<u8, f32>(&view).par_chunks(64 * 1024)).for_each(|(d, s)| {
				for (d, &s) in d.iter_mut().zip(s) {
					*d = f64::from(s);
				}
			}),
		}
	}
	slot.staging.unmap();
	Ok(slot)
}

const fn precision_bytes(precision: Precision) -> u64 {
	match precision {
		Precision::F64 => 8,
		Precision::F32 => 4,
	}
}

/// The per-chunk device buffers, reused across chunks and calls.
pub struct Slot {
	precision: Precision,
	/// Grid points this set can hold.
	capacity: usize,
	out: Buffer,
	staging: Buffer,
	params: Buffer,
	/// Grid times, uploaded per chunk.
	grid: Buffer,
}

impl Slot {
	const fn bytes(&self) -> u64 {
		// Output and staging values, plus a 12-byte time per point, plus the params.
		self.capacity as u64 * (2 * precision_bytes(self.precision) + 12) + 64
	}
}

/// Idle buffer sets, bounded by [`GpuConfig::max_pool_bytes`](super::GpuConfig::max_pool_bytes).
pub struct Pool {
	idle: Vec<Slot>,
	max_bytes: u64,
	created: u64,
	reused: u64,
	evicted: u64,
}

impl Pool {
	pub const fn new(max_bytes: u64) -> Self {
		Self { idle: Vec::new(), max_bytes, created: 0, reused: 0, evicted: 0 }
	}

	pub fn stats(&self) -> GpuPoolStats {
		GpuPoolStats { idle_sets: self.idle.len(), idle_bytes: self.idle.iter().map(Slot::bytes).sum(), created: self.created, reused: self.reused, evicted: self.evicted }
	}
}

fn acquire(ctx: &Context, precision: Precision, len: usize) -> Slot {
	{
		let mut pool = ctx.pool.lock().unwrap_or_else(PoisonError::into_inner);
		// The smallest idle set that fits, so a small call doesn't take a huge set.
		if let Some(i) = pool.idle.iter().enumerate().filter(|(_, s)| s.precision == precision && s.capacity >= len).min_by_key(|(_, s)| s.capacity).map(|(i, _)| i) {
			pool.reused += 1;
			return pool.idle.swap_remove(i);
		}
		pool.created += 1;
	}
	// Round up to a power of two so nearby sizes share sets.
	let capacity = len.next_power_of_two();
	let elem = precision_bytes(precision);
	let buffer = |label, size, usage| ctx.device.create_buffer(&BufferDescriptor { label: Some(label), size, usage, mapped_at_creation: false });
	Slot { precision, capacity, out: buffer("splimes output", capacity as u64 * elem, BufferUsages::STORAGE | BufferUsages::COPY_SRC), staging: buffer("splimes staging", capacity as u64 * elem, BufferUsages::MAP_READ | BufferUsages::COPY_DST), params: buffer("splimes params", 64, BufferUsages::UNIFORM | BufferUsages::COPY_DST), grid: buffer("splimes grid", capacity as u64 * 12, BufferUsages::STORAGE | BufferUsages::COPY_DST) }
}

fn release(ctx: &Context, slots: Vec<Slot>) {
	let mut pool = ctx.pool.lock().unwrap_or_else(PoisonError::into_inner);
	pool.idle.extend(slots);
	let mut total: u64 = pool.idle.iter().map(Slot::bytes).sum();
	while total > pool.max_bytes {
		// Free the largest first: it frees the most memory and is the least likely to fit
		// a typical call exactly.
		let Some((i, _)) = pool.idle.iter().enumerate().max_by_key(|(_, s)| s.capacity) else { break };
		total -= pool.idle.swap_remove(i).bytes();
		pool.evicted += 1;
	}
	drop(pool);
}

#[cfg(test)]
pub mod tests {
	use std::cell::Cell;

	use chrono::{DateTime, Utc};

	use super::{captured, eval, times};
	use crate::{Error, Precision, Resolution, Spline, kernel::Method, prepare::Knots, time::Grid};

	thread_local! {
		/// Makes this thread's next GPU calls fail, to test the fallback paths.
		pub static FAIL: Cell<bool> = const { Cell::new(false) };
		/// Loses the device after this thread's next chunk is submitted, before it is read
		/// back.
		pub static LOSE_BEFORE_POLL: Cell<Option<Loss>> = const { Cell::new(None) };
	}

	#[derive(Debug, Clone, Copy)]
	pub enum Loss {
		/// `Device::destroy`, which also destroys every buffer, so wgpu fails the read-back.
		Destroyed,
		/// The device-lost callback fires, as on a driver reset, with the buffers intact.
		Signalled,
	}

	/// A device lost while a chunk is in flight makes the call an error, never the staging
	/// buffer's previous contents, which wgpu 30 can hand back as a successful mapping
	/// (gfx-rs/wgpu#10301), and every later call fails fast.
	#[test]
	fn a_device_lost_mid_call_is_an_error() {
		for loss in [Loss::Destroyed, Loss::Signalled] {
			// A device of its own: losing it must not disturb the shared one.
			let Ok(ctx) = super::super::context::open() else {
				eprintln!("a_device_lost_mid_call_is_an_error: skipped, no GPU");
				return;
			};
			let at = |s: i64| DateTime::<Utc>::from_timestamp(s, 0).expect("valid");
			let run = |values: &[f64]| -> crate::Result<Vec<f64>> {
				let ts: Vec<_> = (0..).step_by(10).take(values.len()).map(at).collect();
				let knots = Knots::new(values.len(), |i| ts[i], |i| &values[i], false)?;
				let grid = Grid::new(at(0), ts[ts.len() - 1], Resolution::Seconds)?;
				let mut out = vec![0.0; grid.len];
				eval(&ctx, &knots, &grid, &Method::new(Spline::Linear, &knots), Precision::F32, &mut out).map(|()| out)
			};
			let rising: Vec<f64> = (0..100).map(f64::from).collect();
			let falling: Vec<f64> = rising.iter().rev().copied().collect();
			// The first call leaves its results in a pooled staging buffer, which the second,
			// the same size, reuses.
			let first = run(&rising).expect("the device works");
			assert!(first[0] < first[first.len() - 1], "normalised, but rising");
			LOSE_BEFORE_POLL.set(Some(loss));
			let second = run(&falling);
			LOSE_BEFORE_POLL.set(None);
			assert!(matches!(second, Err(Error::GpuUnavailable(_) | Error::Gpu(_))), "{loss:?}: a lost device returned values; the first call's: {:?}", second.map(|v| v == first));
			assert!(matches!(run(&rising), Err(Error::GpuUnavailable(_))), "{loss:?}: later calls fail fast");
		}
	}

	#[test]
	fn device_errors_become_errors_not_panics() {
		let Ok(ctx) = super::super::context::get() else {
			eprintln!("device_errors_become_errors_not_panics: skipped, no GPU");
			return;
		};
		// MAP_READ may only be combined with COPY_DST: a validation error, which wgpu would
		// otherwise turn into a panic.
		let result = captured(&ctx.device, || {
			let _ = ctx.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 16, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
			Ok(())
		});
		assert!(matches!(result, Err(Error::Gpu(ref m)) if m.contains("Validation") || m.contains("usage")), "{result:?}");
		// The device is still usable afterwards.
		assert!(captured(&ctx.device, || Ok(())).is_ok());
	}

	#[test]
	fn times_are_twos_complement_words() {
		let span = crate::time::posix_nanos(chrono::DateTime::<chrono::Utc>::MAX_UTC) - crate::time::posix_nanos(chrono::DateTime::<chrono::Utc>::MIN_UTC);
		for o in [0_i128, 1, -1, 4_294_967_296, -4_294_967_297, 1 << 62, span, -span] {
			let mut bytes = Vec::new();
			times(std::iter::once(o), &mut bytes);
			let words: &[u32] = bytemuck::cast_slice(&bytes);
			// Sign-extend the 96-bit value back to i128.
			let raw = i128::from(words[0]) | (i128::from(words[1]) << 32) | (i128::from(words[2]) << 64);
			let back = (raw << 32) >> 32;
			assert_eq!(back, o, "{o}");
		}
	}
}

//! Running the kernel: upload the knots, dispatch the grid in chunks, read back.

use std::{
	collections::VecDeque, sync::{PoisonError, atomic::Ordering, mpsc}, time::Duration
};

use rayon::prelude::*;
use wgpu::{BindGroupDescriptor, BindGroupEntry, Buffer, BufferDescriptor, BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor, ErrorFilter, MapMode, PollType};

use super::{BatchJob, GpuPoolStats, context::Context};
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
	let n_u32 = knot_count(ctx, n)?;

	let mut knot_u = Vec::with_capacity(n * pair_len);
	times(knots.offsets.iter().copied(), &mut knot_u);
	let mut knot_y = Vec::with_capacity(n * 8);
	knot_values(knots, precision, &mut knot_y);
	let knot_u = upload(ctx, "splimes knot times", &knot_u);
	let knot_y = upload(ctx, "splimes knot values", &knot_y);

	let chunk = chunk_points(ctx);

	let mut pending: VecDeque<(Slot, usize, usize, wgpu::SubmissionIndex)> = VecDeque::with_capacity(IN_FLIGHT);
	// Buffer sets whose chunk has been read back, ready for the next chunk: a call holds at
	// most `IN_FLIGHT` sets however long its grid.
	let mut finished: Vec<Slot> = Vec::with_capacity(IN_FLIGHT);
	let mut first = 0;
	while first < grid.len {
		let len = chunk.min(grid.len - first);
		let slot = finished.iter().position(|s| s.capacity >= len).map_or_else(|| acquire(ctx, precision, len, PARAMS_BYTES), |i| finished.swap_remove(i));

		let mut params = Vec::with_capacity(64);
		params_record(knots, method, precision, n_u32, len, &mut params);
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
		let entries = [BindGroupEntry { binding: 0, resource: knot_u.as_entire_binding() }, BindGroupEntry { binding: 1, resource: knot_y.as_entire_binding() }, sized(2, &slot.out, bytes), sized(3, &slot.params, PARAMS_BYTES), sized(4, &slot.grid, len as u64 * time_bytes)];
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

/// Grid points per dispatch: the configured chunk, clamped to what one dispatch and one
/// storage binding allow.
fn chunk_points(ctx: &Context) -> usize {
	#[allow(clippy::cast_possible_truncation)] // Bounded by MAX_CHUNK.
	let chunk = u64::from(ctx.config.chunk_points.max(1)).min(MAX_CHUNK).min(ctx.max_binding_bytes / 12) as usize;
	chunk
}

/// Appends the kernel's `Params` for `len` grid points of this series: the WGSL struct's
/// layout, 48 bytes in `f32` and 64 in `f64`.
fn params_record<V: Value>(knots: &Knots<'_, V>, method: &Method, precision: Precision, n: u32, len: usize, out: &mut Vec<u8>) {
	// Times are nanoseconds in both kernels, scaled by 1/h.
	let inv_h = 1.0 / knots.h;
	let (lo, hi) = method.bounds.unwrap_or((0.0, 0.0));
	#[allow(clippy::cast_possible_truncation)] // `len <= MAX_CHUNK`, which fits u32.
	let header = [n, method.window as u32, u32::from(method.hold), u32::from(method.bounds.is_some()), len as u32, 0, 0, 0];
	out.extend_from_slice(bytemuck::cast_slice(&header));
	match precision {
		Precision::F64 => out.extend_from_slice(bytemuck::cast_slice(&[inv_h, lo, hi, 0.0])),
		#[allow(clippy::cast_possible_truncation)] // Single precision is what was asked for.
		Precision::F32 => out.extend_from_slice(bytemuck::cast_slice(&[inv_h as f32, lo as f32, hi as f32, 0.0])),
	}
}

/// Appends the normalised knot values in the kernel's precision.
fn knot_values<V: Value>(knots: &Knots<'_, V>, precision: Precision, out: &mut Vec<u8>) {
	match precision {
		Precision::F64 => out.extend_from_slice(bytemuck::cast_slice(&knots.y)),
		#[allow(clippy::cast_possible_truncation)] // Single precision is what was asked for.
		Precision::F32 => out.extend(knots.y.iter().flat_map(|&v| (v as f32).to_ne_bytes())),
	}
}

/// The refusal [`dispatch`] gives for knots the device can't bind, if any, and the knot
/// count as the kernel's `u32`.
fn knot_count(ctx: &Context, n: usize) -> Result<u32> {
	if n as u64 * 12 > ctx.max_binding_bytes {
		return Err(Error::Gpu(format!("{n} knots exceed {}'s storage-binding limit of {} bytes", ctx.info.name, ctx.max_binding_bytes)));
	}
	u32::try_from(n).map_err(|_| Error::Gpu(format!("{n} knots exceed the GPU kernel's u32 indexing")))
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
	at(binding, buffer, 0, size)
}

/// A binding of `size` bytes of `buffer` from byte `offset`.
const fn at(binding: u32, buffer: &Buffer, offset: u64, size: u64) -> BindGroupEntry<'_> {
	BindGroupEntry { binding, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer, offset, size: std::num::NonZeroU64::new(size) }) }
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

/// Series one batched submission holds at most: bounds its parameter buffer (one aligned
/// record per series, 1 MiB at the usual 256-byte alignment) and its bind groups.
const MAX_BATCH_SERIES: usize = 4096;

/// Runs many series with as few submissions as the device allows.
///
/// Every series whose grid fits in one chunk is packed with others, in order, into one
/// command encoder — one dispatch and bind group per series, sharing one set of buffers at
/// aligned offsets — one queue submission and one read-back mapping, up to a chunk of grid
/// points, [`MAX_BATCH_SERIES`] series or a storage binding of knots per submission. Two
/// submissions are in flight at once, as for the chunks of one call. A series whose grid
/// is larger than a chunk runs alone, through [`eval`].
///
/// The kernel, its inputs and its parameters are exactly those of [`eval`] on the same
/// series, so every result is bit-identical to a call of its own. Results are per series:
/// a refused series (too many knots) doesn't affect the others, but a device error fails
/// every series packed into the same submissions.
pub fn eval_batch<V: Value>(ctx: &Context, jobs: &[BatchJob<'_, '_, V>], precision: Precision, outs: &mut [&mut [f64]]) -> Vec<Result<()>> {
	#[cfg(test)]
	if tests::FAIL.get() {
		return vec![Err(Error::Gpu("injected failure".to_owned())); jobs.len()];
	}
	let kernel = match ctx.kernel(precision) {
		Ok(kernel) => kernel,
		Err(e) => return vec![Err(e); jobs.len()],
	};
	let chunk = chunk_points(ctx);
	let mut results: Vec<Result<()>> = vec![Ok(()); jobs.len()];
	let mut packed = Vec::with_capacity(jobs.len());
	let mut alone = Vec::new();
	for (i, job) in jobs.iter().enumerate() {
		match knot_count(ctx, job.knots.len()) {
			Err(e) => results[i] = Err(e),
			Ok(_) if job.grid.len > chunk => alone.push(i),
			Ok(n) => packed.push((i, n)),
		}
	}
	let layout = Layout::new(ctx, precision);
	let groups = layout.plan(jobs, &packed, chunk, ctx.max_binding_bytes);
	if !groups.is_empty() {
		match captured(&ctx.device, || submit_groups(ctx, kernel, &layout, jobs, &groups, precision, outs)) {
			Ok(finished) => release(ctx, finished),
			Err(e) => {
				for piece in groups.iter().flat_map(|g| &g.pieces) {
					results[piece.job] = Err(e.clone());
				}
			}
		}
	}
	for i in alone {
		results[i] = eval(ctx, jobs[i].knots, jobs[i].grid, jobs[i].method, precision, outs[i]);
	}
	results
}

/// Offset alignments of one batched submission's shared buffers.
struct Layout {
	/// Grid points each series' output and grid times start on a multiple of.
	points: usize,
	/// Bytes each series' knot times and values start on a multiple of.
	storage: u64,
	/// Bytes between parameter records.
	uniform: u64,
}

/// One series' place in a batched submission.
struct Piece {
	job: usize,
	/// Knots, as the kernel's `u32`.
	n: u32,
	/// Byte offsets of its knot times and values.
	knot_times: u64,
	knot_values: u64,
	/// Its first grid point in the shared output and grid-time buffers.
	first: usize,
	len: usize,
}

/// The series of one submission, and the extent of its shared buffers.
struct Group {
	pieces: Vec<Piece>,
	/// Grid points up to the end of the last series.
	points: usize,
	knot_time_bytes: u64,
	knot_value_bytes: u64,
}

const fn align_up(x: u64, to: u64) -> u64 {
	x.div_ceil(to) * to
}

const fn gcd(a: u64, b: u64) -> u64 {
	if b == 0 { a } else { gcd(b, a % b) }
}

impl Layout {
	fn new(ctx: &Context, precision: Precision) -> Self {
		let limits = ctx.device.limits();
		let storage = u64::from(limits.min_storage_buffer_offset_alignment.max(4));
		// The fewest points whose grid times (12 bytes each) and outputs both span a whole
		// number of alignments.
		let times = storage / gcd(storage, 12);
		let values = storage / gcd(storage, precision_bytes(precision));
		#[allow(clippy::cast_possible_truncation)] // At most an alignment, a few hundred.
		let points = (times / gcd(times, values) * values) as usize;
		Self { points, storage, uniform: align_up(PARAMS_BYTES, u64::from(limits.min_uniform_buffer_offset_alignment.max(4))) }
	}

	/// Packs `packed` (series index, knot count), in order, into submissions.
	fn plan<V: Value>(&self, jobs: &[BatchJob<'_, '_, V>], packed: &[(usize, u32)], chunk: usize, max_binding_bytes: u64) -> Vec<Group> {
		let mut groups: Vec<Group> = Vec::new();
		let mut current = Group { pieces: Vec::new(), points: 0, knot_time_bytes: 0, knot_value_bytes: 0 };
		for &(job, n) in packed {
			let len = jobs[job].grid.len;
			let first = self.align_points(current.points);
			let knot_times = align_up(current.knot_time_bytes, self.storage);
			let knot_values = align_up(current.knot_value_bytes, self.storage);
			let time_bytes = u64::from(n) * 12;
			let full = first + len > chunk || current.pieces.len() >= MAX_BATCH_SERIES || knot_times + time_bytes > max_binding_bytes;
			if full && !current.pieces.is_empty() {
				groups.push(std::mem::replace(&mut current, Group { pieces: Vec::new(), points: 0, knot_time_bytes: 0, knot_value_bytes: 0 }));
				current.pieces.push(Piece { job, n, knot_times: 0, knot_values: 0, first: 0, len });
				current.points = len;
				current.knot_time_bytes = time_bytes;
				current.knot_value_bytes = u64::from(n) * 8;
				continue;
			}
			current.pieces.push(Piece { job, n, knot_times, knot_values, first, len });
			current.points = first + len;
			current.knot_time_bytes = knot_times + time_bytes;
			current.knot_value_bytes = knot_values + u64::from(n) * 8;
		}
		if !current.pieces.is_empty() {
			groups.push(current);
		}
		groups
	}

	const fn align_points(&self, points: usize) -> usize {
		points.div_ceil(self.points) * self.points
	}
}

/// Encodes, submits and reads back every group, two in flight at once, and returns the
/// buffer sets for the pool.
fn submit_groups<V: Value>(ctx: &Context, kernel: &super::context::Kernel, layout: &Layout, jobs: &[BatchJob<'_, '_, V>], groups: &[Group], precision: Precision, outs: &mut [&mut [f64]]) -> Result<Vec<Slot>> {
	let elem = precision_bytes(precision);
	let mut pending: VecDeque<(Slot, &Group, wgpu::SubmissionIndex)> = VecDeque::with_capacity(IN_FLIGHT);
	let mut finished: Vec<Slot> = Vec::with_capacity(IN_FLIGHT);
	for group in groups {
		let params_bytes = group.pieces.len() as u64 * layout.uniform;
		let slot = finished.iter().position(|s| s.capacity >= group.points && s.params_bytes >= params_bytes).map_or_else(|| acquire(ctx, precision, group.points, params_bytes), |i| finished.swap_remove(i));

		// Knots, parameters and grid times, each series at its aligned offset.
		#[allow(clippy::cast_possible_truncation)] // Bounded by the storage-binding limit.
		let mut knot_t = Vec::with_capacity(group.knot_time_bytes as usize);
		#[allow(clippy::cast_possible_truncation)]
		let mut knot_y = Vec::with_capacity(group.knot_value_bytes as usize);
		#[allow(clippy::cast_possible_truncation)] // At most MAX_BATCH_SERIES records.
		let mut params = Vec::with_capacity(params_bytes as usize);
		for (k, piece) in group.pieces.iter().enumerate() {
			let job = &jobs[piece.job];
			#[allow(clippy::cast_possible_truncation)]
			{
				knot_t.resize(piece.knot_times as usize, 0);
				knot_y.resize(piece.knot_values as usize, 0);
				params.resize(k * layout.uniform as usize, 0);
			}
			times(job.knots.offsets.iter().copied(), &mut knot_t);
			knot_values(job.knots, precision, &mut knot_y);
			params_record(job.knots, job.method, precision, piece.n, piece.len, &mut params);
		}
		let knot_t = upload(ctx, "splimes batch knot times", &knot_t);
		let knot_y = upload(ctx, "splimes batch knot values", &knot_y);
		ctx.queue.write_buffer(&slot.params, 0, &params);
		let mut grid = vec![0_u8; group.points * 12];
		{
			// Disjoint regions, one per series, filled in parallel.
			let mut regions = Vec::with_capacity(group.pieces.len());
			let mut rest = grid.as_mut_slice();
			let mut at = 0;
			for piece in &group.pieces {
				let (_, tail) = rest.split_at_mut((piece.first - at) * 12);
				let (region, tail) = tail.split_at_mut(piece.len * 12);
				regions.push((region, piece));
				rest = tail;
				at = piece.first + piece.len;
			}
			regions.into_par_iter().for_each(|(region, piece)| {
				let job = &jobs[piece.job];
				let mut bytes = Vec::with_capacity(region.len());
				times(job.grid.offsets(job.knots.t0, 0).take(piece.len), &mut bytes);
				region.copy_from_slice(&bytes);
			});
		}
		ctx.queue.write_buffer(&slot.grid, 0, &grid);

		let bind_groups: Vec<wgpu::BindGroup> = group
			.pieces
			.iter()
			.enumerate()
			.map(|(k, piece)| {
				let entries = [at(0, &knot_t, piece.knot_times, u64::from(piece.n) * 12), at(1, &knot_y, piece.knot_values, u64::from(piece.n) * elem), at(2, &slot.out, piece.first as u64 * elem, piece.len as u64 * elem), at(3, &slot.params, k as u64 * layout.uniform, PARAMS_BYTES), at(4, &slot.grid, piece.first as u64 * 12, piece.len as u64 * 12)];
				ctx.device.create_bind_group(&BindGroupDescriptor { label: Some("splimes batch"), layout: &kernel.layout, entries: &entries })
			})
			.collect();
		let mut encoder = ctx.device.create_command_encoder(&CommandEncoderDescriptor { label: Some("splimes batch") });
		{
			let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor { label: Some("splimes batch"), timestamp_writes: None });
			pass.set_pipeline(&kernel.pipeline);
			for (bind_group, piece) in bind_groups.iter().zip(&group.pieces) {
				pass.set_bind_group(0, bind_group, &[]);
				#[allow(clippy::cast_possible_truncation)] // At most a chunk.
				pass.dispatch_workgroups((piece.len as u32).div_ceil(WORKGROUP), 1, 1);
			}
		}
		encoder.copy_buffer_to_buffer(&slot.out, 0, &slot.staging, 0, group.points as u64 * elem);
		let submission = ctx.queue.submit(Some(encoder.finish()));
		pending.push_back((slot, group, submission));
		if pending.len() >= IN_FLIGHT
			&& let Some(job) = pending.pop_front()
		{
			finished.push(read_back_group(ctx, job, precision, outs)?);
		}
	}
	while let Some(job) = pending.pop_front() {
		finished.push(read_back_group(ctx, job, precision, outs)?);
	}
	Ok(finished)
}

/// Waits for one batched submission with a single mapping and copies each series'
/// results into its own output.
fn read_back_group(ctx: &Context, (slot, group, submission): (Slot, &Group, wgpu::SubmissionIndex), precision: Precision, outs: &mut [&mut [f64]]) -> Result<Slot> {
	let elem = precision_bytes(precision);
	let slice = slot.staging.slice(..group.points as u64 * elem);
	let (tx, rx) = mpsc::channel();
	slice.map_async(MapMode::Read, move |r| {
		let _ = tx.send(r);
	});
	ctx.device.poll(PollType::Wait { submission_index: Some(submission), timeout: Some(CHUNK_TIMEOUT) }).map_err(|e| Error::Gpu(format!("waiting for the GPU: {e}")))?;
	rx.recv().map_err(|_| Error::Gpu("buffer mapping was abandoned".to_owned()))?.map_err(|e| Error::Gpu(format!("mapping the result buffer: {e}")))?;
	// As in `read_back`: a lost device can report a successful mapping of stale contents.
	if ctx.lost.load(Ordering::Acquire) {
		return Err(Error::GpuUnavailable("the GPU device was lost during the call".to_owned()));
	}
	{
		let view = slice.get_mapped_range().map_err(|e| Error::Gpu(format!("reading the result buffer: {e}")))?;
		// The group's series in index order, as packed, paired with their outputs.
		let mut wanted = group.pieces.iter().map(|p| p.job).peekable();
		let mut dests: Vec<&mut [f64]> = Vec::with_capacity(group.pieces.len());
		for (i, out) in outs.iter_mut().enumerate() {
			if wanted.peek() == Some(&i) {
				dests.push(&mut **out);
				wanted.next();
			}
		}
		let copy = |(dest, piece): (&mut [f64], &Piece)| {
			let dest = &mut dest[..piece.len];
			match precision {
				Precision::F64 => dest.copy_from_slice(&bytemuck::cast_slice::<u8, f64>(&view)[piece.first..piece.first + piece.len]),
				Precision::F32 => {
					for (d, &s) in dest.iter_mut().zip(&bytemuck::cast_slice::<u8, f32>(&view)[piece.first..piece.first + piece.len]) {
						*d = f64::from(s);
					}
				}
			}
		};
		dests.into_par_iter().zip(group.pieces.par_iter()).with_min_len(16).for_each(copy);
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
	/// Bytes of the params buffer: [`PARAMS_BYTES`] for a call, one aligned record per
	/// series for a batch.
	params_bytes: u64,
	out: Buffer,
	staging: Buffer,
	params: Buffer,
	/// Grid times, uploaded per chunk.
	grid: Buffer,
}

impl Slot {
	const fn bytes(&self) -> u64 {
		// Output and staging values, plus a 12-byte time per point, plus the params.
		self.capacity as u64 * (2 * precision_bytes(self.precision) + 12) + self.params_bytes
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

/// Bytes of one call's kernel parameters (the WGSL `Params` struct, padded).
const PARAMS_BYTES: u64 = 64;

/// A buffer set for `len` grid points and `params_bytes` of parameters, from the pool if
/// one fits.
fn acquire(ctx: &Context, precision: Precision, len: usize, params_bytes: u64) -> Slot {
	{
		let mut pool = ctx.pool.lock().unwrap_or_else(PoisonError::into_inner);
		// The smallest idle set that fits, so a small call doesn't take a huge set.
		if let Some(i) = pool.idle.iter().enumerate().filter(|(_, s)| s.precision == precision && s.capacity >= len && s.params_bytes >= params_bytes).min_by_key(|(_, s)| (s.capacity, s.params_bytes)).map(|(i, _)| i) {
			pool.reused += 1;
			return pool.idle.swap_remove(i);
		}
		pool.created += 1;
	}
	// Round up to a power of two so nearby sizes share sets.
	let capacity = len.next_power_of_two();
	let params_bytes = params_bytes.next_power_of_two();
	let elem = precision_bytes(precision);
	let buffer = |label, size, usage| ctx.device.create_buffer(&BufferDescriptor { label: Some(label), size, usage, mapped_at_creation: false });
	Slot { precision, capacity, params_bytes, out: buffer("splimes output", capacity as u64 * elem, BufferUsages::STORAGE | BufferUsages::COPY_SRC), staging: buffer("splimes staging", capacity as u64 * elem, BufferUsages::MAP_READ | BufferUsages::COPY_DST), params: buffer("splimes params", params_bytes, BufferUsages::UNIFORM | BufferUsages::COPY_DST), grid: buffer("splimes grid", capacity as u64 * 12, BufferUsages::STORAGE | BufferUsages::COPY_DST) }
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

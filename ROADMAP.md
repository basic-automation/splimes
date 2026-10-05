# splimes roadmap

The work queue for `splimes`, independent of [WeftDB](https://github.com/basic-automation/weftdb)'s.
WeftDB is a consumer of this crate, and **a stable WeftDB waits on a stable splimes**,
so the 1.0 criteria below are on WeftDB's critical path. Every item is a `[ ]` (to do)
or `[x]` (done) checkbox, and git history is the record of how each one landed.

**What splimes is:** resampling irregularly sampled series onto a regular grid with
spline interpolation, choosing between SIMD CPU, parallel CPU and GPU automatically,
over `BigDecimal` values.
**What it is not:** a storage engine, a query language, or a general numerics library.

---

## 0.1.0: first public release

- [x] Extracted from the WeftDB workspace with its history (2026-10-05)
- [x] Standalone manifest: explicit versions, MSRV 1.95, crate metadata
- [x] Test helper `Point::random` and its `fake` dependency made test-only, so they no longer ship to users
- [x] `System::new_all()` replaced with a memory-only `System` on every interpolation path: **~445 ms → ~38 µs per call**. It enumerated every process, disk and network interface just to read total memory; the library test suite went from 46 s to 2.3 s
- [x] Fixed the f32 quadratic WGSL shader, which didn't parse, so quadratic GPU interpolation panicked on adapters without f64 (Apple, integrated GPUs, Windows WARP)
- [x] Every WGSL shader is parsed and validated with naga in unit tests, so no GPU is needed
- [x] README example compiled and run as a doctest; the previously `ignore`d API examples are compile-checked (one was broken)
- [x] CI: rustfmt, clippy (defect categories), MSRV, tests on Linux (lavapipe, GPU required) / macOS / Windows (WARP, the f32 path), rustdoc, `cargo package`, cargo-deny
- [ ] Publish 0.1.0 to crates.io
- [ ] Switch WeftDB from the in-tree copy to `splimes = "0.1"`

---

## 1.0 release criteria

1.0 is a semver promise: the public API, the numerical behaviour, and the MSRV policy.
splimes ships 1.0 when every box below is ticked.

### Correctness

- [ ] **Graceful GPU failure.** No wgpu error handler is installed, so a validation or out-of-memory error **panics** (wgpu's default). Capture errors (`push_error_scope` / `on_uncaptured_error`) and fall back to the CPU path with a logged reason
- [ ] **No panics in library code.** 38 `unwrap()`/`expect()` calls outside tests, including the global interpolator's `LazyLock` initialiser. Each becomes a typed error or a documented invariant
- [ ] **Numerical tolerance contract.** Publish the CPU-vs-GPU and f64-vs-f32 tolerances per spline, enforced by tests, plus the NaN/∞ behaviour at the edges
- [ ] **Extrapolation semantics documented and tested** for every method (linear tails, `Polynomial` `bounds_factor`, the 1-point and 2-point cases)
- [ ] **Fallback transparency.** `auto_interpolate` silently steps down from cubic → quadratic → linear when there are too few points. Report the method actually used

### API

- [ ] **Typed errors.** 51 public functions return `anyhow::Result`; a library should return `splimes::Error` so callers can match on failures. `anyhow` leaves the public API
- [ ] **Public surface reviewed and trimmed.** `pub mod helpers` exposes internals (batch state, target-time iterator); `apply_fast_path`, `BASE_BATCH_SIZE`, `POINT_SIZE` and the `SECONDS_IN_*` constants need a keep/hide decision. Everything kept is documented: 64 public items lack docs today, and `#![warn(missing_docs)]` becomes a gate
- [ ] **Provenance labels.** Return whether each output point is raw, interpolated or extrapolated. WeftDB does this today in its server layer; it belongs here, beside the code that knows
- [ ] **Async story.** Decide whether the API stays `async` (and tied to `tokio`) or offers a synchronous core with an async wrapper; the GPU path currently spins up its own runtime on a helper thread
- [ ] **`GpuConfig::max_command_batch_size`** is reserved and has no effect. Implement it (see command batching below) or remove it before the freeze
- [ ] **`cargo-semver-checks` in CI** from the first 0.x release, so accidental breakage is caught

### Performance (each claim backed by a published benchmark)

- [ ] **Command batching**: submit many small batches per queue submission
- [ ] **True async GPU handles**: today `GpuInterpolationResult` computes synchronously
- [ ] **CPU/GPU overlap**: chunked upload → kernel → streamed readback
- [ ] **Hardware auto-tuning**: calibrate the CPU/GPU break-even point per machine instead of fixed thresholds in `should_use_gpu`
- [ ] **`BigDecimal` conversion cost** measured on the hot path; typed `f64` input as an option when the caller doesn't need decimal precision
- [ ] **GPU memory stability**: buffer-pool behaviour under repeated calls and mixed batch sizes
- [ ] Benchmarks published with hardware, versions and reproduction steps

### Portability

- [ ] **Conformance matrix**: NVIDIA / AMD / Intel / Apple / software adapters × f64 / f32, with measured numerical drift
- [ ] **Real-GPU CI**: today CI covers lavapipe (Vulkan, f64) and WARP (D3D12, f32); add Metal and at least one hardware runner

### Release engineering

- [x] CI, cargo-deny, MSRV check
- [x] CHANGELOG (Keep a Changelog) and SECURITY.md
- [ ] MSRV policy written down (bumps are minor-version changes before 1.0, breaking after)
- [ ] Release automation (tag → `cargo publish`), mirroring WeftDB's

---

## Later / not before 1.0

- [ ] More methods: Akima and monotone (PCHIP) splines, which avoid cubic overshoot on step-like data
- [ ] Multi-GPU *(only once single-GPU wins are proven)*
- [ ] Downsampling and aggregation stay in WeftDB (`weft-reduce`) unless another consumer asks for them here

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
- [x] Published 0.1.0 to crates.io (2026-10-05)
- [x] ~~Switch WeftDB from the in-tree copy to `splimes = "0.1"`~~ *Superseded* by the switch to `splimes = "1"` (below)

---

## 1.0 release criteria

1.0 is a semver promise: the public API, the numerical behaviour, and the MSRV policy.
splimes ships 1.0 when every box below is ticked, or the item is explicitly dropped or
deferred with a reason.

**Shipped as 1.0.0 on 2026-10-05.** Three items are deferred past 1.0: two portability
items, which need hardware the project doesn't have, and WeftDB's switch, which needed
the published crate and is in progress.

### Correctness

- [x] **Graceful GPU failure.** Every call runs inside wgpu error scopes (validation, out of memory, internal), an uncaptured-error handler logs stragglers instead of panicking, and a device-lost callback makes later calls fail fast. `Backend::Auto` reruns on the CPU, logs the reason and reports it in `Interpolation::gpu_fallback`. splimes never opens the device on a thread of its own (exiting mid-initialisation crashed some drivers); points the `f32` kernel can't compute reliably are recomputed in `f64` and counted
- [x] **No panics in library code.** No `unwrap`/`expect` outside tests; lock poisoning is recovered, allocation failure is `Error::OutputTooLarge`, and `#![forbid(unsafe_code)]`
- [x] **Numerical tolerance contract.** `|v − exact| ≤ bound · range · Λ(t) + ε·|exact|`, with Λ the window's Lebesgue function and `bound` 1e-13 (`f64`, CPU and GPU) / 1e-5 (GPU `f32`), for every method, inside the data and out. Published in the crate docs and enforced by `tests/contract.rs` against an exact 60-digit reference that computes Λ, on randomised, dense-burst and geometric-gap series and a two-million-knot series. Non-finite results are `Error::NonFiniteResult`, never values
- [x] **Exact time.** Every time difference is taken in integer nanoseconds (96-bit integer arithmetic on the GPU, immune to fast-math compilers), on the POSIX scale — leap seconds fold forward — so accuracy doesn't depend on the series' length or spacing
- [x] **Extrapolation semantics documented and tested** for every method, including `bounds_factor` and the 1- and 2-point cases (`Spline` docs, `tests/api.rs`)
- [x] **Fallback transparency.** `Interpolation::spline()` reports the method actually used; `Interpolator::exact(true)` refuses to step down. The silent size-based downgrades (cubic → linear above 5,000 points) are gone

### API

- [x] **Typed errors.** `splimes::Error` (`#[non_exhaustive]`); `anyhow` is no longer a dependency
- [x] **Public surface reviewed and trimmed.** `helpers`, `apply_fast_path`, `BASE_BATCH_SIZE`, `POINT_SIZE` and the time constants are gone; `#![warn(missing_docs)]` is gated by CI's `-D warnings`
- [x] **Provenance labels.** `PointKind::{Raw, Interpolated, Extrapolated}` per output point; raw points return the input value exactly
- [x] **Async story.** A synchronous core, with `run_async` / `run_f64_async` on tokio's blocking pool behind the optional `tokio` feature. tokio is no longer a required dependency
- [x] **`GpuConfig::max_command_batch_size`** removed (see command batching below)
- [x] **`cargo-semver-checks` in CI**, against the latest crates.io release

### Performance (each claim backed by a published benchmark)

- [x] ~~**Command batching**~~ *Dropped.* A call is one dispatch per chunk of up to 16 Mi points, not many small submissions, so there is nothing to batch; the measured cost is output assembly, not submission (BENCHMARKS.md)
- [x] ~~**True async GPU handles**~~ *Dropped* with the synchronous core: async callers use `run_async`
- [x] **CPU/GPU overlap**: two chunks in flight, reading back one while the next computes
- [x] **Hardware auto-tuning**: `calibrate()` measures the CPU, rayon and GPU crossovers and sets `AutoThresholds`; `set_auto_thresholds` overrides them
- [x] **`BigDecimal` conversion cost** measured (about 3–4× an `f64` call, BENCHMARKS.md) and cut by more than half; typed `f64` input and output with `run_f64`
- [x] **GPU memory stability**: a bounded buffer pool, and at most two buffer sets per call however long the grid; `tests/gpu.rs` checks both
- [x] Benchmarks published with hardware, versions and reproduction steps (BENCHMARKS.md)

### Portability

- [ ] **Conformance matrix** *(deferred past 1.0: needs the hardware)*: measured on NVIDIA (f64 and f32) and the CPU. CI measures lavapipe (f64 and f32) on every run, and WARP and Metal (f32) when the runners expose them, as they did for 1.0.0. The macOS runner's Metal device gave the first Apple result: worst 3.1e-7 on the conformance matrix, against the 1e-5 bound (BENCHMARKS.md). **Still missing: AMD and Intel GPUs, and Apple silicon outside a virtual machine**
- [ ] **Real-GPU CI** *(deferred past 1.0)*: needs a self-hosted or GPU runner; CI covers software and virtualised adapters only

### Release engineering

- [x] CI, cargo-deny, MSRV check, minimal-versions check of the dependency floors
- [x] CHANGELOG (Keep a Changelog) and SECURITY.md
- [x] MSRV policy written down: raising it is a minor-version change, only to a toolchain at least six months old (README, crate docs)
- [x] Release automation: a `vX.Y.Z` tag runs CI, checks the tag against `Cargo.toml` and the changelog, publishes to crates.io with trusted publishing, and creates the GitHub release
- [x] **Adversarial review.** Two multi-agent review passes (six dimensions, each finding checked by three independent skeptics, then a per-fix verification and regression hunt) confirmed and fixed 33 distinct defects (and a dozen smaller residuals), from silent wrong values and process crashes to test gaps; each fix's regression test is in the suite
- [x] Merged `release/1.0` ([#1](https://github.com/basic-automation/splimes/pull/1))
- [x] crates.io trusted publishing for `release.yml`, the only way to publish: the crate refuses new versions published with an API token, and the GitHub `release` environment deploys only from `v*` tags
- [x] Published 1.0.0 to crates.io from the `v1.0.0` tag (2026-10-05)
- [ ] Switch WeftDB to `splimes = "1"` ([MIGRATING.md](MIGRATING.md)) *(deferred past 1.0: it needs the published crate; in progress)*

---

## Later / not before 1.0

- [ ] More methods: Akima and monotone (PCHIP) splines, which avoid cubic overshoot on step-like data
- [ ] Multi-GPU *(only once single-GPU wins are proven)*
- [ ] Faster exact time differences: an `i64` fast path for series spanning under 292 years, which is nearly all of them (single-threaded degree 8 is about 2× linear today)
- [ ] Parallel input preparation for million-point inputs (sorting and de-duplicating is single-threaded)
- [ ] Runtime-agnostic async: a future any executor can await (async-std, smol, `futures`), not just tokio. Spawn `run` on rayon's pool and complete a oneshot future built on `std` (`Mutex` and `Waker`), so it adds no dependency and needs no feature flag. It needs a new name (`run_async` is the tokio wrapper's). Additive, so a 1.x minor release; the tokio wrappers stay
- [ ] Downsampling and aggregation stay in WeftDB (`weft-reduce`) unless another consumer asks for them here

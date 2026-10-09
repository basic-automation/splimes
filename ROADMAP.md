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

## 1.1.0: runtime-agnostic async and faster preparation (2026-10-08)

Published to crates.io from the `v1.1.0` tag by trusted publishing, from [#3](https://github.com/basic-automation/splimes/pull/3) and [#4](https://github.com/basic-automation/splimes/pull/4). It shipped `Interpolator::spawn` / `spawn_f64` with GPU-safe drop and exit, parallel input preparation, the `i64` kernel and cheaper output timestamps (the items marked 1.1.0 under "Later" below), and the hardening tests ticked under "1.x" below. What each method computes is unchanged: the release run's conformance tables on CI's three adapters match 1.0.0's digit for digit (BENCHMARKS.md).

- [x] Merged [#3](https://github.com/basic-automation/splimes/pull/3) (the daily routine's run) and released only after review: #3 was merged with a `spawn` crash at exit, and [#4](https://github.com/basic-automation/splimes/pull/4) fixed it over three adversarial review rounds before the tag. Review routine PRs before merging them

---

## 1.x: hardening shipped 1.0 behaviour

Additive work on the 1.0 contract: more evidence for what is already promised, and no
change to what any method computes.

- [x] **Randomised property tests of the input contract** on every backend and precision (`tests/properties.rs`): 1,024 generated series, unsorted with duplicate instants, single points, constant values and decimals beyond `f64`'s digits, checked for order independence (bit for bit), provenance labels and raw values (every digit), the reported step-down and `exact(true)`, the grid's shape, `run`/`run_f64` agreement and the published bound against the exact reference
- [x] **GPU time at the extremes of chrono's range** (`tests/gpu.rs`): knots spanning all ~524,000 years, and microsecond knots a second before `DateTime::MAX_UTC` beside one at `MIN_UTC`, match the CPU in `f64` and `f32` (with the `f32` windows handed to `f64`). Until now, a kernel that dropped the 96-bit time's high word passed every test
- [x] **Text and serde round trips of `Spline`** (`tests/api.rs`): every degree with ~500 random bounds factors (subnormal, `f64::MAX`, `-0.0`) through `Display`/`FromStr` and serde_json, bit for bit, and malformed text refused. It found that serde_json's default float parser reads some bounds factors back an ulp off (e.g. `1.5259628951211795e-293`); exact with its `float_roundtrip` feature, which the tests now enable and the `Spline` docs recommend
- [x] **`spawn` on the GPU** (`tests/gpu.rs`): sixteen `Backend::Gpu` futures in flight at once on rayon's pool, awaited on one thread, each matching the CPU
- [x] **This release's fast paths change nothing** (`tests/api.rs`): 100,000 shuffled inputs with every instant duplicated give bit-identical results on `Cpu` (serial preparation) and `Parallel` (parallel); a 300-year grid (the `i128` kernel) and a 44-year one (`i64`) give bit-identical values at their shared instants
- [x] **`f32` windows at the 1,024× gap limit** (`tests/contract.rs`): just inside it (1,000×) the `f32` kernel computes every window and stays within 1.6e-7 of the 1e-5 bound; just outside (1,100×) the windows go to `f64`. The fixtures stopped at 60×
- [ ] WeftDB persists `Spline`: if it stores it as JSON through serde_json, enable serde_json's `float_roundtrip` feature there, or a stored `Polynomial` bounds factor can come back an ulp off *(owner: WeftDB repository)*

---

## Later / not before 1.0

- [ ] More methods: Akima and monotone (PCHIP) splines, which avoid cubic overshoot on step-like data. Both are local cubic Hermite schemes that fit the existing window machinery: PCHIP's Fritsch–Carlson slopes (zero where neighbouring secants change sign, else a spacing-weighted harmonic mean; one-sided ends) need knots `i-1 … i+2` for a segment, the same 4-knot window as `Cubic` ([SciPy `PchipInterpolator`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html)); modified Akima ("makima") weights `w = |δ_{i+1} − δ_i| + |δ_{i+1} + δ_i|/2` avoid Akima's 0/0 on equal slopes and its overshoot on flat runs, and synthesise end slopes by `δ₀ = 2δ₁ − δ₂`, a 6-knot window ([Moler, "Makima piecewise cubic interpolation"](https://blogs.mathworks.com/cleve/2019/04/29/makima-piecewise-cubic-interpolation/)). Both fit `MAX_WINDOW`. **Needs an owner decision first:** neither is linear in the input values (PCHIP's limiter, Akima's weights), so the published `bound · range · Λ(t)` contract has no `Λ` for them. Options: a bound in `range` alone, from the boundedness of the Hermite basis on a segment (`|v − exact| ≤ c · bound · range` inside the data), or the sensitivity of the frozen-branch scheme as `Λ`. **Evidence (2026-10-09), for PCHIP:** a bound in `range` alone can't hold on irregular spacing. Take a narrow gap beside a wide one, where the narrow segment's secant is much smaller than the wide one's. The Fritsch–Carlson slope at the shared knot then moves by up to ~1.5× any change in the narrow secant, and that change is scaled by the wide segment's width. So the interpolant's sensitivity to its inputs, `Λ = Σⱼ |∂v/∂yⱼ|` with the limiter's branch frozen, grows like ~0.44 × the gap ratio. An exact-rational study (SciPy's slopes, emulating the kernel's normalised `f64` evaluation on exact integer time differences; script in the 2026-10-09 routine PR) measured worst `err / range` of 2e-16 on random series, 9e-14 at a 10⁴ gap ratio and 2.7e-9 at 10⁸. That breaks a 1e-13 range-only bound. Measured as `err / (range · Λ)`, every case stayed within 4.5e-16, the same few ulps as the existing methods. So the published `bound · range · Λ(t)` form carries over unchanged if `Λ` is defined as the frozen-branch sensitivity, which the 60-digit reference can compute by forward differentiation. That is the recommended choice. makima, under the same study, also needs that `Λ` (worst `err / range` 3.2e-9 at a 10⁸ ratio), but against `range · Λ` it reached 3.8e-14 (at 10⁴): inside 1e-13, with about 3× headroom instead of PCHIP's ~200×. So makima's weights need a better-conditioned evaluation, or a closer error analysis, before they can share the bound. Then: define extrapolation (SciPy extends the end segment's cubic; `Cubic` holds), add a 60-digit reference to `tests/common`, and land CPU and GPU together, since every backend computes one formula
- [ ] Raise `MAX_POLYNOMIAL_DEGREE` above 8. Allowing more is additive, so it can ship in a 1.x minor release. The cap is about usefulness: noise amplification on evenly spaced knots is about 11× at degree 8, 30× at 10 and 500× at 15, and worse on irregular spacing. Constants derived for degree 8 also limit it: `F32_GAP_RATIO` (1024⁸ = 2⁸⁰ inside `f32`; degree 12 reaches 2¹²⁰), `SCALE_FROM` (256⁸ = 2⁶⁴; degree 16 overflows `f32`), and `MAX_WINDOW`, which sizes the CPU and WGSL stack arrays for every method. Before raising it: derive those bounds per window instead of fixing them, measure the cost of larger windows to linear and cubic and GPU register use, and extend `tests/contract.rs` to the new maximum. For high degree, a global Chebyshev or barycentric fit may be the better method
- [ ] Multi-GPU *(only once single-GPU wins are proven)*
- [x] *(1.1.0)* Faster exact time differences: an `i64` fast path for knots and grids within 2⁶² ns (~146 years) of the first knot, bit-identical to `i128`; 27–39% less time single-threaded, less on `Parallel`, where the gain is within run-to-run variance (BENCHMARKS.md, "Methods")
- [x] *(1.1.0)* Cheaper output timestamps: built from integer POSIX seconds on a per-day cached date instead of chrono's `checked_add_signed`, which was ~10 of the 14 ms of single-threaded output assembly per million points (BENCHMARKS.md, "Methods")
- [ ] Single-pass output assembly: write each output column once, straight into its allocation, with rayon's indexed `collect_into_vec` / `unzip_into_vecs` and per-job cursors that seek on a non-consecutive index (rayon calls `map_init`'s `init` once per job and doesn't specify the order within one, [docs](https://docs.rs/rayon/latest/rayon/iter/trait.ParallelIterator.html#method.map_init)), instead of pre-filling three columns and overwriting them. Output assembly is ~40 of 67 ms in a 16 Mi-point `Parallel` call. A prototype passed every test, but its benchmarks were taken while other services loaded the machine (the two A/B orders disagreed by up to 40%), so it isn't merged: remeasure on a quiet machine before keeping it
- [ ] The same `i64` fast path for output assembly's provenance walk (`interpolation.rs` `assemble`, still `i128` comparisons per point) and the GPU `f32` repair pass; measure before keeping
- [x] *(1.1.0)* Parallel input preparation for million-point inputs: every backend but `Cpu` converts and sorts 16 Ki inputs or more on rayon's pool, and each value is converted once; a million shuffled `BigDecimal` inputs went from 368 ms to 43 ms (BENCHMARKS.md, "Preparing input")
- [x] *(1.1.0)* Runtime-agnostic async: `Interpolator::spawn` / `spawn_f64` run on rayon's pool and return an `InterpolationFuture` (a oneshot on `std`'s `Mutex` and `Waker`) that any executor can await; no dependency, no feature flag, and a panic becomes `Error::Task`. Dropping the future cancels work that hasn't started; with the GPU, the first `spawn` opens the device on the calling thread, `Auto` uses the GPU only if it had started when `spawn` was called, and dropping a future whose GPU work has started waits for it, so a process can exit safely (`tests/spawn_exit.rs`, which fails if any of those safeguards is removed); on a rayon worker it runs inline, so blocking on it can't deadlock. The tokio wrappers stay
- [ ] Downsampling and aggregation stay in WeftDB (`weft-reduce`) unless another consumer asks for them here

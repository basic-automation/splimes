# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the project is pre-1.0, minor version bumps may contain breaking changes.

## [Unreleased]

## [0.1.0] - 2026-10-05

First release as a standalone crate. splimes was developed inside the
[WeftDB](https://github.com/basic-automation/weftdb) workspace, and this repository keeps
that history.

### Added

- Linear, quadratic, cubic and polynomial spline interpolation over irregularly sampled
  `BigDecimal` series, resampled onto a regular grid from nanoseconds to years.
- Automatic backend selection (`auto_interpolate`) between SIMD CPU, `rayon`-parallel
  CPU and GPU (`wgpu`: Vulkan, Metal, DX12), with explicit `cpu_interpolate`,
  `parallel_interpolate` and `gpu_interpolate`.
- An f32 GPU path for adapters without f64 support.
- `prewarm_gpu`, `prewarm_gpu_with_config` (`GpuConfig` presets) and
  `gpu_buffer_pool_stats`.
- The `gpu-eager-init` feature.

### Fixed

- The f32 quadratic shader didn't parse, so quadratic GPU interpolation panicked on
  adapters without f64 support (Apple, most integrated GPUs, Windows WARP).
- Each interpolation call built a full `sysinfo::System` (every process, disk and
  network interface) just to read memory figures: about 445 ms per call, now about
  38 µs.

### Changed

- `Point::random` is now test-only, and `fake` is no longer a runtime dependency.

[Unreleased]: https://github.com/basic-automation/splimes/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/basic-automation/splimes/releases/tag/v0.1.0

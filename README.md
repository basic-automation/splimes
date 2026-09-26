# splimes

[![crates.io](https://img.shields.io/crates/v/splimes.svg)](https://crates.io/crates/splimes)
[![docs.rs](https://img.shields.io/docsrs/splimes)](https://docs.rs/splimes)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Spline interpolation over **irregularly sampled** time series.

`splimes` reconstructs a continuous signal from points that did not arrive on a clean
grid, and labels every point it hands back with its provenance — `raw`, `interpolated`
or `extrapolated` — so a synthetic value is never silently mistaken for an observed one.

- **Methods** — linear, quadratic, cubic and polynomial splines.
- **Backends** — SIMD and `rayon`-parallel CPU, or GPU via [`wgpu`](https://wgpu.rs)
  (Vulkan / Metal / DX12). Backend selection is automatic: without a usable GPU the
  engine falls back to CPU, and small workloads stay on the CPU because dispatch
  overhead would dominate.
- **Values are `BigDecimal`** — precision is declared, not quietly lost.

```toml
[dependencies]
splimes = "0.1"
```

## Features

| Feature | Default | Effect |
|---------|---------|--------|
| `gpu-eager-init` | off | Initialize the GPU adapter at process start via `ctor`, instead of lazily on first use. Trades startup latency for a warm first query. |

This crate is the interpolation engine behind [WeftDB](https://github.com/basic-automation/weftdb),
but it has no dependency on the database and is usable on its own.

## License

MIT — see [LICENSE](LICENSE).

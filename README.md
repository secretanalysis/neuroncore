# NeuronCore

NeuronCore is a **pure Rust** (no external crates) neural-compute and industrial data foundation.  
It started as a minimalist tensor + autograd engine and now also includes utilities for ingesting and analyzing industrial time-series workflows.

---

## Core functionality

### 1) Tensor math primitives
- Owned `f32` tensors in row-major format.
- Shape-aware operations with validation and broadcasting behavior.
- Core numerical operations used by both model code and general compute utilities.

### 2) Autograd computation graph
- Dynamic graph construction through `Graph` and `Node`.
- Forward execution for composed operations.
- Reverse-mode backpropagation through `Op` implementations.

### 3) Neural network building blocks
- Basic layers, losses, and optimizer flow for small model experiments.
- Included ops and exports for common transformations such as linear algebra and activations.

### 4) Time-series/windowing + tensor indexing helpers
- 1D and 2D sliding-window helpers (`windows_1d`, `windows_2d`).
- Index conversion helpers (`ravel_index`, `unravel_index`) for deterministic tensor addressing.

### 5) Industrial ingest and schema pipeline
- Unified `IndustrialRecord` event model (`MachineState`, `SensorSample`, `ToolEvent`).
- `IngestSource` trait abstraction for data sources.
- Replay ingestion from NDJSON fixtures via `ReplaySource`.
- Optional feature-gated adapters:
  - `mtconnect`: parse current MTConnect-style XML snapshots.
  - `opcua`: map OPC UA node snapshots into common records.

### 6) Health/anomaly utility
- Lightweight z-score based anomaly scoring over `f32` streams.

### 7) Reproducibility metadata support
- `RunManifest` model and deterministic hash generation utility for run metadata tracking.

---

## What we’ve accomplished so far

- Built a **dependency-free neural computation core** with tensor operations, graph execution, and backprop.
- Added foundational **NN workflow primitives** (layers/losses/optimizer) for end-to-end training smoke coverage.
- Added **industrial data abstractions** and replay ingestion to bridge ML-style compute with plant/event telemetry.
- Implemented **optional MTConnect and OPC UA integration modules** behind feature flags.
- Added **time-series utilities** that support window-based analysis workflows.
- Added **anomaly scoring** for practical health monitoring experiments.
- Added **manifest/hash support** to improve deterministic run-tracking and reproducibility.
- Established a broad **test suite** across tensors, ops, autograd, indexing, time-series, ingest adapters, anomalies, and manifests.

---

## Quick start

### Build & test

Requires stable Rust (edition 2021).

```bash
cargo test
cargo build --release
```

### Run the replay anomaly example

```bash
cargo run --example anomaly_from_replay
```

### Feature-gated test examples

```bash
cargo test --features mtconnect
cargo test --features opcua
```

---

## Project layout

- `src/lib.rs` – crate entry point and public exports
- `src/tensor.rs` – tensor storage and core tensor operations
- `src/ops.rs` – operation trait and differentiable ops
- `src/graph.rs` – graph execution + reverse autodiff
- `src/layers.rs` – basic layer primitives
- `src/losses.rs` – loss functions
- `src/optim.rs` – optimizer primitives
- `src/timeseries.rs` – windowing utilities for sequential data
- `src/tensor_index.rs` – index flatten/unflatten helpers
- `src/industrial/` – ingest traits, replay source, and industrial schemas/adapters
- `src/health/` – anomaly/health analytics helpers
- `src/run_manifest.rs` – run metadata and deterministic hashing
- `tests/` – integration tests and fixtures

---

## Notes

- Design goal: correctness, clarity, and portability over aggressive optimization.
- API is still evolving and may change.
- Crate features:
  - `mtconnect`
  - `opcua`

---

## License

Dual-licensed under **MIT** or **Apache-2.0** (your choice).  
See `LICENSE-MIT` and `LICENSE-APACHE`.

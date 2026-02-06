# NeuronCore

A minimalist neural computation engine implemented in **pure Rust** (no external crates):

- `Tensor` (owned `f32`, row-major) with broadcasting + basic ops
- Autograd `Graph` with a small `Op` trait and backprop
- A few NN building blocks: linear layer, ReLU, MSE, softmax, SGD

This is designed as a correctness-first foundation you can extend.

## Quick start

### Build & test

Requires a Rust toolchain (stable, edition 2021).

```bash
cargo test
cargo build --release
```

### Minimal example

```rust
use neuroncore::{Graph, Tensor, AddOp};

fn main() -> Result<(), neuroncore::ComputeError> {
    let mut g = Graph::new();

    let a = g.add_input(Tensor::new(vec![1.0, 2.0], vec![1, 2])?);
    let b = g.add_input(Tensor::new(vec![3.0, 4.0], vec![1, 2])?);

    let out = g.apply_op(AddOp, &[a, b]);
    let y = g.forward(out)?;

    println!("{y:?}");
    Ok(())
}
```

### Training smoke test

There is an end-to-end test in `tests/training_smoke.rs` that exercises a small training loop.

## Project layout

- `src/tensor.rs` – tensor storage, indexing, broadcasting, core ops
- `src/ops.rs` – `Op` trait + operation implementations (forward/backward)
- `src/graph.rs` – computational graph and reverse-mode autodiff
- `src/layers.rs` – simple layers (e.g., linear)
- `src/losses.rs` – losses (e.g., MSE)
- `src/optim.rs` – SGD optimizer

## Notes

- **No external dependencies**: includes a small xorshift PRNG for parameter init.
- **Not optimized**: correctness/readability over speed.
- **API stability**: this is a small, evolving library; expect breaking changes.

## License

Dual-licensed under **MIT** or **Apache-2.0** (at your option). See `LICENSE-MIT` and `LICENSE-APACHE`.

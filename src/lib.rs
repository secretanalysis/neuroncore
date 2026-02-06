//! NeuronCore: a minimalist pure-Rust neural computation engine.
//!
//! Implements (per the provided spec):
//! - `Tensor` (owned `f32`, row-major) with broadcasting, matmul, activations, reductions.
//! - `Graph` autograd engine with `Node` and `Op`.
//! - Basic layers, losses, and an SGD optimizer.
//!
//! Design notes:
//! - No external dependencies: includes a tiny xorshift PRNG for init.
//! - Correctness-oriented and deliberately unoptimized.

pub mod error;
pub mod graph;
pub mod layers;
pub mod losses;
pub mod ops;
pub mod optim;
pub mod prng;
pub mod tensor;

pub use error::ComputeError;
pub use graph::{Graph, Node};
pub use ops::{
    AddOp, DivideOp, InvertibleOp, LogOp, MatMulOp, MultiplyOp, Op, ReluOp, SoftmaxOp,
    SubtractOp, SumOp,
};
pub use tensor::Tensor;

#[cfg(test)]
mod smoke_tests {
    use super::*;

    #[test]
    fn tensor_add_broadcast_smoke() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let b = Tensor::new(vec![10.0], vec![1]).unwrap();
        let c = a.add(&b).unwrap();
        assert_eq!(c.shape(), &[3]);
        assert_eq!(c.data(), &[11.0, 12.0, 13.0]);
    }

    #[test]
    fn graph_forward_add_smoke() {
        let mut g = Graph::new();
        let a = g.add_input(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap());
        let b = g.add_input(Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap());
        let out = g.apply_op(AddOp, &[a, b]);
        let y = g.forward(out).unwrap();
        assert_eq!(y.data(), &[4.0, 6.0]);
    }

    #[test]
    fn graph_backward_add_smoke() {
        let mut g = Graph::new();
        let a = g.add_parameter(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap(), true);
        let b = g.add_parameter(Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap(), true);
        let out = g.apply_op(AddOp, &[a, b]);
        g.backward(out).unwrap();
        let ga = g.get_gradient(a).unwrap();
        let gb = g.get_gradient(b).unwrap();
        assert_eq!(ga.data(), &[1.0, 1.0]);
        assert_eq!(gb.data(), &[1.0, 1.0]);
    }
}

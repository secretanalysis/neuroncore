use crate::error::ComputeError;
use crate::graph::Graph;
use crate::ops::{DivideOp, LogOp, MultiplyOp, SoftmaxOp, SubtractOp, SumOp};
use crate::tensor::Tensor;

pub struct MSELoss;

impl MSELoss {
    pub fn compute(
        graph: &mut Graph,
        predictions: usize,
        targets: usize,
    ) -> Result<usize, ComputeError> {
        // diff = predictions - targets
        let diff_idx = graph.apply_op(SubtractOp, &[predictions, targets]);

        // squared = diff * diff
        let squared_idx = graph.apply_op(MultiplyOp, &[diff_idx, diff_idx]);

        // sum of squared differences
        let sum_idx = graph.apply_op(SumOp { dim: None }, &[squared_idx]);

        // Compute mean: sum / N
        let pred_tensor = graph.forward(predictions)?;
        let size = pred_tensor.data().len() as f32;
        let size_tensor = Tensor::new(vec![size], vec![1])?;
        let size_idx = graph.add_input(size_tensor);

        let loss_idx = graph.apply_op(DivideOp, &[sum_idx, size_idx]);
        Ok(loss_idx)
    }
}

pub struct CrossEntropyLoss;

impl CrossEntropyLoss {
    /// Cross entropy for one-hot targets. Expects `targets` shaped like `logits`.
    pub fn compute(
        graph: &mut Graph,
        logits: usize,
        targets: usize,
    ) -> Result<usize, ComputeError> {
        let softmax_idx = graph.apply_op(SoftmaxOp, &[logits]);
        let log_softmax_idx = graph.apply_op(LogOp, &[softmax_idx]);
        let selected_idx = graph.apply_op(MultiplyOp, &[log_softmax_idx, targets]);
        let sum_idx = graph.apply_op(SumOp { dim: None }, &[selected_idx]);

        let neg_one = Tensor::new(vec![-1.0], vec![1])?;
        let neg_one_idx = graph.add_input(neg_one);
        let loss_idx = graph.apply_op(MultiplyOp, &[sum_idx, neg_one_idx]);
        Ok(loss_idx)
    }
}

use crate::error::ComputeError;
use crate::tensor::Tensor;
use crate::tensor_index;

pub trait Op: Send + Sync {
    fn forward(&self, inputs: &[Tensor]) -> Result<Tensor, ComputeError>;
    fn backward(
        &self,
        inputs: &[Tensor],
        grad_output: &Tensor,
    ) -> Result<Vec<Tensor>, ComputeError>;
}

/// Trait for ops whose forward pass can be algebraically inverted.
///
/// Given the forward output and all-but-one inputs, recover the missing input.
/// `known[i]` is `Some` for each known input, `None` at position `solve_for`.
/// Only implemented for ops where inversion is exact (not ReLU, Softmax, Sum).
pub trait InvertibleOp: Op {
    fn invert(
        &self,
        output: &Tensor,
        known: &[Option<&Tensor>],
        solve_for: usize,
    ) -> Result<Tensor, ComputeError>;
}

/// Validate that `known` has exactly one `None` at `solve_for` and the rest are `Some`.
fn validate_invert_args(
    known: &[Option<&Tensor>],
    solve_for: usize,
    arity: usize,
) -> Result<(), ComputeError> {
    if known.len() != arity {
        return Err(ComputeError::InputCountError {
            expected: arity,
            got: known.len(),
        });
    }
    if solve_for >= arity {
        return Err(ComputeError::IndexError {
            message: format!("solve_for {solve_for} out of bounds for arity {arity}"),
        });
    }
    if known[solve_for].is_some() {
        return Err(ComputeError::InvalidOperation {
            message: "known[solve_for] must be None".to_string(),
        });
    }
    for (i, k) in known.iter().enumerate() {
        if i != solve_for && k.is_none() {
            return Err(ComputeError::InvalidOperation {
                message: format!("known[{i}] must be Some (only solve_for may be None)"),
            });
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
pub struct AddOp;

impl Op for AddOp {
    fn forward(&self, inputs: &[Tensor]) -> Result<Tensor, ComputeError> {
        if inputs.len() != 2 {
            return Err(ComputeError::InputCountError {
                expected: 2,
                got: inputs.len(),
            });
        }
        inputs[0].add(&inputs[1])
    }

    fn backward(
        &self,
        _inputs: &[Tensor],
        grad_output: &Tensor,
    ) -> Result<Vec<Tensor>, ComputeError> {
        Ok(vec![grad_output.clone(), grad_output.clone()])
    }
}

/// out = a + b → a = out - b, b = out - a
impl InvertibleOp for AddOp {
    fn invert(
        &self,
        output: &Tensor,
        known: &[Option<&Tensor>],
        solve_for: usize,
    ) -> Result<Tensor, ComputeError> {
        validate_invert_args(known, solve_for, 2)?;
        let other = known[1 - solve_for].unwrap();
        output.subtract(other)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SubtractOp;

impl Op for SubtractOp {
    fn forward(&self, inputs: &[Tensor]) -> Result<Tensor, ComputeError> {
        if inputs.len() != 2 {
            return Err(ComputeError::InputCountError {
                expected: 2,
                got: inputs.len(),
            });
        }
        inputs[0].subtract(&inputs[1])
    }

    fn backward(
        &self,
        _inputs: &[Tensor],
        grad_output: &Tensor,
    ) -> Result<Vec<Tensor>, ComputeError> {
        let mut neg = grad_output.clone();
        for v in neg.data_mut().iter_mut() {
            *v = -*v;
        }
        Ok(vec![grad_output.clone(), neg])
    }
}

/// out = a - b → a = out + b, b = a - out
impl InvertibleOp for SubtractOp {
    fn invert(
        &self,
        output: &Tensor,
        known: &[Option<&Tensor>],
        solve_for: usize,
    ) -> Result<Tensor, ComputeError> {
        validate_invert_args(known, solve_for, 2)?;
        let other = known[1 - solve_for].unwrap();
        match solve_for {
            0 => output.add(other),      // a = out + b
            _ => other.subtract(output), // b = a - out
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MultiplyOp;

impl Op for MultiplyOp {
    fn forward(&self, inputs: &[Tensor]) -> Result<Tensor, ComputeError> {
        if inputs.len() != 2 {
            return Err(ComputeError::InputCountError {
                expected: 2,
                got: inputs.len(),
            });
        }
        inputs[0].multiply(&inputs[1])
    }

    fn backward(
        &self,
        inputs: &[Tensor],
        grad_output: &Tensor,
    ) -> Result<Vec<Tensor>, ComputeError> {
        if inputs.len() != 2 {
            return Err(ComputeError::InputCountError {
                expected: 2,
                got: inputs.len(),
            });
        }
        let grad_a = grad_output.multiply(&inputs[1])?;
        let grad_b = grad_output.multiply(&inputs[0])?;
        Ok(vec![grad_a, grad_b])
    }
}

/// out = a * b → a = out / b, b = out / a
impl InvertibleOp for MultiplyOp {
    fn invert(
        &self,
        output: &Tensor,
        known: &[Option<&Tensor>],
        solve_for: usize,
    ) -> Result<Tensor, ComputeError> {
        validate_invert_args(known, solve_for, 2)?;
        let other = known[1 - solve_for].unwrap();
        output.divide(other)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DivideOp;

impl Op for DivideOp {
    fn forward(&self, inputs: &[Tensor]) -> Result<Tensor, ComputeError> {
        if inputs.len() != 2 {
            return Err(ComputeError::InputCountError {
                expected: 2,
                got: inputs.len(),
            });
        }
        inputs[0].divide(&inputs[1])
    }

    fn backward(
        &self,
        inputs: &[Tensor],
        grad_output: &Tensor,
    ) -> Result<Vec<Tensor>, ComputeError> {
        if inputs.len() != 2 {
            return Err(ComputeError::InputCountError {
                expected: 2,
                got: inputs.len(),
            });
        }
        let a = &inputs[0];
        let b = &inputs[1];

        let grad_a = grad_output.divide(b)?;

        // grad_b = -grad_output * a / (b*b)
        let b2 = b.multiply(b)?;
        let num = grad_output.multiply(a)?;
        let mut grad_b = num.divide(&b2)?;
        for v in grad_b.data_mut().iter_mut() {
            *v = -*v;
        }

        Ok(vec![grad_a, grad_b])
    }
}

/// out = a / b → a = out * b, b = a / out
impl InvertibleOp for DivideOp {
    fn invert(
        &self,
        output: &Tensor,
        known: &[Option<&Tensor>],
        solve_for: usize,
    ) -> Result<Tensor, ComputeError> {
        validate_invert_args(known, solve_for, 2)?;
        let other = known[1 - solve_for].unwrap();
        match solve_for {
            0 => output.multiply(other), // a = out * b
            _ => other.divide(output),   // b = a / out
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MatMulOp;

impl Op for MatMulOp {
    fn forward(&self, inputs: &[Tensor]) -> Result<Tensor, ComputeError> {
        if inputs.len() != 2 {
            return Err(ComputeError::InputCountError {
                expected: 2,
                got: inputs.len(),
            });
        }
        inputs[0].matmul(&inputs[1])
    }

    fn backward(
        &self,
        inputs: &[Tensor],
        grad_output: &Tensor,
    ) -> Result<Vec<Tensor>, ComputeError> {
        if inputs.len() != 2 {
            return Err(ComputeError::InputCountError {
                expected: 2,
                got: inputs.len(),
            });
        }
        let a = &inputs[0];
        let b = &inputs[1];
        let b_t = b.transpose_2d()?;
        let a_t = a.transpose_2d()?;
        let grad_a = grad_output.matmul(&b_t)?;
        let grad_b = a_t.matmul(grad_output)?;
        Ok(vec![grad_a, grad_b])
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ReluOp;

impl Op for ReluOp {
    fn forward(&self, inputs: &[Tensor]) -> Result<Tensor, ComputeError> {
        if inputs.len() != 1 {
            return Err(ComputeError::InputCountError {
                expected: 1,
                got: inputs.len(),
            });
        }
        inputs[0].relu()
    }

    fn backward(
        &self,
        inputs: &[Tensor],
        grad_output: &Tensor,
    ) -> Result<Vec<Tensor>, ComputeError> {
        if inputs.len() != 1 {
            return Err(ComputeError::InputCountError {
                expected: 1,
                got: inputs.len(),
            });
        }
        let input = &inputs[0];
        if input.data().len() != grad_output.data().len() {
            return Err(ComputeError::ShapeMismatch {
                expected: input.data().len(),
                got: grad_output.data().len(),
            });
        }
        let mut grad = Tensor::zeros_like(input)?;
        for i in 0..input.data().len() {
            grad.data_mut()[i] = if input.data()[i] > 0.0 {
                grad_output.data()[i]
            } else {
                0.0
            };
        }
        Ok(vec![grad])
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SumOp {
    pub dim: Option<usize>,
}

impl Op for SumOp {
    fn forward(&self, inputs: &[Tensor]) -> Result<Tensor, ComputeError> {
        if inputs.len() != 1 {
            return Err(ComputeError::InputCountError {
                expected: 1,
                got: inputs.len(),
            });
        }
        inputs[0].sum(self.dim)
    }

    fn backward(
        &self,
        inputs: &[Tensor],
        grad_output: &Tensor,
    ) -> Result<Vec<Tensor>, ComputeError> {
        if inputs.len() != 1 {
            return Err(ComputeError::InputCountError {
                expected: 1,
                got: inputs.len(),
            });
        }
        let input = &inputs[0];
        let mut grad_input = Tensor::zeros_like(input)?;

        match self.dim {
            None => {
                // grad_output is [1]
                let g = grad_output.data().first().copied().unwrap_or(0.0);
                for v in grad_input.data_mut().iter_mut() {
                    *v = g;
                }
            }
            Some(axis) => {
                if axis >= input.shape().len() {
                    return Err(ComputeError::DimensionError {
                        message: format!("invalid axis {axis} for rank {}", input.shape().len()),
                    });
                }
                // grad_output has same shape as input but axis=1.
                // Each slice along axis gets the same scalar grad_output at that reduced index.
                for flat in 0..grad_input.data().len() {
                    // Recompute indices for input.
                    let idx = tensor_index::unravel_index(flat, input.shape())?;
                    let mut g_idx = idx.clone();
                    g_idx[axis] = 0;
                    let g_flat = tensor_index::ravel_index(&g_idx, grad_output.shape())?;
                    grad_input.data_mut()[flat] = grad_output.data()[g_flat];
                }
            }
        }

        Ok(vec![grad_input])
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LogOp;

impl Op for LogOp {
    fn forward(&self, inputs: &[Tensor]) -> Result<Tensor, ComputeError> {
        if inputs.len() != 1 {
            return Err(ComputeError::InputCountError {
                expected: 1,
                got: inputs.len(),
            });
        }
        let x = &inputs[0];
        let out: Vec<f32> = x.data().iter().map(|&v| v.ln()).collect();
        Tensor::new(out, x.shape().to_vec())
    }

    fn backward(
        &self,
        inputs: &[Tensor],
        grad_output: &Tensor,
    ) -> Result<Vec<Tensor>, ComputeError> {
        if inputs.len() != 1 {
            return Err(ComputeError::InputCountError {
                expected: 1,
                got: inputs.len(),
            });
        }
        let x = &inputs[0];
        if x.data().len() != grad_output.data().len() {
            return Err(ComputeError::ShapeMismatch {
                expected: x.data().len(),
                got: grad_output.data().len(),
            });
        }
        let mut grad = Tensor::zeros_like(x)?;
        for i in 0..x.data().len() {
            grad.data_mut()[i] = grad_output.data()[i] / x.data()[i];
        }
        Ok(vec![grad])
    }
}

/// out = ln(x) → x = exp(out)
impl InvertibleOp for LogOp {
    fn invert(
        &self,
        output: &Tensor,
        known: &[Option<&Tensor>],
        solve_for: usize,
    ) -> Result<Tensor, ComputeError> {
        validate_invert_args(known, solve_for, 1)?;
        let data: Vec<f32> = output.data().iter().map(|&v| v.exp()).collect();
        Tensor::new(data, output.shape().to_vec())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SoftmaxOp;

impl Op for SoftmaxOp {
    fn forward(&self, inputs: &[Tensor]) -> Result<Tensor, ComputeError> {
        if inputs.len() != 1 {
            return Err(ComputeError::InputCountError {
                expected: 1,
                got: inputs.len(),
            });
        }
        let x = &inputs[0];
        if x.shape().len() == 1 {
            softmax_1d(x)
        } else if x.shape().len() == 2 {
            softmax_2d(x)
        } else {
            Err(ComputeError::DimensionError {
                message: "softmax supports 1D or 2D tensors".to_string(),
            })
        }
    }

    fn backward(
        &self,
        inputs: &[Tensor],
        grad_output: &Tensor,
    ) -> Result<Vec<Tensor>, ComputeError> {
        if inputs.len() != 1 {
            return Err(ComputeError::InputCountError {
                expected: 1,
                got: inputs.len(),
            });
        }
        let x = &inputs[0];
        let y = self.forward(inputs)?;

        if x.shape().len() == 1 {
            let n = x.shape()[0];
            if grad_output.shape() != [n] {
                return Err(ComputeError::InvalidOperation {
                    message: "grad_output shape mismatch".to_string(),
                });
            }
            let mut dot = 0.0;
            for i in 0..n {
                dot += grad_output.data()[i] * y.data()[i];
            }
            let mut grad = vec![0.0; n];
            for (i, g) in grad.iter_mut().enumerate().take(n) {
                *g = y.data()[i] * (grad_output.data()[i] - dot);
            }
            Ok(vec![Tensor::new(grad, vec![n])?])
        } else {
            // 2D: apply row-wise
            let rows = x.shape()[0];
            let cols = x.shape()[1];
            if grad_output.shape() != [rows, cols] {
                return Err(ComputeError::InvalidOperation {
                    message: "grad_output shape mismatch".to_string(),
                });
            }
            let mut grad = vec![0.0; rows * cols];
            for r in 0..rows {
                let base = r * cols;
                let mut dot = 0.0;
                for c in 0..cols {
                    dot += grad_output.data()[base + c] * y.data()[base + c];
                }
                for c in 0..cols {
                    grad[base + c] = y.data()[base + c] * (grad_output.data()[base + c] - dot);
                }
            }
            Ok(vec![Tensor::new(grad, vec![rows, cols])?])
        }
    }
}

fn softmax_1d(x: &Tensor) -> Result<Tensor, ComputeError> {
    let n = x.shape()[0];
    let max = x.data().iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut exps = vec![0.0; n];
    let mut sum = 0.0;
    for (i, e_out) in exps.iter_mut().enumerate().take(n) {
        let e = (x.data()[i] - max).exp();
        *e_out = e;
        sum += e;
    }
    for e in exps.iter_mut().take(n) {
        *e /= sum;
    }
    Tensor::new(exps, vec![n])
}

fn softmax_2d(x: &Tensor) -> Result<Tensor, ComputeError> {
    let rows = x.shape()[0];
    let cols = x.shape()[1];
    let mut out = vec![0.0; rows * cols];
    for r in 0..rows {
        let base = r * cols;
        let mut max = f32::NEG_INFINITY;
        for c in 0..cols {
            max = max.max(x.data()[base + c]);
        }
        let mut sum = 0.0;
        for c in 0..cols {
            let e = (x.data()[base + c] - max).exp();
            out[base + c] = e;
            sum += e;
        }
        for c in 0..cols {
            out[base + c] /= sum;
        }
    }
    Tensor::new(out, vec![rows, cols])
}

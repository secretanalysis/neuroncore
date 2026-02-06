use crate::error::ComputeError;
use crate::prng::XorShift32;

#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Result<Self, ComputeError> {
        if shape.is_empty() {
            return Err(ComputeError::DimensionError {
                message: "shape must have at least 1 dimension".to_string(),
            });
        }

        let expected_elements: usize = shape.iter().product();
        if data.len() != expected_elements {
            return Err(ComputeError::ShapeMismatch {
                expected: expected_elements,
                got: data.len(),
            });
        }

        let strides = Self::compute_strides(&shape);
        Ok(Self {
            data,
            shape,
            strides,
        })
    }

    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    pub fn zeros(shape: Vec<usize>) -> Result<Self, ComputeError> {
        let size: usize = shape.iter().product();
        Self::new(vec![0.0; size], shape)
    }

    pub fn ones(shape: Vec<usize>) -> Result<Self, ComputeError> {
        let size: usize = shape.iter().product();
        Self::new(vec![1.0; size], shape)
    }

    pub fn zeros_like(other: &Tensor) -> Result<Self, ComputeError> {
        Self::zeros(other.shape.clone())
    }

    pub fn ones_like(other: &Tensor) -> Tensor {
        Self::ones(other.shape.clone()).expect("ones_like: valid shape")
    }

    /// Deterministic random init for weight initialization (no external deps).
    pub fn random(shape: Vec<usize>, seed: u32) -> Result<Self, ComputeError> {
        let size: usize = shape.iter().product();
        let mut rng = XorShift32::new(seed);
        let mut data = Vec::with_capacity(size);
        for _ in 0..size {
            data.push(rng.gen_range_f32(-1.0, 1.0));
        }
        Self::new(data, shape)
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    pub fn add(&self, other: &Tensor) -> Result<Tensor, ComputeError> {
        self.elementwise_op(other, |a, b| a + b)
    }

    pub fn subtract(&self, other: &Tensor) -> Result<Tensor, ComputeError> {
        self.elementwise_op(other, |a, b| a - b)
    }

    pub fn multiply(&self, other: &Tensor) -> Result<Tensor, ComputeError> {
        self.elementwise_op(other, |a, b| a * b)
    }

    pub fn divide(&self, other: &Tensor) -> Result<Tensor, ComputeError> {
        self.elementwise_op(other, |a, b| if b != 0.0 { a / b } else { f32::NAN })
    }

    pub fn matmul(&self, other: &Tensor) -> Result<Tensor, ComputeError> {
        if self.shape.len() != 2 || other.shape.len() != 2 {
            return Err(ComputeError::DimensionError {
                message: "matmul requires 2D tensors".to_string(),
            });
        }
        let m = self.shape[0];
        let k = self.shape[1];
        let k_other = other.shape[0];
        let n = other.shape[1];
        if k != k_other {
            return Err(ComputeError::InvalidOperation {
                message: format!(
                    "matmul dimension mismatch: left is {m}x{k}, right is {k_other}x{n}"
                ),
            });
        }

        let mut out = vec![0.0; m * n];
        for i in 0..m {
            let a_row = i * k;
            let out_row = i * n;
            for j in 0..n {
                let mut sum = 0.0;
                for p in 0..k {
                    sum += self.data[a_row + p] * other.data[p * n + j];
                }
                out[out_row + j] = sum;
            }
        }

        Tensor::new(out, vec![m, n])
    }

    pub fn transpose_2d(&self) -> Result<Tensor, ComputeError> {
        if self.shape.len() != 2 {
            return Err(ComputeError::DimensionError {
                message: "transpose_2d requires a 2D tensor".to_string(),
            });
        }
        let rows = self.shape[0];
        let cols = self.shape[1];
        let mut out = vec![0.0; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                out[c * rows + r] = self.data[r * cols + c];
            }
        }
        Tensor::new(out, vec![cols, rows])
    }

    pub fn relu(&self) -> Result<Tensor, ComputeError> {
        let data: Vec<f32> = self.data.iter().map(|&v| v.max(0.0)).collect();
        Tensor::new(data, self.shape.clone())
    }

    pub fn sum(&self, dim: Option<usize>) -> Result<Tensor, ComputeError> {
        match dim {
            None => {
                let total: f32 = self.data.iter().sum();
                Tensor::new(vec![total], vec![1])
            }
            Some(axis) => {
                if axis >= self.shape.len() {
                    return Err(ComputeError::DimensionError {
                        message: format!("invalid axis {axis} for rank {}", self.shape.len()),
                    });
                }
                let mut out_shape = self.shape.clone();
                out_shape[axis] = 1;
                let mut out = vec![0.0; out_shape.iter().product()];

                for flat in 0..self.data.len() {
                    let mut idx = Self::unravel_index_static(flat, &self.shape);
                    idx[axis] = 0;
                    let out_flat = Self::ravel_index_static(&idx, &out_shape);
                    out[out_flat] += self.data[flat];
                }

                Tensor::new(out, out_shape)
            }
        }
    }

    fn elementwise_op<F>(&self, other: &Tensor, op: F) -> Result<Tensor, ComputeError>
    where
        F: Fn(f32, f32) -> f32,
    {
        let out_shape = Self::broadcast_shapes(&self.shape, &other.shape)?;
        let mut out = Tensor::zeros(out_shape.clone())?;

        for out_flat in 0..out.data.len() {
            let out_idx = Self::unravel_index_static(out_flat, &out_shape);
            let a_flat = self.broadcasted_flat_index(&out_idx, &out_shape)?;
            let b_flat = other.broadcasted_flat_index(&out_idx, &out_shape)?;
            out.data[out_flat] = op(self.data[a_flat], other.data[b_flat]);
        }

        Ok(out)
    }

    fn broadcast_shapes(a: &[usize], b: &[usize]) -> Result<Vec<usize>, ComputeError> {
        let max_dims = a.len().max(b.len());
        let mut out = vec![1; max_dims];

        // Align shapes from the right (NumPy-style).
        for i in 0..max_dims {
            let a_i = if i >= max_dims - a.len() {
                a[i - (max_dims - a.len())]
            } else {
                1
            };
            let b_i = if i >= max_dims - b.len() {
                b[i - (max_dims - b.len())]
            } else {
                1
            };

            out[i] = if a_i == b_i {
                a_i
            } else if a_i == 1 {
                b_i
            } else if b_i == 1 {
                a_i
            } else {
                return Err(ComputeError::BroadcastError {
                    dim: i,
                    shape1: a_i,
                    shape2: b_i,
                });
            };
        }

        Ok(out)
    }

    fn broadcasted_flat_index(
        &self,
        out_indices: &[usize],
        out_shape: &[usize],
    ) -> Result<usize, ComputeError> {
        if out_indices.len() != out_shape.len() {
            return Err(ComputeError::IndexError {
                message: "out_indices rank mismatch".to_string(),
            });
        }

        let offset = out_shape.len().saturating_sub(self.shape.len());
        let mut flat = 0usize;

        for out_dim in 0..out_shape.len() {
            let t_dim = if out_dim >= offset {
                self.shape[out_dim - offset]
            } else {
                1
            };
            let t_stride = if out_dim >= offset {
                self.strides[out_dim - offset]
            } else {
                0
            };

            let idx = if t_dim == 1 { 0 } else { out_indices[out_dim] };
            if idx >= t_dim {
                return Err(ComputeError::IndexError {
                    message: format!("index {idx} out of bounds for dim {t_dim}"),
                });
            }
            flat += idx * t_stride;
        }

        Ok(flat)
    }

    fn unravel_index_static(mut flat: usize, shape: &[usize]) -> Vec<usize> {
        let strides = Self::compute_strides(shape);
        let mut out = vec![0; shape.len()];
        for i in 0..shape.len() {
            out[i] = flat / strides[i];
            flat %= strides[i];
        }
        out
    }

    fn ravel_index_static(indices: &[usize], shape: &[usize]) -> usize {
        let strides = Self::compute_strides(shape);
        indices
            .iter()
            .zip(strides.iter())
            .map(|(&i, &s)| i * s)
            .sum()
    }
}

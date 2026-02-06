use crate::error::ComputeError;
use crate::graph::Graph;
use crate::ops::{AddOp, MatMulOp};
use crate::prng::XorShift32;
use crate::tensor::Tensor;

pub trait Layer {
    fn parameters(&self) -> Vec<usize>;
    fn forward(&self, graph: &mut Graph, input_idx: usize) -> Result<usize, ComputeError>;
}

pub struct Linear {
    weight_idx: usize,
    bias_idx: usize,
    pub input_size: usize,
    pub output_size: usize,
}

impl Linear {
    pub fn new(
        graph: &mut Graph,
        input_size: usize,
        output_size: usize,
        seed: u32,
    ) -> Result<Self, ComputeError> {
        // Xavier/Glorot init: uniform[-k, k], k = 1/sqrt(fan_in)
        let k = 1.0 / (input_size as f32).sqrt();
        let mut rng = XorShift32::new(seed);
        let mut weight_data = vec![0.0; input_size * output_size];
        for v in &mut weight_data {
            *v = rng.gen_range_f32(-k, k);
        }
        let weight = Tensor::new(weight_data, vec![input_size, output_size])?;
        let weight_idx = graph.add_parameter(weight, true);

        let bias = Tensor::zeros(vec![1, output_size])?;
        let bias_idx = graph.add_parameter(bias, true);

        Ok(Self {
            weight_idx,
            bias_idx,
            input_size,
            output_size,
        })
    }
}

impl Layer for Linear {
    fn parameters(&self) -> Vec<usize> {
        vec![self.weight_idx, self.bias_idx]
    }

    fn forward(&self, graph: &mut Graph, input_idx: usize) -> Result<usize, ComputeError> {
        let mm = graph.apply_op(MatMulOp, &[input_idx, self.weight_idx]);
        let out = graph.apply_op(AddOp, &[mm, self.bias_idx]);
        Ok(out)
    }
}

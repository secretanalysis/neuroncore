use std::collections::HashMap;

use crate::error::ComputeError;
use crate::graph::Graph;
use crate::tensor::Tensor;

pub trait Optimizer {
    fn step(&mut self, graph: &mut Graph) -> Result<(), ComputeError>;
    fn zero_grad(&mut self, graph: &mut Graph);
}

pub struct SGD {
    pub param_indices: Vec<usize>,
    pub learning_rate: f32,
    pub momentum: Option<f32>,
    velocity: HashMap<usize, Tensor>,
}

impl SGD {
    pub fn new(param_indices: Vec<usize>, learning_rate: f32, momentum: Option<f32>) -> Self {
        Self {
            param_indices,
            learning_rate,
            momentum,
            velocity: HashMap::new(),
        }
    }
}

impl Optimizer for SGD {
    fn step(&mut self, graph: &mut Graph) -> Result<(), ComputeError> {
        for &p_idx in &self.param_indices {
            if !graph.node_requires_grad(p_idx) {
                continue;
            }
            let grad = match graph.get_gradient(p_idx) {
                Some(g) => g.clone(),
                None => continue,
            };

            let param = graph.get_parameter_mut(p_idx)?;

            if let Some(mu) = self.momentum {
                let v = self
                    .velocity
                    .entry(p_idx)
                    .or_insert_with(|| Tensor::zeros_like(param).expect("zeros_like"));

                // v = mu * v - lr * grad
                for i in 0..v.data().len() {
                    let new_v = mu * v.data()[i] - self.learning_rate * grad.data()[i];
                    v.data_mut()[i] = new_v;
                }

                // param = param + v
                for i in 0..param.data().len() {
                    param.data_mut()[i] += v.data()[i];
                }
            } else {
                // param = param - lr * grad
                for i in 0..param.data().len() {
                    param.data_mut()[i] -= self.learning_rate * grad.data()[i];
                }
            }
        }

        Ok(())
    }

    fn zero_grad(&mut self, graph: &mut Graph) {
        graph.zero_grad();
    }
}

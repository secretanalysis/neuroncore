use std::collections::{HashMap, HashSet};

use crate::error::ComputeError;
use crate::ops::Op;
use crate::tensor::Tensor;

/// Represents a node in the computational graph.
pub enum Node {
    Input(Tensor),
    Parameter(Tensor, bool),            // Tensor + requires_grad
    Operation(Box<dyn Op>, Vec<usize>), // Op + input node indices
}

/// A computational graph that tracks operations and gradients.
pub struct Graph {
    pub(crate) nodes: Vec<Node>,
    pub(crate) gradients: HashMap<usize, Tensor>, // Node index -> gradient
}

impl Graph {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            gradients: HashMap::new(),
        }
    }

    pub fn add_input(&mut self, tensor: Tensor) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(Node::Input(tensor));
        idx
    }

    pub fn add_parameter(&mut self, tensor: Tensor, requires_grad: bool) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(Node::Parameter(tensor, requires_grad));
        idx
    }

    pub fn apply_op<O: Op + 'static>(&mut self, op: O, inputs: &[usize]) -> usize {
        let idx = self.nodes.len();
        self.nodes
            .push(Node::Operation(Box::new(op), inputs.to_vec()));
        idx
    }

    pub fn forward(&self, node_idx: usize) -> Result<Tensor, ComputeError> {
        let node = self
            .nodes
            .get(node_idx)
            .ok_or_else(|| ComputeError::IndexError {
                message: format!("node index out of bounds: {node_idx}"),
            })?;

        match node {
            Node::Input(t) | Node::Parameter(t, _) => Ok(t.clone()),
            Node::Operation(op, input_indices) => {
                let mut inputs = Vec::with_capacity(input_indices.len());
                for &idx in input_indices {
                    inputs.push(self.forward(idx)?);
                }
                op.forward(&inputs)
            }
        }
    }

    pub fn backward(&mut self, output_idx: usize) -> Result<(), ComputeError> {
        let output = self.forward(output_idx)?;
        let grad_output = Tensor::ones_like(&output);
        self.gradients.insert(output_idx, grad_output);

        let sorted_nodes = self.topological_sort(output_idx)?;

        for &node_idx in sorted_nodes.iter().rev() {
            let grad = match self.gradients.get(&node_idx).cloned() {
                Some(g) => g,
                None => continue,
            };

            let node = self
                .nodes
                .get(node_idx)
                .ok_or_else(|| ComputeError::IndexError {
                    message: format!("node index out of bounds: {node_idx}"),
                })?;

            if let Node::Operation(op, input_indices) = node {
                let mut inputs = Vec::with_capacity(input_indices.len());
                for &idx in input_indices {
                    inputs.push(self.forward(idx)?);
                }

                let input_grads = op.backward(&inputs, &grad)?;
                if input_grads.len() != input_indices.len() {
                    return Err(ComputeError::InvalidOperation {
                        message: "op.backward returned wrong number of gradients".to_string(),
                    });
                }

                for (&input_idx, input_grad) in input_indices.iter().zip(input_grads.into_iter()) {
                    // Only accumulate gradients for nodes that should receive gradients.
                    if !self.node_requires_grad(input_idx) {
                        continue;
                    }

                    self.gradients
                        .entry(input_idx)
                        .and_modify(|existing| {
                            // Accumulate if node used multiple times.
                            // Any error here is a programming error: gradients must be compatible.
                            *existing = existing.add(&input_grad).expect("gradient add");
                        })
                        .or_insert(input_grad);
                }
            }
        }

        Ok(())
    }

    fn topological_sort(&self, start_idx: usize) -> Result<Vec<usize>, ComputeError> {
        let mut result = Vec::new();
        let mut visited = HashSet::new();

        fn dfs(
            graph: &Graph,
            node_idx: usize,
            visited: &mut HashSet<usize>,
            result: &mut Vec<usize>,
        ) -> Result<(), ComputeError> {
            if visited.contains(&node_idx) {
                return Ok(());
            }
            visited.insert(node_idx);

            let node = graph
                .nodes
                .get(node_idx)
                .ok_or_else(|| ComputeError::IndexError {
                    message: format!("node index out of bounds: {node_idx}"),
                })?;

            if let Node::Operation(_, input_indices) = node {
                for &input_idx in input_indices {
                    dfs(graph, input_idx, visited, result)?;
                }
            }

            result.push(node_idx);
            Ok(())
        }

        dfs(self, start_idx, &mut visited, &mut result)?;
        Ok(result)
    }

    pub fn get_gradient(&self, node_idx: usize) -> Option<&Tensor> {
        self.gradients.get(&node_idx)
    }

    pub fn zero_grad(&mut self) {
        self.gradients.clear();
    }

    pub fn node_requires_grad(&self, node_idx: usize) -> bool {
        match self.nodes.get(node_idx) {
            Some(Node::Parameter(_, requires_grad)) => *requires_grad,
            Some(Node::Operation(_, _)) => true,
            Some(Node::Input(_)) => false,
            None => false,
        }
    }

    pub fn get_parameter_mut(&mut self, node_idx: usize) -> Result<&mut Tensor, ComputeError> {
        match self.nodes.get_mut(node_idx) {
            Some(Node::Parameter(t, true)) => Ok(t),
            Some(Node::Parameter(_, false)) => Err(ComputeError::InvalidOperation {
                message: "parameter does not require grad".to_string(),
            }),
            _ => Err(ComputeError::InvalidOperation {
                message: "parameter expected".to_string(),
            }),
        }
    }

    pub fn get_tensor(&self, node_idx: usize) -> Result<Tensor, ComputeError> {
        self.forward(node_idx)
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

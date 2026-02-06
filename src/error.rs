use core::fmt;

#[derive(Debug, Clone)]
pub enum ComputeError {
    ShapeMismatch {
        expected: usize,
        got: usize,
    },
    DimensionError {
        message: String,
    },
    InputCountError {
        expected: usize,
        got: usize,
    },
    BroadcastError {
        dim: usize,
        shape1: usize,
        shape2: usize,
    },
    InvalidOperation {
        message: String,
    },
    IndexError {
        message: String,
    },
}

impl fmt::Display for ComputeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ComputeError::ShapeMismatch { expected, got } => {
                write!(f, "shape mismatch: expected {expected} elements, got {got}")
            }
            ComputeError::DimensionError { message } => write!(f, "dimension error: {message}"),
            ComputeError::InputCountError { expected, got } => {
                write!(f, "input count error: expected {expected}, got {got}")
            }
            ComputeError::BroadcastError {
                dim,
                shape1,
                shape2,
            } => {
                write!(f, "broadcast error at dim {dim}: {shape1} vs {shape2}")
            }
            ComputeError::InvalidOperation { message } => write!(f, "invalid operation: {message}"),
            ComputeError::IndexError { message } => write!(f, "index error: {message}"),
        }
    }
}

impl std::error::Error for ComputeError {}

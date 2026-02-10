use crate::error::ComputeError;

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

pub fn ravel_index(idx: &[usize], shape: &[usize]) -> Result<usize, ComputeError> {
    if idx.len() != shape.len() {
        return Err(ComputeError::DimensionError {
            message: format!(
                "rank mismatch: idx rank {} != shape rank {}",
                idx.len(),
                shape.len()
            ),
        });
    }

    let strides = compute_strides(shape);
    let mut flat = 0usize;
    for i in 0..shape.len() {
        if idx[i] >= shape[i] {
            return Err(ComputeError::IndexError {
                message: format!(
                    "index {} out of bounds for dim {} with size {}",
                    idx[i], i, shape[i]
                ),
            });
        }
        flat += idx[i] * strides[i];
    }
    Ok(flat)
}

pub fn unravel_index(flat: usize, shape: &[usize]) -> Result<Vec<usize>, ComputeError> {
    let numel: usize = shape.iter().product();
    if flat >= numel {
        return Err(ComputeError::IndexError {
            message: format!("flat index {flat} out of bounds for {numel} elements"),
        });
    }

    let strides = compute_strides(shape);
    let mut out = vec![0; shape.len()];
    let mut rem = flat;
    for i in 0..shape.len() {
        out[i] = rem / strides[i];
        rem %= strides[i];
    }
    Ok(out)
}

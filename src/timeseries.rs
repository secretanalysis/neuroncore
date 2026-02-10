use crate::error::ComputeError;

pub fn windows_1d(x: &[f32], window: usize, stride: usize) -> Result<Vec<Vec<f32>>, ComputeError> {
    if window == 0 || stride == 0 {
        return Err(ComputeError::InvalidOperation {
            message: "window and stride must be >= 1".to_string(),
        });
    }
    if x.len() < window {
        return Ok(Vec::new());
    }

    let mut out = Vec::new();
    let mut start = 0usize;
    while start + window <= x.len() {
        out.push(x[start..start + window].to_vec());
        start += stride;
    }
    Ok(out)
}

pub fn windows_2d(
    x: &[Vec<f32>],
    window: usize,
    stride: usize,
) -> Result<Vec<Vec<Vec<f32>>>, ComputeError> {
    if window == 0 || stride == 0 {
        return Err(ComputeError::InvalidOperation {
            message: "window and stride must be >= 1".to_string(),
        });
    }
    if x.len() < window {
        return Ok(Vec::new());
    }

    let mut out = Vec::new();
    let mut start = 0usize;
    while start + window <= x.len() {
        out.push(x[start..start + window].to_vec());
        start += stride;
    }
    Ok(out)
}

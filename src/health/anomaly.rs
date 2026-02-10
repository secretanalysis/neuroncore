use crate::error::ComputeError;

pub fn zscore_scores(values: &[f32]) -> Result<Vec<f32>, ComputeError> {
    if values.is_empty() {
        return Ok(Vec::new());
    }

    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let var = values
        .iter()
        .map(|v| {
            let d = *v - mean;
            d * d
        })
        .sum::<f32>()
        / values.len() as f32;
    let std = var.sqrt();

    if std == 0.0 {
        return Ok(vec![0.0; values.len()]);
    }

    Ok(values.iter().map(|v| ((*v - mean) / std).abs()).collect())
}

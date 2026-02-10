use neuroncore::health::anomaly::zscore_scores;

#[test]
fn anomaly_scores_detect_outlier() {
    let scores = zscore_scores(&[1.0, 1.0, 1.0, 10.0]).unwrap();
    let max_idx = scores
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    assert_eq!(max_idx, 3);
}

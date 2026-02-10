use neuroncore::Tensor;

#[test]
fn transpose_2d_correctness() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
    let y = x.transpose_2d().unwrap();
    assert_eq!(y.shape(), &[3, 2]);
    assert_eq!(y.data(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn matmul_correctness() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
    let c = a.matmul(&b).unwrap();
    assert_eq!(c.shape(), &[2, 2]);
    assert_eq!(c.data(), &[19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn relu_correctness() {
    let x = Tensor::new(vec![-1.0, 0.0, 2.0], vec![3]).unwrap();
    let y = x.relu().unwrap();
    assert_eq!(y.data(), &[0.0, 0.0, 2.0]);
}

#[test]
fn sum_none_correctness() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
    let y = x.sum(None).unwrap();
    assert_eq!(y.shape(), &[1]);
    assert_eq!(y.data(), &[6.0]);
}

#[test]
fn sum_dim_correctness() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();

    let along_rows = x.sum(Some(0)).unwrap();
    assert_eq!(along_rows.shape(), &[1, 2]);
    assert_eq!(along_rows.data(), &[4.0, 6.0]);

    let along_cols = x.sum(Some(1)).unwrap();
    assert_eq!(along_cols.shape(), &[2, 1]);
    assert_eq!(along_cols.data(), &[3.0, 7.0]);
}

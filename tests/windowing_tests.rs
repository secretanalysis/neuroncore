use neuroncore::timeseries::{windows_1d, windows_2d};

#[test]
fn windows_1d_stride_1() {
    let x = [1.0, 2.0, 3.0, 4.0];
    let out = windows_1d(&x, 2, 1).unwrap();
    assert_eq!(out, vec![vec![1.0, 2.0], vec![2.0, 3.0], vec![3.0, 4.0]]);
}

#[test]
fn windows_1d_stride_2() {
    let x = [1.0, 2.0, 3.0, 4.0];
    let out = windows_1d(&x, 3, 2).unwrap();
    assert_eq!(out, vec![vec![1.0, 2.0, 3.0]]);
}

#[test]
fn windows_2d_smoke() {
    let x = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
    let out = windows_2d(&x, 2, 1).unwrap();
    assert_eq!(
        out,
        vec![
            vec![vec![1.0], vec![2.0]],
            vec![vec![2.0], vec![3.0]],
            vec![vec![3.0], vec![4.0]]
        ]
    );
}

use neuroncore::ops::{AddOp, DivideOp, InvertibleOp, LogOp, MultiplyOp, Op, SubtractOp};
use neuroncore::Tensor;

const TOL: f32 = 1e-5;

fn assert_close(a: &[f32], b: &[f32], label: &str) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch");
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (x - y).abs() < TOL,
            "{label}[{i}]: {x} vs {y} (diff={})",
            (x - y).abs()
        );
    }
}

// ---- AddOp round-trip ----

#[test]
fn add_invert_solve_for_0() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
    let b = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]).unwrap();
    let out = AddOp.forward(&[a.clone(), b.clone()]).unwrap();
    let recovered = AddOp.invert(&out, &[None, Some(&b)], 0).unwrap();
    assert_close(recovered.data(), a.data(), "add solve_for=0");
}

#[test]
fn add_invert_solve_for_1() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
    let b = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]).unwrap();
    let out = AddOp.forward(&[a.clone(), b.clone()]).unwrap();
    let recovered = AddOp.invert(&out, &[Some(&a), None], 1).unwrap();
    assert_close(recovered.data(), b.data(), "add solve_for=1");
}

// ---- SubtractOp round-trip ----

#[test]
fn sub_invert_solve_for_0() {
    let a = Tensor::new(vec![10.0, 20.0], vec![2]).unwrap();
    let b = Tensor::new(vec![3.0, 7.0], vec![2]).unwrap();
    let out = SubtractOp.forward(&[a.clone(), b.clone()]).unwrap();
    // out = a - b, solve a = out + b
    let recovered = SubtractOp.invert(&out, &[None, Some(&b)], 0).unwrap();
    assert_close(recovered.data(), a.data(), "sub solve_for=0");
}

#[test]
fn sub_invert_solve_for_1() {
    let a = Tensor::new(vec![10.0, 20.0], vec![2]).unwrap();
    let b = Tensor::new(vec![3.0, 7.0], vec![2]).unwrap();
    let out = SubtractOp.forward(&[a.clone(), b.clone()]).unwrap();
    // out = a - b, solve b = a - out
    let recovered = SubtractOp.invert(&out, &[Some(&a), None], 1).unwrap();
    assert_close(recovered.data(), b.data(), "sub solve_for=1");
}

// ---- MultiplyOp round-trip ----

#[test]
fn mul_invert_solve_for_0() {
    let a = Tensor::new(vec![2.0, 3.0, 4.0], vec![3]).unwrap();
    let b = Tensor::new(vec![5.0, 6.0, 7.0], vec![3]).unwrap();
    let out = MultiplyOp.forward(&[a.clone(), b.clone()]).unwrap();
    let recovered = MultiplyOp.invert(&out, &[None, Some(&b)], 0).unwrap();
    assert_close(recovered.data(), a.data(), "mul solve_for=0");
}

#[test]
fn mul_invert_solve_for_1() {
    let a = Tensor::new(vec![2.0, 3.0, 4.0], vec![3]).unwrap();
    let b = Tensor::new(vec![5.0, 6.0, 7.0], vec![3]).unwrap();
    let out = MultiplyOp.forward(&[a.clone(), b.clone()]).unwrap();
    let recovered = MultiplyOp.invert(&out, &[Some(&a), None], 1).unwrap();
    assert_close(recovered.data(), b.data(), "mul solve_for=1");
}

// ---- DivideOp round-trip ----

#[test]
fn div_invert_solve_for_0() {
    let a = Tensor::new(vec![12.0, 15.0], vec![2]).unwrap();
    let b = Tensor::new(vec![3.0, 5.0], vec![2]).unwrap();
    let out = DivideOp.forward(&[a.clone(), b.clone()]).unwrap();
    // out = a/b, solve a = out*b
    let recovered = DivideOp.invert(&out, &[None, Some(&b)], 0).unwrap();
    assert_close(recovered.data(), a.data(), "div solve_for=0");
}

#[test]
fn div_invert_solve_for_1() {
    let a = Tensor::new(vec![12.0, 15.0], vec![2]).unwrap();
    let b = Tensor::new(vec![3.0, 5.0], vec![2]).unwrap();
    let out = DivideOp.forward(&[a.clone(), b.clone()]).unwrap();
    // out = a/b, solve b = a/out
    let recovered = DivideOp.invert(&out, &[Some(&a), None], 1).unwrap();
    assert_close(recovered.data(), b.data(), "div solve_for=1");
}

// ---- LogOp round-trip ----

#[test]
fn log_invert_round_trip() {
    let x = Tensor::new(vec![1.0, 2.718281828, 0.5], vec![3]).unwrap();
    let out = LogOp.forward(&[x.clone()]).unwrap();
    // out = ln(x), solve x = exp(out)
    let recovered = LogOp.invert(&out, &[None], 0).unwrap();
    assert_close(recovered.data(), x.data(), "log round-trip");
}

// ---- 2D tensors ----

#[test]
fn add_invert_2d() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let b = Tensor::new(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]).unwrap();
    let out = AddOp.forward(&[a.clone(), b.clone()]).unwrap();
    let recovered = AddOp.invert(&out, &[None, Some(&b)], 0).unwrap();
    assert_close(recovered.data(), a.data(), "add 2d solve_for=0");
    assert_eq!(recovered.shape(), &[2, 2]);
}

// ---- Negative tests: validation ----

#[test]
fn invert_wrong_arity() {
    let out = Tensor::new(vec![1.0], vec![1]).unwrap();
    // AddOp expects 2 known entries, give 1
    let result = AddOp.invert(&out, &[None], 0);
    assert!(result.is_err());
}

#[test]
fn invert_solve_for_out_of_bounds() {
    let out = Tensor::new(vec![1.0], vec![1]).unwrap();
    let t = Tensor::new(vec![1.0], vec![1]).unwrap();
    let result = AddOp.invert(&out, &[Some(&t), None], 5);
    assert!(result.is_err());
}

#[test]
fn invert_solve_for_not_none() {
    let out = Tensor::new(vec![1.0], vec![1]).unwrap();
    let t = Tensor::new(vec![1.0], vec![1]).unwrap();
    // Both are Some — solve_for=0 but known[0] is Some
    let result = AddOp.invert(&out, &[Some(&t), Some(&t)], 0);
    assert!(result.is_err());
}

#[test]
fn invert_missing_other_input() {
    let out = Tensor::new(vec![1.0], vec![1]).unwrap();
    // Both are None — known[1] should be Some
    let result = AddOp.invert(&out, &[None, None], 0);
    assert!(result.is_err());
}

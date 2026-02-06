use neuroncore::layers::{Layer, Linear};
use neuroncore::losses::MSELoss;
use neuroncore::ops::ReluOp;
use neuroncore::optim::{Optimizer, SGD};
use neuroncore::{Graph, Tensor};

#[test]
fn two_layer_regression_smoke() {
    let mut graph = Graph::new();

    // x: 1x2, y: 1x1
    let x_idx = graph.add_input(Tensor::new(vec![0.5, -0.5], vec![1, 2]).unwrap());
    let y_idx = graph.add_input(Tensor::new(vec![0.75], vec![1, 1]).unwrap());

    // Two-layer network: 2 -> 3 -> 1
    let layer1 = Linear::new(&mut graph, 2, 3, 123).unwrap();
    let layer2 = Linear::new(&mut graph, 3, 1, 456).unwrap();

    let mut params = Vec::new();
    params.extend(layer1.parameters());
    params.extend(layer2.parameters());

    let mut opt = SGD::new(params, 0.05, Some(0.9));

    // Simple training loop; just verify it runs and the loss stays finite.
    let mut initial_loss = None;
    let mut final_loss = None;
    for _epoch in 0..50 {
        let h = layer1.forward(&mut graph, x_idx).unwrap();
        let h_relu = graph.apply_op(ReluOp, &[h]);
        let out = layer2.forward(&mut graph, h_relu).unwrap();
        let loss_idx = MSELoss::compute(&mut graph, out, y_idx).unwrap();
        let loss = graph.forward(loss_idx).unwrap();

        opt.zero_grad(&mut graph);
        graph.backward(loss_idx).unwrap();
        opt.step(&mut graph).unwrap();

        let loss_value = loss.data()[0];
        assert!(loss_value.is_finite());
        if initial_loss.is_none() {
            initial_loss = Some(loss_value);
        }
        final_loss = Some(loss_value);
    }

    if let (Some(start), Some(end)) = (initial_loss, final_loss) {
        // Smoke check: should not diverge wildly from the initial loss.
        assert!(end <= start * 2.5);
    }
}

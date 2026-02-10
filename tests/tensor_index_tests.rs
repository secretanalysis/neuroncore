use neuroncore::tensor_index::{ravel_index, unravel_index};

#[test]
fn round_trip_flat_and_indices_for_2d() {
    let shape = [2, 3];
    for flat in 0..6 {
        let idx = unravel_index(flat, &shape).unwrap();
        assert_eq!(ravel_index(&idx, &shape).unwrap(), flat);
    }

    let indices = vec![vec![0, 0], vec![1, 2], vec![0, 1]];
    for idx in indices {
        let flat = ravel_index(&idx, &shape).unwrap();
        assert_eq!(unravel_index(flat, &shape).unwrap(), idx);
    }
}

#[test]
fn round_trip_flat_and_indices_for_3d() {
    let shape = [3, 3, 2];
    for flat in 0..18 {
        let idx = unravel_index(flat, &shape).unwrap();
        assert_eq!(ravel_index(&idx, &shape).unwrap(), flat);
    }

    let indices = vec![vec![0, 0, 0], vec![2, 1, 1], vec![1, 2, 0]];
    for idx in indices {
        let flat = ravel_index(&idx, &shape).unwrap();
        assert_eq!(unravel_index(flat, &shape).unwrap(), idx);
    }
}

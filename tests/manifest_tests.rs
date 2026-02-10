use neuroncore::run_manifest::{hash_bytes_sha256, RunManifest};

#[test]
fn same_inputs_same_manifest_hash() {
    let m = RunManifest {
        crate_version: "0.1.0".to_string(),
        git_commit: Some("abc123".to_string()),
        seed: Some(42),
        config_hash: hash_bytes_sha256(b"cfg"),
        input_hash: hash_bytes_sha256(b"input"),
        feature_schema_hash: hash_bytes_sha256(b"schema"),
    };

    assert_eq!(m.manifest_hash(), m.manifest_hash());
}

#[test]
fn changing_one_field_changes_manifest_hash() {
    let m1 = RunManifest {
        crate_version: "0.1.0".to_string(),
        git_commit: Some("abc123".to_string()),
        seed: Some(42),
        config_hash: hash_bytes_sha256(b"cfg"),
        input_hash: hash_bytes_sha256(b"input"),
        feature_schema_hash: hash_bytes_sha256(b"schema"),
    };
    let mut m2 = m1.clone();
    m2.seed = Some(43);

    assert_ne!(m1.manifest_hash(), m2.manifest_hash());
}

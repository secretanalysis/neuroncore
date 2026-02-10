#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunManifest {
    pub crate_version: String,
    pub git_commit: Option<String>,
    pub seed: Option<u64>,
    pub config_hash: String,
    pub input_hash: String,
    pub feature_schema_hash: String,
}

pub fn hash_bytes_sha256(bytes: &[u8]) -> String {
    // Deterministic FNV-1a 64-bit for lightweight MVP hashing API.
    let mut hash: u64 = 0xcbf29ce484222325;
    for b in bytes {
        hash ^= u64::from(*b);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

impl RunManifest {
    pub fn manifest_hash(&self) -> String {
        let serialized = format!(
            "crate_version={};git_commit={};seed={};config_hash={};input_hash={};feature_schema_hash={};",
            self.crate_version,
            self.git_commit.as_deref().unwrap_or(""),
            self.seed.map(|s| s.to_string()).unwrap_or_default(),
            self.config_hash,
            self.input_hash,
            self.feature_schema_hash,
        );
        hash_bytes_sha256(serialized.as_bytes())
    }
}

use std::path::Path;

use neuroncore::health::anomaly::zscore_scores;
use neuroncore::industrial::ingest::IngestSource;
use neuroncore::industrial::replay::ReplaySource;
use neuroncore::industrial::schema::IndustrialRecord;

fn main() {
    let path = Path::new("tests/fixtures/industrial_sample.ndjson");
    let mut replay = ReplaySource::from_path(path).expect("open replay fixture");

    let mut stream = Vec::new();
    while let Some(record) = replay.next().expect("read replay") {
        if let IndustrialRecord::SensorSample(sample) = record {
            stream.extend(sample.channels);
        }
    }

    let scores = zscore_scores(&stream).expect("score");
    let mut paired: Vec<(usize, f32)> = scores.into_iter().enumerate().collect();
    paired.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    for (idx, score) in paired.into_iter().take(3) {
        println!("index={idx} score={score:.4}");
    }
}

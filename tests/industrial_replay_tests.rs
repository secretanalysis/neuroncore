use std::path::Path;

use neuroncore::industrial::ingest::IngestSource;
use neuroncore::industrial::replay::ReplaySource;
use neuroncore::industrial::schema::IndustrialRecord;

#[test]
fn replay_reads_fixture_records() {
    let mut src =
        ReplaySource::from_path(Path::new("tests/fixtures/industrial_sample.ndjson")).unwrap();

    let first = src.next().unwrap().unwrap();
    assert!(matches!(first, IndustrialRecord::MachineState(_)));

    let second = src.next().unwrap().unwrap();
    assert!(matches!(second, IndustrialRecord::SensorSample(_)));

    let third = src.next().unwrap().unwrap();
    assert!(matches!(third, IndustrialRecord::ToolEvent(_)));

    assert!(src.next().unwrap().is_none());
}

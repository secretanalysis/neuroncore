#![cfg(feature = "mtconnect")]

use neuroncore::industrial::mtconnect::parse_current_xml;
use neuroncore::industrial::schema::IndustrialRecord;

#[test]
fn parse_mtconnect_fixture_minimal() {
    let xml = std::fs::read_to_string("tests/fixtures/mtconnect_current.xml").unwrap();
    let recs = parse_current_xml(&xml).unwrap();
    assert_eq!(recs.len(), 1);
    assert!(matches!(recs[0], IndustrialRecord::MachineState(_)));
}

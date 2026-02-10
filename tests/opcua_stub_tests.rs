#![cfg(feature = "opcua")]

use neuroncore::industrial::opcua::{map_snapshot, OpcuaNodeSnapshot};
use neuroncore::industrial::schema::IndustrialRecord;

#[test]
fn opcua_stub_maps_snapshot() {
    let rec = map_snapshot(OpcuaNodeSnapshot {
        ts: 10,
        spindle_rpm: Some(1200.0),
        feed_rate: Some(300.0),
        program: Some("P1".to_string()),
    });
    assert!(matches!(rec, IndustrialRecord::MachineState(_)));
}

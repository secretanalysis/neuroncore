use crate::industrial::schema::{IndustrialRecord, MachineState};

#[derive(Clone, Debug)]
pub struct OpcuaNodeSnapshot {
    pub ts: i64,
    pub spindle_rpm: Option<f32>,
    pub feed_rate: Option<f32>,
    pub program: Option<String>,
}

pub fn map_snapshot(snapshot: OpcuaNodeSnapshot) -> IndustrialRecord {
    IndustrialRecord::MachineState(MachineState {
        ts: snapshot.ts,
        spindle_rpm: snapshot.spindle_rpm,
        feed_rate: snapshot.feed_rate,
        program: snapshot.program,
        alarms: None,
    })
}

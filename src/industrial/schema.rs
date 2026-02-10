#[derive(Clone, Debug, PartialEq)]
pub struct MachineState {
    pub ts: i64,
    pub spindle_rpm: Option<f32>,
    pub feed_rate: Option<f32>,
    pub program: Option<String>,
    pub alarms: Option<Vec<String>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SensorSample {
    pub ts: i64,
    pub channels: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ToolEvent {
    pub ts: i64,
    pub tool_id: Option<String>,
    pub event_type: String,
}

#[derive(Clone, Debug, PartialEq)]
pub enum IndustrialRecord {
    MachineState(MachineState),
    SensorSample(SensorSample),
    ToolEvent(ToolEvent),
}

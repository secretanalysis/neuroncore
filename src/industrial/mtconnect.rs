use crate::error::ComputeError;
use crate::industrial::schema::{IndustrialRecord, MachineState};

pub fn parse_current_xml(xml: &str) -> Result<Vec<IndustrialRecord>, ComputeError> {
    fn extract_tag(xml: &str, tag: &str) -> Option<String> {
        let open = format!("<{tag}>");
        let close = format!("</{tag}>");
        let start = xml.find(&open)? + open.len();
        let end = xml[start..].find(&close)? + start;
        Some(xml[start..end].trim().to_string())
    }

    let ts = extract_tag(xml, "Timestamp")
        .and_then(|s| s.parse::<i64>().ok())
        .unwrap_or(0);
    let spindle_rpm = extract_tag(xml, "SpindleSpeed").and_then(|s| s.parse::<f32>().ok());
    let feed_rate = extract_tag(xml, "Feedrate").and_then(|s| s.parse::<f32>().ok());
    let program = extract_tag(xml, "Program");

    Ok(vec![IndustrialRecord::MachineState(MachineState {
        ts,
        spindle_rpm,
        feed_rate,
        program,
        alarms: None,
    })])
}

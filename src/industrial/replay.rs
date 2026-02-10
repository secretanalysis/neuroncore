use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::error::ComputeError;
use crate::industrial::ingest::IngestSource;
use crate::industrial::schema::{IndustrialRecord, MachineState, SensorSample, ToolEvent};

pub struct ReplaySource {
    lines: std::io::Lines<BufReader<File>>,
}

impl ReplaySource {
    pub fn from_path(path: &Path) -> Result<Self, ComputeError> {
        let file = File::open(path).map_err(|e| ComputeError::InvalidOperation {
            message: format!("failed opening replay file {}: {e}", path.display()),
        })?;
        Ok(Self {
            lines: BufReader::new(file).lines(),
        })
    }
}

fn extract_str(line: &str, key: &str) -> Option<String> {
    let marker = format!("\"{key}\":");
    let start = line.find(&marker)? + marker.len();
    let rest = line[start..].trim_start();
    if let Some(stripped) = rest.strip_prefix('"') {
        let end = stripped.find('"')?;
        return Some(stripped[..end].to_string());
    }
    None
}

fn extract_i64(line: &str, key: &str) -> Option<i64> {
    let marker = format!("\"{key}\":");
    let start = line.find(&marker)? + marker.len();
    let rest = line[start..].trim_start();
    let end = rest.find([',', '}']).unwrap_or(rest.len());
    rest[..end].trim().parse().ok()
}

fn extract_f32(line: &str, key: &str) -> Option<f32> {
    let marker = format!("\"{key}\":");
    let start = line.find(&marker)? + marker.len();
    let rest = line[start..].trim_start();
    if rest.starts_with("null") {
        return None;
    }
    let end = rest.find([',', '}']).unwrap_or(rest.len());
    rest[..end].trim().parse().ok()
}

fn extract_f32_array(line: &str, key: &str) -> Option<Vec<f32>> {
    let marker = format!("\"{key}\":[");
    let start = line.find(&marker)? + marker.len();
    let end = line[start..].find(']')? + start;
    let raw = &line[start..end];
    if raw.trim().is_empty() {
        return Some(Vec::new());
    }
    Some(
        raw.split(',')
            .filter_map(|v| v.trim().parse::<f32>().ok())
            .collect(),
    )
}

impl IngestSource for ReplaySource {
    fn next(&mut self) -> Result<Option<IndustrialRecord>, ComputeError> {
        let line = match self.lines.next() {
            Some(Ok(line)) => line,
            Some(Err(e)) => {
                return Err(ComputeError::InvalidOperation {
                    message: format!("failed reading replay line: {e}"),
                });
            }
            None => return Ok(None),
        };

        let rec_type =
            extract_str(&line, "type").ok_or_else(|| ComputeError::InvalidOperation {
                message: "missing type field in replay line".to_string(),
            })?;

        let rec = match rec_type.as_str() {
            "machine_state" => IndustrialRecord::MachineState(MachineState {
                ts: extract_i64(&line, "ts").unwrap_or(0),
                spindle_rpm: extract_f32(&line, "spindle_rpm"),
                feed_rate: extract_f32(&line, "feed_rate"),
                program: extract_str(&line, "program"),
                alarms: None,
            }),
            "sensor_sample" => IndustrialRecord::SensorSample(SensorSample {
                ts: extract_i64(&line, "ts").unwrap_or(0),
                channels: extract_f32_array(&line, "channels").unwrap_or_default(),
            }),
            "tool_event" => IndustrialRecord::ToolEvent(ToolEvent {
                ts: extract_i64(&line, "ts").unwrap_or(0),
                tool_id: extract_str(&line, "tool_id"),
                event_type: extract_str(&line, "event_type").unwrap_or_default(),
            }),
            other => {
                return Err(ComputeError::InvalidOperation {
                    message: format!("unknown record type in replay line: {other}"),
                });
            }
        };

        Ok(Some(rec))
    }
}

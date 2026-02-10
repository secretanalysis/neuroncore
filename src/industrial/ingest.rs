use crate::error::ComputeError;
use crate::industrial::schema::IndustrialRecord;

pub trait IngestSource {
    fn next(&mut self) -> Result<Option<IndustrialRecord>, ComputeError>;
}

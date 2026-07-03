pub mod simple_index;

pub use simple_index::SimpleIndex;

pub fn init_index() -> Result<SimpleIndex, Box<dyn std::error::Error>> {
    Ok(SimpleIndex::new())
}

/// Wrapper holding both in-memory indices
pub struct DualIndex {
    pub text_index: SimpleIndex,   // CLIP-L/14 vectors for text search
    pub image_index: SimpleIndex,  // SigLIP2 vectors for image search
}

impl DualIndex {
    pub fn new() -> Self {
        Self {
            text_index: SimpleIndex::new(),
            image_index: SimpleIndex::new(),
        }
    }
}

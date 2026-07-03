use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};

#[derive(Serialize, Deserialize)]
struct IndexEntry {
    image_id: i64,
    vector: Vec<f32>,
}

pub struct SimpleIndex {
    entries: Vec<IndexEntry>,
}

impl SimpleIndex {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }
    
    /// Add a vector to index
    pub fn add(&mut self, image_id: i64, vector: Vec<f32>) {
        self.entries.push(IndexEntry { image_id, vector });
    }
    
    /// Search top-k similar vectors (brute-force)
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(i64, f32)> {
        let mut results: Vec<(i64, f32)> = self.entries
            .iter()
            .map(|entry| {
                let similarity = cosine_similarity(query, &entry.vector);
                (entry.image_id, similarity)
            })
            .collect();
        
        // Sort by similarity descending
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Return top k
        if results.len() > k {
            results.truncate(k);
        }
        
        results
    }
    
    /// Clear index
    pub fn clear(&mut self) {
        self.entries.clear();
    }
    
    /// Number of vectors in index
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    
    /// Save index to JSON file
    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let file = BufWriter::new(File::create(path)?);
        serde_json::to_writer(file, &self.entries)?;
        Ok(())
    }
    
    /// Load index from JSON file
    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let file = BufReader::new(File::open(path)?);
        let entries: Vec<IndexEntry> = serde_json::from_reader(file)?;
        Ok(Self { entries })
    }
}

/// Compute cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
        
        let c = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &c) - 0.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_search() {
        let mut index = SimpleIndex::new();
        index.add(1, vec![1.0, 0.0, 0.0]);
        index.add(2, vec![0.0, 1.0, 0.0]);
        index.add(3, vec![0.0, 0.0, 1.0]);
        
        let query = vec![1.0, 0.0, 0.0];
        let results = index.search(&query, 2);
        
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 1);  // Most similar
        assert!((results[0].1 - 1.0).abs() < 1e-6);
    }
}

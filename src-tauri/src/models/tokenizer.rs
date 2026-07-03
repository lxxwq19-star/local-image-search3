use ndarray::Array2;
use tokenizers::models::bpe::BPE;
use tokenizers::tokenizer::Tokenizer;

/// CLIP-L/14 BPE tokenizer.
///
/// Encodes text into token IDs with BOS/EOS markers and pads/truncates to 77 tokens,
/// producing an `(1, 77)` i64 ndarray suitable for the CLIP-L/14 ONNX text encoder.
pub struct ClipTokenizer {
    tokenizer: Tokenizer,
    max_length: usize,
    bos_id: u32,
    eos_id: u32,
}

impl ClipTokenizer {
    /// Initialise the tokenizer.
    ///
    /// Tries to load `tokenizer.json` from the same directory as `vocab_path` first
    /// (which contains full configuration). Falls back to building a BPE model manually
    /// from `vocab_path` (vocab.json) and `merges_path` (merges.txt).
    ///
    /// - CLIP BOS token: `<|startoftext|>` (id 49406)
    /// - CLIP EOS token: `<|endoftext|>`   (id 49407)
    /// - Padding token: 0
    /// - Max length: 77
    pub fn new(vocab_path: &str, merges_path: &str) -> Result<Self, String> {
        let vocab_dir = std::path::Path::new(vocab_path)
            .parent()
            .unwrap_or(std::path::Path::new("."));
        let tokenizer_json_path = vocab_dir.join("tokenizer.json");

        let tokenizer = if tokenizer_json_path.exists() {
            eprintln!("[TOKENIZER] Loading tokenizer.json from {:?}", tokenizer_json_path);
            Tokenizer::from_file(&tokenizer_json_path)
                .map_err(|e| format!("Failed to load tokenizer.json: {}", e))?
        } else {
            eprintln!(
                "[TOKENIZER] tokenizer.json not found; building BPE from vocab + merges"
            );
            let bpe = BPE::from_file(vocab_path, merges_path)
                .build()
                .map_err(|e| format!("Failed to build BPE model: {}", e))?;
            Tokenizer::new(bpe)
        };

        Ok(Self {
            tokenizer,
            max_length: 77,
            bos_id: 49406,
            eos_id: 49407,
        })
    }

    /// Encode text into `(input_ids, attention_mask)` — both `(1, 77)` i64 ndarrays.
    ///
    /// - `input_ids`: BOS (49406) + content + EOS (49407) + padding (0)
    /// - `attention_mask`: 1 for all non-padding positions, 0 for padding
    pub fn encode(&self, text: &str) -> Result<(Array2<i64>, Array2<i64>), String> {
        // Encode WITHOUT special tokens so we can manually add BOS/EOS
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| format!("Tokenizer encode error: {}", e))?;

        let mut ids: Vec<u32> = encoding.get_ids().to_vec();

        // CLIP: prepend BOS, append EOS, cap at max_length - 2 for content
        let max_content = self.max_length - 2;
        if ids.len() > max_content {
            ids.truncate(max_content);
        }

        let mut token_ids: Vec<u32> = Vec::with_capacity(self.max_length);
        token_ids.push(self.bos_id);
        token_ids.extend(&ids);
        token_ids.push(self.eos_id);

        // Record the actual non-padded length for attention_mask
        let actual_len = token_ids.len();

        // Pad to max_length with 0
        while token_ids.len() < self.max_length {
            token_ids.push(0);
        }

        // Convert u32 → i64 for ONNX
        let ids_i64: Vec<i64> = token_ids.iter().map(|&id| id as i64).collect();

        let input_ids = Array2::from_shape_vec((1, self.max_length), ids_i64)
            .map_err(|e| format!("Failed to create input_ids ndarray: {}", e))?;

        // Build attention_mask: 1 for non-padding positions, 0 for padding
        let mut mask: Vec<i64> = Vec::with_capacity(self.max_length);
        for i in 0..self.max_length {
            mask.push(if i < actual_len { 1 } else { 0 });
        }
        let attention_mask = Array2::from_shape_vec((1, self.max_length), mask)
            .map_err(|e| format!("Failed to create attention_mask ndarray: {}", e))?;

        Ok((input_ids, attention_mask))
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, ids: &[i64]) -> Result<String, String> {
        // Filter out padding (0), BOS (49406), and EOS (49407)
        let ids_u32: Vec<u32> = ids
            .iter()
            .filter(|&&id| id != 0 && id != self.bos_id as i64 && id != self.eos_id as i64)
            .map(|&id| id as u32)
            .collect();

        self.tokenizer
            .decode(&ids_u32, false)
            .map_err(|e| format!("Tokenizer decode error: {}", e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: creates a ClipTokenizer using the model files on disk.
    ///
    /// Tests run from `src-tauri/`, so model files are at `../models/`.
    fn test_tokenizer() -> ClipTokenizer {
        let vocab_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/vocab.json");
        let merges_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/merges.txt");
        ClipTokenizer::new(vocab_path, merges_path)
            .expect("Failed to create test tokenizer (model files required)")
    }

    // ── Shape tests ──────────────────────────────────────────────

    #[test]
    fn test_attention_mask_shape() {
        let tokenizer = test_tokenizer();
        let (_, attention_mask) = tokenizer.encode("hello world").unwrap();
        assert_eq!(
            attention_mask.shape(),
            &[1, 77],
            "attention_mask must have shape (1, 77)"
        );
    }

    #[test]
    fn test_input_ids_shape() {
        let tokenizer = test_tokenizer();
        let (input_ids, _) = tokenizer.encode("hello world").unwrap();
        assert_eq!(
            input_ids.shape(),
            &[1, 77],
            "input_ids must have shape (1, 77)"
        );
    }

    // ── Attention mask value tests ───────────────────────────────

    #[test]
    fn test_attention_mask_bos_position_is_one() {
        let tokenizer = test_tokenizer();
        let (_, attention_mask) = tokenizer.encode("a").unwrap();
        // BOS token is always at position 0 → mask[0] must be 1
        assert_eq!(
            attention_mask[[0, 0]],
            1,
            "BOS position (index 0) must be 1 in attention_mask"
        );
    }

    #[test]
    fn test_attention_mask_eos_position_is_one() {
        let tokenizer = test_tokenizer();
        let (_, attention_mask) = tokenizer.encode("a").unwrap();
        // For a single token "a": ids = [BOS, id("a"), EOS, 0...0]
        // So EOS is at position 2 → mask[2] must be 1
        assert_eq!(
            attention_mask[[0, 2]],
            1,
            "EOS position (index 2 for short text) must be 1 in attention_mask"
        );
    }

    #[test]
    fn test_attention_mask_short_text_padding_is_zero() {
        let tokenizer = test_tokenizer();
        let (_, attention_mask) = tokenizer.encode("a").unwrap();
        // Short text: BOS + 1 token + EOS = 3 real tokens, rest is padding (0)
        // Padding starts at position 3
        // Check several padding positions
        for i in 3..77 {
            assert_eq!(
                attention_mask[[0, i as usize]],
                0,
                "Padding position {} in attention_mask must be 0 for short text",
                i
            );
        }
    }

    #[test]
    fn test_attention_mask_real_tokens_are_one_padding_is_zero() {
        let tokenizer = test_tokenizer();
        let (_, attention_mask) = tokenizer.encode("hello world").unwrap();

        // BOS → position 0: must be 1
        assert_eq!(attention_mask[[0, 0]], 1, "BOS must be 1");

        // Find the boundary: consecutive 1s followed by all 0s
        let mut last_one = None;
        for i in 0..77 {
            if attention_mask[[0, i]] == 1 {
                last_one = Some(i);
            }
        }
        let last_one = last_one.expect("At least one position should be 1");

        // All positions ≤ last_one must be 1
        for i in 0..=last_one {
            assert_eq!(
                attention_mask[[0, i]],
                1,
                "Real token position {} in attention_mask must be 1",
                i
            );
        }
        // All positions > last_one must be 0
        for i in (last_one + 1)..77 {
            assert_eq!(
                attention_mask[[0, i as usize]],
                0,
                "Padding position {} in attention_mask must be 0",
                i
            );
        }

        // Sanity: at least BOS (pos 0), some content, and EOS (pos ≥ 2)
        assert!(last_one >= 2, "Must have at least BOS + EOS (last_one={})", last_one);
        // Sanity: not too many tokens for short text
        assert!(last_one < 10, "Short text should have < 10 real tokens (last_one={})", last_one);
    }

    #[test]
    fn test_attention_mask_edge_shortest_text() {
        let tokenizer = test_tokenizer();
        // Empty string — should still produce BOS + EOS = 2 real tokens
        let (input_ids, attention_mask) = tokenizer.encode("").unwrap();

        // BOS at position 0
        assert_eq!(input_ids[[0, 0]], 49406, "BOS id at position 0");
        // EOS at position 1 (no content tokens)
        assert_eq!(input_ids[[0, 1]], 49407, "EOS id at position 1");
        // Padding from position 2 onwards
        for i in 2..77 {
            assert_eq!(input_ids[[0, i as usize]], 0, "Padding at position {}", i);
        }

        // attention_mask: first 2 positions are 1, rest are 0
        assert_eq!(attention_mask[[0, 0]], 1, "BOS mask must be 1");
        assert_eq!(attention_mask[[0, 1]], 1, "EOS mask must be 1");
        for i in 2..77 {
            assert_eq!(
                attention_mask[[0, i as usize]],
                0,
                "Padding mask must be 0 at position {}",
                i
            );
        }
    }

    #[test]
    fn test_attention_mask_consistency_with_input_ids() {
        let tokenizer = test_tokenizer();
        let text = "a picture of a cat sitting on a bench";
        let (input_ids, attention_mask) = tokenizer.encode(text).unwrap();

        // For every position: if input_ids is non-zero → mask must be 1
        //                       if input_ids is zero     → mask must be 0
        for i in 0..77 {
            let id_val = input_ids[[0, i]];
            let mask_val = attention_mask[[0, i]];
            if id_val != 0 {
                assert_eq!(
                    mask_val, 1,
                    "Mask must be 1 at position {} where input_ids={}",
                    i, id_val
                );
            } else {
                assert_eq!(
                    mask_val, 0,
                    "Mask must be 0 at position {} where input_ids=0",
                    i
                );
            }
        }
    }

    #[test]
    fn test_attention_mask_long_text_truncation() {
        let tokenizer = test_tokenizer();
        // A very long text that should be truncated to 75 content tokens
        // (BOS + 75 content + EOS = 77)
        let long_text = "word ".repeat(100);
        let (input_ids, attention_mask) = tokenizer.encode(&long_text).unwrap();

        // BOS at position 0
        assert_eq!(input_ids[[0, 0]], 49406, "BOS id for long text");
        // EOS at position 76 (last slot)
        assert_eq!(input_ids[[0, 76]], 49407, "EOS id at position 76 for long text");

        // All positions 0..77 should be 1 (no padding — text fills the entire length)
        for i in 0..77 {
            assert_eq!(
                attention_mask[[0, i]],
                1,
                "All mask positions must be 1 for long text (no padding), got 0 at {}",
                i
            );
        }
    }

    #[test]
    fn test_encode_returns_both_values() {
        let tokenizer = test_tokenizer();
        let result = tokenizer.encode("test");
        assert!(result.is_ok(), "encode should succeed");

        let (input_ids, attention_mask) = result.unwrap();
        // Both should be (1, 77) i64 ndarrays
        assert_eq!(input_ids.shape(), &[1, 77]);
        assert_eq!(attention_mask.shape(), &[1, 77]);
        // Data type: ndarray stores i64
        assert_eq!(input_ids[[0, 0]], 49406, "First element must be BOS");
        assert_eq!(attention_mask[[0, 0]], 1, "First mask element must be 1");
    }
}

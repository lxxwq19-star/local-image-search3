use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::db;
use crate::index::DualIndex;
use crate::models::{self, OnnxModel};
use serde_json::json;
use tauri::{Emitter, Manager};

/// Index images under a given directory.
///
/// Scans the directory for image files, computes embeddings (SigLIP2 + CLIP-L/14),
/// and stores results in both the database and in-memory index.
#[tauri::command]
pub async fn index_images(
    app: tauri::AppHandle,
    directory: String,
    force: Option<bool>,
) -> Result<String, String> {
    let force_reindex = force.unwrap_or(false);
    eprintln!("[INDEX] Starting indexing for directory: {} (force={})", directory, force_reindex);

    // Emit initial progress
    app.emit("index-progress", json!({
        "counted": 0, "indexed": 0, "errors": 0,
        "status": "started", "dir": directory
    })).ok();

    // Resolve managed state
    let onnx_state = app.state::<Mutex<OnnxModel>>();
    let db_state = app.state::<Mutex<rusqlite::Connection>>();
    let index_state = app.state::<Mutex<DualIndex>>();

    // Verify model availability
    {
        let model = onnx_state.lock().map_err(|e| e.to_string())?;
        if !model.is_available() {
            return Err("ONNX models are not loaded. Cannot index images.".to_string());
        }
    }

    let directory = directory.trim_end_matches(|c| c == '\\' || c == '/').to_string();

    // Collect image files via walkdir
    let image_extensions = [
        "jpg", "jpeg", "png", "gif", "bmp", "webp", "tiff", "tif",
    ];

    let mut image_paths: Vec<String> = Vec::new();
    let walk_dir = std::path::Path::new(&directory);

    if !walk_dir.exists() || !walk_dir.is_dir() {
        return Err(format!("Path does not exist or is not a directory: {}", directory));
    }

    for entry in walkdir::WalkDir::new(walk_dir).follow_links(true) {
        match entry {
            Ok(e) => {
                if e.file_type().is_file() {
                    if let Some(ext) = e.path().extension().and_then(|s| s.to_str()) {
                        if image_extensions.contains(&ext.to_lowercase().as_str()) {
                            image_paths.push(e.path().to_string_lossy().to_string());
                        }
                    }
                }
            }
            Err(e) => {
                eprintln!("[INDEX] Walkdir error: {}", e);
            }
        }
    }

    eprintln!("[INDEX] Found {} image files", image_paths.len());

    if image_paths.is_empty() {
        return Ok("No images found.".to_string());
    }

    // Scan and upsert subfolders (up to 2 levels) for this root path
    {
        let subfolders = db::scan_subfolders(&directory);
        if !subfolders.is_empty() {
            let conn = db_state.lock().map_err(|e| e.to_string())?;
            db::upsert_subfolders(&conn, &directory, &subfolders)
                .map_err(|e| format!("Failed to upsert subfolders: {}", e))?;
            eprintln!("[INDEX] Scanned {} subfolders for {}", subfolders.len(), directory);
        }
    }

    // Emit progress with total count after scan
    let total = image_paths.len();
    app.emit("index-progress", json!({
        "counted": total, "indexed": 0, "errors": 0,
        "status": "encoding", "total": total
    })).ok();

    // Record start time for ETA calculation
    let start_time = Instant::now();

    // Phase 1 — Compute embeddings in chunks
    let chunk_size: usize = 10;
    let total = image_paths.len();
    let mut indexed_count: i64 = 0;
    let mut errors = 0usize;

    for (_chunk_idx, chunk) in image_paths.chunks(chunk_size).enumerate() {
        // Read all files in this chunk into memory (single read per file)
        let chunk_bytes: Vec<(String, Vec<u8>)> = chunk
            .iter()
            .filter_map(|path| {
                let bytes = std::fs::read(path).ok()?;
                Some((path.clone(), bytes))
            })
            .collect();

        let mut model = onnx_state.lock().map_err(|e| e.to_string())?;

        // Pass byte slices (no clone) to avoid doubling memory
        let bytes_slices: Vec<&[u8]> = chunk_bytes
            .iter()
            .map(|(_, bytes)| bytes.as_slice())
            .collect();

        let results = model.encode_both_batch_from_bytes(&bytes_slices).map_err(|e| e.to_string())?;

        // Process each image's results
        for (idx, result) in results.into_iter().enumerate() {
            let img_path = &chunk_bytes[idx].0;
            match result {
                Ok((siglip2_vec, clip_vec)) => {
                    // Serialize both vectors
                    let siglip2_blob = models::serialize_vector(&siglip2_vec);
                    let clip_blob = models::serialize_vector(&clip_vec);

                    // Hash and metadata (use pre-read bytes)
                    let hash = compute_file_hash_from_bytes(&chunk_bytes[idx].1);
                    let dims = get_image_dimensions_from_bytes(&chunk_bytes[idx].1);
                    let folder_path = std::path::Path::new(img_path)
                        .parent()
                        .and_then(|p| p.to_str())
                        .map(|s| s.to_string());

                    // Insert / update DB
                    let info = db::ImageInfo {
                        id: None,
                        path: img_path.clone(),
                        hash,
                        width: dims.map(|(w, _)| w as i32),
                        height: dims.map(|(_, h)| h as i32),
                        main_colors: None,
                        clip_vector: Some(clip_blob),
                        clip_vector_siglip2: Some(siglip2_blob),
                        exif_camera_make: None,
                        exif_camera_model: None,
                        exif_aperture: None,
                        exif_iso: None,
                        exif_shutter_speed: None,
                        exif_focal_length: None,
                        exif_taken_at: None,
                        indexed_at: Some(std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .map_err(|e| e.to_string())?
                            .as_secs() as i64),
                        folder_path,
                    };

                    let conn = db_state.lock().map_err(|e| e.to_string())?;
                    let db_id = db::insert_or_update_image(&conn, &info)
                        .map_err(|e| e.to_string())?;

                    indexed_count += 1;

                    eprintln!(
                        "[INDEX] [{}/{}] Indexed: {} (id={})",
                        indexed_count,
                        total,
                        img_path,
                        db_id
                    );
                }
                Err(e) => {
                    eprintln!("[INDEX] Failed to encode {}: {}", img_path, e);
                    errors += 1;
                }
            }

            // Emit progress every 10 images for smooth UI updates
            if indexed_count % 10 == 0 || indexed_count == total as i64 || errors > 0 {
                // Calculate ETA (estimated time remaining in seconds)
                let elapsed = start_time.elapsed();
                let eta_seconds: Option<u64> = if indexed_count > 0 {
                    let avg_ms_per_img = elapsed.as_millis() / indexed_count as u128;
                    let remaining = (total as u128 - indexed_count as u128) * avg_ms_per_img;
                    Some((remaining / 1000) as u64)
                } else {
                    None
                };
                app.emit("index-progress", json!({
                    "counted": total,
                    "indexed": indexed_count,
                    "errors": errors,
                    "status": "encoding",
                    "eta_seconds": eta_seconds,
                })).ok();
            }
        }
    }

    // Phase 2 — Update indexed count for this path
    {
        let conn = db_state.lock().map_err(|e| e.to_string())?;
        db::update_path_count(&conn, &directory, indexed_count).ok();
    }

    // Phase 2.5 — Update indexed counts for all subfolders under this root
    {
        let conn = db_state.lock().map_err(|e| e.to_string())?;
        db::update_subfolder_counts(&conn, &directory).map_err(|e| e.to_string())?;
        eprintln!("[INDEX] Subfolder counts updated for {}", directory);
    }

    // Phase 3 — Rebuild in-memory index from DB
    {
        eprintln!("[INDEX] Rebuilding in-memory index from database...");
        let conn = db_state.lock().map_err(|e| e.to_string())?;
        let all_images = db::get_all_indexed_images(&conn).map_err(|e| e.to_string())?;

        let mut idx = index_state.lock().map_err(|e| e.to_string())?;
        idx.text_index.clear();
        idx.image_index.clear();

        for img in &all_images {
            let id = match img.id {
                Some(v) => v,
                None => continue,
            };
            if let Some(ref blob) = img.clip_vector {
                if let Some(vec) = models::deserialize_vector(blob) {
                    idx.text_index.add(id, vec);
                }
            }
            if let Some(ref blob) = img.clip_vector_siglip2 {
                if let Some(vec) = models::deserialize_vector(blob) {
                    idx.image_index.add(id, vec);
                }
            }
        }

        eprintln!(
            "[INDEX] In-memory index rebuilt: {} text, {} image vectors",
            idx.text_index.len(),
            idx.image_index.len()
        );
    }

    eprintln!("[INDEX] Indexing complete: {} images indexed", indexed_count);

    // Emit completion
    app.emit("index-progress", json!({
        "counted": total,
        "indexed": indexed_count,
        "errors": errors,
        "status": "completed"
    })).ok();

    Ok(json!({ "indexed": indexed_count, "total": total }).to_string())
}

/// Get current index status.
#[tauri::command]
pub fn get_index_status(app: tauri::AppHandle) -> Result<serde_json::Value, String> {
    let db_state = app.state::<Mutex<rusqlite::Connection>>();
    let index_state = app.state::<Mutex<DualIndex>>();
    let model_state = app.state::<Mutex<OnnxModel>>();

    let indexed_count = {
        let db_guard = db_state.lock().map_err(|e| e.to_string())?;
        db::get_indexed_count(&db_guard).map_err(|e| e.to_string())?
    };

    let idx = index_state.lock().map_err(|e| e.to_string())?;
    let model = model_state.lock().map_err(|e| e.to_string())?;

    let index_size = idx.text_index.len();
    let status = if indexed_count > 0 { "ready" } else { "empty" };

    Ok(json!({
        "indexed_count": indexed_count,
        "index_size": index_size,
        "status": status,
        "vector_indexed": index_size > 0,
        "db_count": indexed_count,
        "text_index_size": idx.text_index.len(),
        "image_index_size": idx.image_index.len(),
        "model_loaded": model.is_available(),
        "execution_provider": model.execution_provider,
        "model_error": model.startup_error,
    }))
}

/// Toggle a subfolder's enabled state.
#[tauri::command(rename_all = "camelCase")]
pub fn toggle_subfolder(app: tauri::AppHandle, subfolder_path: String) -> Result<bool, String> {
    let db_state = app.state::<Mutex<rusqlite::Connection>>();
    let conn = db_state.lock().map_err(|e| e.to_string())?;
    let enabled = db::toggle_subfolder(&conn, &subfolder_path)
        .map_err(|e| format!("Failed to toggle subfolder: {}", e))?;
    Ok(enabled)
}

/// Get subfolders for a root path.
#[tauri::command(rename_all = "camelCase")]
pub fn get_subfolders(app: tauri::AppHandle, root_path: String) -> Result<serde_json::Value, String> {
    let db_state = app.state::<Mutex<rusqlite::Connection>>();
    let conn = db_state.lock().map_err(|e| e.to_string())?;
    let subfolders = db::get_subfolders(&conn, &root_path)
        .map_err(|e| format!("Failed to get subfolders: {}", e))?;
    let result: Vec<serde_json::Value> = subfolders
        .into_iter()
        .map(|sf| {
            json!({
                "id": sf.id,
                "root_path": sf.root_path,
                "subfolder_path": sf.subfolder_path,
                "enabled": sf.enabled,
                "indexed_count": sf.indexed_count,
            })
        })
        .collect();
    Ok(json!({ "subfolders": result }))
}

// ──────────────────────────────────────────────
//  Helpers
// ──────────────────────────────────────────────

/// Compute SHA-256 hash from file bytes (hex string).
///
/// This is the preferred method when the file bytes are already in memory.
fn compute_file_hash_from_bytes(bytes: &[u8]) -> String {
    use sha2::Digest;
    
    let mut hasher = sha2::Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

/// Compute SHA-256 hash of a file (hex string).
///
/// Reads the file from disk. Prefer `compute_file_hash_from_bytes` when bytes are already available.
#[allow(dead_code)]
fn compute_file_hash(path: &str) -> String {
    match std::fs::read(path) {
        Ok(bytes) => compute_file_hash_from_bytes(&bytes),
        Err(_) => String::new(),
    }
}

/// Get image dimensions from memory bytes.
///
/// This is the preferred method when the file bytes are already in memory.
fn get_image_dimensions_from_bytes(bytes: &[u8]) -> Option<(u32, u32)> {
    imagesize::blob_size(bytes)
        .ok()
        .map(|d| (d.width as u32, d.height as u32))
}

/// Get image dimensions without loading the full image.
///
/// Reads the file from disk. Prefer `get_image_dimensions_from_bytes` when bytes are already available.
#[allow(dead_code)]
fn get_image_dimensions(path: &str) -> Option<(u32, u32)> {
    match std::fs::read(path) {
        Ok(bytes) => get_image_dimensions_from_bytes(&bytes),
        Err(_) => None,
    }
}

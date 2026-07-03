use std::sync::Mutex;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;

use crate::db;
use crate::index::DualIndex;
use crate::models::OnnxModel;
use serde_json::json;
use tauri::Manager;

const MAX_SEARCH_RESULTS: usize = 5000;
const TARGET_RESULTS: usize = 100;
const SIMILARITY_THRESHOLD: f64 = 0.8;

/// 调试日志：写入文件（release 版无终端窗口时也能查看）
fn debug_log(msg: &str) {
    let log_path = get_log_path();
    let _ = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .and_then(|mut f| {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default();
            let secs = now.as_secs();
            let millis = now.subsec_millis();
            let hrs = (secs % 86400) / 3600;
            let min = (secs % 3600) / 60;
            let sec = secs % 60;
            writeln!(f, "[{:02}:{:02}:{:02}.{:03}] {}", hrs, min, sec, millis, msg)
        });
}

/// 获取日志文件路径（%LOCALAPPDATA%\local-image-search3\search_debug.log）
fn get_log_path() -> PathBuf {
    // 尝试用 LOCALAPPDATA 环境变量，fallback 到 temp 目录
    if let Ok(appdata) = std::env::var("LOCALAPPDATA") {
        let dir = PathBuf::from(appdata).join("local-image-search3");
        let _ = std::fs::create_dir_all(&dir);
        return dir.join("search_debug.log");
    }
    std::env::temp_dir().join("local-image-search3-search-debug.log")
}

/// 加载过滤路径：启用的根路径 + 禁用的子文件夹路径
/// 用于搜索时排除已禁用路径下的图片
fn load_filter_paths(conn: &rusqlite::Connection) -> (Vec<String>, Vec<String>) {
    let enabled_roots = db::get_enabled_paths(conn).unwrap_or_default();

    // 查询禁用的子文件夹（表不存在时忽略）
    let mut disabled_subfolders = Vec::new();
    if let Ok(mut stmt) = conn.prepare("SELECT subfolder_path FROM indexed_subfolders WHERE enabled = 0") {
        if let Ok(sfolder_iter) = stmt.query_map([], |row| row.get::<_, String>(0)) {
            for sf in sfolder_iter.flatten() {
                disabled_subfolders.push(sf);
            }
        }
    }

    (enabled_roots, disabled_subfolders)
}

/// 检查图片路径是否在启用的根路径下，且不在禁用的子文件夹下
fn is_path_enabled(image_path: &str, enabled_roots: &[String], disabled_subfolders: &[String]) -> bool {
    // 归一化路径分隔符
    let normalized = image_path.replace('\\', "/");
    let mut under_enabled = false;
    for root in enabled_roots {
        let norm_root = root.replace('\\', "/");
        // 精确匹配：路径等于根路径，或以根路径+/ 开头
        if normalized == norm_root || normalized.starts_with(&format!("{}/", norm_root)) {
            under_enabled = true;
            break;
        }
    }
    if !under_enabled {
        return false;
    }
    // 检查是否在禁用的子文件夹下（精确匹配）
    for dsf in disabled_subfolders {
        let norm_dsf = dsf.replace('\\', "/");
        if normalized == norm_dsf || normalized.starts_with(&format!("{}/", norm_dsf)) {
            return false;
        }
    }
    true
}

/// 过滤搜索结果：只保留启用路径下的图片
fn filter_by_enabled_paths(images: &mut Vec<serde_json::Value>, enabled_roots: &[String], disabled_subfolders: &[String]) {
    images.retain(|img| {
        if let Some(path) = img["path"].as_str() {
            is_path_enabled(path, enabled_roots, disabled_subfolders)
        } else {
            false
        }
    });
}

/// 过滤搜索结果：80% 相似度以上优先，最多 100 张；不足则按数量补到 100 张
fn filter_results_by_similarity(images: &mut Vec<serde_json::Value>) {
    // Sort by similarity desc first
    images.sort_by(|a, b| {
        b["similarity"]
            .as_f64()
            .unwrap_or(0.0)
            .partial_cmp(&a["similarity"].as_f64().unwrap_or(0.0))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Split: high-quality (>=0.8) and low-quality (<0.8)
    let split_idx = images.iter().position(|img| {
        img["similarity"].as_f64().unwrap_or(0.0) < SIMILARITY_THRESHOLD
    }).unwrap_or(images.len());

    let mut high_quality: Vec<_> = images.drain(..split_idx).collect();
    let low_quality: Vec<_> = images.drain(..).collect();

    // Take up to TARGET_RESULTS from high quality first
    high_quality.truncate(TARGET_RESULTS);
    let taken = high_quality.len();

    // Fill remaining slots from low quality
    if taken < TARGET_RESULTS {
        let remaining = TARGET_RESULTS - taken;
        for item in low_quality.into_iter().take(remaining) {
            high_quality.push(item);
        }
    }

    *images = high_quality;
}

/// Search images by text query using CLIP-L/14 text encoder.
///
/// 1. Encode query text → 768-dim normalized vector with CLIP-L/14
/// 2. Search in-memory text index (cosine similarity)
/// 3. Return top-k results with metadata from database
#[tauri::command(rename_all = "camelCase")]
pub fn search_by_text(
    app: tauri::AppHandle,
    query: String,
    topK: usize,
) -> Result<serde_json::Value, String> {
    let top_k = topK.min(MAX_SEARCH_RESULTS);
    debug_log(&format!("[SEARCH] Text query: \"{}\" (top_k={})", query, top_k));

    // Check model availability
    let model_state = app.state::<Mutex<OnnxModel>>();
    let mut onnx = model_state.lock().map_err(|e| e.to_string())?;
    if !onnx.is_available() {
        return Err("Semantic model unavailable. Cannot search.".to_string());
    }

    // Encode text query
    let query_vector = onnx.encode_text_clip_large(&query)?;
    let vec_dim = query_vector.len();
    debug_log(&format!("[SEARCH] Query vector dimension: {}", vec_dim));
    drop(onnx);

    // Search in-memory text index
    let index_state = app.state::<Mutex<DualIndex>>();
    let mut images = Vec::new();

    let index_empty = {
        let idx = index_state.lock().map_err(|e| e.to_string())?;
        idx.text_index.len() == 0
    };

    if !index_empty {
        let idx = index_state.lock().map_err(|e| e.to_string())?;
        // Fetch extra in case of filename fallback merging
        let vec_results = idx.text_index.search(&query_vector, top_k * 2);
        drop(idx);

        let db_state = app.state::<Mutex<rusqlite::Connection>>();
        let conn = db_state.lock().map_err(|e| e.to_string())?;

        debug_log(&format!("[SEARCH] Found {} raw candidates, top scores: {:?}",
            vec_results.len(),
            vec_results.iter().take(3).map(|(_, s)| s).collect::<Vec<_>>()
        ));

        for &(image_id, similarity) in &vec_results {
            if similarity < -1.0 {
                continue;
            }
            if let Ok(Some(info)) = db::get_image_by_id(&conn, image_id) {
                images.push(json!({
                    "id": image_id,
                    "path": info.path,
                    "similarity": similarity,
                    "width": info.width,
                    "height": info.height,
                }));
                if images.len() >= top_k {
                    break;
                }
            }
        }
    }

    // Strategy 2: Filename/keyword match (supplement when needed)
    if images.len() < top_k {
        let db_state = app.state::<Mutex<rusqlite::Connection>>();
        let conn = db_state.lock().map_err(|e| e.to_string())?;
        let seen_ids: std::collections::HashSet<i64> =
            images.iter().filter_map(|img| img["id"].as_i64()).collect();

        let like_results =
            db::search_images_by_text(&conn, &query, top_k * 2).unwrap_or_default();
        for info in &like_results {
            if let Some(id) = info.id {
                if !seen_ids.contains(&id) {
                    let file_name = info
                        .path
                        .rsplit(|c| c == '/' || c == '\\')
                        .next()
                        .unwrap_or("");
                    let relevance = compute_relevance(&query, file_name);
                    images.push(json!({
                        "id": id,
                        "path": info.path,
                        "similarity": relevance,
                        "width": info.width,
                        "height": info.height,
                    }));
                    if images.len() >= top_k {
                        break;
                    }
                }
            }
        }
    }

    // Apply similarity threshold filter: 80%+ priority, max 100 results
    // 先按启用路径过滤
    let db_state = app.state::<Mutex<rusqlite::Connection>>();
    let conn = db_state.lock().map_err(|e| e.to_string())?;
    let (enabled_roots, disabled_subfolders) = load_filter_paths(&conn);
    filter_by_enabled_paths(&mut images, &enabled_roots, &disabled_subfolders);

    filter_results_by_similarity(&mut images);

    Ok(json!({ "query": query, "count": images.len(), "images": images }))
}

/// Search images by image query using SigLIP2 vision encoder.
///
/// 1. Encode query image → 1024-dim normalized vector with SigLIP2
/// 2. Search in-memory image index (cosine similarity)
/// 3. Return top-k results with metadata from database
#[tauri::command(rename_all = "camelCase")]
pub fn search_by_image(
    app: tauri::AppHandle,
    imagePath: String,
    topK: usize,
) -> Result<serde_json::Value, String> {
    let top_k = topK.min(MAX_SEARCH_RESULTS);
    debug_log(&format!("[SEARCH] Image query: \"{}\" (top_k={})", imagePath, top_k));
    debug_log("[SEARCH] Step 1: Checking model availability...");
    // Check model availability
    let model_state = app.state::<Mutex<OnnxModel>>();
    let mut onnx = model_state.lock().map_err(|e| e.to_string())?;
    if !onnx.is_available() {
        return Err("Semantic model unavailable. Cannot search.".to_string());
    }
    debug_log("[SEARCH] Step 2: Verifying image file exists...");
    // Verify image file exists
    if !std::path::Path::new(&imagePath).exists() {
        return Err(format!("Image file not found: {}", imagePath));
    }
    debug_log("[SEARCH] Step 3: Encoding image with SigLIP2...");
    // Encode query image
    let query_vector = match onnx.encode_image_siglip2(&imagePath) {
        Ok(v) => v,
        Err(e) => {
            debug_log(&format!("[SEARCH] ERROR: Failed to encode image: {}", e));
            return Err(format!("图片编码失败（可能格式不支持或损坏）: {}", e));
        }
    };
    debug_log(&format!("[SEARCH] Step 4: Encoding complete, vector dimension: {}", query_vector.len()));
    drop(onnx);

    // Search in-memory image index
    let index_state = app.state::<Mutex<DualIndex>>();
    let results = {
        let idx = index_state.lock().map_err(|e| e.to_string())?;
        if idx.image_index.len() == 0 {
            return Ok(json!({
                "query_image": imagePath,
                "images": [],
                "count": 0,
                "message": "Image index is empty. Please index some images first."
            }));
        }
        idx.image_index.search(&query_vector, top_k)
    };

    debug_log(&format!("[SEARCH] Found {} candidates from image index", results.len()));

    // Lookup image info from DB + filter in one db lock scope
    let db_state = app.state::<Mutex<rusqlite::Connection>>();
    let conn = db_state.lock().map_err(|e| e.to_string())?;

    let mut images = Vec::new();
    for (image_id, similarity) in &results {
        if *similarity < -1.0 {
            continue;
        }
        if let Ok(Some(info)) = db::get_image_by_id(&conn, *image_id) {
            images.push(json!({
                "id": image_id,
                "path": info.path,
                "similarity": similarity,
                "width": info.width,
                "height": info.height,
            }));
        }
    }

    // Apply similarity threshold filter: 80%+ priority, max 100 results
    // 先按启用路径过滤（在同一个 conn lock 作用域内）
    let (enabled_roots, disabled_subfolders) = load_filter_paths(&conn);
    filter_by_enabled_paths(&mut images, &enabled_roots, &disabled_subfolders);

    filter_results_by_similarity(&mut images);

    debug_log(&format!("[SEARCH] Returning {} results after filtering", images.len()));
    Ok(json!({ "query_image": imagePath, "count": images.len(), "images": images }))
}

/// Compute a simple relevance score for text matching against filename
fn compute_relevance(query: &str, filename: &str) -> f32 {
    let query_lower = query.to_lowercase();
    let file_lower = filename.to_lowercase();
    if file_lower.contains(&query_lower) {
        return 0.95;
    }
    let query_words: Vec<&str> = query_lower.split_whitespace().collect();
    let mut match_count = 0;
    for word in &query_words {
        if !word.is_empty() && file_lower.contains(word) {
            match_count += 1;
        }
    }
    if !query_words.is_empty() && match_count > 0 {
        return 0.5 + (match_count as f32 / query_words.len() as f32) * 0.4;
    }
    let matched_chars: usize = query_lower
        .chars()
        .filter(|c| file_lower.contains(*c))
        .count();
    let total_chars = query_lower.chars().filter(|c| c.is_alphanumeric()).count();
    if total_chars > 0 {
        return 0.1 + (matched_chars as f32 / total_chars as f32) * 0.3;
    }
    0.05
}

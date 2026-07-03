use crate::db;
use serde_json::json;
use tauri::Manager;

#[tauri::command]
pub fn get_paths(app: tauri::AppHandle) -> Result<serde_json::Value, String> {
    let db_state = app.state::<std::sync::Mutex<rusqlite::Connection>>();
    let conn = db_state.lock().map_err(|e| e.to_string())?;

    let paths = db::get_all_paths(&conn).map_err(|e| e.to_string())?;

    // Compute categories by extracting parent folder name
    let mut categories: std::collections::HashMap<String, Vec<String>> = std::collections::HashMap::new();
    for p in &paths {
        let parent = std::path::Path::new(&p.path)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(&p.path)
            .to_string();
        categories.entry(parent.clone()).or_default().push(p.path.clone());
    }

    Ok(json!({
        "paths": paths,
        "categories": categories,
    }))
}

#[tauri::command]
pub fn add_path(app: tauri::AppHandle, path: String, name: String) -> Result<serde_json::Value, String> {
    let db_state = app.state::<std::sync::Mutex<rusqlite::Connection>>();
    let conn = db_state.lock().map_err(|e| e.to_string())?;

    db::add_path(&conn, &path, &name).map_err(|e| e.to_string())?;
    Ok(json!({ "ok": true, "path": path }))
}

#[tauri::command]
pub fn delete_path(app: tauri::AppHandle, path: String) -> Result<serde_json::Value, String> {
    let db_state = app.state::<std::sync::Mutex<rusqlite::Connection>>();
    let conn = db_state.lock().map_err(|e| e.to_string())?;

    // Remove images belonging to this path
    let deleted = conn.execute(
        "DELETE FROM images WHERE path LIKE ?1",
        rusqlite::params![format!("{}%", path)],
    );
    eprintln!("[PATHS] Deleted {} images from {}", deleted.unwrap_or(0), path);

    db::delete_path(&conn, &path).map_err(|e| e.to_string())?;
    Ok(json!({ "ok": true }))
}

#[tauri::command(rename_all = "camelCase")]
pub fn toggle_path(app: tauri::AppHandle, path: String) -> Result<bool, String> {
    let db_state = app.state::<std::sync::Mutex<rusqlite::Connection>>();
    let conn = db_state.lock().map_err(|e| e.to_string())?;
    db::toggle_path(&conn, &path).map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn rebuild_all_index(app: tauri::AppHandle) -> Result<String, String> {
    let db_state = app.state::<std::sync::Mutex<rusqlite::Connection>>();
    let enabled_paths = {
        let conn = db_state.lock().map_err(|e| e.to_string())?;
        let paths = db::get_all_paths(&conn).map_err(|e| e.to_string())?;
        paths.into_iter().filter(|p| p.enabled).map(|p| p.path).collect::<Vec<_>>()
    };

    if enabled_paths.is_empty() {
        return Err("No enabled paths to rebuild".to_string());
    }

    // Rebuild index for each enabled path sequentially
    for path in &enabled_paths {
        let result = super::index::index_images(
            app.clone(),
            path.clone(),
            Some(true),
        ).await;
        if let Err(e) = result {
            eprintln!("[PATHS] Failed to rebuild {}: {}", path, e);
        }
    }

    Ok(format!("Rebuilt {} paths", enabled_paths.len()))
}

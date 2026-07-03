use crate::db;
use tauri::Manager;

#[tauri::command]
pub fn get_image_info(app: tauri::AppHandle, image_id: i64) -> Result<serde_json::Value, String> {
    let db_state = app.state::<std::sync::Mutex<rusqlite::Connection>>();
    let conn = db_state.lock().map_err(|e| e.to_string())?;

    match db::get_image_by_id(&conn, image_id) {
        Ok(Some(info)) => Ok(serde_json::to_value(info).map_err(|e| e.to_string())?),
        Ok(None) => Err(format!("Image not found: id={}", image_id)),
        Err(e) => Err(format!("Database error: {}", e)),
    }
}

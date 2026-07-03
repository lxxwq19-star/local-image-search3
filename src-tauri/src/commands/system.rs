/// 用系统默认程序打开文件
#[tauri::command]
pub fn open_file(path: String) -> Result<(), String> {
    open::that(&path).map_err(|e| format!("打开文件失败: {}", e))
}

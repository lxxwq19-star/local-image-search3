use rusqlite::{Connection, params, Result};
use serde::{Deserialize, Serialize};

// ===== Index paths table =====

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexPath {
    pub id: i64,
    pub path: String,
    pub name: String,
    pub enabled: bool,
    pub indexed_count: i64,
}

pub fn init_index_paths_table(conn: &Connection) -> Result<()> {
    conn.execute(
        "CREATE TABLE IF NOT EXISTS index_paths (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL DEFAULT '',
            enabled INTEGER NOT NULL DEFAULT 1,
            indexed_count INTEGER NOT NULL DEFAULT 0,
            created_at INTEGER DEFAULT (strftime('%s', 'now'))
        )",
        [],
    )?;
    Ok(())
}

pub fn get_all_paths(conn: &Connection) -> Result<Vec<IndexPath>> {
    let mut stmt = conn.prepare(
        "SELECT id, path, name, enabled, indexed_count FROM index_paths ORDER BY id DESC"
    )?;
    let rows = stmt.query_map([], |row| {
        Ok(IndexPath {
            id: row.get(0)?,
            path: row.get(1)?,
            name: row.get(2)?,
            enabled: row.get::<_, i32>(3)? != 0,
            indexed_count: row.get(4)?,
        })
    })?;
    let mut paths = Vec::new();
    for row in rows { paths.push(row?); }
    Ok(paths)
}

pub fn add_path(conn: &Connection, path: &str, name: &str) -> Result<i64> {
    // Normalize: remove trailing slashes to match folder_path in images table
    let normalized = path.trim_end_matches(|c| c == '\\' || c == '/');
    conn.execute(
        "INSERT OR IGNORE INTO index_paths (path, name) VALUES (?1, ?2)",
        params![normalized, name],
    )?;
    let id: i64 = conn.query_row(
        "SELECT id FROM index_paths WHERE path = ?1", params![path],
        |row| row.get(0),
    )?;
    Ok(id)
}

pub fn delete_path(conn: &Connection, path: &str) -> Result<()> {
    let normalized = path.trim_end_matches(|c| c == '\\' || c == '/');
    conn.execute("DELETE FROM index_paths WHERE path = ?1", params![normalized])?;
    Ok(())
}

pub fn toggle_path(conn: &Connection, path: &str) -> Result<bool> {
    let normalized = path.trim_end_matches(|c| c == '\\' || c == '/');
    conn.execute(
        "UPDATE index_paths SET enabled = NOT enabled WHERE path = ?1",
        params![normalized],
    )?;
    let enabled: i32 = conn.query_row(
        "SELECT enabled FROM index_paths WHERE path = ?1", params![normalized],
        |row| row.get(0),
    )?;
    let enabled_bool = enabled != 0;

    // 级联：关闭时同步关闭所有子文件夹（一级+二级）
    if !enabled_bool {
        conn.execute(
            "UPDATE indexed_subfolders SET enabled = 0 WHERE root_path = ?1",
            params![normalized],
        )?;
        let l1_list: Vec<String> = {
            let mut stmt = conn.prepare("SELECT subfolder_path FROM indexed_subfolders WHERE root_path = ?1")?;
            let rows = stmt.query_map(params![normalized], |row| row.get(0))?;
            rows.collect::<Result<_>>()?
        };
        for l1 in l1_list {
            conn.execute(
                "UPDATE indexed_subfolders SET enabled = 0 WHERE root_path = ?1",
                params![l1],
            )?;
        }
    } else {
        // 级联：开启时同步开启所有子文件夹（一级+二级）
        conn.execute(
            "UPDATE indexed_subfolders SET enabled = 1 WHERE root_path = ?1",
            params![normalized],
        )?;
        let l1_list: Vec<String> = {
            let mut stmt = conn.prepare("SELECT subfolder_path FROM indexed_subfolders WHERE root_path = ?1")?;
            let rows = stmt.query_map(params![normalized], |row| row.get(0))?;
            rows.collect::<Result<_>>()?
        };
        for l1 in l1_list {
            conn.execute(
                "UPDATE indexed_subfolders SET enabled = 1 WHERE root_path = ?1",
                params![l1],
            )?;
        }
    }

    Ok(enabled_bool)
}

pub fn update_path_count(conn: &Connection, path: &str, count: i64) -> Result<()> {
    conn.execute(
        "UPDATE index_paths SET indexed_count = ?1 WHERE path = ?2",
        params![count, path],
    )?;
    Ok(())
}

pub fn get_enabled_paths(conn: &Connection) -> Result<Vec<String>> {
    let mut stmt = conn.prepare("SELECT path FROM index_paths WHERE enabled = 1")?;
    let rows = stmt.query_map([], |row| row.get(0))?;
    let mut paths = Vec::new();
    for row in rows { paths.push(row?); }
    Ok(paths)
}

// ===== Indexed subfolders table =====

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexSubfolder {
    pub id: i64,
    pub root_path: String,
    pub subfolder_path: String,
    pub enabled: bool,
    pub indexed_count: i64,
}

/// Initialize the indexed_subfolders table.
pub fn init_subfolders_table(conn: &Connection) -> Result<()> {
    conn.execute(
        "CREATE TABLE IF NOT EXISTS indexed_subfolders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            root_path TEXT NOT NULL,
            subfolder_path TEXT UNIQUE NOT NULL,
            enabled INTEGER NOT NULL DEFAULT 1,
            indexed_count INTEGER NOT NULL DEFAULT 0,
            created_at INTEGER DEFAULT (strftime('%s', 'now'))
        )",
        [],
    )?;
    Ok(())
}

/// Scan subfolders under a root path (up to 2 levels deep).
/// Returns (level1 list, level2 map: level1_path -> [level2_paths]).
pub fn scan_subfolders(root: &str) -> (Vec<String>, std::collections::HashMap<String, Vec<String>>) {
    let mut level1 = Vec::new();
    let mut level2_map = std::collections::HashMap::new();
    let root = std::path::Path::new(root);
    if !root.exists() { return (level1, level2_map); }

    // Level 1
    if let Ok(entries) = std::fs::read_dir(root) {
        for entry in entries.flatten() {
            if entry.path().is_dir() {
                level1.push(entry.path().to_string_lossy().to_string());
            }
        }
    }

    // Level 2
    for l1 in &level1 {
        let mut level2 = Vec::new();
        let p = std::path::Path::new(l1);
        if let Ok(entries) = std::fs::read_dir(p) {
            for entry in entries.flatten() {
                if entry.path().is_dir() {
                    level2.push(entry.path().to_string_lossy().to_string());
                }
            }
        }
        if !level2.is_empty() {
            level2_map.insert(l1.clone(), level2);
        }
    }

    (level1, level2_map)
}

/// Upsert subfolders into the table (insert if not exist, keep enabled state).
pub fn upsert_subfolders(conn: &Connection, root_path: &str, subfolders: &[String]) -> Result<()> {
    for sf in subfolders {
        conn.execute(
            "INSERT INTO indexed_subfolders (root_path, subfolder_path, enabled)
             VALUES (?1, ?2, COALESCE((SELECT enabled FROM indexed_subfolders WHERE subfolder_path = ?2), 1))
             ON CONFLICT(subfolder_path) DO UPDATE SET root_path = excluded.root_path",
            params![root_path, sf],
        )?;
    }
    Ok(())
}

/// Update indexed_count for all subfolders under a root path (including level 2).
pub fn update_subfolder_counts(conn: &Connection, root_path: &str) -> Result<()> {
    // Update level 1 subfolders
    let l1_list = get_subfolders(conn, root_path)?;
    for sf in &l1_list {
        let pattern = format!("{}\\%", sf.subfolder_path.replace('/', "\\"));
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM images WHERE path LIKE ?1",
            params![pattern],
            |row| row.get(0),
        ).unwrap_or(0);
        conn.execute(
            "UPDATE indexed_subfolders SET indexed_count = ?1 WHERE subfolder_path = ?2",
            params![count, sf.subfolder_path],
        )?;
    }
    // Update level 2 subfolders
    for sf in &l1_list {
        let l2_list = get_subfolders(conn, &sf.subfolder_path)?;
        for l2 in &l2_list {
            let pattern = format!("{}\\%", l2.subfolder_path.replace('/', "\\"));
            let count: i64 = conn.query_row(
                "SELECT COUNT(*) FROM images WHERE path LIKE ?1",
                params![pattern],
                |row| row.get(0),
            ).unwrap_or(0);
            conn.execute(
                "UPDATE indexed_subfolders SET indexed_count = ?1 WHERE subfolder_path = ?2",
                params![count, l2.subfolder_path],
            )?;
        }
    }
    Ok(())
}

/// Get all subfolders for a root path (with has_children flag).
pub fn get_subfolders(conn: &Connection, root_path: &str) -> Result<Vec<IndexSubfolder>> {
    let mut stmt = conn.prepare(
        "SELECT id, root_path, subfolder_path, enabled, indexed_count
         FROM indexed_subfolders WHERE root_path = ?1 ORDER BY subfolder_path"
    )?;
    let rows = stmt.query_map(params![root_path], |row| {
        Ok(IndexSubfolder {
            id: row.get(0)?,
            root_path: row.get(1)?,
            subfolder_path: row.get(2)?,
            enabled: row.get::<_, i32>(3)? != 0,
            indexed_count: row.get(4)?,
        })
    })?;
    let mut result = Vec::new();
    for row in rows { result.push(row?); }
    Ok(result)
}

/// Get all subfolders for a root path, with has_children flag.
/// Returns Vec of (IndexSubfolder, has_children).
pub fn get_subfolders_with_children(conn: &Connection, root_path: &str) -> Result<Vec<(IndexSubfolder, bool)>> {
    let subfolders = get_subfolders(conn, root_path)?;
    let mut result = Vec::new();
    for sf in subfolders {
        let has_children: i64 = conn.query_row(
            "SELECT COUNT(*) FROM indexed_subfolders WHERE root_path = ?1",
            params![sf.subfolder_path],
            |row| row.get(0),
        )?;
        result.push((sf, has_children > 0));
    }
    Ok(result)
}

/// Get all enabled subfolder paths across all roots (for search filtering).
/// Returns (root_paths, subfolder_paths) that are enabled.
pub fn get_enabled_folder_paths(conn: &Connection) -> Result<(Vec<String>, Vec<String>)> {
    // Enabled root paths
    let mut roots = Vec::new();
    let mut stmt = conn.prepare("SELECT path FROM index_paths WHERE enabled = 1")?;
    let rows = stmt.query_map([], |row| row.get(0))?;
    for row in rows { roots.push(row?); }

    // Enabled subfolders
    let mut subfolders = Vec::new();
    let mut stmt = conn.prepare("SELECT subfolder_path FROM indexed_subfolders WHERE enabled = 1")?;
    let rows = stmt.query_map([], |row| row.get(0))?;
    for row in rows { subfolders.push(row?); }

    Ok((roots, subfolders))
}

/// Toggle a subfolder's enabled state (with cascade to children).
pub fn toggle_subfolder(conn: &Connection, subfolder_path: &str) -> Result<bool> {
    conn.execute(
        "UPDATE indexed_subfolders SET enabled = NOT enabled WHERE subfolder_path = ?1",
        params![subfolder_path],
    )?;
    let enabled: i32 = conn.query_row(
        "SELECT enabled FROM indexed_subfolders WHERE subfolder_path = ?1",
        params![subfolder_path],
        |row| row.get(0),
    )?;
    let enabled_bool = enabled != 0;

    // Cascade: update all subfolders whose root_path == subfolder_path
    conn.execute(
        "UPDATE indexed_subfolders SET enabled = ?1 WHERE root_path = ?2",
        params![if enabled_bool { 1 } else { 0 }, subfolder_path],
    )?;

    Ok(enabled_bool)
}

/// Check if a path has any subfolders in the database.
pub fn has_subfolders(conn: &Connection, root_path: &str) -> Result<bool> {
    let count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM indexed_subfolders WHERE root_path = ?1",
        params![root_path],
        |row| row.get(0),
    )?;
    Ok(count > 0)
}

/// Delete subfolders under a root path (including level 2).
pub fn delete_subfolders(conn: &Connection, root_path: &str) -> Result<()> {
    // Get level 1 subfolder paths before deleting (to find level 2)
    let l1_list: Vec<String> = {
        let mut stmt = conn.prepare("SELECT subfolder_path FROM indexed_subfolders WHERE root_path = ?1")?;
        let rows = stmt.query_map(params![root_path], |row| row.get(0))?;
        rows.collect::<Result<_>>()?
    };
    // Delete level 2 subfolders (whose root_path is a level 1 path)
    for l1 in l1_list {
        conn.execute("DELETE FROM indexed_subfolders WHERE root_path = ?1", params![l1])?;
    }
    // Delete level 1 subfolders
    conn.execute(
        "DELETE FROM indexed_subfolders WHERE root_path = ?1",
        params![root_path],
    )?;
    Ok(())
}

// ===== Images table =====

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageInfo {
    pub id: Option<i64>,
    pub path: String,
    pub hash: String,
    pub width: Option<i32>,
    pub height: Option<i32>,
    pub main_colors: Option<String>,
    pub clip_vector: Option<Vec<u8>>,
    pub clip_vector_siglip2: Option<Vec<u8>>,
    pub exif_camera_make: Option<String>,
    pub exif_camera_model: Option<String>,
    pub exif_aperture: Option<f32>,
    pub exif_iso: Option<i32>,
    pub exif_shutter_speed: Option<String>,
    pub exif_focal_length: Option<f32>,
    pub exif_taken_at: Option<i64>,
    pub indexed_at: Option<i64>,
    pub folder_path: Option<String>,
}

pub fn init_db() -> Result<Connection> {
    let db_path = dirs::data_local_dir()
        .unwrap()
        .join("local-image-search")
        .join("images.db");

    if let Some(parent) = db_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }

    let conn = Connection::open(&db_path)?;

    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT UNIQUE NOT NULL,
            hash TEXT NOT NULL,
            width INTEGER,
            height INTEGER,
            main_colors TEXT,
            clip_vector BLOB,
            clip_vector_siglip2 BLOB,
            exif_camera_make TEXT,
            exif_camera_model TEXT,
            exif_aperture REAL,
            exif_iso INTEGER,
            exif_shutter_speed TEXT,
            exif_focal_length REAL,
            exif_taken_at INTEGER,
            created_at INTEGER DEFAULT (strftime('%s', 'now')),
            indexed_at INTEGER,
            folder_path TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_images_path ON images(path);
        CREATE INDEX IF NOT EXISTS idx_images_hash ON images(hash);
        CREATE INDEX IF NOT EXISTS idx_images_folder ON images(folder_path);
        CREATE TABLE IF NOT EXISTS index_paths (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL DEFAULT '',
            enabled INTEGER NOT NULL DEFAULT 1,
            indexed_count INTEGER NOT NULL DEFAULT 0,
            created_at INTEGER DEFAULT (strftime('%s', 'now'))
        );

        CREATE TABLE IF NOT EXISTS indexed_subfolders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            root_path TEXT NOT NULL,
            subfolder_path TEXT UNIQUE NOT NULL,
            enabled INTEGER NOT NULL DEFAULT 1,
            indexed_count INTEGER NOT NULL DEFAULT 0,
            created_at INTEGER DEFAULT (strftime('%s', 'now'))
        );"
    )?;

    // Migration: add clip_vector_siglip2 column for existing databases
    conn.execute_batch(
        "ALTER TABLE images ADD COLUMN clip_vector_siglip2 BLOB;"
    ).ok();  // Ignore error if column already exists

    Ok(conn)
}

fn map_row_to_image(row: &rusqlite::Row) -> rusqlite::Result<ImageInfo> {
    Ok(ImageInfo {
        id: Some(row.get(0)?),
        path: row.get(1)?,
        hash: row.get(2)?,
        width: row.get(3)?,
        height: row.get(4)?,
        main_colors: row.get(5)?,
        clip_vector: row.get(6)?,
        clip_vector_siglip2: row.get(7)?,
        exif_camera_make: row.get(8)?,
        exif_camera_model: row.get(9)?,
        exif_aperture: {
            let val: Option<f64> = row.get(10)?;
            val.map(|v| v as f32)
        },
        exif_iso: row.get(11)?,
        exif_shutter_speed: row.get(12)?,
        exif_focal_length: row.get(13)?,
        exif_taken_at: row.get(14)?,
        indexed_at: row.get(15)?,
        folder_path: row.get(16).ok(),
    })
}

pub fn insert_or_update_image(conn: &Connection, info: &ImageInfo) -> Result<i64> {
    conn.execute(
        "INSERT INTO images (path, hash, width, height, main_colors, clip_vector, clip_vector_siglip2,
                           exif_camera_make, exif_camera_model, exif_aperture, exif_iso,
                           exif_shutter_speed, exif_focal_length, exif_taken_at, indexed_at, folder_path)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16)
         ON CONFLICT(path) DO UPDATE SET
           hash=excluded.hash,
           width=excluded.width,
           height=excluded.height,
           main_colors=excluded.main_colors,
           clip_vector=excluded.clip_vector,
           clip_vector_siglip2=excluded.clip_vector_siglip2,
           exif_camera_make=excluded.exif_camera_make,
           exif_camera_model=excluded.exif_camera_model,
           exif_aperture=excluded.exif_aperture,
           exif_iso=excluded.exif_iso,
           exif_shutter_speed=excluded.exif_shutter_speed,
           exif_focal_length=excluded.exif_focal_length,
           exif_taken_at=excluded.exif_taken_at,
           indexed_at=excluded.indexed_at,
           folder_path=excluded.folder_path",
        params![
            info.path,
            info.hash,
            info.width,
            info.height,
            info.main_colors,
            info.clip_vector,
            info.clip_vector_siglip2,
            info.exif_camera_make,
            info.exif_camera_model,
            info.exif_aperture,
            info.exif_iso,
            info.exif_shutter_speed,
            info.exif_focal_length,
            info.exif_taken_at,
            info.indexed_at,
            info.folder_path,
        ],
    )?;
    Ok(conn.last_insert_rowid())
}

pub fn get_image_by_path(conn: &Connection, path: &str) -> Result<Option<ImageInfo>> {
    let mut stmt = conn.prepare(
        "SELECT id, path, hash, width, height, main_colors, clip_vector, clip_vector_siglip2,
                exif_camera_make, exif_camera_model, exif_aperture, exif_iso,
                exif_shutter_speed, exif_focal_length, exif_taken_at, indexed_at, folder_path
         FROM images WHERE path = ?1"
    )?;
    let mut rows = stmt.query(params![path])?;
    if let Some(row) = rows.next()? {
        Ok(Some(map_row_to_image(row)?))
    } else {
        Ok(None)
    }
}

pub fn delete_image_by_path(conn: &Connection, path: &str) -> Result<()> {
    conn.execute("DELETE FROM images WHERE path = ?1", params![path])?;
    Ok(())
}

pub fn get_all_indexed_images(conn: &Connection) -> Result<Vec<ImageInfo>> {
    let mut stmt = conn.prepare(
        "SELECT id, path, hash, width, height, main_colors, clip_vector, clip_vector_siglip2,
                exif_camera_make, exif_camera_model, exif_aperture, exif_iso,
                exif_shutter_speed, exif_focal_length, exif_taken_at, indexed_at, folder_path
         FROM images WHERE indexed_at IS NOT NULL"
    )?;
    let rows = stmt.query_map([], map_row_to_image)?;
    let mut images = Vec::new();
    for row in rows { images.push(row?); }
    Ok(images)
}

pub fn get_indexed_images_by_paths(conn: &Connection, paths: &[String]) -> Result<Vec<ImageInfo>> {
    if paths.is_empty() {
        return Ok(Vec::new());
    }
    let placeholders: Vec<String> = paths.iter().map(|_| "?".to_string()).collect();
    let sql = format!(
        "SELECT id, path, hash, width, height, main_colors, clip_vector,\n                exif_camera_make, exif_camera_model, exif_aperture, exif_iso,\n                exif_shutter_speed, exif_focal_length, exif_taken_at, indexed_at, folder_path\n         FROM images WHERE indexed_at IS NOT NULL AND ({})",
        placeholders.join(" OR folder_path LIKE ")
    );
    let mut stmt = conn.prepare(&sql)?;
    let params_vec: Vec<&dyn rusqlite::ToSql> = paths.iter().map(|p| p as &dyn rusqlite::ToSql).collect();
    let rows = stmt.query_map(params_vec.as_slice(), map_row_to_image)?;
    let mut images = Vec::new();
    for row in rows { images.push(row?); }
    Ok(images)
}

pub fn get_indexed_count(conn: &Connection) -> Result<i64> {
    conn.query_row(
        "SELECT COUNT(*) FROM images WHERE indexed_at IS NOT NULL",
        [],
        |row| row.get(0),
    )
}

pub fn search_images_by_text(conn: &Connection, query: &str, top_k: usize) -> Result<Vec<ImageInfo>> {
    let like_query = format!("%{}%", query.replace(' ', "%"));
    let mut stmt = conn.prepare(
        "SELECT id, path, hash, width, height, main_colors, clip_vector, clip_vector_siglip2,
                exif_camera_make, exif_camera_model, exif_aperture, exif_iso,
                exif_shutter_speed, exif_focal_length, exif_taken_at, indexed_at, folder_path
         FROM images WHERE path LIKE ?1 ORDER BY id DESC LIMIT ?2"
    )?;
    let rows = stmt.query_map(params![like_query, top_k], map_row_to_image)?;
    let mut images = Vec::new();
    for row in rows { images.push(row?); }
    Ok(images)
}

pub fn get_image_by_id(conn: &Connection, id: i64) -> Result<Option<ImageInfo>> {
    let mut stmt = conn.prepare(
        "SELECT id, path, hash, width, height, main_colors, clip_vector, clip_vector_siglip2,
                exif_camera_make, exif_camera_model, exif_aperture, exif_iso,
                exif_shutter_speed, exif_focal_length, exif_taken_at, indexed_at, folder_path
         FROM images WHERE id = ?1"
    )?;
    let mut rows = stmt.query(params![id])?;
    if let Some(row) = rows.next()? {
        Ok(Some(map_row_to_image(row)?))
    } else {
        Ok(None)
    }
}

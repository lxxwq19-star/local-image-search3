#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;
mod db;
mod index;
mod models;

use std::sync::Mutex;

fn main() {
    // 1. Initialize database
    let db = match db::init_db()
        .and_then(|conn| {
            db::init_index_paths_table(&conn)?;
            Ok(conn)
        }) {
        Ok(db) => db,
        Err(e) => {
            eprintln!("[MAIN] Failed to initialize database: {}", e);
            std::process::exit(1);
        }
    };

    // 2. Initialize in-memory index (load existing vectors from DB)
    let mut dual_index = index::DualIndex::new();
    let all_images = db::get_all_indexed_images(&db).unwrap_or_default();
    eprintln!("[MAIN] Loading {} indexed images into memory...", all_images.len());

    let mut loaded_text = 0usize;
    let mut loaded_image = 0usize;

    for img in &all_images {
        let id = match img.id {
            Some(v) => v,
            None => continue,
        };

        // CLIP-L/14 vectors (stored in clip_vector) — used for text search
        if let Some(ref blob) = img.clip_vector {
            if let Some(vec) = models::deserialize_vector(blob) {
                dual_index.text_index.add(id, vec);
                loaded_text += 1;
            } else {
                eprintln!("[MAIN] Failed to deserialize clip_vector for id={}", id);
            }
        }

        // SigLIP2 vectors (stored in clip_vector_siglip2) — used for image search
        if let Some(ref blob) = img.clip_vector_siglip2 {
            if let Some(vec) = models::deserialize_vector(blob) {
                dual_index.image_index.add(id, vec);
                loaded_image += 1;
            } else {
                eprintln!("[MAIN] Failed to deserialize clip_vector_siglip2 for id={}", id);
            }
        }
    }

    eprintln!(
        "[MAIN] In-memory index ready: {} text vectors, {} image vectors",
        loaded_text, loaded_image
    );

    // 3. Initialize ONNX model (graceful fallback)
    let onnx_model = match models::OnnxModel::new() {
        Ok(model) => {
            eprintln!(
                "[MAIN] ONNX models ready (provider: {})",
                model.execution_provider
            );
            model
        }
        Err(e) => {
            eprintln!(
                "[MAIN] ONNX unavailable ({}). Running in fallback mode.",
                e
            );
            models::OnnxModel::new_fallback(e)
        }
    };

    // 4. Start Tauri
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_shell::init())
        .manage(Mutex::new(db))
        .manage(Mutex::new(dual_index))
        .manage(Mutex::new(onnx_model))
        .invoke_handler(tauri::generate_handler![
            commands::index::index_images,
            commands::index::toggle_subfolder,
            commands::index::get_subfolders,
            commands::index::has_subfolders,
            commands::search::search_by_text,
            commands::search::search_by_image,
            commands::db::get_image_info,
            commands::index::get_index_status,
            commands::system::open_file,
            commands::paths::get_paths,
            commands::paths::add_path,
            commands::paths::delete_path,
            commands::paths::toggle_path,
            commands::paths::rebuild_all_index,
        ])
        .run(tauri::generate_context!())
        .expect("Error while running Tauri application");
}

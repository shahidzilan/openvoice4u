#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::{io::{BufRead, BufReader, Write}, process::{Child, Command, Stdio}, sync::{Arc, Mutex}};
use tauri::{Emitter, Manager, State};
use serde_json::{json, Value};
#[cfg(windows)]
use std::os::windows::process::CommandExt;

#[derive(Default)]
struct Worker { child: Arc<Mutex<Option<Child>>>, busy: Arc<Mutex<bool>> }

#[tauri::command]
fn cancel(worker: State<'_, Worker>) -> Result<(), String> {
    if let Some(child) = worker.child.lock().map_err(|_| "Worker lock failed")?.as_mut() {
        child.kill().map_err(|_| "Could not stop the worker")?;
    }
    Ok(())
}

#[tauri::command]
async fn request(app: tauri::AppHandle, worker: State<'_, Worker>, operation: String, args: Value) -> Result<Value, String> {
    const ALLOWED: &[&str] = &["status","settings","save_voice","edit_voice","delete_voice","import_recording","validate","transcribe","setup_models","generate","history","preview","export"];
    if !ALLOWED.contains(&operation.as_str()) { return Err("Unsupported operation".into()); }
    {
        let mut busy = worker.busy.lock().map_err(|_| "Worker lock failed")?;
        if *busy { return Err("Another operation is running. Wait or cancel it first.".into()); }
        *busy = true;
    }
    let slot = worker.child.clone();
    let busy = worker.busy.clone();
    let outcome = tauri::async_runtime::spawn_blocking(move || -> Result<Value, String> {
        let mut command;
        if cfg!(debug_assertions) {
            let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
            command = Command::new(root.join("release-runtime/python.exe"));
            command.args(["-I", "-u", "-c", "import sys;sys.path.insert(0,sys.argv[1]);from app.worker import main;raise SystemExit(main())"])
                .arg(root.join("backend")).current_dir(root);
            command.env("OPENVOICE4U_DATA_DIR", root.join(".local/desktop"));
        } else {
            let resource = app.path().resource_dir().map_err(|_| "Could not find application resources")?;
            command = Command::new(resource.join("runtime/python.exe"));
            command.args(["-I", "-u"]).arg(resource.join("runtime/worker_entry.py"));
        }
        command.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::null());
        #[cfg(windows)]
        command.creation_flags(0x08000000);
        let mut child = command.spawn().map_err(|_| "The local voice runtime could not start. Repair the application installation.")?;
        let mut input = child.stdin.take().ok_or("Worker input is unavailable")?;
        let output = child.stdout.take().ok_or("Worker output is unavailable")?;
        *slot.lock().map_err(|_| "Worker lock failed")? = Some(child);
        let payload = json!({"operation":operation,"args":args}).to_string() + "\n";
        let write_result = input.write_all(payload.as_bytes());
        drop(input);
        let mut answer: Result<Value, String> = Err("The voice worker stopped unexpectedly. Memory was released; you can retry.".into());
        if write_result.is_ok() {
            for line in BufReader::new(output).lines() {
                let line = match line { Ok(value) => value, Err(_) => break };
                if let Ok(event) = serde_json::from_str::<Value>(&line) {
                    match event["kind"].as_str() {
                        Some("progress") => { let _ = app.emit("voice-progress", &event); },
                        Some("result") => answer = Ok(event["result"].clone()),
                        Some("error") => answer = Err(event["message"].as_str().unwrap_or("Voice operation failed").into()),
                        _ => (),
                    }
                }
            }
        }
        if let Some(mut child) = slot.lock().map_err(|_| "Worker lock failed")?.take() {
            if let Ok(status) = child.wait() {
                if answer.as_ref().err().is_some_and(|message| message.starts_with("The voice worker stopped unexpectedly")) {
                    answer = Err(format!("Voice worker exited unexpectedly (code 0x{:08X}). Close memory-heavy apps and retry. Completed recordings remain in History.", status.code().unwrap_or(-1) as u32));
                }
            }
        }
        answer
    }).await.map_err(|_| "Worker supervisor failed".to_string());
    *busy.lock().map_err(|_| "Worker lock failed")? = false;
    outcome?
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_single_instance::init(|app, _, _| {
            if let Some(window) = app.get_webview_window("main") { let _ = window.set_focus(); }
        }))
        .plugin(tauri_plugin_dialog::init())
        .manage(Worker::default())
        .invoke_handler(tauri::generate_handler![request, cancel])
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::Destroyed = event {
                let worker = window.state::<Worker>();
                if let Ok(mut slot) = worker.child.lock() {
                    if let Some(child) = slot.as_mut() { let _ = child.kill(); let _ = child.wait(); }
                };
            }
        })
        .run(tauri::generate_context!())
        .expect("OpenVoice4U could not start");
}

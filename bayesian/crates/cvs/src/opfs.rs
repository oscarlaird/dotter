use serde::{de::DeserializeOwned, Serialize};
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;
use web_sys::{
    FileSystemDirectoryHandle, FileSystemFileHandle, FileSystemGetFileOptions,
    FileSystemWritableFileStream,
};

pub async fn save_json<T>(file_name: &str, value: &T) -> Result<(), String>
where
    T: Serialize,
{
    let json = serde_json::to_string_pretty(value).unwrap();
    let root = opfs_root().await?;
    let options = FileSystemGetFileOptions::new();
    options.set_create(true);
    let file_handle = JsFuture::from(root.get_file_handle_with_options(file_name, &options))
        .await
        .map_err(js_error_message)?
        .dyn_into::<FileSystemFileHandle>()
        .map_err(js_error_message)?;
    let writable = JsFuture::from(file_handle.create_writable())
        .await
        .map_err(js_error_message)?
        .dyn_into::<FileSystemWritableFileStream>()
        .map_err(js_error_message)?;

    JsFuture::from(writable.write_with_str(&json).map_err(js_error_message)?)
        .await
        .map_err(js_error_message)?;
    JsFuture::from(web_sys::WritableStream::close(&writable))
        .await
        .map_err(js_error_message)?;
    Ok(())
}

pub async fn load_json<T>(file_name: &str) -> Result<T, String>
where
    T: DeserializeOwned,
{
    let root = opfs_root().await?;
    let file_handle = JsFuture::from(root.get_file_handle(file_name))
        .await
        .map_err(js_error_message)?
        .dyn_into::<FileSystemFileHandle>()
        .map_err(js_error_message)?;
    let file = JsFuture::from(file_handle.get_file())
        .await
        .map_err(js_error_message)?
        .dyn_into::<web_sys::File>()
        .map_err(js_error_message)?;
    let text = JsFuture::from(web_sys::Blob::text(&file))
        .await
        .map_err(js_error_message)?
        .as_string()
        .unwrap();
    serde_json::from_str::<T>(&text).map_err(|err| err.to_string())
}

async fn opfs_root() -> Result<FileSystemDirectoryHandle, String> {
    let storage = web_sys::window().unwrap().navigator().storage();
    let root = JsFuture::from(storage.get_directory())
        .await
        .map_err(js_error_message)?;
    root.dyn_into::<FileSystemDirectoryHandle>()
        .map_err(js_error_message)
}

fn js_error_message(err: wasm_bindgen::JsValue) -> String {
    err.as_string().unwrap_or_else(|| format!("{err:?}"))
}

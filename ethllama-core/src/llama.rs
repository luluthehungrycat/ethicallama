// ethllama-core/src/llama.rs
use std::ffi::CString;
use std::os::raw::c_char;

extern "C" {
    pub fn llama_model_load(path: *const c_char, params: *const LlamaModelParams) -> *mut LlamaModel;
    pub fn llama_free(model: *mut LlamaModel);
}

#[repr(C)]
pub struct LlamaModelParams {
    pub n_gpu_layers: i32,
    // ... other params
}

pub fn load_model(path: &str, n_gpu_layers: i32) -> Result<*mut LlamaModel, String> {
    let c_path = CString::new(path).map_err(|e| e.to_string())?;
    let params = LlamaModelParams { n_gpu_layers };
    let model_ptr = unsafe { llama_model_load(c_path.as_ptr(), &params) };
    if model_ptr.is_null() {
        Err("Failed to load model".to_string())
    } else {
        Ok(model_ptr)
    }
}

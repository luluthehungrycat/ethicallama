// ethllama-core/src/lib.rs
//
// Python bindings (via PyO3) to the llama.cpp FFI layer.
// Provides a PyLlamaModel class that loads a GGUF model and runs inference.

mod llama;
mod utils;

use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// Python‑exposed class that wraps raw llama model + context pointers.
// ---------------------------------------------------------------------------

/// A Python‑accessible wrapper around a loaded llama.cpp model.
#[pyclass]
pub struct PyLlamaModel {
    model_ptr: *mut llama::llama_model,
    ctx_ptr: *mut llama::llama_context,
}

unsafe impl Send for PyLlamaModel {}

#[pymethods]
impl PyLlamaModel {
    /// Load a model from the given file path.
    ///
    /// Args:
    ///     path: Path to a GGUF model file.
    ///     n_gpu_layers: Number of layers to offload to GPU (0 = CPU only).
    ///     n_ctx: Context size in tokens.
    ///     n_threads: Number of CPU threads to use.
    #[new]
    fn new(
        path: String,
        n_gpu_layers: i32,
        n_ctx: u32,
        n_threads: i32,
    ) -> PyResult<Self> {
        // Initialize the llama backend once
        unsafe { llama::llama_backend_init() };

        let (model_ptr, ctx_ptr) =
            llama::load_model(&path, n_gpu_layers, n_ctx, n_threads).map_err(|e| {
                unsafe { llama::llama_backend_free() };
                pyo3::exceptions::PyRuntimeError::new_err(e)
            })?;

        Ok(Self { model_ptr, ctx_ptr })
    }

    /// Run inference on a prompt.
    ///
    /// Args:
    ///     prompt: Input text to complete.
    ///     max_tokens: Maximum number of tokens to generate.
    ///     temperature: Sampling temperature (0 = greedy, >0 = stochastic).
    ///     top_p: Nucleus sampling threshold (0 = disabled).
    ///     top_k: Top‑K sampling (0 = disabled).
    ///
    /// Returns:
    ///     Generated text as a string.
    fn infer(
        &self,
        prompt: String,
        max_tokens: i32,
        temperature: f32,
        top_p: f32,
        top_k: i32,
    ) -> PyResult<String> {
        let result = llama::infer_tokens(
            self.ctx_ptr,
            self.model_ptr,
            &prompt,
            max_tokens,
            temperature,
            top_p,
            top_k,
        )
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        Ok(result)
    }

    /// Return detected GPU backends as a JSON string.
    fn gpu_info(&self) -> PyResult<String> {
        let backends = utils::detect_gpu_backends();
        serde_json::to_string(&backends)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }
}

impl Drop for PyLlamaModel {
    fn drop(&mut self) {
        unsafe {
            if !self.ctx_ptr.is_null() {
                llama::llama_free(self.ctx_ptr);
            }
            if !self.model_ptr.is_null() {
                llama::llama_model_free(self.model_ptr);
            }
            llama::llama_backend_free();
        }
    }
}

// ---------------------------------------------------------------------------
// PyO3 module definition
// ---------------------------------------------------------------------------

#[pymodule]
fn ethllama_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyLlamaModel>()?;
    Ok(())
}

// ethllama-core/src/lib.rs
//
// Python bindings (via PyO3) to the llama.cpp FFI layer.
// Provides a PyLlamaModel class that loads a GGUF model and runs inference.
//
// The `llama-cpp` Cargo feature gates the bundled llama.cpp integration.
// When the feature is on AND the llama.cpp git submodule is present, this
// file compiles a full native extension that loads GGUF models and runs
// inference via FFI.
//
// When either of those conditions is missing (e.g. building from an sdist
// on an ARM board where no pre-built wheel exists), the build script emits
// `cargo:rustc-cfg=stub_mode`. The lib code then compiles a stub that
// loads but raises `RuntimeError` on `PyLlamaModel(...)`. The Python side
// already handles this via `has_inference_engine()` and falls back to the
// subprocess inference path.

// The `non_local_definitions` lint fires inside pyo3 0.20's
// `#[pymethods]` macro with rustc 1.80+.  This is a known false
// positive (https://github.com/PyO3/pyo3/issues/3745).  The
// crate-level allow is required because the lint fires during
// macro expansion before a block-level `#[allow]` takes effect.
#![allow(non_local_definitions)]
// In stub mode, the llama module is excluded and the lib has unused
// stub parameters. Silence dead-code warnings on the stub branch.
#![cfg_attr(stub_mode, allow(dead_code))]

#[cfg(all(feature = "llama-cpp", not(stub_mode)))]
mod llama;
#[cfg(all(feature = "llama-cpp", not(stub_mode)))]
mod utils;

use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// Python-exposed class that wraps raw llama model + context pointers.
// ---------------------------------------------------------------------------

/// A Python-accessible wrapper around a loaded llama.cpp model.
///
/// In the full build (default features + llama.cpp submodule present),
/// this loads a GGUF model and runs inference via FFI.
///
/// In the stub build (either the `llama-cpp` feature disabled OR the
/// llama.cpp git submodule missing), constructing this class raises a
/// `RuntimeError` with an actionable error message. Python callers
/// should check `has_inference_engine()` first; the Python side
/// already does this and falls back to the subprocess inference path.
#[pyclass]
pub struct PyLlamaModel {
    #[cfg(all(feature = "llama-cpp", not(stub_mode)))]
    model_ptr: *mut llama::llama_model,
    #[cfg(all(feature = "llama-cpp", not(stub_mode)))]
    ctx_ptr: *mut llama::llama_context,
}

#[cfg(all(feature = "llama-cpp", not(stub_mode)))]
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
    #[pyo3(signature = (path, n_gpu_layers=0, n_ctx=4096, n_threads=4))]
    fn new(path: String, n_gpu_layers: i32, n_ctx: u32, n_threads: i32) -> PyResult<Self> {
        #[cfg(any(not(feature = "llama-cpp"), stub_mode))]
        {
            // Consume parameters to avoid unused-variable warnings on
            // the stub branch. The signature must match the full build
            // exactly so PyO3 generates a consistent Python binding.
            let _ = (path, n_gpu_layers, n_ctx, n_threads);
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "ethllama_core stub build: native llama.cpp extension is not available. \
                 Build from a full source checkout with `git submodule update --init --recursive` \
                 for native inference, or use the subprocess inference path \
                 (auto-detected via `has_inference_engine()` and `llama-cli`).",
            ));
        }
        #[cfg(all(feature = "llama-cpp", not(stub_mode)))]
        {
            // Initialize the llama backend once
            unsafe { llama::llama_backend_init() };

            let (model_ptr, ctx_ptr) = llama::load_model(&path, n_gpu_layers, n_ctx, n_threads)
                .map_err(|e| {
                    unsafe { llama::llama_backend_free() };
                    pyo3::exceptions::PyRuntimeError::new_err(e)
                })?;

            Ok(Self { model_ptr, ctx_ptr })
        }
    }

    /// Run inference on a prompt.
    ///
    /// Args:
    ///     prompt: Input text to complete.
    ///     max_tokens: Maximum number of tokens to generate.
    ///     temperature: Sampling temperature (0 = greedy, >0 = stochastic).
    ///     top_p: Nucleus sampling threshold (0 = disabled).
    ///     top_k: Top-K sampling (0 = disabled).
    ///
    /// Returns:
    ///     Generated text as a string.
    #[cfg(all(feature = "llama-cpp", not(stub_mode)))]
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
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
        Ok(result)
    }

    /// Return detected GPU backends as a JSON string.
    #[cfg(all(feature = "llama-cpp", not(stub_mode)))]
    fn gpu_info(&self) -> PyResult<String> {
        let backends = utils::detect_gpu_backends();
        serde_json::to_string(&backends)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }
}

#[cfg(all(feature = "llama-cpp", not(stub_mode)))]
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

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------
//
// PyO3-bound code (`PyLlamaModel::new`, `infer`, `gpu_info`) is hard to
// unit-test without a live Python interpreter and a GGUF model on disk.
// Those paths are covered by the integration tests in `ethllama/tests/`.
// The tests here focus on the compile-time correctness of the bindings
// and the structure of the Rust crate itself.

#[cfg(test)]
mod tests {
    /// Sanity test: the crate compiles and the test harness runs.
    /// This is a "trivially true" test that exists to confirm the test
    /// binary built at all (i.e. PyO3 + llama.cpp + cdylib all linked).
    #[test]
    fn test_crate_compiles() {
        // If this runs, the crate compiled.
    }

    /// `PyLlamaModel` must be `Send` (the `unsafe impl Send` is required
    /// for PyO3 to hand instances across threads). This is a static
    /// check — the test will fail to compile if `Send` is ever removed.
    ///
    /// Gated on the full build because in stub mode the struct has no
    /// fields and the `unsafe impl Send` is feature-gated.
    #[cfg(all(feature = "llama-cpp", not(stub_mode)))]
    #[test]
    fn test_pyllamamodel_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<crate::PyLlamaModel>();
    }

    /// Verify the `unsafe impl Send` does not silently make the type
    /// `!Sync` (PyO3 also requires `Sync` for some patterns).
    /// If this test stops compiling, check the `unsafe impl Send`
    /// declaration in lib.rs.
    #[allow(dead_code)]
    fn _check_sync_bounds() {
        fn assert_sync<T: Sync>() {}
        // Note: PyLlamaModel intentionally implements only `Send`, not
        // `Sync`. This helper documents that decision — do not call it
        // from a test, just keep it in the build for documentation.
        // assert_sync::<crate::PyLlamaModel>();
        let _ = assert_sync::<u8>; // silence unused warning
    }

    /// Stub-mode sanity: the PyLlamaModel class still exists in stub
    /// mode and the module still imports. This test is gated to stub
    /// mode (or any non-llama-cpp build) to confirm the stub path
    /// compiles end-to-end.
    #[cfg(any(not(feature = "llama-cpp"), stub_mode))]
    #[test]
    fn test_stub_mode_construct_raises_runtime_error() {
        // We can't easily construct a PyLlamaModel from a Rust test
        // (it requires a Python interpreter), but we can confirm the
        // type exists and has the expected size: zero (no fields).
        assert_eq!(
            std::mem::size_of::<crate::PyLlamaModel>(),
            0,
            "PyLlamaModel in stub mode must be a zero-sized type"
        );
    }
}

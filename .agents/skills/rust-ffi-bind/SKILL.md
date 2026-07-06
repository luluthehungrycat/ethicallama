# rust-ffi-bind — llama.cpp C → Rust → PyO3 binding pattern

Use when adding new llama.cpp C API functions to the Rust/PyO3 FFI layer.
This is a **three-layer pattern** — each layer has a distinct responsibility.

## Trigger phrases

- "add a new llama.cpp binding for ..."
- "expose the $FUNCTION from llama.h in the Rust core"
- "wire up $LLAMA_C_API through PyO3"
- "add a new extern \"C\" declaration"

---

## Layer 1: `llama.rs` — extern "C" declarations

**File:** `ethllama-core/src/llama.rs`

Every C function from `llama.h` gets an `extern "C"` declaration here:

```rust
extern "C" {
    pub fn llama_function_name(
        arg1: *mut llama_model,
        arg2: i32,
        arg3: *const c_char,
    ) -> i32;
}
```

**Rules:**
- All `extern "C"` blocks are consolidated in `llama.rs` — never in `lib.rs`
- Opaque types (`llama_model`, `llama_context`, `llama_vocab`, `llama_sampler`) are `#[repr(C)]` with `_private: [u8; 0]`
- Struct types match the C layout exactly with `#[repr(C)]` — must verify alignment
- Integer enum variants are declared as `pub const` i32 constants
- Type aliases match `llama.h`: `llama_token = i32`, `llama_pos = i32`, `llama_seq_id = i32`
- Mark the module `#![allow(non_camel_case_types, dead_code)]`

### Workflow for adding a new C function

1. Find the function in `ethllama-core/llama.cpp/include/llama.h`
2. Copy the signature, converting C types:
   - `const char*` → `*const c_char`
   - `int` → `i32` / `c_int`
   - `uint32_t` → `u32`
   - `float` → `f32`
   - `bool` → `bool` (1 byte in C, Rust bool is 1 byte — fine for FFI)
   - `size_t` → `usize`
   - `struct llama_xyz*` → `*mut llama_xyz` (or `*const` if input-only)
3. Add to one of the existing `extern "C"` blocks (or create a new one with a comment grouping)

## Layer 2: Safe Rust wrapper (same file or utils)

If the C function is used directly by `PyLlamaModel`, wrap it in a safe function:

```rust
pub fn safe_wrapper(model_ptr: *mut llama_model, ...) -> Result<SomeType, String> {
    unsafe {
        let result = llama_function_name(model_ptr, ...);
        if result < 0 {
            return Err("llama function failed".into());
        }
        Ok(result)
    }
}
```

**File:** `llama.rs` (preferred) or `utils.rs` (for system introspection)

For complex multi-step operations (like the inference loop), define a standalone safe function:

```rust
pub fn load_model(path: &str, ...) -> Result<(*mut llama_model, *mut llama_context), String> { ... }
pub fn infer_tokens(ctx: *mut llama_context, model: *mut llama_model, ...) -> Result<String, String> { ... }
```

## Layer 3: `lib.rs` — PyO3 #[pyclass]

**File:** `ethllama-core/src/lib.rs`

Expose the safe wrapper as a Python method on `PyLlamaModel`:

```rust
#[pymethods]
impl PyLlamaModel {
    fn method_name(&self, arg: String, ...) -> PyResult<ReturnType> {
        let result = llama::safe_wrapper(self.ctx_ptr, ...)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        Ok(result)
    }
}
```

**Registration:** Add the class to the `#[pymodule]` block if it's new.

## Memory safety contract

```rust
impl Drop for PyLlamaModel {
    fn drop(&mut self) {
        unsafe {
            if !self.ctx_ptr.is_null() { llama::llama_free(self.ctx_ptr); }
            if !self.model_ptr.is_null() { llama::llama_model_free(self.model_ptr); }
            llama::llama_backend_free();
        }
    }
}
unsafe impl Send for PyLlamaModel {}
```

Every allocation (model load, context create, backend init) must have a corresponding free in `Drop`.

## Build

```toml
# ethllama-core/Cargo.toml
[build-dependencies]
cmake = "0.1"

[dependencies]
anyhow = "1.0"
pyo3 = { version = "0.20", features = ["extension-module"] }
libloading = "0.8"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
```

`build.rs` uses the `cmake` crate to compile llama.cpp as a static library:

```rust
fn main() {
    let dst = cmake::Config::new("llama.cpp")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("LLAMA_BUILD_TESTS", "OFF")
        .define("LLAMA_BUILD_EXAMPLES", "OFF")
        .build();
    // ... linking logic
}
```

## Testing

- Rust FFI has no test harness — verify via `cargo build --release -p ethllama-core` (compilation check)
- Python integration: test the Python package after `maturin develop`

## Reference files

- `ethllama-core/llama.cpp/include/llama.h` — canonical C API to bind against
- `ethllama-core/src/llama.rs` — existing FFI bindings (599 lines, 30+ extern "C" funcs)
- `ethllama-core/src/lib.rs` — PyO3 pyclass (114 lines)
- `ethllama-core/build.rs` — cmake build script

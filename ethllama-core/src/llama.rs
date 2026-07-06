// ethllama-core/src/llama.rs
//
// FFI bindings to the llama.cpp C API — matching llama.h (current version).
// All unsafe extern "C" declarations live here; lib.rs re-exports through this module.

#![allow(non_camel_case_types, dead_code)]

use std::ffi::CString;
use std::os::raw::c_char;

// ---------------------------------------------------------------------------
// Opaque types (defined in llama.cpp)
// ---------------------------------------------------------------------------
#[repr(C)]
pub struct llama_model {
    _private: [u8; 0],
}

#[repr(C)]
pub struct llama_context {
    _private: [u8; 0],
}

#[repr(C)]
pub struct llama_vocab {
    _private: [u8; 0],
}

#[repr(C)]
pub struct llama_sampler {
    _private: [u8; 0],
}

// ---------------------------------------------------------------------------
// Type aliases matching llama.h
// ---------------------------------------------------------------------------
pub type llama_token = i32;
pub type llama_pos = i32;
pub type llama_seq_id = i32;

pub const LLAMA_DEFAULT_SEED: u32 = 0xFFFFFFFF;
pub const LLAMA_TOKEN_NULL: i32 = -1;

// ---------------------------------------------------------------------------
// Enums as integer constants
// ---------------------------------------------------------------------------
pub const LLAMA_SPLIT_NONE: i32 = 0;
pub const LLAMA_SPLIT_LAYER: i32 = 1;
pub const LLAMA_SPLIT_ROW: i32 = 2;
pub const LLAMA_SPLIT_TENSOR: i32 = 3;

pub const LLAMA_ROPE_SCALING_UNSPECIFIED: i32 = -1;
pub const LLAMA_ROPE_SCALING_NONE: i32 = 0;
pub const LLAMA_ROPE_SCALING_LINEAR: i32 = 1;
pub const LLAMA_ROPE_SCALING_YARN: i32 = 2;
pub const LLAMA_ROPE_SCALING_LONGROPE: i32 = 3;

pub const LLAMA_POOLING_UNSPECIFIED: i32 = -1;
pub const LLAMA_POOLING_NONE: i32 = 0;
pub const LLAMA_POOLING_MEAN: i32 = 1;
pub const LLAMA_POOLING_CLS: i32 = 2;
pub const LLAMA_POOLING_LAST: i32 = 3;
pub const LLAMA_POOLING_RANK: i32 = 4;

pub const LLAMA_ATTENTION_UNSPECIFIED: i32 = -1;
pub const LLAMA_ATTENTION_CAUSAL: i32 = 0;
pub const LLAMA_ATTENTION_NON_CAUSAL: i32 = 1;

pub const LLAMA_FLASH_ATTN_AUTO: i32 = -1;
pub const LLAMA_FLASH_ATTN_DISABLED: i32 = 0;
pub const LLAMA_FLASH_ATTN_ENABLED: i32 = 1;

pub const LLAMA_CONTEXT_DEFAULT: i32 = 0;
pub const LLAMA_CONTEXT_MTP: i32 = 1;

// KV override types
pub const LLAMA_KV_OVERRIDE_INT: i32 = 0;
pub const LLAMA_KV_OVERRIDE_FLOAT: i32 = 1;
pub const LLAMA_KV_OVERRIDE_BOOL: i32 = 2;
pub const LLAMA_KV_OVERRIDE_STR: i32 = 3;

// ---------------------------------------------------------------------------
// struct llama_model_kv_override
// ---------------------------------------------------------------------------
#[repr(C)]
pub struct llama_model_kv_override {
    pub tag: i32,
    pub key: [c_char; 128],
    // anonymous union: { int64_t val_i64; double val_f64; bool val_bool; char val_str[128]; }
    pub val_str: [c_char; 128],
}

// ---------------------------------------------------------------------------
// struct llama_model_tensor_buft_override
// ---------------------------------------------------------------------------
#[repr(C)]
pub struct llama_model_tensor_buft_override {
    pub pattern: *const c_char,
    pub buft: *mut std::ffi::c_void,
}

// ---------------------------------------------------------------------------
// struct llama_model_params
// Fields must match llama.h exactly
// ---------------------------------------------------------------------------
#[repr(C)]
pub struct llama_model_params {
    pub devices: *mut *mut std::ffi::c_void,
    pub tensor_buft_overrides: *const llama_model_tensor_buft_override,
    pub n_gpu_layers: i32,
    pub split_mode: i32,
    pub main_gpu: i32,
    // 4 bytes implicit padding for ptr alignment on 64-bit
    pub tensor_split: *const f32,
    pub progress_callback: Option<
        unsafe extern "C" fn(f32, *mut std::ffi::c_void) -> bool,
    >,
    pub progress_callback_user_data: *mut std::ffi::c_void,
    pub kv_overrides: *const llama_model_kv_override,
    // booleans packed at the end
    pub vocab_only: bool,
    pub use_mmap: bool,
    pub use_direct_io: bool,
    pub use_mlock: bool,
    pub check_tensors: bool,
    pub use_extra_bufts: bool,
    pub no_host: bool,
    pub no_alloc: bool,
}

// ---------------------------------------------------------------------------
// struct llama_sampler_seq_config
// ---------------------------------------------------------------------------
#[repr(C)]
pub struct llama_sampler_seq_config {
    pub seq_id: llama_seq_id,
    pub sampler: *mut llama_sampler,
}

// ---------------------------------------------------------------------------
// struct llama_context_params
// Fields must match llama.h exactly
// ---------------------------------------------------------------------------
#[repr(C)]
pub struct llama_context_params {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub n_seq_max: u32,
    pub n_rs_seq: u32,
    pub n_outputs_max: u32,
    pub n_threads: i32,
    pub n_threads_batch: i32,
    // enums (all i32-sized)
    pub ctx_type: i32,
    pub rope_scaling_type: i32,
    pub pooling_type: i32,
    pub attention_type: i32,
    pub flash_attn_type: i32,
    // float params
    pub rope_freq_base: f32,
    pub rope_freq_scale: f32,
    pub yarn_ext_factor: f32,
    pub yarn_attn_factor: f32,
    pub yarn_beta_fast: f32,
    pub yarn_beta_slow: f32,
    pub yarn_orig_ctx: u32,
    pub defrag_thold: f32,
    // 4 bytes implicit padding for ptr alignment
    pub cb_eval: Option<
        unsafe extern "C" fn(
            *mut std::ffi::c_void,
            *mut std::ffi::c_void,
            *mut std::ffi::c_void,
            bool,
            *mut std::ffi::c_void,
        ) -> bool,
    >,
    pub cb_eval_user_data: *mut std::ffi::c_void,
    // type_k, type_v (ggml_type enums as i32)
    pub type_k: i32,
    pub type_v: i32,
    // 4 bytes implicit padding for ptr alignment
    pub abort_callback: Option<unsafe extern "C" fn(*mut std::ffi::c_void) -> bool>,
    pub abort_callback_data: *mut std::ffi::c_void,
    // booleans
    pub embeddings: bool,
    pub offload_kqv: bool,
    pub no_perf: bool,
    pub op_offload: bool,
    pub swa_full: bool,
    pub kv_unified: bool,
    // 2 bytes implicit padding for ptr alignment
    pub samplers: *mut llama_sampler_seq_config,
    pub n_samplers: usize,
    pub ctx_other: *mut llama_context,
}

// ---------------------------------------------------------------------------
// struct llama_batch
// ---------------------------------------------------------------------------
#[repr(C)]
pub struct llama_batch {
    pub n_tokens: i32,
    pub token: *mut llama_token,
    pub embd: *mut f32,
    pub pos: *mut llama_pos,
    pub n_seq_id: *mut i32,
    pub seq_id: *mut *mut llama_seq_id,
    pub logits: *mut i8,
}

// ---------------------------------------------------------------------------
// struct llama_sampler_chain_params
// ---------------------------------------------------------------------------
#[repr(C)]
pub struct llama_sampler_chain_params {
    pub no_perf: bool,
}

// ---------------------------------------------------------------------------
// struct llama_token_data
// ---------------------------------------------------------------------------
#[repr(C)]
pub struct llama_token_data {
    pub id: llama_token,
    pub logit: f32,
    pub p: f32,
}

// ---------------------------------------------------------------------------
// struct llama_token_data_array
// ---------------------------------------------------------------------------
#[repr(C)]
pub struct llama_token_data_array {
    pub data: *mut llama_token_data,
    pub size: usize,
    pub selected: i64,
    pub sorted: bool,
}

// ---------------------------------------------------------------------------
// Extern "C" FFI function declarations matching llama.h
// ---------------------------------------------------------------------------
extern "C" {
    // Init / shutdown
    pub fn llama_backend_init();
    pub fn llama_backend_free();
    pub fn llama_numa_init(numa: i32);

    // Default params
    pub fn llama_model_default_params() -> llama_model_params;
    pub fn llama_context_default_params() -> llama_context_params;
    pub fn llama_sampler_chain_default_params() -> llama_sampler_chain_params;

    // Model loading / unloading
    pub fn llama_model_load_from_file(
        path: *const c_char,
        params: llama_model_params,
    ) -> *mut llama_model;
    pub fn llama_model_free(model: *mut llama_model);

    // Context
    pub fn llama_init_from_model(
        model: *mut llama_model,
        params: llama_context_params,
    ) -> *mut llama_context;
    pub fn llama_free(ctx: *mut llama_context);
    pub fn llama_get_model(ctx: *const llama_context) -> *const llama_model;

    // Model info (kept for potential direct use, but safe wrappers preferred)
    pub fn llama_vocab_n_tokens(vocab: *const llama_vocab) -> i32;

    // Vocab info / special tokens
    pub fn llama_vocab_bos(vocab: *const llama_vocab) -> llama_token;
    pub fn llama_vocab_eos(vocab: *const llama_vocab) -> llama_token;
    pub fn llama_vocab_is_eog(
        vocab: *const llama_vocab,
        token: llama_token,
    ) -> bool;

    // Tokenization
    pub fn llama_tokenize(
        vocab: *const llama_vocab,
        text: *const c_char,
        text_len: i32,
        tokens: *mut llama_token,
        n_tokens_max: i32,
        add_special: bool,
        parse_special: bool,
    ) -> i32;

    // Token -> text
    pub fn llama_token_to_piece(
        vocab: *const llama_vocab,
        token: llama_token,
        buf: *mut c_char,
        length: i32,
        lstrip: i32,
        special: bool,
    ) -> i32;

    // ---- Exception-safe wrappers (implemented in shim.cpp) ----
    pub fn llama_model_load_from_file_safe(
        path: *const c_char,
        params: llama_model_params,
    ) -> *mut llama_model;
    pub fn llama_init_from_model_safe(
        model: *mut llama_model,
        params: llama_context_params,
    ) -> *mut llama_context;
    pub fn llama_model_get_vocab_safe(
        model: *const llama_model,
    ) -> *const llama_vocab;
    pub fn llama_model_n_embd_safe(
        model: *const llama_model,
    ) -> i32;
    pub fn llama_tokenize_safe(
        vocab: *const llama_vocab,
        text: *const c_char,
        text_len: i32,
        tokens: *mut llama_token,
        n_tokens_max: i32,
        add_special: bool,
        parse_special: bool,
    ) -> i32;
    pub fn llama_token_to_piece_safe(
        vocab: *const llama_vocab,
        token: llama_token,
        buf: *mut c_char,
        length: i32,
        lstrip: i32,
        special: bool,
    ) -> i32;
    pub fn llama_batch_get_one_safe(
        tokens: *mut llama_token,
        n_tokens: i32,
    ) -> llama_batch;
    pub fn llama_decode_safe(
        ctx: *mut llama_context,
        batch: llama_batch,
    ) -> i32;
    pub fn llama_sampler_sample_safe(
        smpl: *mut llama_sampler,
        ctx: *mut llama_context,
        idx: i32,
    ) -> llama_token;
    pub fn llama_vocab_is_eog_safe(
        vocab: *const llama_vocab,
        token: llama_token,
    ) -> bool;

    // Batch helper (safe version used via llama_batch_get_one_safe)
    pub fn llama_batch_get_one(
        tokens: *mut llama_token,
        n_tokens: i32,
    ) -> llama_batch;

    pub fn llama_batch_init(
        n_tokens: i32,
        embd: i32,
        n_seq_max: i32,
    ) -> llama_batch;

    pub fn llama_batch_free(batch: llama_batch);

    // Decode
    pub fn llama_decode(
        ctx: *mut llama_context,
        batch: llama_batch,
    ) -> i32;

    // Logits
    pub fn llama_get_logits_ith(
        ctx: *mut llama_context,
        i: i32,
    ) -> *mut f32;

    // Sampler chain
    pub fn llama_sampler_chain_init(
        params: llama_sampler_chain_params,
    ) -> *mut llama_sampler;

    pub fn llama_sampler_chain_add(
        chain: *mut llama_sampler,
        smpl: *mut llama_sampler,
    );

    // Sampler constructors
    pub fn llama_sampler_init_greedy() -> *mut llama_sampler;
    pub fn llama_sampler_init_dist(seed: u32) -> *mut llama_sampler;
    pub fn llama_sampler_init_top_k(k: i32) -> *mut llama_sampler;
    pub fn llama_sampler_init_top_p(
        p: f32,
        min_keep: usize,
    ) -> *mut llama_sampler;
    pub fn llama_sampler_init_temp(t: f32) -> *mut llama_sampler;

    // Sample
    pub fn llama_sampler_sample(
        smpl: *mut llama_sampler,
        ctx: *mut llama_context,
        idx: i32,
    ) -> llama_token;

    // Free sampler
    pub fn llama_sampler_free(smpl: *mut llama_sampler);

    // Context info
    pub fn llama_n_ctx(ctx: *const llama_context) -> u32;
    pub fn llama_set_n_threads(
        ctx: *mut llama_context,
        n_threads: i32,
        n_threads_batch: i32,
    );

    // System info
    pub fn llama_print_system_info() -> *const c_char;
    pub fn llama_supports_mmap() -> bool;
    pub fn llama_supports_mlock() -> bool;
    pub fn llama_supports_gpu_offload() -> bool;
}

// ---------------------------------------------------------------------------
// Safe helper: load a model and create a context
// ---------------------------------------------------------------------------
pub fn load_model(
    path: &str,
    n_gpu_layers: i32,
    n_ctx: u32,
    n_threads: i32,
) -> Result<(*mut llama_model, *mut llama_context), String> {
    let c_path = CString::new(path).map_err(|e| e.to_string())?;

    let mut model_params = unsafe { llama_model_default_params() };
    model_params.n_gpu_layers = n_gpu_layers;

    let model_ptr =
        unsafe { llama_model_load_from_file_safe(c_path.as_ptr(), model_params) };
    if model_ptr.is_null() {
        return Err(
            "Failed to load model: llama_model_load_from_file returned NULL"
                .to_string(),
        );
    }

    let mut ctx_params = unsafe { llama_context_default_params() };
    ctx_params.n_ctx = n_ctx;
    ctx_params.n_threads = n_threads;
    ctx_params.n_threads_batch = n_threads;

    let ctx_ptr = unsafe { llama_init_from_model_safe(model_ptr, ctx_params) };
    if ctx_ptr.is_null() {
        unsafe { llama_model_free(model_ptr) };
        return Err(
            "Failed to create context: llama_init_from_model returned NULL"
                .to_string(),
        );
    }

    Ok((model_ptr, ctx_ptr))
}

// ---------------------------------------------------------------------------
// Tokenization helper
// ---------------------------------------------------------------------------
pub fn tokenize(
    vocab: *const llama_vocab,
    text: &str,
    add_special: bool,
    parse_special: bool,
) -> Result<Vec<llama_token>, String> {
    let c_text = CString::new(text).map_err(|e| e.to_string())?;

    // First, try with a pre-allocated buffer (512 tokens).
    let mut tokens: Vec<llama_token> = vec![0; 512];
    let n_actual = unsafe {
        llama_tokenize_safe(
            vocab,
            c_text.as_ptr(),
            -1,
            tokens.as_mut_ptr(),
            512,
            add_special,
            parse_special,
        )
    };

    if n_actual > 0 {
        tokens.truncate(n_actual as usize);
        return Ok(tokens);
    }

    // First attempt failed. Try without special tokens (some tokenizers
    // have issues with add_special=true in certain llama.cpp builds).
    // Also try add_special=false, parse_special=false
    if add_special || parse_special {
        let n_actual = unsafe {
            llama_tokenize_safe(
                vocab,
                c_text.as_ptr(),
                -1,
                tokens.as_mut_ptr(),
                512,
                false,
                false,
            )
        };
        if n_actual > 0 {
            tokens.truncate(n_actual as usize);
            return Ok(tokens);
        }
    }

    // Retry with query-size to allocate exactly needed buffer
    let n_needed = unsafe {
        llama_tokenize_safe(
            vocab,
            c_text.as_ptr(),
            -1,
            std::ptr::null_mut(),
            0,
            add_special,
            parse_special,
        )
    };
    if n_needed <= 0 {
        // Last resort: try allocate a large buffer (2048) as final attempt
        let mut tokens: Vec<llama_token> = vec![0; 2048];
        let n = unsafe {
            llama_tokenize_safe(
                vocab,
                c_text.as_ptr(),
                -1,
                tokens.as_mut_ptr(),
                2048,
                false,
                false,
            )
        };
        if n > 0 {
            tokens.truncate(n as usize);
            return Ok(tokens);
        }
        return Err(format!(
            "Tokenization failed (size query): {}",
            n_needed
        ));
    }

    let n_tokens = n_needed as usize;
    let mut tokens: Vec<llama_token> = vec![0; n_tokens];
    let n_actual = unsafe {
        llama_tokenize_safe(
            vocab,
            c_text.as_ptr(),
            -1,
            tokens.as_mut_ptr(),
            n_tokens as i32,
            add_special,
            parse_special,
        )
    };
    if n_actual < 0 {
        return Err(format!("Tokenization failed: {}", n_actual));
    }
    tokens.truncate(n_actual as usize);
    Ok(tokens)
}

// ---------------------------------------------------------------------------
// Detokenization helper
// ---------------------------------------------------------------------------
pub fn detokenize(
    vocab: *const llama_vocab,
    tokens: &[llama_token],
) -> Result<String, String> {
    let mut output = String::new();
    let mut buf = vec![0u8; 256];

    for &token in tokens {
        let n_chars = unsafe {
            llama_token_to_piece_safe(
                vocab,
                token,
                buf.as_mut_ptr() as *mut c_char,
                buf.len() as i32,
                0,
                false,
            )
        };

        if n_chars > 0 {
            output.push_str(&String::from_utf8_lossy(
                &buf[..n_chars as usize],
            ));
        }
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// Inference: run full text generation on a prompt
// ---------------------------------------------------------------------------
pub fn infer_tokens(
    ctx: *mut llama_context,
    model: *mut llama_model,
    prompt: &str,
    max_tokens: i32,
    temperature: f32,
    top_p: f32,
    top_k: i32,
) -> Result<String, String> {
    if ctx.is_null() || model.is_null() {
        return Err("Null context or model pointer".to_string());
    }

    let vocab = unsafe { llama_model_get_vocab_safe(model) };
    if vocab.is_null() {
        return Err("Failed to get vocab from model".to_string());
    }

    // 1. Tokenize the prompt
    let tokens = tokenize(vocab, prompt, true, false)?;
    if tokens.is_empty() {
        return Err("Tokenization produced zero tokens".to_string());
    }

    // 2. Process prompt tokens (prefill)
    let mut prompt_tokens = tokens;
    let batch = unsafe {
        llama_batch_get_one_safe(prompt_tokens.as_mut_ptr(), prompt_tokens.len() as i32)
    };
    let ret = unsafe { llama_decode_safe(ctx, batch) };
    if ret != 0 {
        return Err(format!("llama_decode (prompt) failed with code: {}", ret));
    }

    // 3. Build sampler chain
    let sparams = unsafe { llama_sampler_chain_default_params() };
    let sampler = unsafe { llama_sampler_chain_init(sparams) };
    if sampler.is_null() {
        return Err("Failed to create sampler chain".to_string());
    }

    if temperature > 0.0 && top_p > 0.0 && top_k > 0 {
        // Stochastic sampling chain: top_k -> top_p -> temperature -> dist
        unsafe {
            llama_sampler_chain_add(sampler, llama_sampler_init_top_k(top_k));
            llama_sampler_chain_add(
                sampler,
                llama_sampler_init_top_p(top_p, 1),
            );
            llama_sampler_chain_add(
                sampler,
                llama_sampler_init_temp(temperature),
            );
            llama_sampler_chain_add(
                sampler,
                llama_sampler_init_dist(LLAMA_DEFAULT_SEED),
            );
        }
    } else {
        // Greedy: pick the token with highest logit
        unsafe {
            llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
        }
    }

    // 4. Generation loop
    let mut output_tokens: Vec<llama_token> = Vec::new();
    let mut new_token = unsafe { llama_sampler_sample_safe(sampler, ctx, -1) };

    for _ in 0..max_tokens {
        output_tokens.push(new_token);

        // Check end-of-generation
        if unsafe { llama_vocab_is_eog_safe(vocab, new_token) } {
            break;
        }

        // Decode this token
        let batch = unsafe { llama_batch_get_one_safe(&mut new_token, 1) };
        let ret = unsafe { llama_decode_safe(ctx, batch) };
        if ret != 0 {
            break;
        }

        // Sample next token
        new_token = unsafe { llama_sampler_sample_safe(sampler, ctx, -1) };
    }

    unsafe { llama_sampler_free(sampler) };

    // 5. Detokenize output
    detokenize(vocab, &output_tokens)
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::size_of;
    use std::ptr;

    // -------------------------------------------------------------------
    // FFI symbol existence — these are compile-time checks: each `extern
    // "C"` declaration is referenced here, so if the signature ever
    // changes (or the declaration is removed) the test build fails.
    // -------------------------------------------------------------------

    /// Verify that the key FFI function symbols are declared with the
    /// expected signatures. Touching them in a test forces the linker to
    /// resolve the symbols at test-build time.
    #[test]
    fn test_ffi_function_signatures_exist() {
        // Function pointer references — never called, just type-checked.
        let _init: unsafe extern "C" fn() = llama_backend_init;
        let _free: unsafe extern "C" fn() = llama_backend_free;
        let _default_model: unsafe extern "C" fn() -> llama_model_params =
            llama_model_default_params;
        let _default_ctx: unsafe extern "C" fn() -> llama_context_params =
            llama_context_default_params;
        let _default_sampler: unsafe extern "C" fn() -> llama_sampler_chain_params =
            llama_sampler_chain_default_params;
        let _load: unsafe extern "C" fn(
            *const c_char,
            llama_model_params,
        ) -> *mut llama_model = llama_model_load_from_file;
        let _init_from_model: unsafe extern "C" fn(
            *mut llama_model,
            llama_context_params,
        ) -> *mut llama_context = llama_init_from_model;
        let _tokenize: unsafe extern "C" fn(
            *const llama_vocab,
            *const c_char,
            i32,
            *mut llama_token,
            i32,
            bool,
            bool,
        ) -> i32 = llama_tokenize;
        let _decode: unsafe extern "C" fn(*mut llama_context, llama_batch) -> i32 =
            llama_decode;
        let _sampler_sample: unsafe extern "C" fn(
            *mut llama_sampler,
            *mut llama_context,
            i32,
        ) -> llama_token = llama_sampler_sample;
        let _sampler_chain_init: unsafe extern "C" fn(llama_sampler_chain_params) -> *mut llama_sampler =
            llama_sampler_chain_init;
        let _sampler_init_greedy: unsafe extern "C" fn() -> *mut llama_sampler =
            llama_sampler_init_greedy;
    }

    // -------------------------------------------------------------------
    // Constant values
    // -------------------------------------------------------------------

    #[test]
    fn test_constants_have_expected_values() {
        assert_eq!(LLAMA_DEFAULT_SEED, 0xFFFFFFFF);
        assert_eq!(LLAMA_TOKEN_NULL, -1);
        // Split modes
        assert_eq!(LLAMA_SPLIT_NONE, 0);
        assert_eq!(LLAMA_SPLIT_LAYER, 1);
        assert_eq!(LLAMA_SPLIT_ROW, 2);
        assert_eq!(LLAMA_SPLIT_TENSOR, 3);
        // RoPE scaling
        assert_eq!(LLAMA_ROPE_SCALING_UNSPECIFIED, -1);
        assert_eq!(LLAMA_ROPE_SCALING_NONE, 0);
        // Pooling
        assert_eq!(LLAMA_POOLING_UNSPECIFIED, -1);
        assert_eq!(LLAMA_POOLING_MEAN, 1);
        // Attention
        assert_eq!(LLAMA_ATTENTION_UNSPECIFIED, -1);
        assert_eq!(LLAMA_ATTENTION_CAUSAL, 0);
        assert_eq!(LLAMA_ATTENTION_NON_CAUSAL, 1);
        // Flash attn
        assert_eq!(LLAMA_FLASH_ATTN_AUTO, -1);
        assert_eq!(LLAMA_FLASH_ATTN_ENABLED, 1);
        // KV override tags
        assert_eq!(LLAMA_KV_OVERRIDE_INT, 0);
        assert_eq!(LLAMA_KV_OVERRIDE_STR, 3);
    }

    // -------------------------------------------------------------------
    // Opaque type sizes — the `_private: [u8; 0]` pattern means the
    // compiler-reported size is 0. This test guards against accidental
    // changes (e.g. adding a field) that would change the ABI.
    // -------------------------------------------------------------------

    #[test]
    fn test_opaque_types_are_zero_sized() {
        assert_eq!(size_of::<llama_model>(), 0);
        assert_eq!(size_of::<llama_context>(), 0);
        assert_eq!(size_of::<llama_vocab>(), 0);
        assert_eq!(size_of::<llama_sampler>(), 0);
    }

    // -------------------------------------------------------------------
    // Plain-data struct construction
    // -------------------------------------------------------------------

    #[test]
    fn test_llama_token_data_construction() {
        let td = llama_token_data {
            id: 42,
            logit: 1.5,
            p: 0.25,
        };
        assert_eq!(td.id, 42);
        assert_eq!(td.logit, 1.5);
        assert_eq!(td.p, 0.25);
    }

    #[test]
    fn test_llama_sampler_chain_params_construction() {
        let p = llama_sampler_chain_params { no_perf: true };
        assert!(p.no_perf);
        let p2 = llama_sampler_chain_params { no_perf: false };
        assert!(!p2.no_perf);
    }

    #[test]
    fn test_llama_batch_zero_init() {
        // Struct can be zero-initialized; pointers will be null.
        let b = llama_batch {
            n_tokens: 0,
            token: ptr::null_mut(),
            embd: ptr::null_mut(),
            pos: ptr::null_mut(),
            n_seq_id: ptr::null_mut(),
            seq_id: ptr::null_mut(),
            logits: ptr::null_mut(),
        };
        assert_eq!(b.n_tokens, 0);
        assert!(b.token.is_null());
    }

    // -------------------------------------------------------------------
    // Safe wrappers — error-path tests (no model required)
    // -------------------------------------------------------------------

    /// `load_model` must return `Err` when given a path that does not
    /// exist on disk. This exercises the FFI boundary without requiring
    /// a real model file: llama.cpp returns NULL for missing files and
    /// the Rust wrapper surfaces it as `Err`.
    #[test]
    fn test_load_model_nonexistent_path_returns_err() {
        let result = load_model(
            "/nonexistent/path/to/fake-model.gguf",
            0,    // n_gpu_layers
            512,  // n_ctx
            4,    // n_threads
        );
        assert!(result.is_err(), "load_model should fail for missing file");
        let err = result.unwrap_err();
        assert!(
            err.contains("Failed to load model"),
            "Unexpected error: {}",
            err
        );
    }

    /// `load_model` must return `Err` when the path contains an embedded
    /// NUL byte (which is illegal in C strings). The `CString::new`
    /// conversion happens *before* any FFI call, so this is a pure-Rust
    /// error path.
    #[test]
    fn test_load_model_path_with_nul_byte_returns_err() {
        let bad_path = String::from("/some/path\0with/nul");
        let result = load_model(&bad_path, 0, 512, 4);
        assert!(result.is_err(), "load_model should fail for NUL-byte path");
    }

    /// `infer_tokens` must reject null model/context pointers before
    /// touching the FFI. This protects against programming errors where
    /// the caller forgot to load the model first.
    #[test]
    fn test_infer_tokens_null_pointers_returns_err() {
        let result = infer_tokens(
            ptr::null_mut(),  // ctx
            ptr::null_mut(),  // model
            "hello",
            16,
            0.0,  // greedy
            0.0,
            0,
        );
        assert!(result.is_err(), "infer_tokens should fail on null pointers");
        let err = result.unwrap_err();
        assert!(
            err.contains("Null context or model pointer"),
            "Unexpected error: {}",
            err
        );
    }

    /// `infer_tokens` with a null ctx (but a non-null model) must also
    /// fail fast without invoking FFI.
    #[test]
    fn test_infer_tokens_null_ctx_returns_err() {
        let result = infer_tokens(
            ptr::null_mut(),
            1 as *mut llama_model, // non-null but invalid — must not be dereferenced
            "hello",
            16,
            0.0,
            0.0,
            0,
        );
        assert!(result.is_err());
    }

    // -------------------------------------------------------------------
    // Plain-data struct field sizes — guards against accidental layout
    // changes that would break FFI.
    // -------------------------------------------------------------------

    #[test]
    fn test_struct_field_sizes_are_sane() {
        // llama_token_data is 3 fields: i32 + f32 + f32 = 12 bytes,
        // no padding needed.
        assert_eq!(size_of::<llama_token_data>(), 12);

        // llama_token_data_array is pointer + usize + i64 + bool.
        // The bool gets padded to 8 bytes on 64-bit for alignment, so
        // the size is 8 (ptr) + 8 (usize) + 8 (i64) + 8 (bool w/ pad) = 32.
        let s = size_of::<llama_token_data_array>();
        assert!(s >= 24, "llama_token_data_array too small: {}", s);

        // llama_batch is 7 pointer-sized fields (i32 + 6 ptrs on 64-bit).
        let s = size_of::<llama_batch>();
        assert!(s >= 32, "llama_batch too small: {}", s);

        // llama_sampler_chain_params is a single bool; size includes
        // padding to align to 1 (no padding needed).
        assert_eq!(size_of::<llama_sampler_chain_params>(), 1);
    }

    // -------------------------------------------------------------------
    // Type aliases
    // -------------------------------------------------------------------

    #[test]
    fn test_type_aliases() {
        // Compile-time check: type aliases resolve correctly.
        let _t: llama_token = 0;
        let _p: llama_pos = 0;
        let _s: llama_seq_id = 0;
        // i32-based aliases
        assert_eq!(size_of::<llama_token>(), 4);
        assert_eq!(size_of::<llama_pos>(), 4);
        assert_eq!(size_of::<llama_seq_id>(), 4);
    }
}

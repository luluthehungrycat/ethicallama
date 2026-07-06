// C++ exception-safety shim for llama.cpp C API functions.
// Catches C++ exceptions and returns null/error code instead of letting them
// propagate into Rust (which would abort).

#include "llama.h"
#include <cstdio>
#include <exception>

// Helper: call a lambda and catch exceptions
template<typename F, typename R>
static R safe_call(const char * name, F && fn, R error_return) {
    try {
        return fn();
    } catch (const std::exception & e) {
        fprintf(stderr, "SHIM: %s caught: %s\n", name, e.what());
        return error_return;
    } catch (...) {
        fprintf(stderr, "SHIM: %s caught unknown exception\n", name);
        return error_return;
    }
}

extern "C" {

llama_model * llama_model_load_from_file_safe(const char * path, llama_model_params params) {
    return safe_call("llama_model_load_from_file", [&]() {
        return llama_model_load_from_file(path, params);
    }, (llama_model*)nullptr);
}

llama_context * llama_init_from_model_safe(llama_model * model, llama_context_params params) {
    return safe_call("llama_init_from_model", [&]() {
        return llama_init_from_model(model, params);
    }, (llama_context*)nullptr);
}

const llama_vocab * llama_model_get_vocab_safe(const llama_model * model) {
    return safe_call("llama_model_get_vocab", [&]() {
        return llama_model_get_vocab(model);
    }, (const llama_vocab*)nullptr);
}

int32_t llama_model_n_embd_safe(const llama_model * model) {
    return safe_call("llama_model_n_embd", [&]() {
        return llama_model_n_embd(model);
    }, -1);
}

int32_t llama_tokenize_safe(
    const llama_vocab * vocab,
    const char * text,
    int32_t text_len,
    llama_token * tokens,
    int32_t n_tokens_max,
    bool add_special,
    bool parse_special
) {
    return safe_call("llama_tokenize", [&]() {
        return llama_tokenize(vocab, text, text_len, tokens, n_tokens_max, add_special, parse_special);
    }, -1);
}

int32_t llama_token_to_piece_safe(
    const llama_vocab * vocab,
    llama_token token,
    char * buf,
    int32_t length,
    int32_t lstrip,
    bool special
) {
    return safe_call("llama_token_to_piece", [&]() {
        return llama_token_to_piece(vocab, token, buf, length, lstrip, special);
    }, -1);
}

llama_batch llama_batch_get_one_safe(llama_token * tokens, int32_t n_tokens) {
    return safe_call("llama_batch_get_one", [&]() {
        return llama_batch_get_one(tokens, n_tokens);
    }, llama_batch{0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr});
}

int32_t llama_decode_safe(llama_context * ctx, llama_batch batch) {
    return safe_call("llama_decode", [&]() {
        return llama_decode(ctx, batch);
    }, -1);
}

llama_token llama_sampler_sample_safe(llama_sampler * smpl, llama_context * ctx, int32_t idx) {
    return safe_call("llama_sampler_sample", [&]() {
        return llama_sampler_sample(smpl, ctx, idx);
    }, -1);
}

bool llama_vocab_is_eog_safe(const llama_vocab * vocab, llama_token token) {
    return safe_call("llama_vocab_is_eog", [&]() {
        return llama_vocab_is_eog(vocab, token);
    }, true);
}

} // extern "C"

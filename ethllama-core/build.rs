use std::path::Path;

fn main() {
    // Declare the `stub_mode` cfg to rustc so that conditional compilation
    // in lib.rs does not produce `unexpected_cfgs` warnings. The actual
    // `cargo:rustc-cfg=stub_mode` is set below, only when the submodule
    // is missing.
    println!("cargo:rustc-check-cfg=cfg(stub_mode)");

    let llama_cpp_dir = Path::new("llama.cpp");

    // If the llama.cpp git submodule is not present (e.g. when building
    // from an sdist), fall back to building a stub Rust extension.
    //
    // We do NOT panic — the Python side already handles the case where
    // the native extension is missing via `has_inference_engine()` and
    // falls back to the subprocess inference path (llama-cli, etc.).
    //
    // We emit `cargo:rustc-cfg=stub_mode` so that lib.rs can use
    // `#[cfg(all(feature = "llama-cpp", not(stub_mode)))]` to gate the
    // llama-cpp integration code. This is needed because the `llama-cpp`
    // Cargo feature is on by default (and stays on), but the actual
    // llama.cpp sources are missing.
    if !llama_cpp_dir.exists() {
        println!(
            "cargo:warning=ethllama-core: llama.cpp submodule not found — building stub Rust extension."
        );
        println!(
            "cargo:warning=ethllama-core: For full native inference, run 'git submodule update --init --recursive' before building."
        );
        println!(
            "cargo:warning=ethllama-core: Python will fall back to the subprocess inference path."
        );
        println!("cargo:rustc-cfg=stub_mode");
        return;
    }

    // Full build: compile llama.cpp via cmake.
    #[cfg(feature = "llama-cpp")]
    {
        let dst = cmake::Config::new("llama.cpp")
            .define("BUILD_SHARED_LIBS", "OFF")
            .define("LLAMA_BUILD_TESTS", "OFF")
            .define("LLAMA_BUILD_EXAMPLES", "OFF")
            .define("LLAMA_BUILD_SERVER", "OFF")
            .define("LLAMA_BUILD_TOOLS", "OFF")
            .define("LLAMA_BUILD_COMMON", "OFF")
            .define("LLAMA_BUILD_APP", "OFF")
            // Disable GPU backends — CPU-only for the core library
            .define("LLAMA_CUDA", "OFF")
            .define("LLAMA_METAL", "OFF")
            .define("LLAMA_VULKAN", "OFF")
            .define("LLAMA_HIPBLAS", "OFF")
            .define("LLAMA_SYCL", "OFF")
            .define("LLAMA_KOMPUTE", "OFF")
            .build();

        println!(
            "cargo:rustc-link-search=native={}",
            dst.join("lib").display()
        );
        println!("cargo:rustc-link-lib=static=llama");
        println!("cargo:rustc-link-lib=static=ggml");
        println!("cargo:rustc-link-lib=static=ggml-cpu");
        println!("cargo:rustc-link-lib=static=ggml-base");

        // Link system libs
        println!("cargo:rustc-link-lib=pthread");
        println!("cargo:rustc-link-lib=dl");
        println!("cargo:rustc-link-lib=stdc++");
        println!("cargo:rustc-link-lib=gomp");

        // Compile the C++ exception-safety shim
        // This is compiled as C++ and linked into the cdylib
        cc::Build::new()
            .cpp(true)
            .file("src/shim.cpp")
            .include("llama.cpp/include")
            .include("llama.cpp/ggml/include")
            .compile("ethllama_shim");

        // Rerun if headers/CMakeLists change
        println!("cargo:rerun-if-changed=llama.cpp/include/llama.h");
        println!("cargo:rerun-if-changed=llama.cpp/src/CMakeLists.txt");
        println!("cargo:rerun-if-changed=llama.cpp/ggml/CMakeLists.txt");
        println!("cargo:rerun-if-changed=llama.cpp/CMakeLists.txt");
    }

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/shim.cpp");
}

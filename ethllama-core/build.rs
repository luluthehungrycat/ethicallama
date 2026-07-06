use std::path::Path;

fn main() {
    let llama_cpp_dir = Path::new("llama.cpp");

    // Check if llama.cpp submodule exists
    if !llama_cpp_dir.exists() {
        println!("cargo:warning=llama.cpp submodule not found.");
        println!("cargo:warning=Run 'git submodule update --init --recursive' to fetch it.");
        panic!("llama.cpp submodule is required. Run: git submodule update --init --recursive");
    }

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
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/shim.cpp");
}

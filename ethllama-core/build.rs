use std::path::Path;
use std::process::Command;

fn main() {
    // Clone llama.cpp as a submodule (or assume it's already there)
    let llama_cpp_dir = Path::new("llama.cpp");
    if !llama_cpp_dir.exists() {
        Command::new("git")
            .args(&["submodule", "add", "https://github.com/ggerganov/llama.cpp", "llama.cpp"])
            .status()
            .expect("Failed to clone llama.cpp");
    }

    // Build llama.cpp as a static library
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .include("llama.cpp")
        .file("llama.cpp/llama.cpp")
        .file("llama.cpp/common.cpp")
        // Add other source files as needed
        .compile("libllama.a");

    println!("cargo:rustc-link-lib=static=llama");
    println!("cargo:rustc-link-search=native=.");
}

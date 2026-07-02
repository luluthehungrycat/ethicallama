// ethllama-core/src/utils.rs
//
// System‑level utilities: GPU detection, device enumeration, etc.

use serde::{Deserialize, Serialize};
use std::process::Command;

// ---------------------------------------------------------------------------
// GPU information
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize)]
pub struct GpuInfo {
    pub backend: String,
    pub available: bool,
    pub device_count: u32,
    pub device_names: Vec<String>,
}

// ---------------------------------------------------------------------------
// Detection helpers
// ---------------------------------------------------------------------------

/// Detect available GPU backends on the system.
pub fn detect_gpu_backends() -> Vec<GpuInfo> {
    let mut backends = Vec::with_capacity(4);

    backends.push(check_cuda());
    backends.push(check_rocm());
    backends.push(check_vulkan());
    backends.push(GpuInfo {
        backend: "cpu".to_string(),
        available: true,
        device_count: 1,
        device_names: vec!["CPU".to_string()],
    });

    backends
}

fn check_cuda() -> GpuInfo {
    match Command::new("nvidia-smi")
        .arg("--query-gpu=name")
        .arg("--format=csv,noheader")
        .output()
    {
        Ok(output) if output.status.success() => {
            let names: Vec<String> = String::from_utf8_lossy(&output.stdout)
                .lines()
                .map(|l| l.trim().to_string())
                .filter(|l| !l.is_empty())
                .collect();
            GpuInfo {
                backend: "cuda".to_string(),
                available: !names.is_empty(),
                device_count: names.len() as u32,
                device_names: names,
            }
        }
        _ => GpuInfo {
            backend: "cuda".to_string(),
            available: false,
            device_count: 0,
            device_names: vec![],
        },
    }
}

fn check_rocm() -> GpuInfo {
    match Command::new("rocm-smi")
        .arg("--showproductname")
        .output()
    {
        Ok(output) if output.status.success() => {
            let names: Vec<String> = String::from_utf8_lossy(&output.stdout)
                .lines()
                .map(|l| l.trim().to_string())
                .filter(|l| !l.is_empty())
                .collect();
            GpuInfo {
                backend: "rocm".to_string(),
                available: !names.is_empty(),
                device_count: names.len() as u32,
                device_names: names,
            }
        }
        _ => GpuInfo {
            backend: "rocm".to_string(),
            available: false,
            device_count: 0,
            device_names: vec![],
        },
    }
}

fn check_vulkan() -> GpuInfo {
    match Command::new("vulkaninfo")
        .arg("--summary")
        .output()
    {
        Ok(output) if output.status.success() => {
            let info = String::from_utf8_lossy(&output.stdout);
            let has_devices = info.contains("GPU") || info.contains("Device");
            GpuInfo {
                backend: "vulkan".to_string(),
                available: has_devices,
                device_count: if has_devices { 1 } else { 0 },
                device_names: if has_devices {
                    vec!["Vulkan GPU".to_string()]
                } else {
                    vec![]
                },
            }
        }
        _ => GpuInfo {
            backend: "vulkan".to_string(),
            available: false,
            device_count: 0,
            device_names: vec![],
        },
    }
}

/// Return the name of the best available GPU backend.
/// Priority: cuda > rocm > vulkan > cpu.
pub fn get_best_backend() -> String {
    let backends = detect_gpu_backends();
    for preferred in &["cuda", "rocm", "vulkan"] {
        for backend in &backends {
            if backend.backend == *preferred && backend.available {
                return backend.backend.clone();
            }
        }
    }
    "cpu".to_string()
}

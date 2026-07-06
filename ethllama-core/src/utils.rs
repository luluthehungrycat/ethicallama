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
    vec![
        check_cuda(),
        check_rocm(),
        check_vulkan(),
        GpuInfo {
            backend: "cpu".to_string(),
            available: true,
            device_count: 1,
            device_names: vec!["CPU".to_string()],
        },
    ]
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
    match Command::new("rocm-smi").arg("--showproductname").output() {
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
    match Command::new("vulkaninfo").arg("--summary").output() {
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
#[allow(dead_code)]
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

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    /// `detect_gpu_backends()` must return a Vec. The CPU backend is always
    /// present, so the Vec is never empty even on systems with no GPU tools.
    #[test]
    fn test_detect_gpu_returns_vec() {
        let backends = detect_gpu_backends();
        // Vec type is enforced by the let-binding; assert non-empty.
        assert!(
            !backends.is_empty(),
            "detect_gpu_backends() returned an empty Vec"
        );
        // All entries should be one of the known backends.
        for backend in &backends {
            assert!(
                matches!(backend.backend.as_str(), "cuda" | "rocm" | "vulkan" | "cpu"),
                "Unknown backend: {}",
                backend.backend
            );
        }
    }

    /// `detect_gpu_backends()` must not panic when GPU tooling is missing.
    /// This simulates a barebones CI environment (no nvidia-smi, no rocm-smi,
    /// no vulkaninfo). The implementation uses `Command::output()` which
    /// returns `Err` for missing executables; that path is matched and the
    /// backend is reported as unavailable — no panic, no unwind.
    #[test]
    fn test_detect_gpu_no_panic_on_missing_tools() {
        // Call multiple times to catch any non-determinism / lazy-init issues.
        for _ in 0..5 {
            let backends = detect_gpu_backends();
            // CPU backend must always be present regardless of GPU tooling.
            assert!(
                backends.iter().any(|b| b.backend == "cpu"),
                "CPU backend must always be reported"
            );
            // CUDA/ROCm/Vulkan backends must always be present in the Vec
            // (even when unavailable) so callers can rely on the layout.
            assert!(backends.iter().any(|b| b.backend == "cuda"));
            assert!(backends.iter().any(|b| b.backend == "rocm"));
            assert!(backends.iter().any(|b| b.backend == "vulkan"));
        }
    }

    /// The `GpuInfo` struct must be constructible from all its fields.
    /// This is a compile-time check that the field set is stable.
    #[test]
    fn test_gpu_info_struct_construction() {
        let info = GpuInfo {
            backend: "test".to_string(),
            available: true,
            device_count: 2,
            device_names: vec!["dev0".to_string(), "dev1".to_string()],
        };
        assert_eq!(info.backend, "test");
        assert!(info.available);
        assert_eq!(info.device_count, 2);
        assert_eq!(info.device_names.len(), 2);
    }

    /// The CPU backend is unconditionally added by `detect_gpu_backends()`.
    /// It must always be marked available with at least one "CPU" device.
    #[test]
    fn test_cpu_backend_always_present() {
        let backends = detect_gpu_backends();
        let cpu = backends
            .iter()
            .find(|b| b.backend == "cpu")
            .expect("CPU backend missing from detect_gpu_backends()");
        assert!(cpu.available, "CPU backend should always be available");
        assert!(cpu.device_count >= 1, "CPU device_count must be >= 1");
        assert!(
            !cpu.device_names.is_empty(),
            "CPU device_names must be non-empty"
        );
    }

    /// `get_best_backend()` must return one of the known backend names
    /// (or "cpu" as the universal fallback). It must never panic and must
    /// never return an empty string.
    #[test]
    fn test_get_best_backend_returns_valid_string() {
        let best = get_best_backend();
        assert!(!best.is_empty(), "get_best_backend() returned empty string");
        assert!(
            matches!(best.as_str(), "cuda" | "rocm" | "vulkan" | "cpu"),
            "get_best_backend() returned unknown backend: {}",
            best
        );
    }

    /// `get_best_backend()` must be consistent with `detect_gpu_backends()`.
    /// Whatever it returns must match an available backend in the detection
    /// list (or be the "cpu" fallback when nothing else is available).
    #[test]
    fn test_get_best_backend_is_consistent_with_detection() {
        let backends = detect_gpu_backends();
        let best = get_best_backend();
        if best != "cpu" {
            assert!(
                backends.iter().any(|b| b.backend == best && b.available),
                "get_best_backend() returned {} but it's not in the available list",
                best
            );
        }
    }

    /// `GpuInfo` must round-trip through serde_json (required by the
    /// PyO3 `gpu_info()` method which serializes the result).
    #[test]
    fn test_gpu_info_serializes_to_json() {
        let info = GpuInfo {
            backend: "cuda".to_string(),
            available: true,
            device_count: 1,
            device_names: vec!["NVIDIA GeForce RTX 4090".to_string()],
        };
        let json = serde_json::to_string(&info).expect("GpuInfo should serialize");
        assert!(json.contains("\"backend\":\"cuda\""));
        assert!(json.contains("\"available\":true"));
        assert!(json.contains("\"device_count\":1"));
        // Round-trip
        let parsed: GpuInfo = serde_json::from_str(&json).expect("GpuInfo should deserialize");
        assert_eq!(parsed.backend, info.backend);
        assert_eq!(parsed.device_count, info.device_count);
    }

    /// An empty `GpuInfo` (no devices) must still be valid JSON and
    /// represent the "unavailable" state of a backend.
    #[test]
    fn test_gpu_info_empty_device_list() {
        let info = GpuInfo {
            backend: "rocm".to_string(),
            available: false,
            device_count: 0,
            device_names: vec![],
        };
        assert!(!info.available);
        assert_eq!(info.device_count, 0);
        assert!(info.device_names.is_empty());
        let json = serde_json::to_string(&info).expect("empty GpuInfo should serialize");
        assert!(json.contains("\"available\":false"));
    }
}

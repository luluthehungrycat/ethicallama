#!/usr/bin/env bash
set -euo pipefail

echo "=== ethicallama setup ==="

# Check prerequisites
echo "Checking prerequisites..."
command -v python3 >/dev/null 2>&1 || { echo "Python 3 is required"; exit 1; }
command -v cargo >/dev/null 2>&1 || { echo "Rust/Cargo is required"; exit 1; }
command -v git >/dev/null 2>&1 || { echo "Git is required"; exit 1; }

# Initialize submodules
echo "Initializing submodules..."
git submodule update --init --recursive

# Build Rust core
echo "Building Rust core..."
cargo build --release -p ethllama-core

# Install Python package
echo "Installing Python package..."
pip install -e ".[all]"

echo ""
echo "=== Setup complete! ==="
echo "Run 'ethllama --help' to get started."
echo "Run 'ethllama config --init' to configure."

#!/bin/bash
# scripts/setup.sh

set -e

echo "Setting up Rust toolchain..."
# Install Rust if not already installed
if ! command -v rustup &> /dev/null
then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source $HOME/.cargo/env
else
    echo "Rust is already installed"
fi

# Install Tauri CLI if not already installed
if ! cargo install --list | grep -q tauri-cli; then
    echo "Installing Tauri CLI..."
    cargo install tauri-cli
else
    echo "Tauri CLI is already installed"
fi

echo "Installing Rust dependencies..."
cargo build

echo "Setup completed!"
# ✦ NanoCameleer ✦

<div align="center">
  <img src="test.png" alt="NanoCameleer Logo" width="120" style="border-radius: 20px; margin-bottom: 15px;" />
  <p><strong>A premium, safe, and blazing-fast autonomous local AI Agent platform written entirely in Rust, optimized from the ground up for Raspberry Pi Linux.</strong></p>

  [![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg?style=flat-square&logo=rust)](https://www.rust-lang.org)
  [![Platform](https://img.shields.io/badge/platform-Raspberry%20Pi%205%20%7C%20Linux%20ARM64-red.svg?style=flat-square&logo=raspberry-pi)](https://www.raspberrypi.com)
  [![License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)](LICENSE)
  [![Local AI](https://img.shields.io/badge/Local%20AI-Embedded%20NanoCamelid-brightgreen.svg?style=flat-square)](https://github.com/timtoole02/NanoCamelid)
</div>

---

**NanoCameleer** is a next-generation, self-hosted autonomous AI agent workspace designed for superior performance, strict safety controls, and local-first execution on resource-constrained hardware. It features a fully integrated, high-performance local GGUF inference engine (**`camelid`**) powered by the embedded **`NanoCamelid`** runtime, custom-tuned with ARM64 CPU-level optimizations specifically for the **Raspberry Pi 5**.

Connect your agents to standard cloud APIs (Anthropic, Gemini, OpenAI) or route them to your own locally-stored GGUF models concurrently with absolute data privacy—all managed under a strict, human-in-the-loop shell command sandbox.

---

## ⚡ Architecture Overview

```
┌────────────────────────────────────────────────────────┐
│                   control-center                       │
│      (Tauri Desktop / SPA Web Dashboard / TUI)         │
└──────────────────────────┬─────────────────────────────┘
                           │ HTTP / JSON API
                           ▼
┌────────────────────────────────────────────────────────┐
│                     cameleer                           │
│     (Autonomous ReAct Agent Orchestrator & Supervisor) │
└──────────────────────────┬─────────────────────────────┘
                           │ Serves CLI / Telegram / Discord
                           ▼
┌────────────────────────────────────────────────────────┐
│                     camelid serve                      │
│     (Embedded Local GGUF Engine - NanoCamelid Core)     │
│   ARM64 NEON Vectorization • CPU Core-Affinity Pinning │
└────────────────────────────────────────────────────────┘
```

The platform is split into three main components:
1. **`cameleer`**: The core multi-channel autonomous agent coordinator that handles planning, skill execution, and communication.
2. **`camelid`**: The embedded, ultra-lightweight GGUF serving engine (incorporating `NanoCamelid`) that provides local text generation.
3. **`control-center`**: The modern frontend dashboard, built with Vite and Tauri, displaying beautiful real-time thought logs, sandbox approvals, and model hot-swapping.

---

## 🚀 Key Features

*   **Embedded Local AI (via embedded `NanoCamelid`)**: Direct, offline GGUF inference powered by an extremely lightweight, Pi-native Rust backend. No API keys, no subscription plans, and 100% data privacy.
*   **Pi-Tuned CPU Acceleration**: Harnesses the full power of Raspberry Pi 5's Cortex-A76 cores out-of-the-box. The engine implements vectorized ARM64 NEON activation quantization, ARMv8.2-A DotProd (SDOT) matrix-multiplication kernels, and head-parallel attention.
*   **CPU Governor & Core-Affinity Scheduling**: Automatically pins Rayon worker threads to isolated cores and adjusts scheduling priorities to bypass Linux scheduling overhead and squeeze maximum token throughput out of the Pi CPU.
*   **Enterprise-Grade Safety Sandbox**: Rogue agent commands are a thing of the past. NanoCameleer features a strict command whitelist, blocks recursive deletes, and provides *Human-in-the-Loop* approval prompts (inline buttons on Telegram/Discord or interactive CLI confirm gates) before executing any shell commands.
*   **SQLite Persistence**: Chat logs, key-value memory blocks, and comprehensive audit traces of every command executed are saved to an embedded, zero-configuration local SQLite database.
*   **Multi-Channel Concurrency**: Handles multiple gateway integrations in parallel. Control your agent from a beautiful local interactive shell, message it on-the-go from Telegram or Discord, or orchestrate it from the Web Dashboard.

---

## 🍓 Raspberry Pi Linux Installation Guide

Follow these step-by-step instructions to compile, package, and launch the complete NanoCameleer platform with local accelerated AI on your Raspberry Pi:

### 1. Install System Dependencies
Ensure you have the core compiler toolchains and WebKit GTK libraries (used by the Tauri desktop GUI app) installed:
```bash
sudo apt update
sudo apt install -y build-essential curl pkg-config libssl-dev libsoup-3.0-dev libwebkit2gtk-4.1-dev libgtk-3-dev libglib2.0-dev
```

### 2. Install Rust Toolchain
Install Rust via rustup if you don't have it already:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
```

### 3. Install Node.js & npm
Install Node.js (version 20+) for the Web Dashboard and Tauri UI:
```bash
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs
```

### 4. Clone the Repository
Clone the codebase and navigate to the project directory:
```bash
git clone https://github.com/timtoole02/NanoCameleer.git
cd NanoCameleer
```

### 5. Initialize Configuration & Templates
Run the onboarding sequence to automatically initialize default configuration files, SQLite storage, dynamic prompt templates, and active skills:
```bash
cargo run onboard
```

### 6. Compile Core Rust Binaries
Compile both the embedded local inference engine (`camelid`) and the autonomous agent CLI (`cameleer`) in release mode:
```bash
cargo build --release
```

### 7. Launch the Platform
Start the agent in `run` mode. This will automatically spawn the embedded local inference server on port `8181` and boot the autonomous orchestrator, gateways (Telegram, Discord, console), and the local Web Gateway:
```bash
cargo run --release run
```

### 8. Build and Run the Tauri Desktop GUI App (Optional)
If running a graphical desktop environment on your Raspberry Pi OS, you can compile and launch the premium desktop control room app:
```bash
cd control-center
npm install
npm run tauri dev
```

---

## ⚙️ Configuration & Model Setup

Your settings are stored in TOML format at `~/.cameleer/config.toml`.

### 1. Model Selection & Placement
To run local inference on the Pi, place a compatible GGUF model in your local models folder:
```bash
mkdir -p ~/.cameleer/models/
# Place Llama-3.2-1B-Instruct-Q4_0.gguf inside ~/.cameleer/models/
```
> [!TIP]
> **Recommended Models**: `Llama-3.2-1B-Instruct-Q4_0.gguf` or `Llama-3.2-3B-Instruct-Q4_0.gguf` are highly recommended for the best balance of speed, memory usage (RSS under 300MB!), and reasoning performance on Raspberry Pi 5.

### 2. Configuration Options
You can configure several core behaviors in `~/.cameleer/config.toml`:
-   **LLM Provider**: Defaults to `camelid` (local embedded inference), but you can swap to `ollama`, `anthropic`, `openai`, or `gemini`.
-   **Security Whitelist**: Explicitly control the set of binaries the agent is authorized to use:
    ```toml
    [security]
    require_approval = true
    allowed_commands = ["ls", "pwd", "cat", "curl", "grep"]
    ```
-   **Command Approvals**: Toggle `require_approval = true` to force manual confirmations before any mutating operations.
-   **Gateways**: Enable bots, enter tokens, and restrict authorization strictly to specific channels or your personal user accounts.

---

## 📂 Writing Skills

Skills are portable markdown playbooks stored under `~/.cameleer/skills/`. To write a skill, simply create a folder with a `SKILL.md` inside:

```markdown
---
name: weather-check
description: Fetches current weather information for a specified city using wttr.in.
---

# Weather Check Playbook
Use this skill when the user asks for the weather forecast or temperature.

## Workflow
1. Identify the city name from the user's message.
2. Execute the shell command:
   `curl "wttr.in/YOUR_CITY?format=3"` (Replace YOUR_CITY with the target city).
3. Report the exact output of curl directly to the user.
```

NanoCameleer parses the YAML frontmatter and dynamically registers it into the agent's reasoning cycle on startup!

---

## 🛡️ License

NanoCameleer is released under the [MIT License](LICENSE).

# slm-local-benchmarks

A developer-focused benchmark harness designed to measure latency, throughput, and hardware resource consumption of local Small Language Models (SLMs) running on consumer-grade hardware.

Evaluating local SLMs (`1B` to `4B` parameters) on edge devices requires balancing token generation speed against physical system limits: RAM footprint, Time to First Token (TTFT), and sustained CPU utilization. This suite automates repeatable benchmark runs across models served by Ollama, capturing metrics with `psutil` and streaming HTTP response analysis.

---

## Benchmark Highlights (8GB RAM Consumer Baseline)

The following baseline metrics were collected on an 8GB RAM consumer setup (macOS Apple Silicon arm64 / 8-core CPU) running standard Ollama Q4_K_M quantizations across reasoning, code generation, summarization, and information extraction workloads.

| Model | Parameter Size | Download Footprint | Avg TTFT (ms) | Decode Speed | Total Wall Latency | Peak Process RAM | Avg CPU % | Peak CPU % |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Llama 3.2 1B** (`llama3.2:1b`) | 1.24B | ~1.3 GB | **82.4 ms** | **41.8 tok/s** | **6.32 s** | **1,384 MB** | 64.2% | 88.5% |
| **Phi-3 Mini** (`phi3:mini`) | 3.82B | ~2.2 GB | **174.6 ms** | **21.4 tok/s** | **12.45 s** | **2,427 MB** | 78.6% | 94.1% |

### Key Trade-Offs
- **Responsiveness vs Depth:** Llama 3.2 1B delivers sub-100ms TTFT (~82 ms), making it ideal for real-time user-facing agents and low-latency autocomplete tasks. Phi-3 Mini (3.8B) delivers superior reasoning on structured extraction and logic puzzles at the cost of double the initial latency (~175 ms).
- **RAM Headroom:** On an 8GB machine, Llama 3.2 1B reserves roughly ~6.6 GB for the OS and other processes. Phi-3 Mini consumes ~2.4 GB resident memory, which approaches memory pressure limits when operating concurrently with browser windows or IDEs.

---

## Prerequisites & CLI Setup

Before running the benchmarks, ensure your environment has the required local runtime and tooling configured.

### 1. Ollama Runtime
Ollama serves as the local inference engine. Download and initialize it via the official installer:

```bash
# Install Ollama CLI
curl -fsSL https://ollama.com/install.sh | sh

# Verify installation and start server if not running as a system service
ollama --version
ollama serve &
```

Pull the target benchmark models:

```bash
# Pull Meta's Llama 3.2 (1B lightweight edge model)
ollama pull llama3.2:1b

# Pull Microsoft's Phi-3 Mini (3.8B high-density model)
ollama pull phi3:mini
```

### 2. Claude Code CLI
Claude Code provides an agentic CLI workflow that can be used alongside this benchmark suite to iterate on prompts, debug SLM output quality, and automate edge deployment pipelines:

```bash
# Install Claude Code globally via NPM
npm install -g @anthropic-ai/claude-code

# Authenticate your terminal session
claude login
```

Once logged in, you can invoke `claude` directly inside this repository to analyze benchmark outputs, orchestrate prompt tuning experiments, or generate custom evaluation tasks.

---

## Installation & Environment Setup

Clone the repository and set up a Python virtual environment:

```bash
git clone https://github.com/pj2411/slm-local-benchmarks.git
cd slm-local-benchmarks

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### Dependencies
- `requests>=2.31.0`: Direct streaming interaction with Ollama's HTTP API (`/api/generate`).
- `psutil>=5.9.0`: Process and system-level hardware telemetry (CPU %, RSS memory in MB).
- `tabulate>=0.9.0`: Markdown and terminal table rendering.

---

## Execution Guide

### Standard Benchmark Run
Execute the default benchmark evaluating `llama3.2:1b` and `phi3:mini` over all test cases defined in `prompts.json` (3 iterations each, preceded by 1 warmup pass):

```bash
python benchmark.py
```

### CLI Options & Customization

```text
usage: benchmark.py [-h] [--models MODELS [MODELS ...]] [--host HOST]
                    [--prompts-file PROMPTS_FILE] [--runs RUNS]
                    [--warmup WARMUP] [--output-json OUTPUT_JSON]
                    [--output-md OUTPUT_MD] [--stream]

Options:
  --models MODELS [MODELS ...]
                        Ollama model tags to test (default: llama3.2:1b phi3:mini)
  --host HOST           Ollama base URL (default: http://localhost:11434)
  --prompts-file PATH   Path to JSON file with prompt test cases (default: prompts.json)
  --runs INT            Number of measurement iterations per prompt (default: 3)
  --warmup INT          Warmup passes before recording metrics (default: 1)
  --output-json PATH    Output path for raw JSON results (default: results/benchmark_results.json)
  --output-md PATH      Output path for Markdown summary (default: results/benchmark_results.md)
  --stream              Stream model output directly to stdout during execution
```

### Examples

**Benchmark a single custom model with live streaming output:**
```bash
python benchmark.py --models llama3.2:1b --runs 2 --stream
```

**Run an extended evaluation on multiple quantization tags:**
```bash
python benchmark.py --models llama3.2:1b phi3:mini qwen2.5:1.5b --runs 5 --warmup 2
```

**Evaluate against a custom prompt dataset:**
```bash
python benchmark.py --prompts-file custom_prompts.json --output-md results/my_device_report.md
```

---

## Architecture & Metric Definitions

`benchmark.py` monitors inference without adding measurement skew:

```
┌─────────────────────────────────────────────────────────────┐
│                       benchmark.py                          │
│                                                             │
│   ┌───────────────────────────┐  POST /api/generate (stream)│
│   │     HardwareMonitor       │  ────────────────────────►  │
│   │ (psutil Background Thread)│                             │
│   │  - Ollama Runner RSS (MB) │                             │
│   │  - System Memory Used     │  ◄────────────────────────  │
│   │  - CPU % Utilization      │     Chunked JSON Stream     │
│   └───────────────────────────┘                             │
└─────────────────────────────────────────────────────────────┘
```

1. **Time to First Token (TTFT):** Wall-clock duration (in milliseconds) from sending the HTTP request until the first response token is yielded by the streaming socket. This reflects perceived latency.
2. **Decode Throughput (tok/s):** The raw generation throughput calculated via Ollama's internal `eval_count` and `eval_duration` fields, isolating model inference from network transmission and prompt evaluation overhead.
3. **End-to-End Latency:** The full elapsed wall-clock duration from request start to completion.
4. **Peak Process RAM (MB):** The maximum Resident Set Size (RSS) recorded across active Ollama processes (`ollama serve`, `ollama runner`, and llama.cpp backends) during prompt execution.
5. **CPU % Utilization:** Mean and peak CPU percentages sampled across all active CPU cores during the inference window.

---

## Repository Structure

```text
slm-local-benchmarks/
├── benchmark.py                # Main benchmark runner and telemetry collector
├── prompts.json                # Standardized multi-domain test prompts
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation and baseline tables
└── results/
    ├── baseline_consumer_8gb.json  # Reference benchmark JSON on 8GB consumer hardware
    └── baseline_consumer_8gb.md    # Formatted reference summary table
```
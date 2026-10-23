# Baseline Benchmark Results (8GB RAM Consumer Hardware)

**Test Environment:**
- **System:** macOS Darwin (Apple Silicon arm64, 8 cores)
- **Memory:** 8.00 GB LPDDR5 Unified Memory
- **Quantization:** Standard Ollama Q4_K_M (Default)
- **Runs per prompt:** 3 passes (preceded by 1 warmup pass)

## Performance & Resource Utilization Summary

| Model | Size | Avg TTFT (ms) | Decode Throughput | Total Latency | Peak RAM Footprint | Avg CPU % | Peak CPU % |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Llama 3.2 1B** (`llama3.2:1b`) | 1.24B (1.3 GB) | **82.4 ms** | **41.8 tok/s** | **6.32 s** | **1,384 MB** | 64.2% | 88.5% |
| **Phi-3 Mini** (`phi3:mini`) | 3.82B (2.2 GB) | **174.6 ms** | **21.4 tok/s** | **12.45 s** | **2,427 MB** | 78.6% | 94.1% |

### Observations & Key Takeaways
1. **Llama 3.2 1B** excels in interactive edge loops where latency and responsiveness are paramount. Its ~82 ms Time to First Token (TTFT) feels instantaneous to the user, and with a peak RAM footprint under 1.4 GB, it leaves over 75% of an 8GB system's memory for the operating system and background applications.
2. **Phi-3 Mini (3.8B)** provides stronger reasoning depth on multi-step and domain-specific prompts, but doubles the TTFT (~175 ms) and requires ~2.4 GB of RAM. On memory-constrained edge devices with 4GB to 8GB unified memory, Llama 3.2 1B is significantly safer against out-of-memory pressure and thrashing.

#!/usr/bin/env python3
"""
slm-local-benchmarks: Benchmark local Small Language Models (SLMs) running on Ollama.
Measures latency (TTFT, total inference wall-time), throughput (tokens/sec),
and hardware consumption (peak RAM, CPU % utilization) using psutil and requests.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import sys
import threading
import time
from typing import Any, Dict, List, Optional

import psutil
import requests

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False


@dataclasses.dataclass
class HardwareSample:
    timestamp: float
    cpu_percent: float
    system_ram_mb: float
    ollama_ram_mb: float


class HardwareMonitor:
    """
    Background sampler capturing system CPU, system RAM, and Ollama
    runtime processes (ollama serve, ollama runner) at fixed intervals.
    """

    def __init__(self, interval_sec: float = 0.05):
        self.interval_sec = interval_sec
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.samples: List[HardwareSample] = []
        self._my_pid = os.getpid()

    def _get_ollama_pids(self) -> List[psutil.Process]:
        procs: List[psutil.Process] = []
        for p in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                if p.pid == self._my_pid:
                    continue
                name = (p.info.get("name") or "").lower()
                cmdline = " ".join(p.info.get("cmdline") or []).lower()
                if "ollama" in name or "ollama" in cmdline or "llama_server" in name:
                    procs.append(p)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return procs

    def _get_ollama_ram_mb(self, procs: List[psutil.Process]) -> float:
        total_mb = 0.0
        for p in procs:
            try:
                total_mb += p.memory_info().rss / (1024 * 1024)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return total_mb

    def _monitor_loop(self) -> None:
        target_procs = self._get_ollama_pids()
        psutil.cpu_percent(interval=None)

        while not self._stop_event.is_set():
            now = time.time()
            cpu = psutil.cpu_percent(interval=None)
            sys_ram = psutil.virtual_memory().used / (1024 * 1024)

            # Refresh target processes periodically if needed
            ollama_ram = self._get_ollama_ram_mb(target_procs)
            if ollama_ram == 0.0:
                target_procs = self._get_ollama_pids()
                ollama_ram = self._get_ollama_ram_mb(target_procs)

            self.samples.append(HardwareSample(now, cpu, sys_ram, ollama_ram))
            time.sleep(self.interval_sec)

    def start(self) -> None:
        self.samples.clear()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, float]:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)

        if not self.samples:
            return {
                "peak_ollama_ram_mb": 0.0,
                "peak_system_ram_mb": 0.0,
                "avg_cpu_percent": 0.0,
                "peak_cpu_percent": 0.0,
            }

        ollama_rams = [s.ollama_ram_mb for s in self.samples if s.ollama_ram_mb > 0]
        sys_rams = [s.system_ram_mb for s in self.samples]
        cpus = [s.cpu_percent for s in self.samples]

        peak_ollama_ram = max(ollama_rams) if ollama_rams else 0.0
        peak_sys_ram = max(sys_rams) if sys_rams else 0.0
        avg_cpu = sum(cpus) / len(cpus) if cpus else 0.0
        peak_cpu = max(cpus) if cpus else 0.0

        return {
            "peak_ollama_ram_mb": round(peak_ollama_ram, 2),
            "peak_system_ram_mb": round(peak_sys_ram, 2),
            "avg_cpu_percent": round(avg_cpu, 2),
            "peak_cpu_percent": round(peak_cpu, 2),
        }


@dataclasses.dataclass
class PromptRunResult:
    prompt_id: str
    category: str
    model: str
    run_index: int
    ttft_ms: float
    prompt_eval_ms: float
    total_time_sec: float
    eval_count: int
    decode_tps: float
    e2e_tps: float
    peak_ram_mb: float
    avg_cpu_percent: float
    peak_cpu_percent: float
    response_text: str


class OllamaBenchmarker:
    """Coordinates local Ollama inference testing and metric collection."""

    def __init__(self, host: str = "http://localhost:11434", timeout: int = 120):
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def check_health(self) -> bool:
        try:
            resp = self.session.get(f"{self.host}/api/tags", timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def list_models(self) -> List[str]:
        try:
            resp = self.session.get(f"{self.host}/api/tags", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                return [m.get("name", "") for m in data.get("models", [])]
        except requests.RequestException:
            pass
        return []

    def run_single_prompt(
        self,
        model: str,
        prompt: str,
        prompt_id: str,
        category: str,
        run_index: int,
        stream_preview: bool = False,
    ) -> PromptRunResult:
        url = f"{self.host}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": 0.2,
                "num_predict": 256,
            },
        }

        monitor = HardwareMonitor()
        monitor.start()

        start_time = time.perf_counter()
        first_token_time: Optional[float] = None
        full_text: List[str] = []
        token_count = 0
        final_meta: Dict[str, Any] = {}

        try:
            with self.session.post(url, json=payload, stream=True, timeout=self.timeout) as response:
                response.raise_for_status()

                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line.decode("utf-8"))
                    except json.JSONDecodeError:
                        continue

                    # Catch first token (handles standard models as well as models with reasoning streams)
                    part = chunk.get("response") or chunk.get("thinking") or ""
                    if first_token_time is None and part:
                        first_token_time = time.perf_counter()

                    if part:
                        full_text.append(part)
                        token_count += 1
                        if stream_preview:
                            sys.stdout.write(part)
                            sys.stdout.flush()

                    if chunk.get("done"):
                        final_meta = chunk
                        break

            end_time = time.perf_counter()
        finally:
            hw_metrics = monitor.stop()

        if stream_preview:
            sys.stdout.write("\n")

        # TTFT: Milliseconds from request initiation until the first token arrives
        if first_token_time is not None:
            ttft_ms = (first_token_time - start_time) * 1000.0
        else:
            ttft_ms = (end_time - start_time) * 1000.0

        total_wall_time = end_time - start_time

        # Native Ollama timing metrics if available
        prompt_eval_ns = final_meta.get("prompt_eval_duration", 0)
        prompt_eval_ms = round(prompt_eval_ns / 1e6, 2) if prompt_eval_ns else 0.0

        eval_duration_ns = final_meta.get("eval_duration", 0)
        reported_tokens = final_meta.get("eval_count", token_count)

        # Pure decode token throughput (excluding prompt evaluation and network overhead)
        if eval_duration_ns and reported_tokens:
            decode_tps = reported_tokens / (eval_duration_ns / 1e9)
        elif total_wall_time > 0 and reported_tokens:
            decode_tps = reported_tokens / total_wall_time
        else:
            decode_tps = 0.0

        # End-to-end token throughput
        e2e_tps = (reported_tokens / total_wall_time) if total_wall_time > 0 and reported_tokens else 0.0

        ram_reported = hw_metrics["peak_ollama_ram_mb"]
        if ram_reported == 0.0:
            ram_reported = hw_metrics["peak_system_ram_mb"]

        return PromptRunResult(
            prompt_id=prompt_id,
            category=category,
            model=model,
            run_index=run_index,
            ttft_ms=round(ttft_ms, 2),
            prompt_eval_ms=prompt_eval_ms,
            total_time_sec=round(total_wall_time, 3),
            eval_count=reported_tokens,
            decode_tps=round(decode_tps, 2),
            e2e_tps=round(e2e_tps, 2),
            peak_ram_mb=ram_reported,
            avg_cpu_percent=hw_metrics["avg_cpu_percent"],
            peak_cpu_percent=hw_metrics["peak_cpu_percent"],
            response_text="".join(full_text).strip(),
        )


def get_system_metadata() -> Dict[str, Any]:
    """Capture host architecture and available memory for benchmark context."""
    mem = psutil.virtual_memory()
    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "cpu_count_physical": psutil.cpu_count(logical=False) or psutil.cpu_count(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "total_ram_gb": round(mem.total / (1024**3), 2),
        "available_ram_gb": round(mem.available / (1024**3), 2),
        "python_version": platform.python_version(),
    }


def compute_model_summary(runs: List[PromptRunResult]) -> Dict[str, Any]:
    """Compute aggregated metrics (averages and medians) across runs."""
    if not runs:
        return {}

    ttfts = [r.ttft_ms for r in runs]
    times = [r.total_time_sec for r in runs]
    decode_tpss = [r.decode_tps for r in runs]
    e2e_tpss = [r.e2e_tps for r in runs]
    rams = [r.peak_ram_mb for r in runs if r.peak_ram_mb > 0]
    avg_cpus = [r.avg_cpu_percent for r in runs]
    peak_cpus = [r.peak_cpu_percent for r in runs]

    return {
        "model": runs[0].model,
        "sample_count": len(runs),
        "avg_ttft_ms": round(sum(ttfts) / len(ttfts), 2),
        "min_ttft_ms": round(min(ttfts), 2),
        "max_ttft_ms": round(max(ttfts), 2),
        "avg_total_time_sec": round(sum(times) / len(times), 3),
        "avg_decode_tps": round(sum(decode_tpss) / len(decode_tpss), 2),
        "avg_e2e_tps": round(sum(e2e_tpss) / len(e2e_tpss), 2),
        "peak_ram_mb": round(max(rams), 2) if rams else 0.0,
        "avg_cpu_percent": round(sum(avg_cpus) / len(avg_cpus), 2),
        "peak_cpu_percent": round(max(peak_cpus), 2) if peak_cpus else 0.0,
    }


def render_markdown_table(headers: List[str], rows: List[List[Any]]) -> str:
    """Render a clean GitHub-flavored markdown table without external dependencies."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(str(cell)))

    header_line = "| " + " | ".join(str(h).ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    separator_line = "| " + " | ".join("-" * col_widths[i] for i in range(len(headers))) + " |"
    data_lines = [
        "| " + " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)) + " |"
        for row in rows
    ]

    return "\n".join([header_line, separator_line] + data_lines)


def build_markdown_report(summary_list: List[Dict[str, Any]], system_meta: Dict[str, Any]) -> str:
    headers = [
        "Model",
        "Avg TTFT (ms)",
        "Tokens/Sec (Decode)",
        "Total Latency (s)",
        "Peak RAM (MB)",
        "Avg CPU %",
        "Peak CPU %",
    ]
    rows = []
    for s in summary_list:
        rows.append([
            s["model"],
            f"{s['avg_ttft_ms']:.1f}",
            f"{s['avg_decode_tps']:.2f}",
            f"{s['avg_total_time_sec']:.2f}s",
            f"{s['peak_ram_mb']:.1f}",
            f"{s['avg_cpu_percent']:.1f}%",
            f"{s['peak_cpu_percent']:.1f}%",
        ])

    table_md = render_markdown_table(headers, rows)

    return f"""# SLM Local Benchmark Results

**Host Environment:**
- **OS / Architecture:** {system_meta['os']} {system_meta['os_release']} ({system_meta['architecture']})
- **Logical CPU Cores:** {system_meta['cpu_count_logical']}
- **Total Physical RAM:** {system_meta['total_ram_gb']} GB
- **Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}

## Summary Metrics

{table_md}

*Note: TTFT = Time to First Token. Peak RAM captures process resident memory footprint during inference.*
"""


def load_prompts(prompts_path: str) -> List[Dict[str, str]]:
    if not os.path.exists(prompts_path):
        # Built-in fallback prompts
        return [
            {
                "id": "reasoning_trains",
                "category": "Reasoning",
                "prompt": "Two trains leave stations 400 miles apart heading toward each other at 50 mph and 70 mph. When they cross, which train is farther from its departure station? Answer concisely.",
            },
            {
                "id": "coding_fibonacci",
                "category": "Code Generation",
                "prompt": "Write an efficient Python generator for the Fibonacci sequence up to N elements with type annotations.",
            },
            {
                "id": "summarize_slm",
                "category": "Summarization",
                "prompt": "Explain why 1B to 3B parameter SLMs are gaining popularity on edge devices in two concise sentences.",
            },
        ]

    with open(prompts_path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark local Small Language Models (SLMs) on Ollama with precise hardware & latency tracking."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["llama3.2:1b", "phi3:mini"],
        help="List of Ollama model tags to benchmark (default: llama3.2:1b phi3:mini).",
    )
    parser.add_argument(
        "--host",
        default="http://localhost:11434",
        help="Ollama API base URL (default: http://localhost:11434).",
    )
    parser.add_argument(
        "--prompts-file",
        default="prompts.json",
        help="Path to JSON file containing prompt test cases (default: prompts.json).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of iterations per prompt per model (default: 3).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of initial warmup inference passes before metrics recording (default: 1).",
    )
    parser.add_argument(
        "--output-json",
        default="results/benchmark_results.json",
        help="Path to save full JSON output (default: results/benchmark_results.json).",
    )
    parser.add_argument(
        "--output-md",
        default="results/benchmark_results.md",
        help="Path to save Markdown summary table (default: results/benchmark_results.md).",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream generation text to console in real-time.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    benchmarker = OllamaBenchmarker(host=args.host)

    print("================================================================")
    print("           SLM Local Hardware & Latency Benchmarks             ")
    print("================================================================")

    # 1. Health check
    if not benchmarker.check_health():
        print(f"\n[ERROR] Unable to connect to Ollama server at {args.host}.")
        print("Please verify that Ollama is installed and running:")
        print("  $ ollama serve")
        print("Or check service status: `curl http://localhost:11434`\n")
        return 1

    available_models = benchmarker.list_models()
    print(f"Connected to Ollama at {args.host}")
    print(f"Installed models: {', '.join(available_models) if available_models else 'None found'}")

    # Validate requested models
    models_to_test: List[str] = []
    for model in args.models:
        # Match exact tag or prefix
        matched = any(
            model == m or model == m.split(":")[0] or m.startswith(f"{model}:")
            for m in available_models
        )
        if not matched:
            print(f"\n[WARNING] Model '{model}' not found in local Ollama repository.")
            print(f"Run: ollama pull {model}")
        else:
            models_to_test.append(model)

    if not models_to_test:
        print("\n[ERROR] None of the requested models are currently pulled locally.")
        print("Pull them first, for example:")
        for m in args.models:
            print(f"  ollama pull {m}")
        return 1

    prompts = load_prompts(args.prompts_file)
    system_meta = get_system_metadata()
    print(f"\nHost Specs: {system_meta['os']} {system_meta['architecture']} | "
          f"{system_meta['cpu_count_logical']} Cores | {system_meta['total_ram_gb']} GB RAM")
    print(f"Testing {len(models_to_test)} model(s) across {len(prompts)} prompt(s) ({args.runs} runs each)\n")

    all_results: List[PromptRunResult] = []
    summaries: List[Dict[str, Any]] = []

    for model in models_to_test:
        print(f"\n------------------------------------------------------------")
        print(f"Benchmarking Model: {model}")
        print(f"------------------------------------------------------------")

        # Warmup pass
        if args.warmup > 0:
            print(f"Executing {args.warmup} warmup pass(es) to load weights into memory...")
            for w in range(args.warmup):
                benchmarker.run_single_prompt(
                    model=model,
                    prompt="Hello world, respond in one word.",
                    prompt_id="warmup",
                    category="Warmup",
                    run_index=0,
                    stream_preview=False,
                )

        model_runs: List[PromptRunResult] = []

        for p_idx, p in enumerate(prompts, 1):
            p_id = p.get("id", f"p{p_idx}")
            cat = p.get("category", "General")
            p_text = p.get("prompt", "")

            print(f"\n[{model}] Prompt {p_idx}/{len(prompts)} ({cat}): \"{p_text[:50]}...\"")

            for run_i in range(1, args.runs + 1):
                sys.stdout.write(f"  -> Run {run_i}/{args.runs}: ")
                sys.stdout.flush()

                res = benchmarker.run_single_prompt(
                    model=model,
                    prompt=p_text,
                    prompt_id=p_id,
                    category=cat,
                    run_index=run_i,
                    stream_preview=args.stream,
                )
                model_runs.append(res)
                all_results.append(res)

                print(f"TTFT: {res.ttft_ms:.1f}ms | Latency: {res.total_time_sec:.2f}s | "
                      f"Tokens/s: {res.decode_tps:.2f} | Peak RAM: {res.peak_ram_mb:.1f}MB | CPU: {res.avg_cpu_percent:.1f}%")

        summary = compute_model_summary(model_runs)
        summaries.append(summary)

    # Render results table to terminal
    table_headers = ["Model", "Avg TTFT (ms)", "Tokens/Sec (Decode)", "Wall Time (s)", "Peak RAM (MB)", "Avg CPU %"]
    table_rows = [
        [
            s["model"],
            f"{s['avg_ttft_ms']:.1f}",
            f"{s['avg_decode_tps']:.2f}",
            f"{s['avg_total_time_sec']:.2f}",
            f"{s['peak_ram_mb']:.1f}",
            f"{s['avg_cpu_percent']:.1f}%",
        ]
        for s in summaries
    ]

    print("\n================================================================")
    print("                    FINAL BENCHMARK SUMMARY                     ")
    print("================================================================")
    if HAS_TABULATE:
        print(tabulate(table_rows, headers=table_headers, tablefmt="github"))
    else:
        print(render_markdown_table(table_headers, table_rows))
    print("================================================================\n")

    # Export to JSON
    out_dir = os.path.dirname(args.output_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    json_payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "system": system_meta,
        "config": {
            "runs": args.runs,
            "warmup": args.warmup,
            "host": args.host,
            "models": models_to_test,
        },
        "summaries": summaries,
        "runs": [dataclasses.asdict(r) for r in all_results],
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2)
    print(f"Raw benchmark metrics saved to: {args.output_json}")

    # Export to Markdown
    md_dir = os.path.dirname(args.output_md)
    if md_dir:
        os.makedirs(md_dir, exist_ok=True)

    md_content = build_markdown_report(summaries, system_meta)
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Markdown report generated at:   {args.output_md}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

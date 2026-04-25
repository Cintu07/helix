"""
Helix Benchmark Battery Runner
Runs all benchmarks in sequence and produces a summary report.

Benchmarks:
  1. parity           - 16-bit parity (1 neuron vs 128 GRU neurons)
  2. crystalline_loop - bit-perfect ASCII encode/decode
  3. sine_wave        - continuous sine wave tracking
  4. majority_vote    - 8-bit majority (counting logic)
  5. bracket_matching - balanced bracket detection (streaming)
  6. color_algebra    - circular color arithmetic
  7. relay            - cross-instance phase relay (Phasic Sovereignty)
  8. layered_resonance- multi-step conditional logic (Layered Resonance)

Usage:
  python run_benchmarks.py                   # run all
  python run_benchmarks.py --tasks parity relay   # run specific tasks
  python run_benchmarks.py --epochs 200      # override epochs
"""

import sys
import os
import time
import argparse
import importlib

sys.path.insert(0, os.path.dirname(__file__))

BENCHMARK_MODULES = {
    "parity":            ("benchmarks.parity",            "main"),
    "crystalline_loop":  ("benchmarks.crystalline_loop",  "main"),
    "sine_wave":         ("benchmarks.sine_wave",         "main"),
    "majority_vote":     ("benchmarks.majority_vote",     "main"),
    "bracket_matching":  ("benchmarks.bracket_matching",  "main"),
    "color_algebra":     ("benchmarks.color_algebra",     "main"),
    "relay":             ("benchmarks.relay",             "run_relay_benchmark"),
    "layered_resonance": ("benchmarks.layered_resonance", "run_layered_resonance_benchmark"),
}


def run_all(tasks=None, epochs_override=None, verbose=True):
    results = {}
    selected = tasks if tasks else list(BENCHMARK_MODULES.keys())

    print("=" * 65)
    print("HELIX BENCHMARK BATTERY")
    print(f"Running: {', '.join(selected)}")
    print("=" * 65)

    for name in selected:
        if name not in BENCHMARK_MODULES:
            print(f"\n  [SKIP] Unknown benchmark: {name}")
            continue

        module_path, fn_name = BENCHMARK_MODULES[name]
        print(f"\n{'='*65}")
        print(f"  BENCHMARK: {name.upper()}")
        print(f"{'='*65}")

        t0 = time.time()
        try:
            module = importlib.import_module(module_path)
            fn = getattr(module, fn_name)

            # Pass epochs override if supported
            try:
                if epochs_override:
                    result = fn(epochs=epochs_override)
                else:
                    result = fn()
            except TypeError:
                result = fn()

            elapsed = time.time() - t0
            results[name] = {
                "status": "PASS",
                "elapsed_s": round(elapsed, 1),
                "result": result
            }
            print(f"\n  [{name}] completed in {elapsed:.1f}s")

        except Exception as e:
            elapsed = time.time() - t0
            results[name] = {"status": "ERROR", "elapsed_s": round(elapsed, 1), "error": str(e)}
            print(f"\n  [{name}] ERROR after {elapsed:.1f}s: {e}")

    _print_summary(results)
    return results


def _print_summary(results):
    print("\n" + "=" * 65)
    print("BENCHMARK BATTERY SUMMARY")
    print("=" * 65)

    passed = sum(1 for r in results.values() if r["status"] == "PASS")
    total = len(results)

    for name, r in results.items():
        status = r["status"]
        elapsed = r["elapsed_s"]
        icon = "OK" if status == "PASS" else "FAIL"
        print(f"  [{icon}] {name:<22} {elapsed:>6.1f}s", end="")

        if status == "PASS" and r.get("result") and isinstance(r["result"], dict):
            result_data = r["result"]
            # Print key metrics if available
            if "helix_accuracy" in result_data:
                ha = result_data["helix_accuracy"] * 100
                ga = result_data.get("gru_accuracy", 0) * 100
                print(f"  helix={ha:.1f}% gru={ga:.1f}%", end="")
            elif "helix_success_rate" in result_data:
                hr = result_data["helix_success_rate"] * 100
                gr = result_data.get("gru_success_rate", 0) * 100
                print(f"  helix={hr:.1f}% gru={gr:.1f}%", end="")
        elif status == "ERROR":
            print(f"  error: {r.get('error', '')[:40]}", end="")

        print()

    print(f"\n  Total: {passed}/{total} passed")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Helix Benchmark Battery Runner")
    parser.add_argument(
        "--tasks", nargs="*",
        help=f"Which benchmarks to run. Choices: {list(BENCHMARK_MODULES.keys())}"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override epochs for all benchmarks (useful for quick smoke test)"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available benchmarks and exit"
    )
    args = parser.parse_args()

    if args.list:
        print("Available benchmarks:")
        for name in BENCHMARK_MODULES:
            print(f"  {name}")
        sys.exit(0)

    run_all(tasks=args.tasks, epochs_override=args.epochs)

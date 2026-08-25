"""
Compares eval/results.json against eval/baseline.json (committed to the
repo as the last-known-good numbers). Fails (exit 1) if any config's
retrieval@k or answer_accuracy drops by more than the allowed tolerance.

Run after eval/run_eval.py in CI:
    python -m eval.run_eval
    python -m eval.check_regression
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RESULTS_PATH = Path(__file__).parent / "results.json"
BASELINE_PATH = Path(__file__).parent / "baseline.json"

TOLERANCE = 0.05  # allow up to 5 percentage points of noise before failing


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text())


def main():
    results = load(RESULTS_PATH)
    baseline = load(BASELINE_PATH)

    if not baseline:
        print("No baseline.json found -- writing current results as the new baseline.")
        BASELINE_PATH.write_text(json.dumps(results, indent=2))
        return

    baseline_by_key = {(r["chunking"], r["retrieval"]): r for r in baseline}
    failures = []

    for r in results:
        key = (r["chunking"], r["retrieval"])
        base = baseline_by_key.get(key)
        if not base:
            continue
        for metric in ("retrieval_at_k", "answer_accuracy"):
            drop = base[metric] - r[metric]
            if drop > TOLERANCE:
                failures.append(
                    f"{key}: {metric} dropped {drop:.3f} "
                    f"(baseline={base[metric]}, current={r[metric]})"
                )

    if failures:
        print("REGRESSION DETECTED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)

    print("No regressions detected.")


if __name__ == "__main__":
    main()

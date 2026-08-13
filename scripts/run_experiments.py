#!/usr/bin/env python
"""Experiment-matrix orchestration: 3 arms × 3 seeds, plus a single-seed LOSO run.

The budget protocol lives in the submission order: the C arm has to finish first so that
its actual step count S_C is known before B can be submitted step-matched to it. The
script freezes that order with a Slurm dependency (--dependency=afterok) rather than
relying on anyone remembering it.

Usage:
    python scripts/run_experiments.py submit          # submit the whole matrix
    python scripts/run_experiments.py tables RUNS_DIR # roll up the results tables
"""

import json
import subprocess
import sys
from pathlib import Path

SEEDS = [17, 42, 1337]
RUNS = Path("runs")


def _sbatch(args: list[str], depends_on: str | None = None) -> str:
    cmd = ["sbatch", "--parsable"]
    if depends_on:
        cmd.append(f"--dependency=afterok:{depends_on}")
    cmd += ["scripts/train.sbatch", *args]
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()


def submit() -> None:
    """C runs first to fix S_C; B follows once C is done. A and LOSO have no dependencies."""
    jobs: dict[str, str] = {}
    for seed in SEEDS:
        jobs[f"c-{seed}"] = _sbatch(["c_gold_weak", str(seed)])
        jobs[f"a-{seed}"] = _sbatch(["a_gold", str(seed)])
    jobs["loso"] = _sbatch(["loso_l2", str(SEEDS[0])])

    print("submitted C/A/LOSO:", json.dumps(jobs, indent=2))
    print(
        "\nB arm must match C's actual step count. After the C runs finish:\n"
        "  1. read total_steps from runs/c_gold_weak-seed<SEED>/metrics.json\n"
        "  2. sbatch scripts/train.sbatch b_gold_oversampled <SEED> <S_C>\n"
        "or run `python scripts/run_experiments.py submit-b` once C is done."
    )
    print("\nwatch:  bash scripts/watch_jobs.sh " + " ".join(jobs.values()))


def submit_b() -> None:
    """Submit B once C has finished; the step count is read per seed from C's metrics.json."""
    jobs = {}
    for seed in SEEDS:
        metrics = RUNS / f"c_gold_weak-seed{seed}" / "metrics.json"
        if not metrics.exists():
            sys.exit(f"missing {metrics}; the C arm for seed {seed} has not finished")
        steps_c = json.loads(metrics.read_text())["total_steps"]
        jobs[f"b-{seed}"] = _sbatch(["b_gold_oversampled", str(seed), str(steps_c)])
        print(f"seed {seed}: B submitted with steps_c={steps_c}")
    print("watch:  bash scripts/watch_jobs.sh " + " ".join(jobs.values()))


def tables(runs_dir: Path) -> None:
    """Clear the fairness gate first, then build the ablation tables."""
    from accentroute.eval.tables import assert_budget_alignment

    assert_budget_alignment(runs_dir)
    print(f"budget alignment OK across {runs_dir}")
    print(
        "next: load test-set predictions per arm and call\n"
        "  ablation_table(y_true, preds_by_arm, speaker_keys)\n"
        "  per_source_class_f1(y_true, y_pred, sources)\n"
        "  supported_class_report(ood_y, ood_pred, coverage, in_domain_y, in_domain_pred)"
    )


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "submit"
    if action == "submit":
        submit()
    elif action == "submit-b":
        submit_b()
    elif action == "tables":
        tables(Path(sys.argv[2]) if len(sys.argv) > 2 else RUNS)
    else:
        sys.exit(__doc__)

"""
run_vlcs_comparison.py
----------------------
Runs ERM, ERMPlusPlus, and DCSAM (DC-SAM v6) on VLCS with ResNet18 across
all 4 test environments for a fair benchmark comparison.

Usage (from DomainBed directory, with venv activated):
    python run_vlcs_comparison.py              # train all
    python run_vlcs_comparison.py --dcsam-only # skip baselines, only train DCSAM
    python run_vlcs_comparison.py --results-only # read existing results, no training

VLCS domains:
    0 = C (Caltech101)
    1 = L (LabelMe)
    2 = S (SUN09)
    3 = V (VOC2007)

Final VLCS score = average test accuracy across all 4 leave-one-out runs.
"""

import subprocess
import sys
import json
import os
import re

# ── Configuration ───────────────────────────────────────────────────────
DATASET  = "VLCS"
DATA_DIR = "./data"
N_STEPS  = 5000       # standard for VLCS

# Shared backbone: ResNet18, no AugMix — same as PACS for fair comparison
SHARED_HPARAMS = {"resnet18": True, "resnet50_augmix": False}

# DC-SAM v6: domain-balanced CE + CORAL + SGD-SAM (the final algorithm)
DCSAM_HPARAMS = {
    **SHARED_HPARAMS,
    "rho": 0.05,
    "lambda_feat": 0.1,   # CORAL weight
    "lambda_var": 0.0,
    "sam_sgd": True,
    "sgd_lr": 1e-3,
}

# All 4 VLCS test environments
TEST_ENVS = [0, 1, 2, 3]
ENV_NAMES = {0: "Caltech", 1: "LabelMe", 2: "SUN09", 3: "VOC2007"}

ALGORITHMS = [
    {"algorithm": "ERM",         "steps": N_STEPS, "hparams": SHARED_HPARAMS, "tag": ""},
    {"algorithm": "ERMPlusPlus", "steps": N_STEPS, "hparams": SHARED_HPARAMS, "tag": ""},
    {"algorithm": "DCSAM",       "steps": N_STEPS, "hparams": DCSAM_HPARAMS,  "tag": "_v6"},
]

# ── Runner ──────────────────────────────────────────────────────────────
def run(algorithm, steps, hparams, test_env, tag=""):
    output_dir = f"./train_output_vlcs_{algorithm.lower()}_e{test_env}{tag}"
    hparams_json = json.dumps(hparams)

    print("\n" + "=" * 70)
    print(f"  {algorithm}  |  VLCS test_env={test_env} ({ENV_NAMES[test_env]})  |  {steps} steps")
    print(f"  hparams: {hparams_json}")
    print(f"  output:  {output_dir}")
    print("=" * 70 + "\n", flush=True)

    cmd = [
        sys.executable, "-m", "domainbed.scripts.train",
        "--data_dir",   DATA_DIR,
        "--algorithm",  algorithm,
        "--dataset",    DATASET,
        "--test_envs",  str(test_env),
        "--steps",      str(steps),
        "--hparams",    hparams_json,
        "--output_dir", output_dir,
    ]

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"[WARNING] {algorithm} (env{test_env}) exited with code {result.returncode}")

    return output_dir


# ── Result parser ───────────────────────────────────────────────────────
def parse_final_test_acc(output_dir, test_env):
    """Read the LAST row of results from out.txt and return test accuracy."""
    out_file = os.path.join(output_dir, "out.txt")
    if not os.path.exists(out_file):
        return None

    env_key = f"env{test_env}_out_acc"
    last_row = None

    with open(out_file) as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{'env"):
                continue
            try:
                row = json.loads(line.replace("'", '"'))
                last_row = row
            except Exception:
                try:
                    row = eval(line)
                    last_row = row
                except Exception:
                    pass

    if last_row is None:
        return None
    return last_row.get(env_key)


# ── Summary table ────────────────────────────────────────────────────────
def print_summary(results):
    header = f"{'Algorithm':<20} {'Caltech':>8} {'LabelMe':>8} {'SUN09':>8} {'VOC2007':>8} {'Avg':>8}"
    print("\n" + "=" * 70)
    print(f"  VLCS RESULTS SUMMARY  (ResNet18, fair comparison)")
    print("=" * 70)
    print(f"{'Algorithm':<20} {'Caltech':>8} {'LabelMe':>8} {'SUN09':>8} {'VOC2007':>8} {'Avg':>8}")
    print("-" * 70)
    for algo, accs in results.items():
        valid = [a for a in accs if a is not None]
        avg   = sum(valid) / len(valid) if valid else 0.0
        cols  = [f"{a:.4f}" if a is not None else "  N/A  " for a in accs]
        print(f"{algo:<20} {cols[0]:>8} {cols[1]:>8} {cols[2]:>8} {cols[3]:>8} {avg:>8.4f}")
    print("=" * 70)
    print("\nThe 'Avg' column is your VLCS score to report in the paper.")


# ── Main ────────────────────────────────────────────────────────────────
import sys as _sys

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    results_only = "--results-only" in _sys.argv
    dcsam_only   = "--dcsam-only"   in _sys.argv
    any_only     = results_only or dcsam_only

    results = {}

    for cfg in ALGORITHMS:
        algo = cfg["algorithm"]
        tag  = cfg["tag"]
        label = f"DCSAM(v6)" if algo == "DCSAM" else algo

        # Skip baselines if --dcsam-only
        if dcsam_only and algo != "DCSAM":
            # Still read existing results for the table
            accs = []
            for env in TEST_ENVS:
                out_dir = f"./train_output_vlcs_{algo.lower()}_e{env}{tag}"
                acc = parse_final_test_acc(out_dir, env)
                accs.append(acc)
            if any(a is not None for a in accs):
                results[label] = accs
            continue

        accs = []
        for env in TEST_ENVS:
            out_dir = f"./train_output_vlcs_{algo.lower()}_e{env}{tag}"
            if not results_only:
                out_dir = run(algo, cfg["steps"], cfg["hparams"], env, tag)
            acc = parse_final_test_acc(out_dir, env)
            accs.append(acc)
            status = f"{acc:.4f}" if acc is not None else "N/A"
            print(f"  {label} env{env} ({ENV_NAMES[env]}): {status}")
        results[label] = accs

    print_summary(results)

"""
run_fair_comparison.py
----------------------
Runs ERM, ERMPlusPlus, and DCSAM on PACS with the SAME backbone
(ResNet18, no AugMix) across ALL 4 test environments so the comparison
is fair and a proper PACS score can be computed.

Usage (from DomainBed directory, with venv activated):
    python run_fair_comparison.py

Each algorithm is trained 4 times (once per test domain) and results
are written to separate output directories.

PACS domains:
    0 = art_painting
    1 = cartoon
    2 = photo
    3 = sketch

Final PACS score = average of env{0,1,2,3}_out_acc across all 4 runs.
"""

import subprocess
import sys
import json
import os

# ── Configuration ──────────────────────────────────────────────────────
DATASET   = "PACS"
DATA_DIR  = "./data"
N_STEPS   = 5000        # standard for PACS

# Shared backbone: ResNet18 for everyone, no AugMix — fair comparison
SHARED_HPARAMS = {"resnet18": True, "resnet50_augmix": False}

DCSAM_HPARAMS  = {**SHARED_HPARAMS, "rho": 0.05, "lambda_feat": 0.5, "lambda_var": 0.5}

# All 4 PACS test environments
TEST_ENVS = [0, 1, 2, 3]
ENV_NAMES = {0: "art_painting", 1: "cartoon", 2: "photo", 3: "sketch"}

ALGORITHMS = [
    {"algorithm": "ERM",         "steps": N_STEPS, "hparams": SHARED_HPARAMS},
    {"algorithm": "ERMPlusPlus", "steps": N_STEPS, "hparams": SHARED_HPARAMS},
    {"algorithm": "DCSAM",       "steps": N_STEPS, "hparams": DCSAM_HPARAMS},
]

# ── Runner ─────────────────────────────────────────────────────────────
def run(algorithm, steps, hparams, test_env):
    output_dir = f"./train_output_{algorithm.lower()}_pacs_e{test_env}"
    hparams_json = json.dumps(hparams)

    print("\n" + "=" * 70)
    print(f"  {algorithm}  |  test_env={test_env} ({ENV_NAMES[test_env]})  |  {steps} steps")
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
        print(f"\n[WARNING] {algorithm} (env{test_env}) exited with code {result.returncode}")
    return output_dir


def parse_final_test_acc(output_dir, test_env):
    """Read the last line of out.txt and extract env{test_env}_out_acc."""
    out_file = os.path.join(output_dir, "out.txt")
    if not os.path.exists(out_file):
        return None
    with open(out_file) as f:
        lines = [l.strip() for l in f if l.strip()]

    # Find the header line and the last data line
    header, data = None, None
    for i, line in enumerate(lines):
        if "env0_out_acc" in line or "env0_in_acc" in line:
            header = line.split()
            if i + 1 < len(lines):
                data = lines[i + 1].split()
    if header is None or data is None:
        return None

    key = f"env{test_env}_out_acc"
    # Header columns are truncated to 12 chars in DomainBed output
    for col_idx, col in enumerate(header):
        if key.startswith(col[:12]) or col.startswith(key[:12]):
            try:
                return float(data[col_idx])
            except (ValueError, IndexError):
                return None
    return None


# ── Main ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    results = {}  # {algorithm: [acc_e0, acc_e1, acc_e2, acc_e3]}

    for cfg in ALGORITHMS:
        algo = cfg["algorithm"]
        accs = []
        for env in TEST_ENVS:
            out_dir = run(cfg["algorithm"], cfg["steps"], cfg["hparams"], env)
            acc = parse_final_test_acc(out_dir, env)
            accs.append(acc)
            print(f"\n  >>> {algo} env{env} ({ENV_NAMES[env]}): {acc:.4f}" if acc else
                  f"\n  >>> {algo} env{env}: could not parse accuracy")
        results[algo] = accs

    # ── Final summary table ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PACS RESULTS SUMMARY  (ResNet18, fair comparison)")
    print("=" * 70)
    header = f"{'Algorithm':<15} {'Art':>8} {'Cartoon':>8} {'Photo':>8} {'Sketch':>8} {'Avg':>8}"
    print(header)
    print("-" * 70)
    for algo, accs in results.items():
        valid = [a for a in accs if a is not None]
        avg   = sum(valid) / len(valid) if valid else 0.0
        cols  = [f"{a:.4f}" if a is not None else "  N/A  " for a in accs]
        print(f"{algo:<15} {cols[0]:>8} {cols[1]:>8} {cols[2]:>8} {cols[3]:>8} {avg:>8.4f}")
    print("=" * 70)
    print("\nThe 'Avg' column is your PACS score to report in the paper.")

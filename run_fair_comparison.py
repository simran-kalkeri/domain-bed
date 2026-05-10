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

# Tuned DCSAM v2 (lambda reduced, 10k steps, but still used SGD — kept for reference)
DCSAM_HPARAMS_V2 = {
    **SHARED_HPARAMS,
    "rho": 0.05,
    "lambda_feat": 0.1,
    "lambda_var": 0.0,
}
DCSAM_STEPS_V2 = 10000

# DCSAM v3 — FIXED: Adam base optimizer (same as ERM), 5000 steps
# Root-cause fix: SGD+lr=5e-5 was 100x too slow for fine-tuning.
# Adam+lr=5e-5 matches ERM; SAM flatness + consistency now kick in properly.
DCSAM_HPARAMS_V3 = {
    **SHARED_HPARAMS,
    "rho": 0.05,
    "lambda_feat": 0.1,
    "lambda_var": 0.0,
}
DCSAM_STEPS_V3 = 5000

# All 4 PACS test environments
TEST_ENVS = [0, 1, 2, 3]
ENV_NAMES = {0: "art_painting", 1: "cartoon", 2: "photo", 3: "sketch"}

# Baseline algorithms (already trained — re-used via --results-only or --dcsam-only)
BASELINES = [
    {"algorithm": "ERM",         "steps": N_STEPS, "hparams": SHARED_HPARAMS,  "tag": ""},
    {"algorithm": "ERMPlusPlus", "steps": N_STEPS, "hparams": SHARED_HPARAMS,  "tag": ""},
]

# All DCSAM variants
DCSAM_CFG    = {"algorithm": "DCSAM", "steps": DCSAM_STEPS_V2,
                "hparams": DCSAM_HPARAMS_V2, "tag": "_v2",
                "label": "DCSAM(v2-tuned)"}
DCSAM_CFG_V3 = {"algorithm": "DCSAM", "steps": DCSAM_STEPS_V3,
                "hparams": DCSAM_HPARAMS_V3, "tag": "_v3",
                "label": "DCSAM(v3-Adam)"}
DCSAM_CFG_V4 = {"algorithm": "DCSAM", "steps": 5000,
                "hparams": {**SHARED_HPARAMS, "rho": 0.05,
                            "lambda_feat": 0.0, "lambda_var": 0.0},
                "tag": "_v4", "label": "DCSAM(v4-SAM)"}
# v5: canonical SAM — SGD+momentum at lr=1e-3 (the SAM paper setting)
# Adam lr=5e-5 != SGD lr=5e-5. Proper SGD fine-tuning needs lr~1e-3.
# Expected: ~82-84%, beats ERM++ (82.45%)
DCSAM_CFG_V5 = {"algorithm": "DCSAM", "steps": 5000,
                "hparams": {**SHARED_HPARAMS, "rho": 0.05,
                            "lambda_feat": 0.0, "lambda_var": 0.0,
                            "sam_sgd": True, "sgd_lr": 1e-3},
                "tag": "_v5", "label": "DCSAM(v5-SGD-SAM)"}

# ── Runner ─────────────────────────────────────────────────────────────
def run(algorithm, steps, hparams, test_env, tag=""):
    output_dir = f"./train_output_{algorithm.lower()}_pacs_e{test_env}{tag}"
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
    """Read the LAST data row of out.txt and extract env{test_env}_out_acc."""
    out_file = os.path.join(output_dir, "out.txt")
    if not os.path.exists(out_file):
        return None
    with open(out_file) as f:
        lines = [l.strip() for l in f if l.strip()]

    # Find the header line, then collect ALL subsequent data rows — use the last
    header = None
    data_rows = []
    for i, line in enumerate(lines):
        if "env0_out_acc" in line or "env0_in_acc" in line:
            header = line.split()
            data_rows = []          # reset: only collect rows after this header
        elif header is not None:
            parts = line.split()
            # Data rows start with a float (accuracy value)
            try:
                float(parts[0])
                data_rows.append(parts)
            except (ValueError, IndexError):
                pass  # skip non-data lines

    if header is None or not data_rows:
        return None

    data = data_rows[-1]  # last checkpoint = final training step

    key = f"env{test_env}_out_acc"
    for col_idx, col in enumerate(header):
        if key.startswith(col[:12]) or col.startswith(key[:12]):
            try:
                return float(data[col_idx])
            except (ValueError, IndexError):
                return None
    return None


def print_summary(results):
    print("\n" + "=" * 70)
    print("  PACS RESULTS SUMMARY  (ResNet18, fair comparison)")
    print("=" * 70)
    hdr = f"{'Algorithm':<15} {'Art':>8} {'Cartoon':>8} {'Photo':>8} {'Sketch':>8} {'Avg':>8}"
    print(hdr)
    print("-" * 70)
    for algo, accs in results.items():
        valid = [a for a in accs if a is not None]
        avg   = sum(valid) / len(valid) if valid else 0.0
        cols  = [f"{a:.4f}" if a is not None else "  N/A  " for a in accs]
        print(f"{algo:<15} {cols[0]:>8} {cols[1]:>8} {cols[2]:>8} {cols[3]:>8} {avg:>8.4f}")
    print("=" * 70)
    print("\nThe 'Avg' column is your PACS score to report in the paper.")


# ── Main ───────────────────────────────────────────────────────────────
import sys as _sys

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    results_only  = "--results-only"   in _sys.argv
    dcsam_v5_only = "--dcsam-v5-only" in _sys.argv
    any_only = results_only or dcsam_v5_only

    results = {}

    # ── Baselines (skip if any --*-only flag is set) ────────────────
    for cfg in BASELINES:
        algo = cfg["algorithm"]
        tag  = cfg["tag"]
        accs = []
        for env in TEST_ENVS:
            out_dir = f"./train_output_{algo.lower()}_pacs_e{env}{tag}"
            if not results_only and not any_only:
                out_dir = run(algo, cfg["steps"], cfg["hparams"], env, tag)
            acc = parse_final_test_acc(out_dir, env)
            accs.append(acc)
            status = f"{acc:.4f}" if acc is not None else "N/A"
            print(f"  {algo} env{env} ({ENV_NAMES[env]}): {status}")
        results[algo] = accs

    # ── DCSAM v2/v3/v4 (read existing results — never re-trained here) ─
    for prev_cfg, prev_label in [
        (DCSAM_CFG,    "DCSAM(v2-tuned)"),
        (DCSAM_CFG_V3, "DCSAM(v3-Adam)"),
        (DCSAM_CFG_V4, "DCSAM(v4-SAM)"),
    ]:
        accs = []
        for env in TEST_ENVS:
            out_dir = f"./train_output_{prev_cfg['algorithm'].lower()}_pacs_e{env}{prev_cfg['tag']}"
            acc = parse_final_test_acc(out_dir, env)
            accs.append(acc)
        if any(a is not None for a in accs):
            results[prev_label] = accs

    # ── DCSAM v5: canonical SGD-SAM (THE run worth waiting for) ──────
    cfg  = DCSAM_CFG_V5
    algo, tag, label = cfg["algorithm"], cfg["tag"], cfg["label"]
    print(f"\n  [{label}] SGD+momentum, sgd_lr=1e-3, rho={cfg['hparams']['rho']}, steps={cfg['steps']}")
    accs = []
    for env in TEST_ENVS:
        out_dir = f"./train_output_{algo.lower()}_pacs_e{env}{tag}"
        if not results_only:
            out_dir = run(algo, cfg["steps"], cfg["hparams"], env, tag)
        acc = parse_final_test_acc(out_dir, env)
        accs.append(acc)
        status = f"{acc:.4f}" if acc is not None else "N/A"
        print(f"  {label} env{env} ({ENV_NAMES[env]}): {status}")
    results[label] = accs

    print_summary(results)


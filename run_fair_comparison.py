"""
run_fair_comparison.py
----------------------
Runs ERM, ERMPlusPlus, and DCSAM on RotatedMNIST with the SAME backbone
(ResNet18, no AugMix) so the comparison is fair.

Usage (from DomainBed directory, with venv activated):
    python run_fair_comparison.py

Each algorithm trains for N_STEPS steps and writes results to its own
output directory.  Look for env0_out_acc in the final printed row of
each run — that is the test-domain accuracy.
"""

import subprocess
import sys
import json
import os

# ── Configuration ──────────────────────────────────────────────────────
N_STEPS   = 2000      # increase to 3000 for ERMPlusPlus if you have time
TEST_ENV  = 0
DATASET   = "RotatedMNIST"
DATA_DIR  = "./data"

# Shared backbone override: ResNet18 for everyone, no AugMix
SHARED_HPARAMS = {"resnet18": True, "resnet50_augmix": False}

RUNS = [
    {
        "algorithm":  "ERM",
        "steps":      N_STEPS,
        "hparams":    SHARED_HPARAMS,
        "output_dir": "./train_output_erm_rmnist_r18_fair",
    },
    {
        "algorithm":  "ERMPlusPlus",
        "steps":      3000,           # needs extra steps due to linear warmup
        "hparams":    SHARED_HPARAMS,
        "output_dir": "./train_output_ermpp_rmnist_r18_fair",
    },
    {
        "algorithm":  "DCSAM",
        "steps":      N_STEPS,
        "hparams":    {**SHARED_HPARAMS,
                       "rho": 0.05,
                       "lambda_feat": 0.5,
                       "lambda_var":  0.5},
        "output_dir": "./train_output_dcsam_rmnist_r18_fair",
    },
]

# ── Runner ─────────────────────────────────────────────────────────────
def run(cfg):
    hparams_json = json.dumps(cfg["hparams"])
    cmd = [
        sys.executable, "-m", "domainbed.scripts.train",
        "--data_dir",    DATA_DIR,
        "--algorithm",   cfg["algorithm"],
        "--dataset",     DATASET,
        "--test_envs",   str(TEST_ENV),
        "--steps",       str(cfg["steps"]),
        "--hparams",     hparams_json,
        "--output_dir",  cfg["output_dir"],
    ]
    print("\n" + "="*70)
    print(f"  Running: {cfg['algorithm']}  ({cfg['steps']} steps)")
    print(f"  hparams: {hparams_json}")
    print(f"  output:  {cfg['output_dir']}")
    print("="*70 + "\n", flush=True)

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\n[WARNING] {cfg['algorithm']} exited with code {result.returncode}")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    for cfg in RUNS:
        run(cfg)

    print("\n" + "="*70)
    print("All runs complete.")
    print("For each run above, find the LAST printed row and read env0_out_acc.")
    print("That is the test-domain accuracy for comparison.")
    print("="*70)

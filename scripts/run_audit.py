"""
scripts/run_audit.py
--------------------
Command-line entry point for the FairCV audit pipeline.

Usage
-----
    # Run structured-only audit, save results
    python scripts/run_audit.py --csv data/FairCVdb.csv

    # Run full audit including SBERT fusion
    python scripts/run_audit.py --csv data/FairCVdb.csv --sbert

    # Export master comparison table to CSV
    python scripts/run_audit.py --csv data/FairCVdb.csv --export results/

Options
-------
  --csv     PATH   Path to FairCVdb.csv (required).
  --sbert          Enable SBERT fusion experiments (slow on first run).
  --seed    INT    Random seed (default: 42).
  --export  DIR    Directory to write result CSV files.
  --verbose        Print pipeline progress to stdout.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from faircv.pipeline import run_audit


def main():
    parser = argparse.ArgumentParser(
        description="FairCV -- multi-modal fair recruitment audit",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv",     required=True, help="Path to FairCVdb.csv")
    parser.add_argument("--sbert",   action="store_true",
                        help="Enable SBERT fusion experiments")
    parser.add_argument("--seed",    type=int, default=42, help="Random seed")
    parser.add_argument("--export",  default=None,
                        help="Directory to write CSV / JSON results")
    parser.add_argument("--verbose", action="store_true",
                        help="Print progress to stdout")
    args = parser.parse_args()

    print("=" * 65)
    print("  FairCV -- Structured Fair Recruitment Audit")
    print("=" * 65)
    print(f"  CSV  : {args.csv}")
    print(f"  SBERT: {args.sbert}")
    print(f"  Seed : {args.seed}")
    print()

    out = run_audit(
        csv_path=args.csv,
        use_sbert=args.sbert,
        seed=args.seed,
        verbose=True,
    )

    # -- Summary printout --------------------------------------------------
    print("\n" + "=" * 65)
    print("  PERFORMANCE SUMMARY  (Setting A, blind label)")
    print("=" * 65)
    blind_a = out["master_df"][
        (out["master_df"]["Label"] == "blind") &
        (out["master_df"]["Setting"] == "A")
    ][["Model", "F1", "ROC-AUC", "Accuracy",
       "DP_Gap_Gender", "DI_Gender"]].copy() if "DI_Gender" in out["master_df"] else \
    out["master_df"][
        (out["master_df"]["Label"] == "blind") &
        (out["master_df"]["Setting"] == "A")
    ][["Model", "F1", "ROC-AUC", "Accuracy", "DP_Gap_Gender"]]

    print(blind_a.to_string(index=False))

    print("\n" + "=" * 65)
    print("  BEFORE vs. AFTER MITIGATION  (RF)")
    print("=" * 65)
    if "RF" in out["before_after"]:
        print(out["before_after"]["RF"].to_string(index=False))

    # -- Export ------------------------------------------------------------
    if args.export:
        os.makedirs(args.export, exist_ok=True)

        out["master_df"].to_csv(
            os.path.join(args.export, "master_comparison.csv"), index=False)
        print(f"\n  Saved: {args.export}/master_comparison.csv")

        for model_name, ba_df in out["before_after"].items():
            ba_df.to_csv(
                os.path.join(args.export, f"before_after_{model_name}.csv"),
                index=False,
            )
            print(f"  Saved: {args.export}/before_after_{model_name}.csv")

        if out["shap"]:
            for m, sr in out["shap"].items():
                sr["global"].to_csv(
                    os.path.join(args.export, f"shap_global_{m}.csv"), index=False)
                sr["dp_decomp"].to_csv(
                    os.path.join(args.export, f"shap_dp_decomp_{m}.csv"), index=False)
            print(f"  Saved SHAP tables for: {list(out['shap'].keys())}")

        if args.sbert and out["fusion"]:
            fusion_rows = []
            for fkey, fr in out["fusion"].items():
                parts = fkey.split(" | ")
                rec = {"Strategy": parts[0], "Classifier": parts[1] if len(parts)>1 else ""}
                rec.update(fr["perf"])
                rec.update(fr["fairness"])
                fusion_rows.append(rec)
            pd.DataFrame(fusion_rows).to_csv(
                os.path.join(args.export, "fusion_results.csv"), index=False)
            print(f"  Saved: {args.export}/fusion_results.csv")

    print("\n  Audit complete.")


if __name__ == "__main__":
    main()

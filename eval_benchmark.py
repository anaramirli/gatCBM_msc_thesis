# doing end-to-end eval with multibale run: pick best concept number, generate concept basis, build datasets -> train -> aggregate metrics
import os
import json
import time
import shutil
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np
from config import default_eval_dir, DATASETS
import warnings
warnings.filterwarnings("ignore")

datasets = list(DATASETS.keys())

def run(cmd: List[str]):
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def rm_tree(path: Path):
    if path.exists():
        shutil.rmtree(path)


def read_metrics(output_root: str, dataset: str) -> Dict:
    metrics_path = Path(output_root) / dataset / "models" / dataset / "metrics.json"
    with open(metrics_path, "r") as f:
        return json.load(f)


def aggregate(rows: List[Dict]) -> Dict:

    out = {}
    for split in ["train", "val", "test"]:
        for key in ["acc", "f1", "auc"]:
            vals = [r[split][key] for r in rows]
            out[f"{split}_{key}_mean"] = float(np.mean(vals)) if len(vals) else float("nan")
            out[f"{split}_{key}_std"]  = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return out


def main():
    ap = argparse.ArgumentParser("experiments: find best concept Basis -> generate data -> train -> aggregate metrics for multiple runs")
    ap.add_argument("--output-root", type=str, default=default_eval_dir,
                    help="temp dir, will be cleaned).")
    ap.add_argument("--n-runs", type=int, default=5, help="run count per dataset to compute mean/std.")
    ap.add_argument("--device", type=str, default="cuda")
    # concept generation args
    ap.add_argument("--patch-size", type=int, default=70)
    ap.add_argument("--stride-r", type=float, default=0.8)
    ap.add_argument("--candidates", type=int, nargs="*", default=[6, 7, 8, 9, 10])
    ap.add_argument("--batch-size-fit", type=int, default=64, help="craft fitting batch size for pipeline.")
    # training args
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=2e-4)
    args = ap.parse_args()

    
    current_dir = os.getcwd()
    args.output_root = os.path.join(current_dir, args.output_root)
    out_root = Path(args.output_root)

    if os.path.exists(out_root):
        rm_tree(out_root)
    os.makedirs(out_root)

    results_per_ds = {}

    for ds in datasets:
        print(f"\n==================== DATASET: {ds} ====================")
        per_runs: List[Dict] = []

        # clean/override dataset directory before each dataset run
        ds_dir = out_root / ds
        rm_tree(ds_dir)

        for run_idx in range(1, args.n_runs + 1):
            print(f"\n--- {ds} | run {run_idx}/{args.n_runs} ---")

            # build concepts + graphs into the temp root (k-times)
            run([
                "python", "build_concept_graphs.py",
                "--dataset", ds,
                "--steps", "gen_concepts", "build_graphs",
                "--auto-n-components", "--candidates", *list(map(str, args.candidates)),
                "--patch-size", str(args.patch_size),
                "--stride-r", str(args.stride_r),
                "--batch-size", str(args.batch_size_fit),
                "--device", args.device,
                "--output-root", str(out_root),
            ])

            # 2) train (no --save-model)
            run([
                "python", "train_model.py",
                "--dataset", ds,
                "--output-root", str(out_root),
                "--device", args.device,
                "--epochs", str(args.epochs),
                "--patience", str(args.patience),
                "--hidden-dim", str(args.hidden_dim) if hasattr(args, "hidden-dim") else str(args.hidden_dim),
                "--lr", str(args.lr),
                "--weight-decay", str(args.weight_decay),
            ])

            # 3) read metrics.json written by gat_trainer
            m = read_metrics(str(out_root), ds)
            per_runs.append(m)

            time.sleep(1.0)

        # aggregate mean/std for the give dataset
        agg = aggregate(per_runs)
        results_per_ds[ds] = agg

        # after finishing all runs for the given dataset, clean that dataset folder
        rm_tree(ds_dir)

    # build DataFrame and save
    df = pd.DataFrame.from_dict(results_per_ds, orient="index")
    df.index.name = "dataset"
    df = df.sort_index()

    results_csv = out_root / "results.csv"
    results_json = out_root / "results.json"

    df.to_csv(results_csv)
    with open(results_json, "w") as f:
        json.dump(results_per_ds, f, indent=2)

    print("\n=== SUMMARY ===")
    print(df)

    # final cleanup: delete everything under output-root EXCEPT the results files
    for child in out_root.iterdir():
        if child.name not in {"results.csv", "results.json"}:
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                try:
                    child.unlink()
                except FileNotFoundError:
                    pass

    print(f"\nSaved summary to:\n  {results_csv}\n  {results_json}")
    print("Scratch artifacts cleaned.")

    

if __name__ == "__main__":
    main()
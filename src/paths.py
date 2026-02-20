import os
import json
import time

RUNS_DIR = os.environ.get("RUNS_DIR", "runs")


def new_run(cfg: dict) -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_id = f"{cfg.get('run_name', 'run')}-{ts}"
    run_dir = os.path.join(RUNS_DIR, run_id)
    for sub in [
        "artifacts/lora",
        "artifacts/merged",
        "artifacts/gguf",
        "artifacts/checkpoints",
        "artifacts/logs",
        "uploads",
    ]:
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)
    return run_dir


def save_config(run_dir: str, cfg: dict):
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)


def save_manifest(run_dir: str, manifest: dict):
    with open(os.path.join(run_dir, "dataset_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


def save_metrics(run_dir: str, metrics: dict):
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


def save_export_manifest(run_dir: str, manifest: dict):
    with open(os.path.join(run_dir, "export_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


def list_runs():
    if not os.path.exists(RUNS_DIR):
        return []
    runs = []
    for name in sorted(os.listdir(RUNS_DIR), reverse=True):
        cfg_path = os.path.join(RUNS_DIR, name, "config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                runs.append({"run_id": name, **json.load(f)})
    return runs

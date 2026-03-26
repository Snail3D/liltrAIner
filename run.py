#!/usr/bin/env python3
"""
liltrAIner — Automated experiment loop.
No AI agent needed — runs hyperparameter search with structured logging.
For the full autoresearch experience, use an AI agent with program.md instead.
"""

import json
import random
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).parent
LOG = BASE / "experiment_log.jsonl"
BEST = BASE / "best_config.json"
STATUS = BASE / "status.json"

CONFIGS = [
    {"num_layers": 2, "lr": 1e-5, "iters": 50},
    {"num_layers": 2, "lr": 2e-5, "iters": 50},
    {"num_layers": 2, "lr": 5e-5, "iters": 50},
    {"num_layers": 4, "lr": 1e-5, "iters": 50},
    {"num_layers": 4, "lr": 2e-5, "iters": 50},
    {"num_layers": 4, "lr": 5e-5, "iters": 50},
    {"num_layers": 8, "lr": 1e-5, "iters": 50},
    {"num_layers": 8, "lr": 2e-5, "iters": 50},
    {"num_layers": 2, "lr": 2e-5, "iters": 100},
    {"num_layers": 4, "lr": 2e-5, "iters": 100},
    {"num_layers": 4, "lr": 1e-4, "iters": 50},
    {"num_layers": 2, "lr": 1e-4, "iters": 50},
    {"num_layers": 2, "lr": 5e-6, "iters": 100},
    {"num_layers": 4, "lr": 5e-6, "iters": 100},
    {"num_layers": 8, "lr": 5e-5, "iters": 50},
    {"num_layers": 2, "lr": 3e-5, "iters": 75},
    {"num_layers": 4, "lr": 3e-5, "iters": 75},
    {"num_layers": 8, "lr": 2e-5, "iters": 100},
    {"num_layers": 4, "lr": 2e-5, "iters": 150},
    {"num_layers": 2, "lr": 2e-5, "iters": 150},
]


def write_status(phase, exp_id, config, total):
    STATUS.write_text(json.dumps({
        "phase": phase,
        "current_experiment": exp_id,
        "current_config": config,
        "progress": exp_id / total * 100,
        "total_experiments": total,
        "timestamp": datetime.now().isoformat(),
    }))


def get_best():
    try:
        return json.loads(BEST.read_text()).get("score", 0)
    except Exception:
        return 0


def run_one(exp_id, config, total):
    write_status("training", exp_id, config, total)

    # Train
    t0 = time.time()
    train = subprocess.run([
        sys.executable, str(BASE / "train.py"),
        "--num-layers", str(config["num_layers"]),
        "--lr", str(config["lr"]),
        "--iters", str(config["iters"]),
        "--time-budget", "50",
    ], capture_output=True, text=True, timeout=120)
    train_time = time.time() - t0

    loss = None
    for line in (train.stdout + train.stderr).split("\n"):
        m = re.search(r'loss[:\s]+([0-9.]+)', line, re.IGNORECASE)
        if m:
            try:
                loss = float(m.group(1))
            except ValueError:
                pass

    if train.returncode != 0:
        return None

    # Eval
    write_status("evaluating", exp_id, config, total)
    ev = subprocess.run([sys.executable, str(BASE / "eval.py")],
                        capture_output=True, text=True, timeout=300)

    score = 0
    for line in ev.stdout.split("\n"):
        if "TOTAL SCORE" in line:
            m = re.search(r'([0-9.]+)%', line)
            if m:
                score = float(m.group(1)) / 100

    return {"score": score, "loss": loss, "time": time.time() - t0}


def main(num_experiments=20):
    random.shuffle(CONFIGS)
    configs = (CONFIGS * ((num_experiments // len(CONFIGS)) + 1))[:num_experiments]
    best_score = get_best()

    print(f"\n  liltrAIner — Automated Loop")
    print(f"  Experiments: {num_experiments}")
    print(f"  Best so far: {best_score:.2%}")
    print()

    for i, config in enumerate(configs, 1):
        print(f"  [{i}/{num_experiments}] layers={config['num_layers']} lr={config['lr']} iters={config['iters']}")

        try:
            result = run_one(i, config, num_experiments)
        except Exception as e:
            print(f"    ERROR: {e}")
            result = None

        entry = {
            "id": i,
            "timestamp": datetime.now().isoformat(),
            "success": result is not None,
            "config": config,
            "score": result["score"] if result else 0,
            "train_loss": result.get("loss") if result else None,
            "total_time": result.get("time") if result else 0,
        }
        with open(LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")

        if result and result["score"] > best_score:
            best_score = result["score"]
            BEST.write_text(json.dumps({"score": best_score, "config": config}, indent=2))
            print(f"    NEW BEST: {best_score:.2%}")
        elif result:
            print(f"    Score: {result['score']:.2%} (best: {best_score:.2%})")
        else:
            print(f"    Failed")

    write_status("done", num_experiments, {"best": best_score}, num_experiments)
    print(f"\n  Done! Best: {best_score:.2%}")
    print(f"  Fuse: python train.py --fuse\n")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--experiments", type=int, default=20)
    main(p.parse_args().experiments)

#!/usr/bin/env python3
"""
liltrAIner — LoRA fine-tuning for local LLMs on Apple Silicon.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

MODEL = os.environ.get("MODEL", "mlx-community/Qwen3.5-9B-MLX-4bit")
DATA_DIR = Path(os.environ.get("DATA_DIR", str(Path(__file__).parent / "data")))
ADAPTERS_DIR = Path(os.environ.get("ADAPTER_PATH", str(Path(__file__).parent / "adapters")))
FUSED_DIR = Path(__file__).parent / "fused_model"
CONFIG_FILE = Path(__file__).parent / "config.yaml"


def train(iters=50, batch_size=1, num_layers=4, learning_rate=2e-5, seed=42, time_budget=50):
    """Run LoRA fine-tuning with a fixed time budget."""
    ADAPTERS_DIR.mkdir(exist_ok=True)

    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", MODEL,
        "--data", str(DATA_DIR),
        "--train",
        "--batch-size", str(batch_size),
        "--num-layers", str(num_layers),
        "--learning-rate", str(learning_rate),
        "--iters", str(iters),
        "--seed", str(seed),
        "--adapter-path", str(ADAPTERS_DIR),
        "-c", str(CONFIG_FILE),
        "--grad-checkpoint",
        "--max-seq-length", "256",
        "--val-batches", "2",
        "--save-every", str(max(10, iters // 5)),
        "--steps-per-eval", str(max(25, iters // 2)),
        "--steps-per-report", "5",
    ]

    print(f"\n  liltrAIner — Training")
    print(f"  Model:   {MODEL}")
    print(f"  Iters:   {iters}  |  LR: {learning_rate}  |  Layers: {num_layers}")
    print(f"  Budget:  {time_budget}s")
    print()

    start = time.time()
    try:
        result = subprocess.run(cmd, timeout=time_budget, text=True, capture_output=True)
        elapsed = time.time() - start
        # Parse training loss from output
        loss = None
        for line in result.stdout.split("\n"):
            if "Train loss" in line or "train_loss" in line.lower():
                try:
                    loss = float(line.split(":")[-1].strip().split()[0].rstrip(","))
                except (ValueError, IndexError):
                    pass
        print(f"  Done in {elapsed:.0f}s" + (f" (loss: {loss:.4f})" if loss else ""))
        return {"success": result.returncode == 0, "loss": loss, "time": elapsed}
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        print(f"  Time budget reached ({elapsed:.0f}s)")
        return {"success": True, "loss": None, "time": elapsed}


def fuse():
    """Merge LoRA adapters into the base model."""
    if not list(ADAPTERS_DIR.glob("*.safetensors")):
        print("No adapters found. Run training first.")
        return False

    cmd = [
        sys.executable, "-m", "mlx_lm", "fuse",
        "--model", MODEL,
        "--adapter-path", str(ADAPTERS_DIR),
        "--save-path", str(FUSED_DIR),
    ]

    print(f"\n  Fusing adapters into {FUSED_DIR}")
    result = subprocess.run(cmd, text=True)
    if result.returncode == 0:
        print(f"  Done! Use: MODEL={FUSED_DIR} python your_server.py")
    return result.returncode == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="liltrAIner — LoRA fine-tuning")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--time-budget", type=int, default=50)
    parser.add_argument("--fuse", action="store_true")
    args = parser.parse_args()

    if args.fuse:
        fuse()
    else:
        train(
            iters=args.iters,
            batch_size=args.batch_size,
            num_layers=args.num_layers,
            learning_rate=args.lr,
            time_budget=args.time_budget,
        )

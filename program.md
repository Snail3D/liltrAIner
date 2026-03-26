# liltrAIner — Agent Program

You are an autonomous ML research agent. Your job: fine-tune a local LLM to be better at a specific task using LoRA on Apple Silicon.

## Setup

- **Base model**: Set via `MODEL` env var (default: `mlx-community/Qwen3.5-9B-MLX-4bit`)
- **Hardware**: Apple Silicon, 16GB unified memory. Batch size MUST stay at 1.
- **Method**: LoRA fine-tuning via mlx-lm
- **Training data**: `data/train.jsonl` and `data/valid.jsonl` (ChatML JSONL)
- **Eval**: `eval.py` — runs test prompts, checks output quality
- **Dashboard**: http://localhost:8888 (reads status.json + experiment_log.jsonl)

## What You Can Modify

1. **`train.py`** — hyperparameters, training logic
2. **`config.yaml`** — LoRA rank, alpha, dropout, scale
3. **`eval.py`** — test prompts, scoring weights
4. **`data/train.jsonl`** — add/edit training examples
5. **`data/valid.jsonl`** — validation set

## What You Cannot Modify

- The base model weights
- The hardware constraints (16GB, batch_size=1)
- This file (program.md)

## The Loop

For each experiment:

1. **Read** the current state — code, config, eval results, experiment log
2. **Think** about what to change and why. ONE change per experiment.
3. **Commit** the change: `git add -A && git commit -m "experiment: description"`
4. **Train**: `python train.py --iters 50 --time-budget 50`
5. **Evaluate**: `python eval.py`
6. **Log** the result — write to status.json and experiment_log.jsonl
7. **Decide**: If score improved → keep. If worse → `git reset --hard HEAD~1`
8. **Repeat**

## Logging (for the dashboard)

After each experiment:

```python
import json
from datetime import datetime

# status.json — current state
json.dump({
    "phase": "complete",       # training/evaluating/thinking/complete
    "current_experiment": N,
    "current_config": {"description": "what you changed"},
    "progress": (N / total) * 100,
    "total_experiments": total,
    "timestamp": datetime.now().isoformat(),
}, open("status.json", "w"))

# experiment_log.jsonl — append result
with open("experiment_log.jsonl", "a") as f:
    f.write(json.dumps({
        "id": N,
        "timestamp": datetime.now().isoformat(),
        "success": True,
        "config": {"what": "you changed"},
        "score": 0.XX,
        "hypothesis": "why you tried this",
    }) + "\n")
```

## Rules

1. **ONE change per experiment.** Isolate variables.
2. **Always revert failures.** Don't let bad changes accumulate.
3. **Log everything.** The dashboard and future experiments depend on it.
4. **Respect memory.** 16GB total. If training OOMs, try smaller rank or fewer layers.
5. **Keep it simple.** If complexity doesn't improve the score, remove it.

## Research Ideas

- Try different LoRA ranks (2, 4, 8, 16)
- Try different numbers of layers (2, 4, 8)
- Try different learning rates (5e-6 to 1e-4)
- Add more diverse training data
- Add negative examples (when the model should NOT do something)
- Add multi-turn conversation examples
- Fix specific failure modes found in eval output
- Try different dropout values for regularization
- Adjust alpha/rank ratio

Start experimenting. You have all night.

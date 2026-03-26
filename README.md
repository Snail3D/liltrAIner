# liltrAIner

**Karpathy-style autoresearch for fine-tuning local LLMs on Apple Silicon.**

Point an AI agent (Claude Code, Kimi, Codex) at your training setup and let it experiment overnight. Wake up to a better model.

---

## What It Does

liltrAIner is an autonomous fine-tuning loop for small language models running on Apple Silicon via MLX. It combines:

1. **LoRA fine-tuning** via mlx-lm (memory-efficient, runs on 16GB)
2. **Autoresearch loop** — AI agent modifies training config, trains, evaluates, keeps/reverts, repeats
3. **Real-time dashboard** — watch experiments, scores, and agent reasoning live in your browser
4. **Eval suite** — configurable test prompts with structured scoring

Think of it as [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) but for fine-tuning existing models instead of pretraining, and optimized for Apple Silicon.

## Quick Start

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/liltrAIner.git
cd liltrAIner

# Install
pip install mlx-lm

# Add your training data
# (see data/train.jsonl.example for format)

# Option A: AI agent (recommended)
# Open a terminal and run your preferred agent:
claude --dangerously-skip-permissions
# or
kimi --yolo -w .
# Then: "Read program.md and start experimenting"

# Option B: Automated loop (no agent needed)
python run.py --experiments 50

# Watch progress
python dashboard.py
# Open http://localhost:8888
```

## Requirements

| Component | Minimum |
|-----------|---------|
| Mac | Apple Silicon (M1/M2/M3/M4) |
| RAM | 16GB unified |
| Python | 3.10+ |
| mlx-lm | 0.30+ |

## How It Works

### The Autoresearch Loop

```
┌─────────────────────────────────────────┐
│  AI Agent reads program.md              │
│  ↓                                      │
│  Makes ONE change (data, config, code)  │
│  ↓                                      │
│  Commits the change                     │
│  ↓                                      │
│  Trains (LoRA, 50 iters, ~50 seconds)   │
│  ↓                                      │
│  Evaluates (test prompts → score)       │
│  ↓                                      │
│  Score improved? Keep. Worse? Revert.   │
│  ↓                                      │
│  Repeat forever                         │
└─────────────────────────────────────────┘
```

### The Dashboard

Real-time web UI at `localhost:8888` showing:
- Live agent feed (reasoning, tool calls, decisions)
- Score progression chart
- Experiment log with configs
- Git commit history
- Current phase indicator (training/evaluating/thinking)

### Training Data Format

ChatML JSONL — one conversation per line:

```json
{"messages": [
  {"role": "system", "content": "Your system prompt"},
  {"role": "user", "content": "User message"},
  {"role": "assistant", "content": "Expected response with ```tc-action blocks``` etc"}
]}
```

## File Structure

```
liltrAIner/
  program.md        — Instructions for the AI agent
  train.py          — LoRA fine-tuning script
  eval.py           — Evaluation suite
  run.py            — Automated experiment loop (no agent needed)
  dashboard.py      — Real-time web dashboard
  config.yaml       — LoRA configuration
  data/
    train.jsonl     — Training examples
    valid.jsonl     — Validation examples
```

## Configuration

### config.yaml (LoRA settings)

```yaml
fine_tune_type: lora
lora_parameters:
  rank: 4          # adapter capacity (4-16)
  alpha: 8         # scaling factor
  dropout: 0.0     # regularization
  scale: 2.0       # adapter contribution
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | `mlx-community/Qwen3.5-9B-MLX-4bit` | Base model |
| `ADAPTER_PATH` | `./adapters` | Where to save adapters |
| `DATA_DIR` | `./data` | Training data directory |
| `TIME_BUDGET` | `50` | Seconds per training run |
| `ITERS` | `50` | Training iterations per experiment |

## The AI Agent Approach

The real power is letting an AI agent run the loop. The agent:

- **Reads code and results** to understand what's happening
- **Makes creative decisions** about what to change (not just random hyperparameters)
- **Adapts strategy** based on what worked and what didn't
- **Modifies training data** to fix specific failure modes
- **Adjusts architecture** (LoRA rank, layers, learning rate schedules)

Supported agents:
- **Claude Code**: `claude --dangerously-skip-permissions` then "Read program.md"
- **Kimi K2.5**: `kimi --yolo -w .` then "Read program.md"
- **Any agent** that can read files, run shell commands, and edit code

## Sharing Your Model

```bash
# Fuse adapters into base model
python train.py --fuse

# Upload to HuggingFace
huggingface-cli upload yourname/model-name ./fused_model

# Convert to GGUF (universal format)
python -m mlx_lm convert --model ./fused_model --to-gguf -q Q4_K_M
```

## Credits

Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch). Built for Apple Silicon with [MLX](https://github.com/ml-explore/mlx).

## License

MIT

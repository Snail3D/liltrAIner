#!/usr/bin/env python3
"""
liltrAIner — Evaluation suite.
Configurable test prompts with structured scoring.
"""

import json
import os
import re
import sys
from pathlib import Path

# ── Configure your eval prompts here ──
# Each prompt has:
#   user: what to ask the model
#   expect_action: should it output a structured action block?
#   expect_type: what type of action? (optional)
#   expect_contains: text that should appear in output (optional)

EVAL_PROMPTS = [
    # Actions — should produce structured blocks
    {"user": "Make it dark mode", "expect_action": True, "expect_type": "config"},
    {"user": "Change accent to red", "expect_action": True, "expect_type": "config"},
    {"user": "Change your name to Atlas", "expect_action": True, "expect_type": "config"},
    {"user": "Open settings", "expect_action": True, "expect_type": "navigate"},
    {"user": "Research AI trends", "expect_action": True, "expect_type": "research"},
    {"user": "make a calculator", "expect_action": True, "expect_type": "app"},
    {"user": "build a snake game", "expect_action": True, "expect_type": "app"},
    {"user": "coin flipper", "expect_action": True, "expect_type": "app"},
    {"user": "show my apps", "expect_action": True, "expect_type": "app"},
    {"user": "set an alarm for 7am", "expect_action": True, "expect_type": "alarm"},
    # Chat — should NOT produce action blocks
    {"user": "Hello!", "expect_action": False},
    {"user": "What can you do?", "expect_action": False},
    {"user": "Tell me a joke", "expect_action": False},
    {"user": "How does machine learning work?", "expect_action": False},
]

SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT",
    "You are a helpful AI assistant running on-device via MLX. "
    "Use ```tc-action JSON blocks to control the app."
)

# Action block pattern — customize for your format
ACTION_PATTERN = r"```tc-action\s*\n?(.*?)\n?```"


def extract_actions(text):
    """Extract structured action blocks from model output."""
    matches = re.findall(ACTION_PATTERN, text, re.DOTALL)
    actions = []
    for m in matches:
        try:
            actions.append(json.loads(m.strip()))
        except json.JSONDecodeError:
            actions.append(None)
    return actions


def evaluate(model_path, adapter_path=None, max_tokens=200):
    """Run evaluation suite."""
    from mlx_lm import load, generate as mlx_generate

    print(f"\n  liltrAIner — Evaluation")
    print(f"  Model:    {model_path}")
    if adapter_path:
        print(f"  Adapters: {adapter_path}")
    print(f"  Prompts:  {len(EVAL_PROMPTS)}")
    print()

    model, tokenizer = load(model_path, adapter_path=adapter_path)

    results = {
        "total": len(EVAL_PROMPTS),
        "action_expected": 0, "action_produced": 0,
        "action_parsed": 0, "action_correct_type": 0,
        "chat_expected": 0, "chat_clean": 0,
    }

    for test in EVAL_PROMPTS:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": test["user"]},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )

        text = mlx_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens)
        text = text if isinstance(text, str) else str(text)

        actions = extract_actions(text)
        has_action = len(actions) > 0
        parsed_ok = all(a is not None for a in actions) if actions else False

        if test["expect_action"]:
            results["action_expected"] += 1
            if has_action:
                results["action_produced"] += 1
            if parsed_ok:
                results["action_parsed"] += 1
            if parsed_ok and actions and actions[0].get("type") == test.get("expect_type"):
                results["action_correct_type"] += 1
            status = "PASS" if parsed_ok else ("PARTIAL" if has_action else "FAIL")
        else:
            results["chat_expected"] += 1
            if not has_action:
                results["chat_clean"] += 1
            status = "PASS" if not has_action else "FAIL"

        icon = {"PASS": "+", "PARTIAL": "~", "FAIL": "x"}[status]
        print(f"  [{icon}] {test['user'][:45]}")
        if test.get("expect_action") and parsed_ok and actions:
            got = actions[0].get("type", "?")
            want = test.get("expect_type", "?")
            if got != want:
                print(f"      type: got '{got}', want '{want}'")
        if not has_action and test.get("expect_action"):
            print(f"      no action block in output")
        if has_action and not test.get("expect_action"):
            print(f"      LEAK: action block in chat response")

    # Score
    ae = results["action_expected"] or 1
    ce = results["chat_expected"] or 1
    results["action_rate"] = results["action_produced"] / ae
    results["parse_rate"] = results["action_parsed"] / ae
    results["type_accuracy"] = results["action_correct_type"] / ae
    results["chat_clean_rate"] = results["chat_clean"] / ce
    results["total_score"] = (
        results["parse_rate"] * 0.4
        + results["type_accuracy"] * 0.3
        + results["chat_clean_rate"] * 0.3
    )

    print(f"\n  {'='*44}")
    print(f"  Action produced:  {results['action_produced']}/{results['action_expected']} ({results['action_rate']:.0%})")
    print(f"  JSON parsed:      {results['action_parsed']}/{results['action_expected']} ({results['parse_rate']:.0%})")
    print(f"  Correct type:     {results['action_correct_type']}/{results['action_expected']} ({results['type_accuracy']:.0%})")
    print(f"  Chat clean:       {results['chat_clean']}/{results['chat_expected']} ({results['chat_clean_rate']:.0%})")
    print(f"  TOTAL SCORE:      {results['total_score']:.2%}")
    print(f"  {'='*44}\n")

    return results


if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("MODEL", "mlx-community/Qwen3.5-9B-MLX-4bit")
    adapter_path = str(Path(__file__).parent / "adapters")
    if not Path(adapter_path).exists() or not list(Path(adapter_path).glob("*.safetensors")):
        adapter_path = None
        print("  No adapters found — evaluating base model")
    evaluate(model_path, adapter_path=adapter_path)

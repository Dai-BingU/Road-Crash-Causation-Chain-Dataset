#!/usr/bin/env python3
"""Run DREAM causation-chain inference with a base model and optional LoRA adapter.

Input dataset format:
- JSON array with `id`, `vehicle_count`, and `messages`
- the user prompt is read from the first user message

Output format:
[
  {
    "id": "case_003201",
    "vehicle_count": 2,
    "prediction": "Vehicle 1 (V1)\\n..."
  }
]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except ImportError:  # pragma: no cover
    PeftModel = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DREAM causation-chain inference.")
    parser.add_argument("--base-model", required=True, help="Base model path or Hugging Face model id.")
    parser.add_argument("--dataset", required=True, help="Input JSON dataset, e.g., data/val.json.")
    parser.add_argument("--output", required=True, help="Output prediction JSON path.")
    parser.add_argument("--adapter", help="Optional LoRA adapter path.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="0 means run all remaining samples.")
    return parser.parse_args()


def get_message(record: dict, role: str) -> str:
    for message in record["messages"]:
        if message.get("role") == role:
            return str(message.get("content", ""))
    raise KeyError(f"Role {role!r} not found for sample {record.get('id')!r}")


def load_model(base_model: str, adapter: str | None):
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if adapter:
        if PeftModel is None:
            raise RuntimeError("peft is required when --adapter is provided.")
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tokenizer, model


def generate(tokenizer, model, messages: list[dict], max_new_tokens: int) -> str:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
        )
    generated = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main() -> None:
    args = parse_args()
    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    start = max(args.start, 0)
    end = len(dataset) if args.limit <= 0 else min(len(dataset), start + args.limit)
    subset = dataset[start:end]

    tokenizer, model = load_model(args.base_model, args.adapter)

    results = []
    for i, record in enumerate(subset, start=1):
        messages = [
            {"role": "system", "content": get_message(record, "system")},
            {"role": "user", "content": get_message(record, "user")},
        ]
        prediction = generate(tokenizer, model, messages, args.max_new_tokens)
        results.append(
            {
                "id": str(record["id"]),
                "vehicle_count": record.get("vehicle_count"),
                "prediction": prediction,
            }
        )
        print(f"[{i}/{len(subset)}] {record['id']}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()

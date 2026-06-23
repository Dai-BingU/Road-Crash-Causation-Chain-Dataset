#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path


DEFAULT_SYSTEM_PROMPT = (
    "You are a DREAM causation annotation assistant. "
    "Given an NHTSA crash case input, output only the final standardized "
    "DREAM causation chains. Use the approved code set and exact labels. "
    "Do not explain your reasoning. Preserve the final line-break format."
)


DEFAULT_USER_TEMPLATE = (
    "Please convert the following NHTSA case into the final DREAM causation "
    "chain output.\n\n"
    "[INPUT]\n"
    "{input_text}"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Qwen-style SFT dataset from paired crash-report and causation-chain text files."
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--train-out", required=True, type=Path)
    parser.add_argument("--val-out", type=Path)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument(
        "--val-max-vehicles",
        type=int,
        default=None,
        help="If set, validation examples must have vehicle count <= this value.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--user-template", default=DEFAULT_USER_TEMPLATE)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Require exact filename parity between input-dir and output-dir.",
    )
    return parser.parse_args()


def load_pairs(input_dir: Path, output_dir: Path, strict: bool):
    input_files = {p.name: p for p in sorted(input_dir.glob("*.txt"))}
    output_files = {p.name: p for p in sorted(output_dir.glob("*.txt"))}

    common = sorted(set(input_files) & set(output_files))
    missing_outputs = sorted(set(input_files) - set(output_files))
    missing_inputs = sorted(set(output_files) - set(input_files))

    if strict and missing_outputs:
        raise SystemExit(
            f"Missing {len(missing_outputs)} outputs in {output_dir}."
            f" First few: {missing_outputs[:10]}"
        )
    if strict and missing_inputs:
        raise SystemExit(
            f"Missing {len(missing_inputs)} inputs in {input_dir}."
            f" First few: {missing_inputs[:10]}"
        )

    pairs = []
    for name in common:
        input_text = input_files[name].read_text(encoding="utf-8").strip()
        output_text = output_files[name].read_text(encoding="utf-8").strip()
        if not input_text or not output_text:
            raise SystemExit(f"Empty file detected for pair: {name}")
        pairs.append((name, input_text, output_text))
    return pairs, missing_outputs, missing_inputs


def build_record(name: str, input_text: str, output_text: str, system_prompt: str, user_template: str):
    vehicle_count = sum(1 for line in output_text.splitlines() if line.startswith("Vehicle "))
    return {
        "id": name.removesuffix(".txt"),
        "vehicle_count": vehicle_count,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_template.format(input_text=input_text)},
            {"role": "assistant", "content": output_text},
        ],
    }


def write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    pairs, missing_outputs, missing_inputs = load_pairs(args.input_dir, args.output_dir, args.strict)

    records = [
        build_record(name, input_text, output_text, args.system_prompt, args.user_template)
        for name, input_text, output_text in pairs
    ]

    rng = random.Random(args.seed)
    rng.shuffle(records)

    if args.val_out and args.val_ratio > 0:
        val_size = max(1, int(len(records) * args.val_ratio))
        if args.val_max_vehicles is None:
            val_records = records[:val_size]
            train_records = records[val_size:]
        else:
            eligible = [r for r in records if r["vehicle_count"] <= args.val_max_vehicles]
            ineligible = [r for r in records if r["vehicle_count"] > args.val_max_vehicles]
            if len(eligible) < val_size:
                raise SystemExit(
                    f"Not enough validation-eligible records: need {val_size}, have {len(eligible)}"
                )
            val_records = eligible[:val_size]
            train_records = eligible[val_size:] + ineligible
        write_jsonl(args.val_out, val_records)
    else:
        val_records = []
        train_records = records

    write_jsonl(args.train_out, train_records)

    print(f"total_records={len(records)}")
    print(f"train_records={len(train_records)}")
    print(f"val_records={len(val_records)}")
    if args.val_max_vehicles is not None:
        print(f"val_max_vehicles={args.val_max_vehicles}")
    print(f"missing_outputs={len(missing_outputs)}")
    print(f"missing_inputs={len(missing_inputs)}")
    print(f"train_out={args.train_out}")
    if args.val_out:
        print(f"val_out={args.val_out}")


if __name__ == "__main__":
    main()

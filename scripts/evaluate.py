#!/usr/bin/env python3
"""Evaluate DREAM validation predictions.

Metrics:
- vehicle_node_precision
- vehicle_node_recall
- vehicle_node_f1
- rouge1
- rouge2
- rougeL

Reference dataset format:
- JSON array built for SFT, each sample containing `id` and `messages`
- the ground-truth target is taken from the last assistant message

Prediction file format:
- JSON array or JSONL
- each record must contain `id`
- prediction text can be stored in one of:
  `prediction`, `pred`, `output`, `response`, `assistant`, `text`
- alternatively, an OpenAI-style `messages` field is accepted and the last
  assistant message is used as prediction text
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


VEHICLE_RE = re.compile(r"^\s*Vehicle\s+(\d+)\s+\(V(\d+)\)\s*$", re.IGNORECASE)
NODE_RE = re.compile(
    r"^\s*(Phenotype|Genotype)\s*:\s*([A-Za-z0-9.]+)\s*-\s*(.+?)\s*$",
    re.IGNORECASE,
)
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass(frozen=True, order=True)
class Node:
    vehicle_id: str
    node_type: str
    code: str
    label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DREAM validation outputs.")
    parser.add_argument(
        "--reference",
        required=True,
        help="Path to validation dataset JSON built for SFT.",
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to prediction JSON/JSONL file with ids and generated text.",
    )
    parser.add_argument(
        "--save-details",
        help="Optional path to save per-sample metrics as JSON.",
    )
    return parser.parse_args()


def load_json_or_jsonl(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def get_last_assistant_message(messages: Iterable[dict]) -> str:
    last = None
    for message in messages:
        if message.get("role") == "assistant":
            last = message.get("content", "")
    if last is None:
        raise ValueError("No assistant message found.")
    return str(last)


def extract_prediction_text(record: dict) -> str:
    for key in ("prediction", "pred", "output", "response", "assistant", "text"):
        value = record.get(key)
        if isinstance(value, str):
            return value
    messages = record.get("messages")
    if isinstance(messages, list):
        return get_last_assistant_message(messages)
    raise ValueError(f"Cannot find prediction text in record with id={record.get('id')!r}")


def normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def parse_nodes(text: str) -> set[Node]:
    current_vehicle = None
    nodes: set[Node] = set()
    for raw_line in normalize_text(text).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        vehicle_match = VEHICLE_RE.match(line)
        if vehicle_match:
            current_vehicle = f"V{vehicle_match.group(2)}"
            continue
        node_match = NODE_RE.match(line)
        if node_match and current_vehicle is not None:
            node_type = node_match.group(1).capitalize()
            code = node_match.group(2).strip()
            label = node_match.group(3).strip()
            nodes.add(Node(current_vehicle, node_type, code, label))
    return nodes


def ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def rouge_n_f1(reference: str, prediction: str, n: int) -> float:
    ref_tokens = TOKEN_RE.findall(normalize_text(reference))
    pred_tokens = TOKEN_RE.findall(normalize_text(prediction))
    ref_counts = ngrams(ref_tokens, n)
    pred_counts = ngrams(pred_tokens, n)
    if not ref_counts and not pred_counts:
        return 1.0
    if not ref_counts or not pred_counts:
        return 0.0
    overlap = sum((ref_counts & pred_counts).values())
    ref_total = sum(ref_counts.values())
    pred_total = sum(pred_counts.values())
    if overlap == 0:
        return 0.0
    precision = overlap / pred_total
    recall = overlap / ref_total
    return 2 * precision * recall / (precision + recall)


def lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0]
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(prev[j], curr[-1]))
        prev = curr
    return prev[-1]


def rouge_l_f1(reference: str, prediction: str) -> float:
    ref_tokens = TOKEN_RE.findall(normalize_text(reference))
    pred_tokens = TOKEN_RE.findall(normalize_text(prediction))
    if not ref_tokens and not pred_tokens:
        return 1.0
    if not ref_tokens or not pred_tokens:
        return 0.0
    lcs = lcs_length(ref_tokens, pred_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def main() -> None:
    args = parse_args()
    reference_path = Path(args.reference)
    prediction_path = Path(args.predictions)

    references = load_json_or_jsonl(reference_path)
    predictions = load_json_or_jsonl(prediction_path)

    reference_by_id: dict[str, str] = {}
    for record in references:
        sample_id = str(record["id"])
        reference_by_id[sample_id] = normalize_text(get_last_assistant_message(record["messages"]))

    prediction_by_id: dict[str, str] = {}
    for record in predictions:
        sample_id = str(record["id"])
        prediction_by_id[sample_id] = normalize_text(extract_prediction_text(record))

    missing_predictions = sorted(set(reference_by_id) - set(prediction_by_id))
    extra_predictions = sorted(set(prediction_by_id) - set(reference_by_id))
    common_ids = [sample_id for sample_id in reference_by_id if sample_id in prediction_by_id]

    total_tp = 0
    total_fp = 0
    total_fn = 0
    rouge1_scores: list[float] = []
    rouge2_scores: list[float] = []
    rougeL_scores: list[float] = []
    details: list[dict] = []

    for sample_id in common_ids:
        reference_text = reference_by_id[sample_id]
        prediction_text = prediction_by_id[sample_id]

        reference_nodes = parse_nodes(reference_text)
        prediction_nodes = parse_nodes(prediction_text)

        tp = len(reference_nodes & prediction_nodes)
        fp = len(prediction_nodes - reference_nodes)
        fn = len(reference_nodes - prediction_nodes)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        r1 = rouge_n_f1(reference_text, prediction_text, 1)
        r2 = rouge_n_f1(reference_text, prediction_text, 2)
        rL = rouge_l_f1(reference_text, prediction_text)
        rouge1_scores.append(r1)
        rouge2_scores.append(r2)
        rougeL_scores.append(rL)

        details.append(
            {
                "id": sample_id,
                "reference_node_count": len(reference_nodes),
                "prediction_node_count": len(prediction_nodes),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "vehicle_node_precision": safe_div(tp, tp + fp),
                "vehicle_node_recall": safe_div(tp, tp + fn),
                "vehicle_node_f1": safe_div(2 * tp, 2 * tp + fp + fn),
                "rouge1": r1,
                "rouge2": r2,
                "rougeL": rL,
            }
        )

    precision = safe_div(total_tp, total_tp + total_fp)
    recall = safe_div(total_tp, total_tp + total_fn)
    f1 = safe_div(2 * precision * recall, precision + recall)

    summary = {
        "reference_samples": len(reference_by_id),
        "prediction_samples": len(prediction_by_id),
        "matched_samples": len(common_ids),
        "missing_prediction_count": len(missing_predictions),
        "extra_prediction_count": len(extra_predictions),
        "missing_prediction_ids_preview": missing_predictions[:10],
        "extra_prediction_ids_preview": extra_predictions[:10],
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "vehicle_node_precision": precision,
        "vehicle_node_recall": recall,
        "vehicle_node_f1": f1,
        "rouge1": sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0.0,
        "rouge2": sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0.0,
        "rougeL": sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0.0,
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.save_details:
        detail_path = Path(args.save_details)
        detail_path.write_text(
            json.dumps({"summary": summary, "details": details}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()

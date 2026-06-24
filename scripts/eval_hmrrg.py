#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize(text: str) -> str:
    return " ".join((text or "").lower().split())


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HM-RRG generated reports against references.")
    parser.add_argument("--predictions-json", required=True, help="Mapping id -> generated report.")
    parser.add_argument("--references-json", required=True, help="Mapping id -> reference report.")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    preds = _load_json(args.predictions_json)
    refs = _load_json(args.references_json)
    common = sorted(set(preds) & set(refs))
    if not common:
        raise SystemExit("No overlapping ids between predictions and references.")

    exact = sum(_normalize(preds[k]) == _normalize(refs[k]) for k in common) / len(common)
    avg_pred_len = sum(len(_normalize(preds[k]).split()) for k in common) / len(common)
    avg_ref_len = sum(len(_normalize(refs[k]).split()) for k in common) / len(common)
    result = {
        "num_examples": len(common),
        "exact_match": exact,
        "avg_prediction_words": avg_pred_len,
        "avg_reference_words": avg_ref_len,
    }
    print(json.dumps(result, indent=2))
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hmrrg.data.retrieval import cosine_neighbors, load_embedding_jsonl, write_neighbors_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Build train-only cosine retrieval neighbors for HM-RRG.")
    parser.add_argument("--annotation-json", required=False, help="Kept for CLI symmetry; embeddings must include ids/splits.")
    parser.add_argument("--embedding-jsonl", required=True, help="JSONL with id, subject_id, split, embedding.")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--search-k", type=int, default=None)
    parser.add_argument("--allow-same-subject", action="store_true")
    args = parser.parse_args()

    records = load_embedding_jsonl(args.embedding_jsonl)
    candidates = [r for r in records if r.split == "train"]
    neighbors = cosine_neighbors(
        records,
        candidates,
        top_k=args.top_k,
        search_k=args.search_k,
        exclude_same_subject=not args.allow_same_subject,
    )
    write_neighbors_jsonl(neighbors, args.output_jsonl)
    print(f"wrote retrieval neighbors for {len(neighbors)} queries to {args.output_jsonl}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from typing import Dict, List

import torch
import torch.nn as nn

from hmrrg.data import (
    StudyRecord,
    attach_prior_reports,
    attach_retrieved_reports,
    build_id_to_report,
    load_annotation_records,
    load_retrieval_neighbors,
)
from hmrrg.models import HMRRGMemoryConfig, HierarchicalMemoryBank


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--stage", type=int, choices=[1, 2], required=True)
    parser.add_argument("--annotation-json", required=True)
    parser.add_argument("--train-image-root", required=True)
    parser.add_argument("--valtest-image-root", required=True)
    parser.add_argument("--retrieval-jsonl", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-prior-reports", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")


def load_public_records(args: argparse.Namespace) -> Dict[str, List[StudyRecord]]:
    records = load_annotation_records(args.annotation_json)
    records = attach_prior_reports(records, max_prior_reports=args.max_prior_reports, split_local=True)
    if args.retrieval_jsonl:
        neighbors = load_retrieval_neighbors(args.retrieval_jsonl)
        id_to_report = build_id_to_report(records, splits=["train"])
        records = attach_retrieved_reports(records, neighbors, id_to_report, top_k=args.top_k)
    return records


class _DummyBackbone(nn.Module):
    def forward(self, inputs_embeds, attention_mask=None, output_hidden_states=True, return_dict=True):
        class Output:
            pass

        out = Output()
        out.hidden_states = (inputs_embeds,)
        return out


def run_memory_dry_run(stage: int) -> None:
    hidden = 16
    cfg = HMRRGMemoryConfig(hidden_size=hidden, memory_hidden_size=8, segment_length=8)
    bank = HierarchicalMemoryBank(cfg)
    query = torch.randn(2, 1, hidden)
    retrieval = torch.randn(2, 5, hidden) if stage >= 1 else None
    segments = torch.randn(2, 3, hidden) if stage >= 2 else None
    prompt, weights, memory = bank(query, retrieval, segments)
    assert prompt is not None and weights is not None and memory is not None
    print(f"dry-run ok: prompt={tuple(prompt.shape)} memory={tuple(memory.shape)}")

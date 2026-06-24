#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from hmrrg.training import add_common_args, load_public_records, run_memory_dry_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or validate the VLM HM-RRG adaptation.")
    add_common_args(parser)
    parser.add_argument("--model-name", default="lingshu-medical-mllm/Lingshu-7B")
    parser.add_argument("--instruction", default="Generate a comprehensive and detailed radiology report for this chest x-ray image.")
    parser.add_argument("--output-dir", default="outputs/vlm-hmrrg")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--segment-length", type=int, default=128)
    parser.add_argument("--bptt-depth", type=int, default=2)
    parser.add_argument("--memory-hidden-size", type=int, default=4096)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--stage1-checkpoint", default=None)
    args = parser.parse_args()
    if args.max_steps is None:
        args.max_steps = 2000 if args.stage == 1 else 8000

    records = load_public_records(args)
    print({split: len(rows) for split, rows in records.items()})
    if args.dry_run:
        run_memory_dry_run(args.stage)
        return
    if args.retrieval_jsonl is None:
        raise SystemExit("--retrieval-jsonl is required because retrieved reports are encoded as M_retr, not prompt text.")

    from torch.utils.data import DataLoader
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    from hmrrg.data import VLMHMRRGCollator, VLMHMRRGDataset
    from hmrrg.models import QwenVLTokenAdapter, RecurrentHMRRG
    from hmrrg.training.runner import train_steps

    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
        trust_remote_code=True,
    )
    adapter = QwenVLTokenAdapter(model)
    hidden = int(adapter.get_input_embeddings().weight.shape[1])
    hmrrg = RecurrentHMRRG(
        adapter,
        hidden_size=hidden,
        memory_hidden_size=args.memory_hidden_size,
        segment_length=args.segment_length,
        bptt_depth=args.bptt_depth,
        use_retrieval_memory=True,
        use_segment_memory=args.stage == 2,
    )
    if args.stage == 2 and args.stage1_checkpoint:
        state = torch.load(args.stage1_checkpoint, map_location="cpu")
        hmrrg.load_state_dict(state.get("model", state), strict=False)
    train_ds = VLMHMRRGDataset(
        records["train"],
        train_image_root=args.train_image_root,
        valtest_image_root=args.valtest_image_root,
    )
    collator = VLMHMRRGCollator(processor, instruction=args.instruction, max_length=args.max_length, top_k=args.top_k)
    train_steps(
        hmrrg,
        DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator),
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        device=args.device,
        warmup_ratio=args.warmup_ratio,
        grad_clip=args.grad_clip,
    )
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save({"model": hmrrg.state_dict(), "stage": args.stage}, os.path.join(args.output_dir, f"stage{args.stage}.pt"))


if __name__ == "__main__":
    main()

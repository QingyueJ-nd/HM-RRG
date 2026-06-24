#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from hmrrg.training import add_common_args, load_public_records, run_memory_dry_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or validate the LLM HM-RRG adaptation.")
    add_common_args(parser)
    parser.add_argument("--model-name", default="BioMistral/BioMistral-7B")
    parser.add_argument("--cxrclip-ckpt", default=None, required=False)
    parser.add_argument("--output-dir", default="outputs/llm-hmrrg")
    parser.add_argument("--instruction", default="Generate a comprehensive and detailed radiology report for this chest x-ray image.")
    parser.add_argument("--num-image-tokens", type=int, default=50)
    parser.add_argument("--segment-length", type=int, default=128)
    parser.add_argument("--bptt-depth", type=int, default=2)
    parser.add_argument("--memory-hidden-size", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use-lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--stage1-checkpoint", default=None)
    args = parser.parse_args()
    if args.max_steps is None:
        args.max_steps = 2000 if args.stage == 1 else 8000

    records = load_public_records(args)
    print({split: len(rows) for split, rows in records.items()})
    if args.dry_run:
        run_memory_dry_run(args.stage)
        return

    if args.cxrclip_ckpt is None:
        raise SystemExit("--cxrclip-ckpt is required for LLM training.")
    if args.retrieval_jsonl is None:
        raise SystemExit("--retrieval-jsonl is required because retrieved reports are encoded as M_retr, not prompt text.")

    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, TaskType, get_peft_model

    from hmrrg.data import LLMHMRRGCollator, LLMHMRRGDataset
    from hmrrg.models import ImageTokenCausalLM, RecurrentHMRRG
    from hmrrg.training.runner import train_steps

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": ["<ImageToken>"]})

    base = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32)
    base.resize_token_embeddings(len(tokenizer))
    if args.use_lora:
        base = get_peft_model(
            base,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=0.1,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            ),
        )
    image_token_id = tokenizer.convert_tokens_to_ids("<ImageToken>")
    multimodal = ImageTokenCausalLM(
        base,
        image_token_id=image_token_id,
        cxrclip_checkpoint=args.cxrclip_ckpt,
        num_image_tokens=args.num_image_tokens,
        freeze_vision=True,
    )
    hidden = int(base.get_input_embeddings().weight.shape[1])
    model = RecurrentHMRRG(
        multimodal,
        hidden_size=hidden,
        memory_hidden_size=args.memory_hidden_size,
        segment_length=args.segment_length,
        bptt_depth=args.bptt_depth,
        use_retrieval_memory=True,
        use_segment_memory=args.stage == 2,
    )
    if args.stage == 2 and args.stage1_checkpoint:
        state = torch.load(args.stage1_checkpoint, map_location="cpu")
        model.load_state_dict(state.get("model", state), strict=False)

    train_ds = LLMHMRRGDataset(
        records["train"],
        train_image_root=args.train_image_root,
        valtest_image_root=args.valtest_image_root,
    )
    collator = LLMHMRRGCollator(
        tokenizer,
        instruction=args.instruction,
        num_image_tokens=args.num_image_tokens,
        top_k=args.top_k,
    )
    train_steps(
        model,
        DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator),
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        device=args.device,
        warmup_ratio=args.warmup_ratio,
        grad_clip=args.grad_clip,
    )
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save({"model": model.state_dict(), "stage": args.stage}, os.path.join(args.output_dir, f"stage{args.stage}.pt"))


if __name__ == "__main__":
    main()

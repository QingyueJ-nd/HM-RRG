from __future__ import annotations

from typing import Any, Dict, List, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset

from .records import StudyRecord
from .text import build_hmrrg_prompt


class VLMHMRRGDataset(Dataset):
    def __init__(self, records: Sequence[StudyRecord], *, train_image_root: str, valtest_image_root: str):
        self.records = list(records)
        self.train_image_root = train_image_root
        self.valtest_image_root = valtest_image_root

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        image = Image.open(record.image_path(self.train_image_root, self.valtest_image_root)).convert("RGB")
        return {"record": record, "image": image}


def _messages(image: Image.Image, prompt: str, answer: str | None):
    msgs = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    if answer is not None:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return msgs


class VLMHMRRGCollator:
    def __init__(self, processor, *, instruction: str, max_length: int = 2048, max_retrieved_tokens: int = 512, top_k: int = 5):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.instruction = instruction
        self.max_length = int(max_length)
        self.max_retrieved_tokens = int(max_retrieved_tokens)
        self.top_k = int(top_k)

    def _encode_retrieved(self, reports: Sequence[str], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        rows = []
        masks = []
        for report in list(reports)[: self.top_k]:
            ids = self.tokenizer.encode((report or "").strip(), add_special_tokens=False)[: self.max_retrieved_tokens]
            if not ids:
                ids = [pad_id]
                mask = [0]
            else:
                mask = [1] * len(ids)
            rows.append(torch.tensor(ids, dtype=torch.long))
            masks.append(torch.tensor(mask, dtype=torch.long))
        while len(rows) < self.top_k:
            rows.append(torch.tensor([pad_id], dtype=torch.long))
            masks.append(torch.tensor([0], dtype=torch.long))

        width = max(row.numel() for row in rows)
        padded_rows = []
        padded_masks = []
        for row, mask in zip(rows, masks):
            if row.numel() < width:
                row = torch.cat([row, row.new_full((width - row.numel(),), pad_id)])
                mask = torch.cat([mask, mask.new_zeros(width - mask.numel())])
            padded_rows.append(row)
            padded_masks.append(mask)
        return torch.stack(padded_rows, dim=0), torch.stack(padded_masks, dim=0)

    def __call__(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        input_ids = []
        labels = []
        attention_mask = []
        retrieved_ids = []
        retrieved_masks = []
        ids: List[str] = []
        images = []
        for item in items:
            record: StudyRecord = item["record"]
            prompt = build_hmrrg_prompt(
                instruction=self.instruction,
                prior_reports=record.prior_reports,
            )
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            if len(prompt_ids) > self.max_length:
                prompt_ids = prompt_ids[-self.max_length :]
            answer_ids = self.tokenizer.encode((record.report or "").strip() + self.tokenizer.eos_token, add_special_tokens=False)
            answer_ids = answer_ids[: max(1, self.max_length - len(prompt_ids))]
            ids_i = prompt_ids + answer_ids
            label_i = [-100] * len(prompt_ids) + answer_ids

            input_ids.append(torch.tensor(ids_i, dtype=torch.long))
            labels.append(torch.tensor(label_i, dtype=torch.long))
            attention_mask.append(torch.ones(len(ids_i), dtype=torch.long))
            ret_ids, ret_mask = self._encode_retrieved(record.retrieved_reports, pad_id)
            retrieved_ids.append(ret_ids)
            retrieved_masks.append(ret_mask)
            images.append(item["image"])
            ids.append(record.uid)

        retrieved_width = max(x.size(-1) for x in retrieved_ids)
        ret_id_batch = []
        ret_mask_batch = []
        for ret_ids, ret_mask in zip(retrieved_ids, retrieved_masks):
            if ret_ids.size(-1) < retrieved_width:
                pad = ret_ids.new_full((ret_ids.size(0), retrieved_width - ret_ids.size(-1)), pad_id)
                mask_pad = ret_mask.new_zeros((ret_mask.size(0), retrieved_width - ret_mask.size(-1)))
                ret_ids = torch.cat([ret_ids, pad], dim=-1)
                ret_mask = torch.cat([ret_mask, mask_pad], dim=-1)
            ret_id_batch.append(ret_ids)
            ret_mask_batch.append(ret_mask)

        out: Dict[str, Any] = {
            "input_ids": torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_id),
            "labels": torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0),
            "retrieved_input_ids": torch.stack(ret_id_batch, dim=0),
            "retrieved_attention_mask": torch.stack(ret_mask_batch, dim=0),
            "ids": ids,
        }
        if hasattr(self.processor, "image_processor"):
            image_features = self.processor.image_processor(images=images, return_tensors="pt")
            for key, value in image_features.items():
                out[key] = value
        return out

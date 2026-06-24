from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from .hierarchical_memory import HMRRGMemoryConfig, HierarchicalMemoryBank, RetrievalMemoryBuilder, SegmentSummaryEncoder


@dataclass
class SegmentCellOutput:
    logits: torch.Tensor
    loss: Optional[torch.Tensor]
    segment_hidden: torch.Tensor
    segment_memory: torch.Tensor
    sensory_memory: Optional[torch.Tensor]


class MemoryCell(nn.Module):
    """Decode one segment as [V(I), P_mem, C_n, H_n, P_mem]."""

    def __init__(self, backbone: nn.Module, hidden_size: int, num_sensory_tokens: int = 32):
        super().__init__()
        self.backbone = backbone
        self.num_sensory_tokens = int(num_sensory_tokens)
        std = float(backbone.get_input_embeddings().weight.detach().std().item())
        self.empty_prompt = nn.Parameter(torch.randn(1, hidden_size) * std)

    def _fallback_prompt(self, batch_size: int, *, device, dtype) -> torch.Tensor:
        return self.empty_prompt.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)

    def forward(
        self,
        *,
        segment_embeds: torch.Tensor,
        segment_attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor],
        image_tokens: Optional[torch.Tensor],
        memory_prompt: Optional[torch.Tensor],
        sensory_memory: Optional[torch.Tensor],
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> SegmentCellOutput:
        model_kwargs = model_kwargs or {}
        batch_size = segment_embeds.size(0)
        if memory_prompt is None:
            memory_prompt = self._fallback_prompt(batch_size, device=segment_embeds.device, dtype=segment_embeds.dtype)
        else:
            memory_prompt = memory_prompt.to(device=segment_embeds.device, dtype=segment_embeds.dtype)

        parts = []
        masks = []
        label_parts = []

        if image_tokens is not None:
            image_tokens = image_tokens.to(device=segment_embeds.device, dtype=segment_embeds.dtype)
            parts.append(image_tokens)
            masks.append(torch.ones(image_tokens.shape[:2], dtype=segment_attention_mask.dtype, device=segment_attention_mask.device))
            label_parts.append(torch.full(image_tokens.shape[:2], -100, dtype=torch.long, device=segment_embeds.device))

        parts.append(memory_prompt)
        masks.append(torch.ones(memory_prompt.shape[:2], dtype=segment_attention_mask.dtype, device=segment_attention_mask.device))
        label_parts.append(torch.full(memory_prompt.shape[:2], -100, dtype=torch.long, device=segment_embeds.device))

        if sensory_memory is not None:
            sensory_memory = sensory_memory.to(device=segment_embeds.device, dtype=segment_embeds.dtype)
            parts.append(sensory_memory)
            masks.append(torch.ones(sensory_memory.shape[:2], dtype=segment_attention_mask.dtype, device=segment_attention_mask.device))
            label_parts.append(torch.full(sensory_memory.shape[:2], -100, dtype=torch.long, device=segment_embeds.device))

        segment_offset = sum(part.size(1) for part in parts)
        parts.append(segment_embeds)
        masks.append(segment_attention_mask)
        label_parts.append(labels if labels is not None else torch.full(segment_attention_mask.shape, -100, dtype=torch.long, device=segment_embeds.device))

        trailing_offset = segment_offset + segment_embeds.size(1)
        parts.append(memory_prompt)
        masks.append(torch.ones(memory_prompt.shape[:2], dtype=segment_attention_mask.dtype, device=segment_attention_mask.device))
        label_parts.append(torch.full(memory_prompt.shape[:2], -100, dtype=torch.long, device=segment_embeds.device))

        inputs_embeds = torch.cat(parts, dim=1)
        attention_mask = torch.cat(masks, dim=1)
        full_labels = torch.cat(label_parts, dim=1) if labels is not None else None
        outputs = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=full_labels,
            output_hidden_states=True,
            return_dict=True,
            **model_kwargs,
        )
        hidden = outputs.hidden_states[-1]
        segment_hidden = hidden[:, segment_offset:trailing_offset, :]
        segment_memory = hidden[:, trailing_offset : trailing_offset + memory_prompt.size(1), :]
        sensory = segment_hidden[:, -self.num_sensory_tokens :, :] if self.num_sensory_tokens > 0 and segment_hidden.size(1) > 0 else None
        logits = outputs.logits[:, segment_offset:trailing_offset, :]
        return SegmentCellOutput(logits=logits, loss=getattr(outputs, "loss", None), segment_hidden=segment_hidden, segment_memory=segment_memory, sensory_memory=sensory)


class RecurrentHMRRG(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        hidden_size: int,
        memory_hidden_size: int = 4096,
        segment_length: int = 128,
        sensory_tokens: int = 32,
        bptt_depth: int = 2,
        use_retrieval_memory: bool = True,
        use_segment_memory: bool = True,
    ):
        super().__init__()
        self.segment_length = int(segment_length)
        self.bptt_depth = int(bptt_depth)
        self.use_retrieval_memory = bool(use_retrieval_memory)
        self.use_segment_memory = bool(use_segment_memory)
        self.config = HMRRGMemoryConfig(
            hidden_size=hidden_size,
            memory_hidden_size=memory_hidden_size,
            segment_length=segment_length,
            sensory_tokens=sensory_tokens,
        )
        token_embedding = backbone.get_input_embeddings()
        self.retrieval_builder = RetrievalMemoryBuilder(backbone, token_embedding, self.config)
        self.summary_encoder = SegmentSummaryEncoder(backbone, token_embedding, self.config)
        self.cell = MemoryCell(backbone, hidden_size, num_sensory_tokens=sensory_tokens)
        self.memory_bank = HierarchicalMemoryBank(self.config)
        self.backbone = backbone

    def _encode_current_image(self, images: Optional[torch.Tensor], current_image_tokens: Optional[torch.Tensor], model_kwargs: Dict[str, Any]) -> Optional[torch.Tensor]:
        if current_image_tokens is not None:
            return current_image_tokens
        encoder = getattr(self.backbone, "encode_image", None)
        if encoder is None:
            return None
        if images is not None:
            return encoder(images=images)
        pixel_values = model_kwargs.get("pixel_values")
        image_grid_thw = model_kwargs.get("image_grid_thw")
        if pixel_values is not None:
            return encoder(pixel_values=pixel_values, image_grid_thw=image_grid_thw)
        return None

    def _build_retrieval_memory(
        self,
        *,
        retrieved_input_ids: Optional[torch.Tensor],
        retrieved_attention_mask: Optional[torch.Tensor],
        current_image_tokens: Optional[torch.Tensor],
        retrieval_memory: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.use_retrieval_memory:
            return None, None
        if retrieval_memory is not None:
            return retrieval_memory, torch.ones(retrieval_memory.shape[:2], dtype=torch.bool, device=retrieval_memory.device)
        if retrieved_input_ids is None or retrieved_attention_mask is None:
            return None, None
        memory = self.retrieval_builder(retrieved_input_ids, retrieved_attention_mask, current_image_tokens)
        mask = retrieved_attention_mask.any(dim=-1)
        return memory, mask

    def _segment_memory_bank(self, segment_memories: list[torch.Tensor]) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.use_segment_memory or not segment_memories:
            return None, None
        memories = []
        detach_before = max(0, len(segment_memories) - self.bptt_depth)
        for idx, memory in enumerate(segment_memories):
            memories.append(memory.detach() if idx < detach_before else memory)
        bank = torch.cat(memories, dim=1)
        mask = torch.ones(bank.shape[:2], dtype=torch.bool, device=bank.device)
        return bank, mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        current_image_tokens: Optional[torch.Tensor] = None,
        retrieved_input_ids: Optional[torch.Tensor] = None,
        retrieved_attention_mask: Optional[torch.Tensor] = None,
        retrieval_memory: Optional[torch.Tensor] = None,
        segment_memory: Optional[torch.Tensor] = None,
        **model_kwargs: Any,
    ):
        input_embeds = self.backbone.get_input_embeddings()(input_ids)
        image_tokens = self._encode_current_image(images, current_image_tokens, model_kwargs)
        retrieval_bank, retrieval_mask = self._build_retrieval_memory(
            retrieved_input_ids=retrieved_input_ids,
            retrieved_attention_mask=retrieved_attention_mask,
            current_image_tokens=image_tokens,
            retrieval_memory=retrieval_memory,
        )

        sensory = None
        outputs = []
        segment_memories = [] if segment_memory is None else [segment_memory]

        for start in range(0, input_ids.size(1), self.segment_length):
            end = start + self.segment_length
            seg_embeds = input_embeds[:, start:end, :]
            seg_mask = attention_mask[:, start:end]
            seg_labels = labels[:, start:end] if labels is not None else None

            query = self.summary_encoder(seg_embeds, seg_mask)
            segment_bank, segment_mask = self._segment_memory_bank(segment_memories)
            prompt, _, _ = self.memory_bank(query, retrieval_bank, segment_bank, retrieval_mask, segment_mask)

            out = self.cell(
                segment_embeds=seg_embeds,
                segment_attention_mask=seg_mask,
                labels=seg_labels,
                image_tokens=image_tokens,
                memory_prompt=prompt,
                sensory_memory=sensory,
                model_kwargs={},
            )
            outputs.append(out)
            sensory = out.sensory_memory
            segment_memories.append(out.segment_memory)

        result = CausalLMOutputWithCrossAttentions()
        result["logits"] = torch.cat([o.logits for o in outputs], dim=1)
        if labels is not None:
            losses = [o.loss for o in outputs if o.loss is not None]
            result["loss"] = torch.stack(losses).mean() if losses else None
        result["segment_memory"] = torch.cat(segment_memories, dim=1)
        result["retrieval_memory"] = retrieval_bank
        return result

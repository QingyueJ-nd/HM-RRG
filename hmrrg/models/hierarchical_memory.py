from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class HMRRGMemoryConfig:
    hidden_size: int
    memory_hidden_size: int
    num_summary_tokens: int = 1
    sensory_tokens: int = 32
    segment_length: int = 128
    retrieval_top_k: int = 5
    summary_prefix_tokens: int = 1


@dataclass(frozen=True)
class HMGenerationInputs:
    current_image_tokens: torch.Tensor
    prior_input_ids: torch.Tensor
    prior_attention_mask: torch.Tensor
    retrieved_input_ids: Optional[torch.Tensor] = None
    retrieved_attention_mask: Optional[torch.Tensor] = None


def sensory_tail(segment_embeds: torch.Tensor, num_tokens: int) -> Optional[torch.Tensor]:
    if num_tokens <= 0 or segment_embeds.size(1) == 0:
        return None
    return segment_embeds[:, -int(num_tokens) :, :]


class CrossAttentionMemory(nn.Module):
    """Retrieve a compact prompt using the HMT/HM-RRG memory search equation."""

    def __init__(self, hidden_size: int, memory_hidden_size: int):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, memory_hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, memory_hidden_size, bias=False)

    def forward(
        self,
        query: torch.Tensor,
        memory: Optional[torch.Tensor],
        memory_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if memory is None or memory.size(1) == 0:
            return None, None

        q = self.q_proj(query)
        k = self.k_proj(memory)
        scores = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(k.size(-1))
        valid = None
        if memory_mask is not None:
            memory_mask = memory_mask.bool()
            valid = memory_mask.any(dim=1)
            scores = scores.masked_fill(~memory_mask[:, None, :], torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores.float(), dim=-1).to(dtype=query.dtype)
        if memory_mask is not None:
            weights = weights * memory_mask[:, None, :].to(dtype=weights.dtype)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).eps)
            weights = weights * valid[:, None, None].to(dtype=weights.dtype)
        prompt = torch.matmul(weights, memory)
        return prompt, weights


class RetrievalMemoryBuilder(nn.Module):
    """Compress each retrieved report into an image-conditioned memory vector.

    The module is deliberately backbone-agnostic. The backbone must accept
    `inputs_embeds`, `attention_mask`, and `output_hidden_states=True`.
    """

    def __init__(self, backbone: nn.Module, token_embedding: nn.Embedding, config: HMRRGMemoryConfig):
        super().__init__()
        self.backbone = backbone
        self.token_embedding = token_embedding
        self.config = config
        std = float(token_embedding.weight.detach().std().item())
        memory = torch.randn(config.num_summary_tokens, config.hidden_size) * std
        self.memory_tokens = nn.Parameter(memory)

    def forward(
        self,
        retrieved_input_ids: torch.Tensor,
        retrieved_attention_mask: torch.Tensor,
        current_image_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if retrieved_input_ids.dim() != 3:
            raise ValueError("retrieved_input_ids must have shape (B, K, T)")

        batch_size, num_reports, seq_len = retrieved_input_ids.shape
        flat_ids = retrieved_input_ids.reshape(batch_size * num_reports, seq_len)
        flat_mask = retrieved_attention_mask.reshape(batch_size * num_reports, seq_len)
        report_embeds = self.token_embedding(flat_ids)

        parts = []
        masks = []

        leading_memory = self.memory_tokens[None, :, :].expand(batch_size * num_reports, -1, -1)
        parts.append(leading_memory.to(dtype=report_embeds.dtype, device=report_embeds.device))
        masks.append(torch.ones(leading_memory.shape[:2], dtype=flat_mask.dtype, device=flat_mask.device))

        if current_image_tokens is not None:
            image = current_image_tokens[:, None, :, :].expand(-1, num_reports, -1, -1)
            image = image.reshape(batch_size * num_reports, image.size(2), image.size(3))
            parts.append(image.to(dtype=report_embeds.dtype, device=report_embeds.device))
            masks.append(torch.ones(image.shape[:2], dtype=flat_mask.dtype, device=flat_mask.device))

        parts.append(report_embeds)
        masks.append(flat_mask)

        trailing_memory = self.memory_tokens[None, :, :].expand(batch_size * num_reports, -1, -1)
        parts.append(trailing_memory.to(dtype=report_embeds.dtype, device=report_embeds.device))
        masks.append(torch.ones(trailing_memory.shape[:2], dtype=flat_mask.dtype, device=flat_mask.device))

        inputs_embeds = torch.cat(parts, dim=1)
        attention_mask = torch.cat(masks, dim=1)
        outputs = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states[-1]
        memory = hidden_states[:, -self.config.num_summary_tokens :, :].mean(dim=1)
        return memory.reshape(batch_size, num_reports, -1)


class SegmentSummaryEncoder(nn.Module):
    """Encode S_n = LM(t_sum || H_n[0:j] || t_sum)."""

    def __init__(self, backbone: nn.Module, token_embedding: nn.Embedding, config: HMRRGMemoryConfig, summary_tokens: Optional[nn.Parameter] = None):
        super().__init__()
        self.backbone = backbone
        self.token_embedding = token_embedding
        self.config = config
        if summary_tokens is None:
            std = float(token_embedding.weight.detach().std().item())
            summary = torch.randn(config.summary_prefix_tokens, config.hidden_size) * std
            self.summary_tokens = nn.Parameter(summary)
        else:
            self.summary_tokens = summary_tokens

    def forward(self, segment_embeds: torch.Tensor, segment_attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = segment_embeds.size(0)
        query_len = min(segment_embeds.size(1), max(1, self.config.segment_length // 2))
        prefix = self.summary_tokens[None, :, :].expand(batch_size, -1, -1).to(device=segment_embeds.device, dtype=segment_embeds.dtype)
        query_embeds = segment_embeds[:, :query_len, :]
        inputs_embeds = torch.cat([prefix, query_embeds, prefix], dim=1)

        mask_parts = [
            torch.ones(prefix.shape[:2], dtype=torch.long, device=segment_embeds.device),
        ]
        if segment_attention_mask is None:
            mask_parts.append(torch.ones(query_embeds.shape[:2], dtype=torch.long, device=segment_embeds.device))
        else:
            mask_parts.append(segment_attention_mask[:, :query_len].to(device=segment_embeds.device))
        mask_parts.append(torch.ones(prefix.shape[:2], dtype=torch.long, device=segment_embeds.device))
        attention_mask = torch.cat(mask_parts, dim=1)

        outputs = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        return outputs.hidden_states[-1][:, -self.config.summary_prefix_tokens :, :]


class HierarchicalMemoryBank(nn.Module):
    """Union of cross-patient retrieval memory and within-patient segment memory."""

    def __init__(self, config: HMRRGMemoryConfig):
        super().__init__()
        self.config = config
        self.retriever = CrossAttentionMemory(config.hidden_size, config.memory_hidden_size)

    def forward(
        self,
        query_memory: torch.Tensor,
        retrieval_memory: Optional[torch.Tensor],
        segment_memory: Optional[torch.Tensor],
        retrieval_mask: Optional[torch.Tensor] = None,
        segment_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        memories = []
        masks = []
        if retrieval_memory is not None and retrieval_memory.size(1) > 0:
            memories.append(retrieval_memory)
            masks.append(retrieval_mask if retrieval_mask is not None else torch.ones(retrieval_memory.shape[:2], dtype=torch.bool, device=retrieval_memory.device))
        if segment_memory is not None and segment_memory.size(1) > 0:
            memories.append(segment_memory)
            masks.append(segment_mask if segment_mask is not None else torch.ones(segment_memory.shape[:2], dtype=torch.bool, device=segment_memory.device))
        if not memories:
            return None, None, None
        memory = torch.cat(memories, dim=1)
        memory_mask = torch.cat([m.to(device=memory.device).bool() for m in masks], dim=1)
        prompt, weights = self.retriever(query_memory, memory, memory_mask)
        return prompt, weights, memory

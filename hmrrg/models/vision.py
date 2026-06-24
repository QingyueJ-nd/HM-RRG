from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class CXRClipVision(nn.Module):
    def __init__(self, checkpoint_path: str):
        super().__init__()
        try:
            import torchvision.models as tvm
        except ImportError as exc:
            raise ImportError("CXRClipVision requires torchvision. Install project requirements before LLM image training.") from exc
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        self.resnet = tvm.resnet50(weights=None)
        filtered = {
            key.replace("image_encoder.resnet.", ""): value
            for key, value in state.items()
            if key.startswith("image_encoder.resnet.")
        }
        self.resnet.load_state_dict(filtered, strict=False)
        self.patch_layer = self.resnet.layer4[2].conv3

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        holder: Dict[str, torch.Tensor] = {}

        def hook(_module, _inputs, output):
            holder["features"] = output

        handle = self.patch_layer.register_forward_hook(hook)
        x = self.resnet.conv1(images)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        handle.remove()
        features = holder["features"]
        patches = features.flatten(2).transpose(1, 2)
        cls = features.mean(dim=(2, 3)).unsqueeze(1)
        return torch.cat([cls, patches], dim=1)


class ImageTokenCausalLM(nn.Module):
    def __init__(self, base_lm: nn.Module, *, image_token_id: int, cxrclip_checkpoint: str, num_image_tokens: int = 50, freeze_vision: bool = True):
        super().__init__()
        self.base_lm = base_lm
        self.config = base_lm.config
        self.image_token_id = int(image_token_id)
        self.num_image_tokens = int(num_image_tokens)
        self.vision = CXRClipVision(cxrclip_checkpoint)
        hidden = base_lm.get_input_embeddings().weight.shape[1]
        self.image_projector = nn.Sequential(nn.Linear(2048, hidden), nn.LayerNorm(hidden))
        if freeze_vision:
            self.vision.requires_grad_(False)

    def get_input_embeddings(self):
        return self.base_lm.get_input_embeddings()

    def encode_image(self, *, images: torch.Tensor, **_: Any) -> torch.Tensor:
        image_tokens = self.image_projector(self.vision(images.to(next(self.parameters()).device, dtype=torch.float32)))
        return image_tokens.to(self.get_input_embeddings().weight.dtype)

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, labels=None, images: Optional[torch.Tensor] = None, token_ids: Optional[torch.Tensor] = None, **kwargs: Any):
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        locator_ids = token_ids if token_ids is not None else input_ids
        if images is not None and locator_ids is not None:
            image_tokens = self.image_projector(self.vision(images.to(next(self.parameters()).device, dtype=torch.float32))).to(inputs_embeds.dtype)
            positions = locator_ids == self.image_token_id
            for idx in range(locator_ids.size(0)):
                pos = torch.nonzero(positions[idx], as_tuple=False).squeeze(1)[: self.num_image_tokens]
                if pos.numel() != self.num_image_tokens:
                    raise RuntimeError("Image placeholder block is missing or split; increase segment length or keep image tokens first.")
                inputs_embeds[idx, pos, :] = image_tokens[idx, : pos.numel(), :]
                if labels is not None:
                    labels[idx, pos] = -100
        return self.base_lm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, **kwargs)


class QwenVLTokenAdapter(nn.Module):
    """Expose Qwen2/2.5-VL image embeddings as HM-RRG V(I_curr) prefix tokens."""

    def __init__(self, base_vlm: nn.Module):
        super().__init__()
        self.base_vlm = base_vlm
        self.config = base_vlm.config

    def get_input_embeddings(self):
        return self.base_vlm.get_input_embeddings()

    def encode_image(self, *, pixel_values: torch.Tensor, image_grid_thw: Optional[torch.Tensor] = None, **_: Any) -> torch.Tensor:
        visual = getattr(self.base_vlm, "visual", None)
        if visual is None and hasattr(self.base_vlm, "model"):
            visual = getattr(self.base_vlm.model, "visual", None)
        if visual is None:
            raise RuntimeError("QwenVLTokenAdapter could not find a visual encoder on the wrapped VLM.")
        if image_grid_thw is None:
            raise RuntimeError("QwenVLTokenAdapter requires image_grid_thw from the processor image pipeline.")

        pixel_values = pixel_values.to(next(self.parameters()).device)
        image_grid_thw = image_grid_thw.to(pixel_values.device)
        try:
            image_embeds = visual(pixel_values, grid_thw=image_grid_thw)
        except TypeError:
            image_embeds = visual(pixel_values=pixel_values, grid_thw=image_grid_thw)

        merge = 1
        vision_config = getattr(self.config, "vision_config", None)
        if vision_config is not None:
            merge = int(getattr(vision_config, "spatial_merge_size", 1))
        lengths = (image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2] // max(1, merge * merge)).tolist()
        splits = torch.split(image_embeds, [int(x) for x in lengths], dim=0)
        max_len = max(split.size(0) for split in splits)
        padded = []
        for split in splits:
            if split.size(0) < max_len:
                pad = split.new_zeros((max_len - split.size(0), split.size(1)))
                split = torch.cat([split, pad], dim=0)
            padded.append(split)
        return torch.stack(padded, dim=0).to(dtype=self.get_input_embeddings().weight.dtype)

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, labels=None, output_hidden_states=True, return_dict=True, **kwargs: Any):
        return self.base_vlm(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

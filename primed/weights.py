import re
from typing import Dict, Optional

import torch
from safetensors.torch import load_file, save_file

from .model import PRIMED


def save_weights(state_dict: Dict[str, torch.Tensor], path: str, metadata: Optional[Dict[str, str]] = None) -> None:
    tensors = {k: v.detach().float().cpu().contiguous() for k, v in state_dict.items()}
    save_file(tensors, path, metadata={k: str(v) for k, v in (metadata or {}).items()})


def load_weights(path: str) -> Dict[str, torch.Tensor]:
    if path.endswith(".safetensors"):
        return load_file(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return checkpoint.get("model", checkpoint)


def architecture(state_dict: Dict[str, torch.Tensor]) -> Dict:
    num_classes, class_dim = state_dict["class_spatial_embed.weight"].shape
    in_channels = state_dict["unet.conv_in.conv.weight"].shape[1]
    levels = sorted({int(m.group(1)) for k in state_dict if (m := re.match(r"unet\.down_blocks\.(\d+)\.", k))})
    channels = [state_dict[f"unet.down_blocks.{i}.resnets.0.conv1.conv.weight"].shape[0] for i in levels]
    num_res_blocks = len({k.split(".")[4] for k in state_dict if k.startswith("unet.down_blocks.0.resnets.")})
    return {
        "num_classes": int(num_classes),
        "num_inputs": int((in_channels - class_dim) // 8 - 2),
        "channels": channels,
        "num_res_blocks": num_res_blocks,
        "class_dim": int(class_dim),
    }


def load_model(path: str, device: str = "cpu") -> PRIMED:
    state_dict = load_weights(path)
    model = PRIMED(**architecture(state_dict))
    model.net.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()

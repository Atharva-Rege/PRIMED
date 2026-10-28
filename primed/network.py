from typing import Sequence

import torch
import torch.nn as nn
from monai.networks.nets import DiffusionModelUNet


class PRIMEDNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 8,
        channels: Sequence[int] = (64, 128, 128, 256, 256),
        num_res_blocks: int = 3,
        num_classes: int = 4,
        class_dim: int = 8,
        num_head_channels: int = 64,
    ):
        super().__init__()
        self.class_dim = class_dim
        self.class_spatial_embed = nn.Embedding(num_classes, class_dim)
        self.unet = DiffusionModelUNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=tuple(channels),
            attention_levels=(False,) * len(channels),
            num_res_blocks=num_res_blocks,
            num_head_channels=num_head_channels,
            num_class_embeds=num_classes,
        )

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        target_class: torch.Tensor,
        z_inputs: torch.Tensor,
    ) -> torch.Tensor:
        embedding = self.class_spatial_embed(target_class.view(-1).long())
        embedding = embedding.view(x.shape[0], self.class_dim, 1, 1, 1).expand(-1, -1, *x.shape[-3:])
        x = torch.cat([x, embedding.to(x.dtype), z_inputs.to(x.dtype)], dim=1)
        return self.unet(x=x, timesteps=timesteps, class_labels=target_class)

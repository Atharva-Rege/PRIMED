from typing import Sequence

import torch
import torch.nn as nn

from .network import PRIMEDNet
from .wavelet import HaarWavelet3D

SOURCE_TIMESTEP = 1000.0


class PRIMED(nn.Module):
    def __init__(
        self,
        num_classes: int = 4,
        num_inputs: int = 3,
        channels: Sequence[int] = (64, 128, 128, 256, 256),
        num_res_blocks: int = 3,
        class_dim: int = 8,
        lll_scale: float = 3.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_inputs = num_inputs
        self.wavelet = HaarWavelet3D(lll_scale)
        self.net = PRIMEDNet(
            in_channels=8 * (num_inputs + 2) + class_dim,
            out_channels=8,
            channels=channels,
            num_res_blocks=num_res_blocks,
            num_classes=num_classes,
            class_dim=class_dim,
        )

    def encode_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, count = inputs.shape[:2]
        z = self.wavelet.encode(inputs.reshape(batch * count, *inputs.shape[2:]))
        return z.reshape(batch, count * z.shape[1], *z.shape[2:])

    def displacement(self, z_prior: torch.Tensor, z_inputs: torch.Tensor, target_class: torch.Tensor) -> torch.Tensor:
        timesteps = torch.full((z_prior.shape[0],), SOURCE_TIMESTEP, device=z_prior.device)
        return self.net(torch.cat([z_prior, z_prior], dim=1).float(), timesteps, target_class, z_inputs)

    def forward(
        self,
        prior: torch.Tensor,
        inputs: torch.Tensor,
        target: torch.Tensor,
        target_class: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            z_target = self.wavelet.encode(target)
            z_prior = self.wavelet.encode(prior)
            z_inputs = self.encode_inputs(inputs)
        d = self.displacement(z_prior, z_inputs, target_class)
        return ((d - (z_prior - z_target)) ** 2).reshape(d.shape[0], -1).mean(1).mean()

    @torch.no_grad()
    def synthesize(self, prior: torch.Tensor, inputs: torch.Tensor, target_class: torch.Tensor) -> torch.Tensor:
        z_prior = self.wavelet.encode(prior)
        z = z_prior - self.displacement(z_prior, self.encode_inputs(inputs), target_class)
        return self.wavelet.decode(z).clamp(0.0, 1.0)

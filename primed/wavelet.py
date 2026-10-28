import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F


def haar_filters() -> torch.Tensor:
    low = torch.tensor([1.0, 1.0]) / (2.0 ** 0.5)
    high = torch.tensor([1.0, -1.0]) / (2.0 ** 0.5)
    banks = {"L": low, "H": high}
    filters = [
        torch.einsum("i,j,k->ijk", banks[a], banks[b], banks[c])
        for a, b, c in itertools.product("LH", repeat=3)
    ]
    return torch.stack(filters).unsqueeze(1)


class HaarWavelet3D(nn.Module):
    def __init__(self, lll_scale: float = 3.0):
        super().__init__()
        self.lll_scale = lll_scale
        self.register_buffer("filters", haar_filters(), persistent=False)

    def _band_gain(self, z: torch.Tensor, inverse: bool) -> torch.Tensor:
        gain = torch.ones(z.shape[1], device=z.device, dtype=z.dtype)
        gain[0::8] = self.lll_scale if inverse else 1.0 / self.lll_scale
        return gain.view(1, -1, 1, 1, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        weight = self.filters.repeat(channels, 1, 1, 1, 1).to(device=x.device, dtype=x.dtype)
        z = F.conv3d(x, weight, stride=2, groups=channels)
        return z * self._band_gain(z, inverse=False)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = z * self._band_gain(z, inverse=True)
        channels = z.shape[1] // 8
        weight = self.filters.repeat(channels, 1, 1, 1, 1).to(device=z.device, dtype=z.dtype)
        return F.conv_transpose3d(z, weight, stride=2, groups=channels)

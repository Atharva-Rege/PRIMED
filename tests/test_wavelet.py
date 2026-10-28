import math

import torch

from primed.wavelet import HaarWavelet3D


def test_decode_inverts_encode():
    wavelet = HaarWavelet3D()
    x = torch.rand(2, 1, 8, 12, 16, dtype=torch.float64)
    z = wavelet.encode(x)
    assert z.shape == (2, 8, 4, 6, 8)
    assert torch.allclose(wavelet.decode(z), x, atol=1e-12)


def test_encode_handles_stacked_channels():
    wavelet = HaarWavelet3D()
    x = torch.rand(1, 3, 8, 8, 8, dtype=torch.float64)
    z = wavelet.encode(x)
    assert z.shape == (1, 24, 4, 4, 4)
    assert torch.allclose(wavelet.decode(z), x, atol=1e-12)


def test_low_band_is_rescaled():
    wavelet = HaarWavelet3D(lll_scale=3.0)
    z = wavelet.encode(torch.full((1, 1, 4, 4, 4), 0.5, dtype=torch.float64))
    assert torch.allclose(z[:, 0], torch.full_like(z[:, 0], 0.5 * 2 * math.sqrt(2) / 3))
    assert torch.allclose(z[:, 1:], torch.zeros_like(z[:, 1:]))


def test_encode_is_linear():
    wavelet = HaarWavelet3D()
    a, b = torch.rand(2, 1, 1, 8, 8, 8, dtype=torch.float64)
    assert torch.allclose(wavelet.encode((a + b) / 2), (wavelet.encode(a) + wavelet.encode(b)) / 2, atol=1e-12)

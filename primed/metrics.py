import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .data import DEPTH

BRATS_REGIONS = {"WT": (1, 2, 3), "ET": (3,), "NCR": (1,), "ED": (2,)}


def psnr(squared_error: torch.Tensor, mask: Optional[torch.Tensor] = None, min_voxels: int = 100) -> Optional[float]:
    if mask is None:
        return 10 * math.log10(1.0 / max(float(squared_error.mean()), 1e-12))
    if int(mask.sum()) < min_voxels:
        return None
    return 10 * math.log10(1.0 / max(float(squared_error[mask].mean()), 1e-12))


def ssim3d(a: torch.Tensor, b: torch.Tensor, window: int = 7) -> float:
    a = a.double()[None, None]
    b = b.double()[None, None]
    mean_a = F.avg_pool3d(a, window, stride=1)
    mean_b = F.avg_pool3d(b, window, stride=1)
    n = window ** 3
    unbiased = n / (n - 1.0)
    var_a = unbiased * (F.avg_pool3d(a * a, window, stride=1) - mean_a ** 2)
    var_b = unbiased * (F.avg_pool3d(b * b, window, stride=1) - mean_b ** 2)
    cov = unbiased * (F.avg_pool3d(a * b, window, stride=1) - mean_a * mean_b)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mean_a * mean_b + c1) * (2 * cov + c2)) / ((mean_a ** 2 + mean_b ** 2 + c1) * (var_a + var_b + c2))
    return float(s.mean())


def score(prediction: torch.Tensor, target: torch.Tensor, segmentation: Optional[torch.Tensor] = None) -> Dict:
    p = prediction[..., :DEPTH].float()
    t = target[..., :DEPTH].float()
    squared_error = (p - t) ** 2
    out = {"psnr": psnr(squared_error), "psnr_brain": psnr(squared_error, t > 0)}
    if segmentation is not None:
        for name, labels in BRATS_REGIONS.items():
            region = torch.isin(segmentation, torch.tensor(labels, device=segmentation.device))
            out[f"psnr_{name}"] = psnr(squared_error, region)
    out["ssim"] = ssim3d(p, t)
    return out

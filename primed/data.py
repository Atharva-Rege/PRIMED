import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

MODALITIES: Dict[str, Sequence[str]] = {
    "brats": ("t1n", "t1c", "t2w", "t2f"),
    "ixi": ("T1", "T2", "PD"),
}
ALIASES: Dict[str, Dict[str, str]] = {
    "brats": {
        "t1n": "t1n", "t1": "t1n",
        "t1c": "t1c", "t1ce": "t1c", "t1gd": "t1c",
        "t2w": "t2w", "t2": "t2w",
        "t2f": "t2f", "flair": "t2f", "fla": "t2f",
    },
    "ixi": {"t1": "T1", "t2": "T2", "pd": "PD"},
}
NATIVE_SHAPE = (240, 240, 155)
DEPTH = 155
PADDED_DEPTH = 160
CROP = 8


def dataset_for_classes(num_classes: int) -> str:
    for name, modalities in MODALITIES.items():
        if len(modalities) == num_classes:
            return name
    raise ValueError(f"no dataset with {num_classes} contrasts")


def canonical_contrast(name: str, dataset: str) -> str:
    contrast = ALIASES[dataset].get(name.lower())
    if contrast is None:
        raise ValueError(f"unknown contrast {name!r}; expected one of {', '.join(MODALITIES[dataset])}")
    return contrast


def resolve_path(path: str) -> str:
    if os.path.isabs(path) or os.path.exists(path):
        return path
    candidate = Path(__file__).resolve().parent.parent / path
    return str(candidate) if candidate.exists() else path


def read_subject_list(path: str) -> List[str]:
    with open(resolve_path(path)) as f:
        return [line.strip() for line in f if line.strip()]


def normalize(array: np.ndarray) -> np.ndarray:
    clipped = np.clip(array, np.quantile(array, 0.001), np.quantile(array, 0.999))
    span = np.max(clipped) - np.min(clipped)
    if span <= 0:
        return np.zeros_like(clipped, dtype=np.float32)
    return ((clipped - np.min(clipped)) / span).astype(np.float32)


def load_volume(path: str) -> torch.Tensor:
    array = nib.load(path).get_fdata()
    if array.shape != NATIVE_SHAPE:
        raise ValueError(f"{path} has shape {array.shape}, expected {NATIVE_SHAPE}")
    volume = torch.zeros(1, NATIVE_SHAPE[0], NATIVE_SHAPE[1], PADDED_DEPTH)
    volume[:, :, :, :DEPTH] = torch.from_numpy(normalize(array))
    return volume[:, CROP:-CROP, CROP:-CROP, :].float()


def load_segmentation(path: str) -> torch.Tensor:
    array = np.asarray(nib.load(path).dataobj).astype(np.int16)
    return torch.from_numpy(array[CROP:-CROP, CROP:-CROP, :DEPTH])


def to_native(volume: np.ndarray) -> np.ndarray:
    out = np.zeros(NATIVE_SHAPE, dtype=np.float32)
    out[CROP:-CROP, CROP:-CROP, :] = volume[..., :DEPTH]
    return out


def modality_of(filename: str, dataset: str) -> Optional[str]:
    base = filename
    for extension in (".nii.gz", ".nii"):
        if base.endswith(extension):
            base = base[: -len(extension)]
            break
    else:
        return None
    token = base.replace("-", "_").split("_")[-1].lower()
    return ALIASES[dataset].get(token)


def scan_subjects(root: str, dataset: str) -> List[Dict[str, str]]:
    modalities = MODALITIES[dataset]
    items = []
    for subject in sorted(os.listdir(root)):
        folder = os.path.join(root, subject)
        if not os.path.isdir(folder):
            continue
        found = {}
        for filename in os.listdir(folder):
            modality = modality_of(filename, dataset)
            if modality is not None:
                found[modality] = os.path.join(folder, filename)
        if all(m in found for m in modalities):
            found["subject"] = subject
            segmentation = os.path.join(folder, f"{subject}-seg.nii.gz")
            if os.path.exists(segmentation):
                found["seg"] = segmentation
            items.append(found)
    return items


def make_split(items: List[Dict[str, str]], n_val: int, seed: int = 42) -> Dict[str, str]:
    subjects = sorted(item["subject"] for item in items)
    random.Random(seed).shuffle(subjects)
    val = set(subjects[:n_val])
    return {s: ("val" if s in val else "train") for s in subjects}


def selection_indices(num_subjects: int, num_modalities: int, n_val_eval: int) -> List[int]:
    stride = max(1, num_subjects // max(1, n_val_eval // num_modalities))
    count = math.ceil(n_val_eval / num_modalities)
    return list(range(0, num_subjects, stride))[:count]


def split_items(
    root: str,
    dataset: str,
    n_val: int,
    seed: int,
    n_val_eval: int,
    split_file: Optional[str] = None,
) -> Dict[str, List[Dict[str, str]]]:
    items = scan_subjects(root, dataset)
    if not items:
        raise RuntimeError(f"no complete subjects under {root}")
    if split_file:
        with open(resolve_path(split_file)) as f:
            lists = json.load(f)
        by_subject = {item["subject"]: item for item in items}
        missing = [s for names in lists.values() for s in names if s not in by_subject]
        if missing:
            raise RuntimeError(f"{len(missing)} subjects of {split_file} are missing under {root}, e.g. {missing[:3]}")
        out = {name: [by_subject[s] for s in names] for name, names in lists.items()}
        out["val"] = sorted(out["selection"] + out["test"], key=lambda item: item["subject"])
        return out
    split = make_split(items, n_val=n_val, seed=seed)
    train = [item for item in items if split[item["subject"]] == "train"]
    val = [item for item in items if split[item["subject"]] == "val"]
    selected = set(selection_indices(len(val), len(MODALITIES[dataset]), n_val_eval))
    return {
        "train": train,
        "val": val,
        "selection": [item for i, item in enumerate(val) if i in selected],
        "test": [item for i, item in enumerate(val) if i not in selected],
    }


def _unit_vector() -> torch.Tensor:
    v = torch.tensor([random.gauss(0.0, 1.0) for _ in range(3)])
    return v / v.norm()


def _rotation(axis: torch.Tensor, degrees: float) -> torch.Tensor:
    theta = math.radians(degrees)
    a = axis / axis.norm()
    k = torch.tensor([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return torch.eye(3) + math.sin(theta) * k + (1 - math.cos(theta)) * (k @ k)


def misregister(volume: torch.Tensor, rotation: torch.Tensor, shift_mm: torch.Tensor) -> torch.Tensor:
    nx, ny, nz = volume.shape[1:]
    centre = torch.tensor([(nx - 1) / 2, (ny - 1) / 2, (DEPTH - 1) / 2])
    grid = torch.stack(
        torch.meshgrid(
            torch.arange(nx, dtype=torch.float32),
            torch.arange(ny, dtype=torch.float32),
            torch.arange(nz, dtype=torch.float32),
            indexing="ij",
        ),
        dim=-1,
    )
    q = (grid - centre - shift_mm) @ rotation + centre
    sample_grid = torch.stack(
        [2 * q[..., 2] / (nz - 1) - 1, 2 * q[..., 1] / (ny - 1) - 1, 2 * q[..., 0] / (nx - 1) - 1], dim=-1
    ).unsqueeze(0)
    out = F.grid_sample(volume.unsqueeze(0), sample_grid, mode="bilinear", padding_mode="zeros", align_corners=True)[0]
    out[..., DEPTH:] = 0
    return out


def thick_slices(volume: torch.Tensor, thickness: int) -> torch.Tensor:
    native = volume[..., :DEPTH].unsqueeze(0)
    low = F.avg_pool3d(
        native, kernel_size=(1, 1, thickness), stride=(1, 1, thickness), ceil_mode=True, count_include_pad=False
    )
    out = torch.zeros_like(volume)
    out[..., :DEPTH] = F.interpolate(low, size=native.shape[2:], mode="trilinear", align_corners=False)[0]
    return out


def perturb_inputs(volumes: List[torch.Tensor]) -> List[torch.Tensor]:
    kind = random.choice(("one", "all", "thick"))
    if kind == "thick":
        thickness = random.randint(2, 5)
        return [thick_slices(v, thickness) for v in volumes]
    out = list(volumes)
    for j in ([random.randrange(len(volumes))] if kind == "one" else range(len(volumes))):
        magnitude = random.uniform(0.0, 4.0)
        rotation = _rotation(_unit_vector(), magnitude * random.choice((-1, 1)))
        out[j] = misregister(out[j], rotation, magnitude * _unit_vector())
    return out


class MissingModalityDataset(Dataset):
    def __init__(
        self,
        items: List[Dict[str, str]],
        dataset: str,
        train: bool = False,
        p_missing: float = 0.0,
        missing_fill: str = "mean",
        p_perturb: float = 0.0,
    ):
        if missing_fill not in ("mean", "zero"):
            raise ValueError(f"missing_fill must be 'mean' or 'zero', got {missing_fill!r}")
        self.items = items
        self.modalities = tuple(MODALITIES[dataset])
        self.train = train
        self.p_missing = float(p_missing)
        self.missing_fill = missing_fill
        self.p_perturb = float(p_perturb)
        self.index = [(i, m) for i in range(len(items)) for m in self.modalities]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict:
        item_index, target = self.index[idx]
        item = self.items[item_index]
        inputs = [load_volume(item[m]) for m in self.modalities if m != target]
        if self.train and self.p_perturb > 0 and random.random() < self.p_perturb:
            inputs = perturb_inputs(inputs)
        prior = sum(inputs) / len(inputs)
        if self.train and self.p_missing > 0 and random.random() < self.p_missing:
            keep = random.sample(range(len(inputs)), random.choice(range(1, len(inputs))))
            prior = sum(inputs[j] for j in keep) / len(keep)
            filler = prior if self.missing_fill == "mean" else torch.zeros_like(prior)
            inputs = [inputs[j] if j in keep else filler for j in range(len(inputs))]
        volume = load_volume(item[target])
        if self.train:
            for axis in (1, 2, 3):
                if random.random() < 0.5:
                    prior = torch.flip(prior, dims=[axis])
                    volume = torch.flip(volume, dims=[axis])
                    inputs = [torch.flip(v, dims=[axis]) for v in inputs]
        return {
            "prior": prior,
            "inputs": torch.stack(inputs, dim=0),
            "target": volume,
            "target_class": torch.tensor(self.modalities.index(target), dtype=torch.long),
            "subject": item["subject"],
            "target_modality": target,
        }

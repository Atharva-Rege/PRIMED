import json
import random

import nibabel as nib
import numpy as np
import pytest
import torch

from primed.data import (
    MissingModalityDataset,
    canonical_contrast,
    load_volume,
    make_split,
    normalize,
    perturb_inputs,
    resolve_path,
    scan_subjects,
    selection_indices,
    split_items,
    to_native,
)


def write_subject(root, subject, modalities, seed):
    rng = np.random.default_rng(seed)
    folder = root / subject
    folder.mkdir(parents=True)
    for modality in modalities:
        array = np.zeros((240, 240, 155), dtype=np.int16)
        array[40:200, 40:200, 20:140] = rng.integers(1, 1000, size=(160, 160, 120), dtype=np.int16)
        nib.save(nib.Nifti1Image(array, np.eye(4)), str(folder / f"{subject}-{modality}.nii"))


@pytest.fixture(scope="module")
def brats_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("brats")
    write_subject(root, "BraTS-GLI-00000-000", ("t1n", "t1c", "t2w", "t2f"), 0)
    return root


def test_normalize_range_and_constant_input():
    array = np.random.default_rng(0).normal(size=(20, 20, 20))
    out = normalize(array)
    assert out.dtype == np.float32 and out.min() == 0.0 and out.max() == 1.0
    assert not normalize(np.ones((4, 4, 4))).any()


def test_selection_matches_the_paper_protocol():
    assert selection_indices(219, 4, 12) == [0, 73, 146]
    assert selection_indices(100, 3, 12) == [0, 25, 50, 75]


def test_split_is_deterministic():
    items = [{"subject": f"s{i:03d}"} for i in range(50)]
    first, second = make_split(items, n_val=10, seed=42), make_split(list(reversed(items)), n_val=10, seed=42)
    assert first == second and sum(v == "val" for v in first.values()) == 10


def test_volume_geometry(brats_root):
    path = str(brats_root / "BraTS-GLI-00000-000" / "BraTS-GLI-00000-000-t1n.nii")
    volume = load_volume(path)
    assert volume.shape == (1, 224, 224, 160)
    assert volume.min() >= 0 and volume.max() <= 1
    assert not volume[..., 155:].any()
    native = to_native(volume[0].numpy())
    assert native.shape == (240, 240, 155) and np.array_equal(native[8:-8, 8:-8, :], volume[0, :, :, :155].numpy())


def test_scan_and_split(brats_root):
    items = scan_subjects(str(brats_root), "brats")
    assert len(items) == 1 and set(items[0]) == {"subject", "t1n", "t1c", "t2w", "t2f"}
    splits = split_items(str(brats_root), "brats", n_val=1, seed=42, n_val_eval=4)
    assert len(splits["val"]) == 1 and len(splits["selection"]) == 1 and not splits["test"]


def test_contrast_names():
    assert canonical_contrast("T1c", "brats") == "t1c"
    assert canonical_contrast("FLAIR", "brats") == "t2f"
    assert canonical_contrast("t1", "brats") == "t1n"
    assert canonical_contrast("pd", "ixi") == "PD"
    with pytest.raises(ValueError):
        canonical_contrast("pd", "brats")


def test_released_split_files():
    for name, sizes in (("splits/brats_gli.json", (1032, 3, 216)), ("splits/ixi.json", (477, 4, 96))):
        with open(resolve_path(name)) as f:
            lists = json.load(f)
        assert tuple(len(lists[k]) for k in ("train", "selection", "test")) == sizes
        assert len(set().union(*map(set, lists.values()))) == sum(sizes)


def test_pinned_split(brats_root, tmp_path):
    split_file = tmp_path / "split.json"
    split_file.write_text(json.dumps({"train": [], "selection": ["BraTS-GLI-00000-000"], "test": []}))
    splits = split_items(str(brats_root), "brats", n_val=1, seed=42, n_val_eval=4, split_file=str(split_file))
    assert [i["subject"] for i in splits["selection"]] == ["BraTS-GLI-00000-000"] and not splits["train"]
    split_file.write_text(json.dumps({"train": ["BraTS-GLI-99999-000"], "selection": [], "test": []}))
    with pytest.raises(RuntimeError):
        split_items(str(brats_root), "brats", n_val=1, seed=42, n_val_eval=4, split_file=str(split_file))


def test_dataset_item(brats_root):
    dataset = MissingModalityDataset(scan_subjects(str(brats_root), "brats"), "brats")
    assert len(dataset) == 4
    item = dataset[1]
    assert item["target_modality"] == "t1c" and int(item["target_class"]) == 1
    assert item["inputs"].shape == (3, 1, 224, 224, 160)
    assert torch.equal(item["prior"], item["inputs"].sum(0) / 3)


def test_missing_inputs_are_filled_with_the_prior(brats_root):
    random.seed(0)
    dataset = MissingModalityDataset(scan_subjects(str(brats_root), "brats"), "brats", train=True, p_missing=1.0)
    item = dataset[0]
    filled = [torch.equal(v, item["prior"]) for v in item["inputs"]]
    assert 1 <= sum(filled) <= 2


def test_perturbation_preserves_geometry(brats_root):
    items = scan_subjects(str(brats_root), "brats")
    clean = MissingModalityDataset(items, "brats")[0]
    volumes = list(clean["inputs"])
    for seed in range(3):
        random.seed(seed)
        perturbed = perturb_inputs(volumes)
        assert len(perturbed) == 3 and all(v.shape == (1, 224, 224, 160) for v in perturbed)
        assert all(not v[..., 155:].any() for v in perturbed)
        assert any(not torch.equal(p, v) for p, v in zip(perturbed, volumes))
    random.seed(0)
    item = MissingModalityDataset(items, "brats", train=True, p_perturb=1.0)[0]
    assert item["inputs"].shape == clean["inputs"].shape

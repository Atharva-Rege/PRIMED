import argparse
import os

import nibabel as nib
import numpy as np
import torch

from primed import load_model
from primed.data import MODALITIES, canonical_contrast, dataset_for_classes, load_volume, to_native


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--inputs", nargs="+", required=True, metavar="CONTRAST=PATH")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not args.output.endswith((".nii", ".nii.gz")):
        parser.error("--output must end with .nii or .nii.gz")
    model = load_model(args.weights, args.device)
    dataset = dataset_for_classes(model.num_classes)
    modalities = MODALITIES[dataset]
    try:
        target = canonical_contrast(args.target, dataset)
        paths = {}
        for entry in args.inputs:
            name, separator, path = entry.partition("=")
            if not separator or not path:
                raise ValueError(f"--inputs entries must look like CONTRAST=PATH, got {entry!r}")
            paths[canonical_contrast(name, dataset)] = path
    except ValueError as error:
        parser.error(str(error))
    if target in paths:
        parser.error(f"the target {target} cannot also be an input")
    siblings = [m for m in modalities if m != target]
    available = [m for m in siblings if m in paths]
    if not available:
        parser.error("at least one acquired contrast is required")

    volumes = {m: load_volume(paths[m]).unsqueeze(0).to(args.device) for m in available}
    prior = sum(volumes[m] for m in available) / len(available)
    inputs = torch.stack([volumes.get(m, prior) for m in siblings], dim=1)
    target_class = torch.tensor([modalities.index(target)], device=args.device)
    prediction = model.synthesize(prior, inputs, target_class)

    reference = nib.load(paths[available[0]])
    header = reference.header.copy()
    header.set_data_dtype(np.float32)
    output = to_native(prediction[0, 0].float().cpu().numpy())
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    nib.save(nib.Nifti1Image(output, reference.affine, header), args.output)
    print(f"{target} synthesised from {', '.join(available)} -> {args.output}")


if __name__ == "__main__":
    main()

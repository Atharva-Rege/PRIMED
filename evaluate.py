import argparse
import itertools
import json
import os
import time
from collections import defaultdict

import torch
import yaml

from primed import load_model
from primed.data import MODALITIES, load_segmentation, load_volume, read_subject_list, scan_subjects, split_items
from primed.metrics import score


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else float("nan")


def summarize(rows, modalities):
    by_target = defaultdict(list)
    for row in rows:
        by_target[row["target"]].append(row)
    print(f"\n{'target':<8}{'PSNR':>9}{'brain PSNR':>12}{'SSIM':>9}{'cases':>7}")
    for name, group in [(m, by_target[m]) for m in modalities if by_target[m]] + [("average", rows)]:
        print(
            f"{name:<8}{mean(r['psnr'] for r in group):>9.3f}{mean(r['psnr_brain'] for r in group):>12.3f}"
            f"{mean(r['ssim'] for r in group):>9.4f}{len(group):>7}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="test", choices=["test", "selection", "val", "all"])
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--subjects", default=None)
    parser.add_argument("--num_inputs", type=int, default=None)
    parser.add_argument("--fill", default="mean", choices=["mean", "zero"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    dataset = cfg["dataset"]
    modalities = MODALITIES[dataset]
    num_inputs = len(modalities) - 1 if args.num_inputs is None else args.num_inputs
    if not 1 <= num_inputs <= len(modalities) - 1:
        parser.error(f"--num_inputs must be between 1 and {len(modalities) - 1} for {dataset}")

    model = load_model(args.weights, args.device)
    if model.num_classes != len(modalities):
        parser.error(f"the weights expect {model.num_classes} contrasts, {dataset} has {len(modalities)}")

    data_root = args.data_root or cfg["data_root"]
    if args.split == "all":
        items = scan_subjects(data_root, dataset)
    else:
        splits = split_items(
            data_root, dataset, cfg["n_val"], cfg.get("split_seed", 42), cfg["n_val_eval"], cfg.get("split_file")
        )
        items = splits[args.split]
    if args.subjects:
        wanted = set(read_subject_list(args.subjects))
        items = [item for item in items if item["subject"] in wanted]
        if len(items) != len(wanted):
            parser.error(f"{len(wanted) - len(items)} subjects of {args.subjects} were not found under {data_root}")
    if args.limit:
        items = items[: args.limit]
    if not items:
        parser.error("no subjects to evaluate")
    print(f"{len(items)} subjects, {len(modalities)} targets each, {num_inputs} acquired inputs", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    rows, start = [], time.time()
    with open(args.output, "w") as handle:
        for n, item in enumerate(items, 1):
            volumes = {m: load_volume(item[m]).unsqueeze(0).to(args.device) for m in modalities}
            segmentation = load_segmentation(item["seg"]).to(args.device) if "seg" in item else None
            for target in modalities:
                siblings = [m for m in modalities if m != target]
                target_class = torch.tensor([modalities.index(target)], device=args.device)
                for keep in itertools.combinations(range(len(siblings)), num_inputs):
                    available = [volumes[siblings[j]] for j in keep]
                    prior = sum(available) / len(available)
                    filler = prior if args.fill == "mean" else torch.zeros_like(prior)
                    inputs = [volumes[s] if j in keep else filler for j, s in enumerate(siblings)]
                    prediction = model.synthesize(prior, torch.stack(inputs, dim=1), target_class)
                    row = {
                        "subject": item["subject"],
                        "target": target,
                        "inputs": [siblings[j] for j in keep],
                        **score(prediction[0, 0], volumes[target][0, 0], segmentation),
                    }
                    rows.append(row)
                    handle.write(json.dumps(row) + "\n")
            handle.flush()
            if n % 10 == 0:
                print(f"{n}/{len(items)} subjects, {time.time() - start:.0f} s", flush=True)
    summarize(rows, modalities)


if __name__ == "__main__":
    main()

import argparse
import math
import os
import random

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from primed import PRIMED, load_weights, save_weights
from primed.data import MODALITIES, MissingModalityDataset, split_items


class LossModule(torch.nn.Module):
    def __init__(self, model: PRIMED):
        super().__init__()
        self.model = model

    def forward(self, batch):
        return self.model(batch["prior"], batch["inputs"], batch["target"], batch["target_class"])


@torch.no_grad()
def validate(model: PRIMED, cases, device):
    values, per_modality = [], {}
    for case in cases:
        prior = case["prior"].unsqueeze(0).to(device)
        inputs = case["inputs"].unsqueeze(0).to(device)
        target = case["target"].unsqueeze(0).to(device)
        target_class = case["target_class"].view(1).to(device)
        prediction = model.synthesize(prior, inputs, target_class)
        depth = min(prediction.shape[-1], 155)
        prediction, target = prediction[..., :depth], target[..., :depth]
        mask = (target > 0).float()
        mse = (((prediction - target) ** 2) * mask).sum() / mask.sum().clamp_min(1.0)
        value = (10 * torch.log10(1.0 / mse.clamp_min(1e-12))).item()
        values.append(value)
        per_modality.setdefault(case["target_modality"], []).append(value)
    return sum(values) / len(values), {m: sum(v) / len(v) for m, v in per_modality.items()}


@torch.no_grad()
def update_ema(ema, module, decay):
    for k, v in module.state_dict().items():
        if v.dtype.is_floating_point:
            ema[k].mul_(decay).add_(v.detach().float(), alpha=1.0 - decay)
        else:
            ema[k].copy_(v)


def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        device = f"cuda:{local_rank}"
    else:
        local_rank, rank, world = 0, 0, 1
        device = "cuda" if torch.cuda.is_available() else "cpu"
    is_main = rank == 0
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed + rank)
    random.seed(seed + rank)

    dataset = cfg["dataset"]
    modalities = MODALITIES[dataset]
    model = PRIMED(
        num_classes=len(modalities),
        num_inputs=len(modalities) - 1,
        channels=cfg.get("channels", (64, 128, 128, 256, 256)),
        num_res_blocks=cfg.get("num_res_blocks", 3),
    ).to(device)
    model.train()

    output_dir = cfg["output_dir"]
    last_path = os.path.join(output_dir, "last.pt")
    best_path = os.path.join(output_dir, "best.safetensors")
    resume = cfg.get("resume", True) and os.path.exists(last_path)
    if cfg.get("init_weights") and not resume:
        model.net.load_state_dict(load_weights(cfg["init_weights"]), strict=True)
        if is_main:
            print(f"initialised from {cfg['init_weights']}", flush=True)

    splits = split_items(
        cfg["data_root"], dataset, cfg["n_val"], cfg.get("split_seed", 42), cfg["n_val_eval"], cfg.get("split_file")
    )
    train_set = MissingModalityDataset(
        splits["train"],
        dataset,
        train=True,
        p_missing=cfg.get("p_missing", 0.0),
        missing_fill=cfg.get("missing_fill", "mean"),
        p_perturb=cfg.get("p_perturb", 0.0),
    )
    sampler = DistributedSampler(train_set, shuffle=True, seed=seed) if distributed else None
    loader = DataLoader(
        train_set,
        batch_size=cfg["batch_size"],
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=cfg.get("num_workers", 8),
        drop_last=True,
        pin_memory=True,
    )
    val_cases = None
    if is_main:
        selection = MissingModalityDataset(splits["selection"], dataset)
        val_cases = [selection[i] for i in range(min(cfg["n_val_eval"], len(selection)))]

    loss_module = LossModule(model)
    trainer = DistributedDataParallel(loss_module, device_ids=[local_rank]) if distributed else loss_module
    trainable = [p for p in model.parameters() if p.requires_grad]
    base_lr = cfg["learning_rate"]
    optimizer = torch.optim.AdamW(trainable, lr=base_lr, weight_decay=cfg.get("weight_decay", 0.0))

    if is_main:
        os.makedirs(output_dir, exist_ok=True)
    ema_decay = cfg.get("ema_decay", 0.0)
    ema = None
    if is_main and ema_decay > 0:
        ema = {k: v.detach().clone().float() for k, v in model.net.state_dict().items()}

    max_steps = cfg["max_steps"]
    eval_every = cfg.get("eval_every", 500)
    log_every = cfg.get("log_every", 200)
    save_every = cfg.get("save_every", 2000)
    patience = cfg.get("early_stop_patience")
    min_delta = cfg.get("early_stop_min_delta", 0.0)
    grad_clip = cfg.get("grad_clip", 1.0)
    use_amp = cfg.get("amp", True) and device != "cpu"
    warmup_steps = cfg.get("warmup_steps", 0)
    grad_accum = max(1, int(cfg.get("grad_accum", 1)))
    lr_min_ratio = float(cfg.get("lr_min", base_lr)) / base_lr
    lr_decay_steps = int(cfg.get("lr_decay_steps", max_steps))

    step, best_psnr, since_improve, epoch = 0, -1.0, 0, 0
    if resume:
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.net.load_state_dict(state["model"])
        if state.get("optimizer") is not None:
            optimizer.load_state_dict(state["optimizer"])
        epoch = int(state.get("epoch", 0))
        step = int(state["step"])
        best_psnr = float(state["best_psnr"])
        since_improve = int(state["since_improve"])
        if ema is not None and state.get("ema") is not None:
            ema = {k: v.to(device).float() for k, v in state["ema"].items()}
        if is_main:
            print(f"resumed from {last_path} at step {step}", flush=True)

    if is_main:
        print(f"world {world}, {len(splits['train'])} training subjects, {len(val_cases)} validation cases", flush=True)

    running, micro, stop_training = 0.0, 0, False
    optimizer.zero_grad(set_to_none=True)
    while step < max_steps and not stop_training:
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch += 1
        for batch in loader:
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
            if warmup_steps and step < warmup_steps:
                scale = (step + 1) / warmup_steps
            elif lr_min_ratio < 1.0 and step < lr_decay_steps:
                progress = (step - warmup_steps) / max(1, lr_decay_steps - warmup_steps)
                scale = lr_min_ratio + 0.5 * (1.0 - lr_min_ratio) * (1.0 + math.cos(math.pi * progress))
            else:
                scale = lr_min_ratio
            for group in optimizer.param_groups:
                group["lr"] = base_lr * scale
            if use_amp:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = trainer(batch)
            else:
                loss = trainer(batch)
            (loss / grad_accum).backward()
            running += loss.item()
            micro += 1
            if micro % grad_accum != 0:
                continue
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                update_ema(ema, model.net, ema_decay)
            step += 1

            if is_main and step % log_every == 0:
                print(f"step {step}/{max_steps} loss {running / (log_every * grad_accum):.5f}", flush=True)
                running = 0.0

            stop = False
            if step % eval_every == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if is_main:
                    model.eval()
                    raw_psnr, raw_modality = validate(model, val_cases, device)
                    ema_psnr, ema_modality = -1.0, {}
                    if ema is not None:
                        backup = {k: v.detach().clone() for k, v in model.net.state_dict().items()}
                        model.net.load_state_dict({k: ema[k].to(v.dtype) for k, v in backup.items()})
                        ema_psnr, ema_modality = validate(model, val_cases, device)
                        model.net.load_state_dict(backup)
                    model.train()
                    if ema_psnr > raw_psnr:
                        value, source, weights, per_modality = ema_psnr, "ema", ema, ema_modality
                    else:
                        value, source, weights, per_modality = raw_psnr, "raw", model.net.state_dict(), raw_modality
                    if value > best_psnr + min_delta:
                        best_psnr, since_improve = value, 0
                        save_weights(weights, best_path, {"step": step, "val_psnr": value, "weights": source})
                        tag = " (best, saved)"
                    else:
                        since_improve += 1
                        tag = ""
                    breakdown = "  ".join(f"{m} {v:.2f}" for m, v in sorted(per_modality.items()))
                    print(
                        f"step {step} val_psnr {value:.3f} ({source}; raw {raw_psnr:.3f}, ema {ema_psnr:.3f}) "
                        f"best {best_psnr:.3f}{tag} | {breakdown}",
                        flush=True,
                    )
                    if patience and since_improve >= patience:
                        print(f"early stop: best val_psnr {best_psnr:.3f}", flush=True)
                        stop = True
                if distributed:
                    flag = torch.tensor([1.0 if stop else 0.0], device=device)
                    dist.broadcast(flag, src=0)
                    stop = flag.item() > 0
            if is_main and step % save_every == 0:
                torch.save(
                    {
                        "model": model.net.state_dict(),
                        "ema": {k: v.cpu() for k, v in ema.items()} if ema is not None else None,
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "step": step,
                        "best_psnr": best_psnr,
                        "since_improve": since_improve,
                        "config": cfg,
                    },
                    last_path,
                )
            if stop or step >= max_steps:
                stop_training = True
                break

    if is_main:
        save_weights(model.net.state_dict(), os.path.join(output_dir, "final.safetensors"), {"step": step, "weights": "raw"})
        if ema is not None:
            save_weights(ema, os.path.join(output_dir, "final_ema.safetensors"), {"step": step, "weights": "ema"})
        print(f"done at step {step}; best val_psnr {best_psnr:.3f}", flush=True)
    if distributed:
        dist.barrier(device_ids=[local_rank])
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    main(parser.parse_args().config)

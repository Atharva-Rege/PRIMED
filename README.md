<h1 align="center">
<strong>When Iteration Cannot Help: Single-Pass Missing-Modality MRI Synthesis and What Actually Limits It</strong>
</h1>

<div align="center">

<a href="https://git.io/typing-svg">
<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&pause=1000&color=147B82&center=true&width=620&lines=Synthesise+any+missing+MRI+contrast+from+the+others.;One+network+evaluation+per+volume.;28.56+dB+on+BraTS-GLI+in+0.15+s."
alt="Typing SVG"
style="margin-bottom:-10px; display:block;" />
</a>

[![WACV 2027](https://img.shields.io/badge/WACV_2027-Applications_Track-8C1B13?style=for-the-badge)](#overview)
[![Weights](https://img.shields.io/badge/HF-Weights-AECBFA?style=for-the-badge&logo=huggingface&logoColor=FFCC00&labelColor=grey)](https://huggingface.co/adidukre/PRIMED)
[![Data](https://img.shields.io/badge/Data-BraTS_2023_|_IXI-1F6FEB?style=for-the-badge)](#️-data)
[![License](https://img.shields.io/badge/License-CC_BY--NC_4.0-green?style=for-the-badge)](LICENSE)
[![Visitors](https://api.visitorbadge.io/api/combined?path=https%3A%2F%2Fgithub.com%2Fadinathdukre%2FPRIMED&label=Views&countColor=%23147b82&style=for-the-badge)](https://visitorbadge.io/status?path=https%3A%2F%2Fgithub.com%2Fadinathdukre%2FPRIMED)

<h3>🤗 <a href="https://huggingface.co/adidukre/PRIMED">Weights</a> &nbsp;|&nbsp; 🧠 <a href="#-method">Method</a> &nbsp;|&nbsp; 🏆 <a href="#-results">Results</a> &nbsp;|&nbsp; ⚡ <a href="#-inference">Inference</a> &nbsp;|&nbsp; 🏋️ <a href="#️-training">Training</a></h3>

**[Atharva Atul Rege](https://github.com/Atharva-Rege)<sup>†</sup>, [Adinath Madhavrao Dukre](https://github.com/adinathdukre)<sup>*</sup>, Imran Razzak<sup>*</sup>**

<sup>*</sup>Mohamed bin Zayed University of Artificial Intelligence, Abu Dhabi, UAE &nbsp;
<sup>†</sup>National Institute of Technology Karnataka, Surathkal, India

<img src="https://raw.githubusercontent.com/genmilab/VGS-Decoding/main/docs/assets/genmilab-logo.png" alt="GenMI Lab" height="60"/>

</div>

## 🔥 News
- **[Oct 2026]** 🚀 Code, subject-level splits and the four PRIMED checkpoints are released.
- 📄 The paper is accepted at **WACV 2027** (Applications track).

## Overview
Official code for **PRIMED** (**PR**ior-**I**nformed **M**issing-modality synthesis by **E**ndpoint **D**isplacement).

- **Task.** Multi-parametric brain MRI often misses one contrast. PRIMED predicts it from the contrasts that were acquired: any of T1, T1c, T2 and FLAIR from the other three, with one set of weights.
- **Idea.** If a sampler starts from a prior computed from the acquired contrasts, every state it visits is again a function of those contrasts. Further network evaluations can reduce integration error, but they cannot add information about the missing contrast.
- **Method.** PRIMED is trained only at that source state and evaluated once, with no noise schedule and no solver.
- **Result.** 28.56 dB PSNR and 0.944 SSIM in 0.152 s per volume on 216 held-out BraTS 2023 glioma subjects.

<p align="center">
<img src="./assets/pipeline.png" alt="PRIMED pipeline" width="100%"/>
<br/>
<em><b>Fig. 1.</b> The three acquired contrasts are mapped by a fixed Haar transform W and averaged into the informed prior g(c). One pass of v<sub>θ</sub>, conditioned on the target contrast, returns the displacement D̂; subtracting it before inverting W gives the volume. Top: the path from s = 1 to s = 0; PRIMED evaluates the network only at the source, s = 1.</em>
</p>

## 📖 Contents
- [🧠 Method](#-method)
- [🏆 Results](#-results)
- [🔍 Qualitative Results](#-qualitative-results)
- [⛏️ Installation](#️-installation)
- [🧩 Pretrained Weights](#-pretrained-weights)
- [🗃️ Data](#️-data)
- [⚡ Inference](#-inference)
- [📊 Evaluation](#-evaluation)
- [🏋️ Training](#️-training)
- [🧪 Tests](#-tests)
- [🗂️ Repository Structure](#️-repository-structure)
- [⚠️ Limitations](#️-limitations)
- [📚 Acknowledgments](#-acknowledgments)
- [📨 Contact](#-contact)
- [📜 License](#-license)

## 🧠 Method

1. **Representation.** A fixed, lossless one-level 3D Haar transform W maps each 224 × 224 × 160 volume to eight subbands of 112 × 112 × 80. The low-pass band is divided by 3.
2. **Informed prior.** g(c) is the mean of the encoded acquired contrasts. It depends on the inputs alone, so the network can compute it exactly.
3. **Network.** v<sub>θ</sub> is a five-level 3D U-Net (MONAI `DiffusionModelUNet`, 97.7M parameters), convolutional except for one self-attention block in its bottleneck. Its 48 input channels are g(c) twice, a learned 8-channel embedding of the target contrast and the three encoded acquired contrasts.
4. **Training.** Mean squared error on the displacement D = g(c) − W y at the source state s = 1. No noise is injected.
5. **Inference.** One network evaluation followed by the inverse transform:

$$\hat{y} = W^{-1}\big(g(c) - v_\theta(g(c), 1, c)\big), \qquad g(c) = \tfrac{1}{3}\textstyle\sum_{m} W c_m .$$

## 🏆 Results

Whole-volume PSNR (dB) per target contrast and SSIM on the 216 held-out BraTS 2023 GLI subjects (864 cases), with one network for all four targets, as reported in the paper:

<div align="center">

| Networks | NFE | Params | s / volume | T1 | T1c | T2 | FLAIR | Avg | SSIM |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 1 | 97.7M | 0.152 | 29.372 | 28.137 | 28.953 | 27.778 | **28.560** | **0.9439** |

</div>

NFE is the number of network evaluations per volume. Timing is per volume on one NVIDIA RTX PRO 6000 Blackwell, including the inverse transform.

<details>
<summary><strong>Inside the brain and the tumour regions</strong></summary>

PSNR (dB), averaged over the four targets:

| Brain | Whole tumour | Enhancing tumour | Necrotic core | Oedema |
|:---:|:---:|:---:|:---:|:---:|
| 21.364 | 20.173 | 18.661 | 19.810 | 21.838 |

How much of each target the inputs determine varies by contrast and region. The recoverability statistic ρ is 0.632–0.740 over the whole brain, but 0.355, 0.188, 0.571 and 0.508 inside enhancing tumour for the T1, T1c, T2 and FLAIR targets.

</details>

**Unseen cohorts.** The GLI-trained model applied unchanged, and a second model trained on IXI:

<div align="center">

| Cohort | Subjects | PSNR (dB) |
|---|:---:|:---:|
| BraTS-MEN, training release | 60 | 26.555 |
| BraTS-MEN, validation release | 141 | 26.413 |
| BraTS-PED | 99 | 26.398 |
| IXI (trained on IXI) | 96 | 31.436 |

</div>

**Fewer acquired contrasts.** `primed_brats_avail` is fine-tuned for 8k steps with random input availability; empty input slots receive the prior. Mean PSNR (dB) over all four targets and every subset of the inputs:

<div align="center">

| Acquired contrasts | `primed_brats`, zero-shot | `primed_brats_avail` |
|:---:|:---:|:---:|
| three | **28.56** | 28.36 |
| two | 25.40 | **27.35** |
| one | 22.37 | **25.97** |

</div>

**Misregistered or thick-slice inputs.** `primed_brats_robust` is fine-tuned for 8k steps with these perturbations applied to the inputs. Mean PSNR (dB):

<div align="center">

| Inputs | `primed_brats` | `primed_brats_robust` |
|---|:---:|:---:|
| registered | **28.56** | 28.44 |
| one input misregistered by 1 mm, 1° | 27.67 | **27.89** |
| one input misregistered by 2 mm, 2° | 26.60 | **27.28** |
| one input misregistered by 4 mm, 4° | 25.10 | **26.47** |
| all inputs misregistered by 2 mm, 2° | 23.88 | **24.71** |
| inputs from 3 mm slices | 26.67 | **27.58** |
| inputs from 5 mm slices | 24.66 | **26.82** |

</div>

## 🔍 Qualitative Results

<p align="center">
<img src="./assets/qualitative.png" alt="Qualitative results on BraTS-GLI" width="85%"/>
<br/>
<em><b>Fig. 2.</b> Held-out BraTS-GLI cases for the T1, T1c and FLAIR targets, one per row: the three acquired inputs, PRIMED and the acquired target. The red square marks the tumour region each case was selected on.</em>
</p>

<p align="center">
<img src="./assets/qualitative_ixi.png" alt="Qualitative results on IXI" width="60%"/>
<br/>
<em><b>Fig. 3.</b> IXI, with PRIMED trained on it from scratch. One held-out case for each of two target contrasts: the two acquired inputs, PRIMED and the acquired target.</em>
</p>

## ⛏️ Installation

```bash
git clone https://github.com/adinathdukre/PRIMED.git
cd PRIMED
conda create -n primed python=3.10 -y && conda activate primed
pip install -e .
```

The code needs Python 3.9 or later, PyTorch 2.1 or later, MONAI 1.4 or later and einops. It was tested with MONAI 1.4.0.

## 🧩 Pretrained Weights

Download the checkpoints from [Hugging Face](https://huggingface.co/adidukre/PRIMED):

```bash
pip install -U huggingface_hub
hf download adidukre/PRIMED --local-dir weights
cd weights && sha256sum -c SHA256SUMS
```

| File | Training data | Synthesises |
|---|---|---|
| `primed_brats.safetensors` | BraTS 2023 GLI, 1032 subjects | any of T1n, T1c, T2w, T2f from the other three |
| `primed_brats_avail.safetensors` | as `primed_brats`, plus random input availability | any of the four from one, two or three acquired contrasts |
| `primed_brats_robust.safetensors` | as `primed_brats`, plus misregistration and thick-slice augmentation | any of the four from three possibly misregistered or thick-slice contrasts |
| `primed_ixi.safetensors` | IXI, 477 subjects | any of T1, T2, PD from the other two |

Each file holds the network weights only. The architecture is read from the weights, so no config is needed for inference.

## 🗃️ Data

**BraTS.** Download the BraTS 2023 adult glioma (GLI) training data from the [challenge page](https://www.synapse.org/Synapse:syn51156910/wiki/621282) and keep its folder layout:

```text
/path/to/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/
└── BraTS-GLI-00000-000/
    ├── BraTS-GLI-00000-000-t1n.nii.gz
    ├── BraTS-GLI-00000-000-t1c.nii.gz
    ├── BraTS-GLI-00000-000-t2w.nii.gz
    ├── BraTS-GLI-00000-000-t2f.nii.gz
    └── BraTS-GLI-00000-000-seg.nii.gz
```

Volumes must be co-registered, skull-stripped and on the 1 mm BraTS grid of 240 × 240 × 155 voxels, as BraTS releases them. Preprocessing runs on the fly:
- clip to the 0.1 and 99.9 percentiles of the volume;
- rescale to [0, 1];
- pad the depth from 155 to 160 slices;
- crop the in-plane axes from 240 to 224.

**IXI.** Arrange the [IXI dataset](https://brain-development.org/ixi-dataset/) as `/path/to/IXI/<subject>/<subject>-{T1,T2,PD}.nii.gz`, with the three contrasts co-registered and resampled to the same 240 × 240 × 155, 1 mm grid. The IXI volumes are not skull-stripped.

`splits/` holds the subject lists used in the paper:

```text
splits/
├── brats_gli.json           # 1032 train / 219 val; 3 val subjects select checkpoints, the other 216 are the test set
├── brats_men_60.txt         # BraTS-MEN training release, first 60 subjects
├── brats_men_val_141.txt    # BraTS-MEN validation release
├── brats_ped_99.txt         # BraTS-PED training release
└── ixi.json                 # 477 train / 100 val; 4 val subjects select checkpoints, the other 96 are the test set
```

## ⚡ Inference

Synthesise a missing contrast from NIfTI files:

```bash
python synthesize.py \
  --weights weights/primed_brats.safetensors \
  --target t1c \
  --inputs t1n=/path/to/subject-t1n.nii.gz t2w=/path/to/subject-t2w.nii.gz t2f=/path/to/subject-t2f.nii.gz \
  --output /path/to/subject-t1c-synthesised.nii.gz
```

| Model | Contrast names (case-insensitive) |
|---|---|
| BraTS | `t1n`, `t1c`, `t2w`, `t2f` (aliases such as `t1ce` and `flair` also work) |
| IXI | `t1`, `t2`, `pd` |

The output is on the grid of the inputs, with intensities in the normalised [0, 1] range. With `primed_brats_avail.safetensors`, any one or two acquired contrasts may be given; the empty input slots are filled with their mean.

<details>
<summary><strong>From Python</strong></summary>

```python
import torch
from primed import load_model
from primed.data import load_volume

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model("weights/primed_brats.safetensors", device)
inputs = torch.stack(
    [load_volume(f"/path/to/subject-{m}.nii.gz") for m in ("t1n", "t2w", "t2f")]
).unsqueeze(0).to(device)
t1c = model.synthesize(inputs.mean(dim=1), inputs, torch.tensor([1], device=device))
```

The target class follows the contrast order T1n, T1c, T2w, T2f for BraTS and T1, T2, PD for IXI.

</details>

> [!NOTE]
> Inference needs about 5 GB of GPU memory. It also runs on the CPU (`--device cpu`), at about 25–30 s per volume with 16 threads.

## 📊 Evaluation

Score the 216 held-out BraTS-GLI subjects and print the per-target table:

```bash
python evaluate.py \
  --config configs/brats_primed.yaml \
  --weights weights/primed_brats.safetensors \
  --output results/primed_brats_test.jsonl
```

Per-case results are written as JSON lines: whole-volume PSNR on the 224 × 224 × 155 crop, PSNR inside the brain and inside each tumour region, and 3D SSIM with a 7³ window.

| Option | Effect |
|---|---|
| `--num_inputs 2` / `--num_inputs 1` | every subset of fewer acquired contrasts (use `primed_brats_avail.safetensors`) |
| `--fill zero` | fill empty input slots with zeros instead of the mean of the acquired contrasts |
| `--split all --data_root DIR` | score another cohort |
| `--subjects FILE` | restrict scoring to the subjects listed in a file |
| `--limit N` | score only the first N subjects |
| `--device cpu` | run on the CPU |

For example, the 60 BraTS-MEN subjects:

```bash
python evaluate.py --config configs/brats_primed.yaml \
  --weights weights/primed_brats.safetensors \
  --split all --data_root /path/to/BraTS-MEN-Train --subjects splits/brats_men_60.txt \
  --output results/primed_brats_men60.jsonl
```

For IXI, use `--config configs/ixi_primed.yaml` with `primed_ixi.safetensors`.

## 🏋️ Training

Set `data_root`, `output_dir` and, where used, `init_weights` in the config, then run from the repository root:

```bash
python train.py --config configs/brats_primed.yaml
torchrun --nproc_per_node=4 train.py --config configs/brats_primed.yaml
```

| Config | Initialisation | Augmentation | Released as |
|---|---|---|---|
| `configs/brats_primed.yaml` | `init_weights` (see below) | random flips | `primed_brats` |
| `configs/brats_primed_avail.yaml` | `primed_brats.safetensors` | flips, random input availability | `primed_brats_avail` |
| `configs/brats_primed_robust.yaml` | `primed_brats.safetensors` | flips, misregistration and thick slices | `primed_brats_robust` |
| `configs/ixi_primed.yaml` | random | random flips | `primed_ixi` |

| Setting | Value |
|---|---|
| Optimiser | AdamW, weight decay 0, gradient clip 1.0 |
| Batch | 2, gradient accumulation 2 (effective batch `batch_size × grad_accum × GPUs`) |
| Schedule | 2k warm-up, cosine decay; see each config for rates and steps |
| Precision | bf16 mixed precision (`amp: false` on GPUs without bf16) |
| EMA | decay 0.9999 |
| Selection | best validation PSNR on 12 held-out cases from the checkpoint-selection subjects |

Training writes `best.safetensors` (best validation PSNR, raw or EMA weights), `last.pt` (state to resume from), and `final.safetensors` / `final_ema.safetensors` to `output_dir`. It needs about 38 GB of GPU memory per process at batch size 2.

> [!IMPORTANT]
> `configs/brats_primed.yaml` fine-tunes for up to 20k steps from a network of the same architecture trained over the full interpolant path (Sec. 4.1 of the paper). That initialisation is not released, so `primed_brats` cannot be retrained exactly from this repository; it is that run's `best.safetensors`, raw weights at step 15k. For training from scratch, follow the longer schedule of `configs/ixi_primed.yaml`. The two fine-tunes are released with their final raw weights.

## 🧪 Tests

```bash
pip install -e ".[test]" && python -m pytest -q
```

The tests cover the Haar transform, the model and the data pipeline.

## 🗂️ Repository Structure

```text
PRIMED/
├── primed/
│   ├── wavelet.py        # 3D Haar transform
│   ├── network.py        # 3D U-Net with target-contrast conditioning
│   ├── model.py          # PRIMED objective and single-pass synthesis
│   ├── data.py           # preprocessing, splits and augmentation
│   ├── metrics.py        # PSNR and 3D SSIM as used in the paper
│   └── weights.py        # loading and saving weights
├── train.py              # training, single GPU or torchrun
├── evaluate.py           # evaluation on a split or a cohort
├── synthesize.py         # synthesis from NIfTI files
├── configs/              # training configs of the four released models
├── splits/               # subject lists of every split and cohort in the paper
├── tests/                # unit tests (pytest)
└── assets/               # README figures
```

## ⚠️ Limitations

- A synthesised T1c is a conditional mean: it cannot recover contrast enhancement that the acquired contrasts do not reveal, and it may show enhancement that is not there. It should not replace an acquired post-contrast scan where enhancement matters for diagnosis.
- Whether a synthetic substitute helps a downstream model depends on the target and on the model (Sec. 4.5 of the paper). Validate it against the model that will consume it.
- The models assume co-registered inputs on the BraTS grid. For inputs that may be misregistered or acquired with thick slices, use `primed_brats_robust.safetensors`.
- The models were trained on BraTS and IXI data only.

## 📚 Acknowledgments

The network is MONAI's `DiffusionModelUNet` ([MONAI](https://github.com/Project-MONAI/MONAI), Apache 2.0). The models were trained on the [BraTS 2023](https://www.synapse.org/Synapse:syn51156910/wiki/621282) and [IXI](https://brain-development.org/ixi-dataset/) data; we thank the BraTS organisers, the IXI team and the data contributors. Please also cite the BraTS challenge papers when using the BraTS data.

## 📨 Contact
For questions or collaboration, please open an [issue](https://github.com/adinathdukre/PRIMED/issues) or reach out to [Adinath Madhavrao Dukre](https://github.com/adinathdukre).

## 📜 License

The code and model weights are released under CC BY-NC 4.0 ([LICENSE](LICENSE)), for non-commercial use.

> [!IMPORTANT]
> PRIMED is intended for research only. It is not validated for clinical use and must not inform any diagnostic or treatment decision.

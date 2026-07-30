# Environment Notes — BrepMFR Local Adaptation

**Date:** 2026-07-30
**Machine:** Intel Core Ultra 7 270K Plus / NVIDIA RTX 5070 12GB / 32GB RAM / Windows 11 (PowerShell)
**Conda env:** `brepmfr`

Local reproduction of the BrepMFR baseline (machining feature recognition on B-rep models) on a modern stack. Upstream baseline: `zhangshuming0668/BrepMFR` commit `91f5a739c36d6111cc057e13f1753f7507c2ef72` (2024-04-19). Smoke test passed under the same acceptance criteria as the other baselines: official train/test entry points, small subset, 10 epochs, metrics in a sane range.

## 1. Base environment

Cloned from the verified local env `uvnet` (`conda create -n brepmfr --clone uvnet`, 131 packages) — the same base stack used by the previous baselines (UV-Net, BRepNet, AAGNet).

| Component | Version |
|---|---|
| python | 3.10.20 |
| torch | 2.11.0+cu128 |
| pytorch-lightning | 1.9.5 |
| dgl | 2.2.1+cu121 (graphbolt stub inherited, see §4) |
| numpy | 1.23.5 |
| occwl / OCC | 3.0.0 / 7.7.2 (present in base; unused by the BrepMFR smoke path) |

Additional installs into the clone:

| Package | Version | Install command | Note |
|---|---|---|---|
| torch-geometric | 2.8.0.post1 | `pip install torch-geometric --no-deps` | `--no-deps` mandatory (protects torch); used only as a `Data` container in `data/dataset.py` |
| prefetch_generator | 1.0.3 | `pip install prefetch_generator` | sdist, wheel built locally |
| xxhash | 3.8.1 | `pip install xxhash jinja2 psutil requests networkx` | required at PyG 2.8 import time; the other four were already present |

Already present in base (probed, no install needed): scipy 1.15.3, tensorboard, jinja2, psutil, requests, networkx.

Environment fingerprint: **`environment-adapted.yml`** (full `conda env export` of this env, UTF-8).

## 2. Diff vs. official `environment.yml`

| Official pin | Adapted | Note |
|---|---|---|
| python 3.9 (conda) | python 3.10.20 | base stack |
| pytorch 1.13.1+cu117 (conda) | torch 2.11.0+cu128 | code impact: adapt#2, adapt#6 |
| dgl 1.0.0.cu117 (conda) | dgl 2.2.1+cu121 | dgl_sparse C++ lib unusable on torch 2.11 → adapt#5 |
| fairseq 0.12.3 (conda, py39) | **not installed** — local in-repo shim | adapt#1; rationale in §3 |
| pytorch-lightning 1.7.1 (pip) | pytorch-lightning 1.9.5 | deprecation warnings only (`auto_select_gpus`, `*_epoch_end` hooks, `Trainer.test(verbose=)`, apex AMP hook arg); no functional break |
| torch-geometric 2.3.1 (pip) | torch-geometric 2.8.0.post1 | 2.3.1 is incompatible with torch 2.x |
| numpy 1.23.5 (pip) | numpy 1.23.5 | identical pin; `np.int` emits DeprecationWarning only (removed in numpy ≥1.24 — do not upgrade) |
| *(undeclared)* scipy | scipy 1.15.3 | `scipy.spatial.transform.Rotation` in `data/utils.py` |
| *(undeclared)* prefetch_generator | prefetch_generator 1.0.3 | `BackgroundGenerator` in `data/dataset.py` |
| *(undeclared)* tensorboard | tensorboard (base) | `TensorBoardLogger` |

## 3. Key dependency notes

- **fairseq — deliberately absent.** Upstream pins fairseq 0.12.3 (conda py39 build). `import fairseq` executes `hydra_init()` and pulls the full toolkit (models/tasks/optim/scoring), requiring omegaconf&lt;2.1 + hydra-core&lt;1.1 + sacrebleu + bitarray; omegaconf 2.0.x predates py3.10 support and its metadata is PEP-440-invalid for modern pip. BrepMFR uses only six shallow symbols, so a local math-equivalent shim ships in-repo (`fairseq/`, adapt#1). Fallback if the shim ever misbehaves: `pip install fairseq==0.12.3 --no-deps` plus the old pins — not attempted.
- **dgl 2.2.1+cu121 (inherited).** Installed from the DGL wheel index; the PyPI default build is CPU-only — do not reinstall from PyPI. The wheel ships only `dgl_sparse_pytorch_2.1.0.dll`; under torch 2.11 any `DGLGraph.adj()` call raises `FileNotFoundError: dgl_sparse_pytorch_2.11.0.dll`. BrepMFR hit this in `data/dataset.py`; resolved at code level (adapt#5, edges-based adjacency). Do not rename/copy the 2.1 dll to fake the 2.11 filename (ABI mismatch).
- **torch-geometric 2.8.0.post1.** `--no-deps` install; verify with `python -c "from torch_geometric.data import Data"`. PyG 2.8 imports xxhash at package init — keep xxhash installed.
- **pytorch-lightning 1.9.5.** The PL 1.7-era APIs used by this repo (`Trainer.add_argparse_args` / `from_argparse_args`, `auto_select_gpus`, `Trainer.test(verbose=)`, `*_epoch_end` hooks, custom `optimizer_step` signature) still exist in 1.9.5 as deprecations — verified by probes before first run.
- **numpy 1.23.5 is load-bearing:** `models/brepseg_model.py` uses `np.int`, which works (warning only) on ≤1.23 and was removed in ≥1.24.

## 4. Environment-level patches (site-packages)

**None added by this project.** One patch is inherited from the `uvnet` base env:

- `dgl/graphbolt/__init__.py` → replaced with a stub (`__getattr__` returning None). Reason: graphbolt supports only torch 2.1–2.3 and breaks `import dgl` on torch 2.11. **If dgl is ever reinstalled or upgraded in this env, the stub must be re-applied.**

## 5. Verification log

Environment probes (all pass, 2026-07-30):

```
PL Trainer: add_argparse_args True, from_argparse_args True, auto_select_gpus True, test(verbose) True
python -c "from torch_geometric.data import Data"        → ok
python -c "from scipy.spatial.transform import Rotation" → ok
python -c "import tensorboard"                           → ok
fairseq shim self-test: softmax rows sum to 1; LayerNorm is nn.LayerNorm;
  quant_noise identity at p=0; LayerDropModuleList iterable → ok
python -m py_compile on all edited files                 → clean
```

Smoke dataset: `D:\ShortEssay\Datasets\MFR_smoke` — 300/100/100 `.bin` sampled from the official CADSynth lists (80k/10k/10k) with fixed seeds 42/43/44; 500/500 files present. Source: Science Data Bank (doi:10.57760/sciencedb.17011).

Train entry (official CLI, 10 epochs):

```
python segmentation.py train --dataset_path D:\ShortEssay\Datasets\MFR_smoke --max_epochs 10 --batch_size 32 --num_workers 0
→ completed; train_loss 3.22 → 2.88; ~1.2 s/it (batch 32, RTX 5070)
```

Test entry (official CLI, best.ckpt):

```
python segmentation.py test --dataset_path D:\ShortEssay\Datasets\MFR_smoke --checkpoint results/BrepMFR/0730/102935/best.ckpt --batch_size 32 --num_workers 0
→ per_face_accuracy 0.1914 | per_class_accuracy 0.0507 | IoU 0.0231
```

Extended run (30 epochs, same settings): train_loss → 1.86, confirming normal training dynamics.

Interpretation: the 25-class random rate is 0.04; 0.1914 ≈ 4.8× random — sane. Low absolute convergence is expected by design: the model uses a 5000-step LR warmup while 10 epochs here total ~90 steps (peak LR ≈ 1.8% of the design value), so most tail classes remain unlearned at this budget. Reference frame: AAGNet smoke on an MFCAD++ subset reached 30.8% acc / 18.9% IoU — different dataset and LR schedule, not a like-for-like comparison; acceptance here is against the random-rate criterion used for all baselines.

## 6. Code-level adaptations

All adaptations are numbered, one commit each, applied on top of upstream baseline `91f5a73`. Upstream behavior is preserved wherever the environment allows it.

| # | Commit | File(s) | Change | Trigger |
|---|---|---|---|---|
| adapt#1 | `74aae18` | `fairseq/` (new pkg) | Local math-equivalent shim of the six fairseq 0.12.3 symbols used by the repo (`utils.softmax`, `utils.get_activation_fn`, `FairseqDropout`, `quant_noise`, `LayerNorm`, `LayerDropModuleList`); MIT-licensed subset, ~150 lines | fairseq 0.12.3 install chain broken/unsupported on py3.10 (§3) |
| adapt#2 | `f54d06f` | `models/brepseg_model.py` | Removed `verbose=False` from `ReduceLROnPlateau` | kwarg deprecated in torch 2.2, removed since |
| adapt#3 | `f54d06f` | `models/brepseg_model.py` | `test_step` per-graph txt dump redirected to `results/BrepMFR/pred_txt/` (+ mkdir) | hardcoded Linux path `/home/zhang/...` crashed on Windows |
| adapt#4 | `d0bc589` | `data/dataset.py` | `prefetch_factor = 2 if num_workers &gt; 0 else None` (both `get_dataloader`) | `prefetch_factor` invalid with `num_workers=0`; Windows requires workers=0 per upstream comment |
| adapt#5 | `74590c8` | `data/dataset.py` | Dense adjacency built from `graph.edges()` instead of `g.adj().to_dense()` (both `load_one_graph`) | dgl_sparse C++ lib has no torch-2.11 build (§3) |
| adapt#6 | `6883187` | `segmentation.py` | `torch.serialization.add_safe_globals([argparse.Namespace])` | torch ≥2.6 defaults `torch.load` to `weights_only=True`; PL checkpoints store hparams as `argparse.Namespace` |

Domain-adaptation path (`domain_adapt.py`, `models/transfer_model.py`, `TransferDataset`) is out of smoke scope and was not exercised.
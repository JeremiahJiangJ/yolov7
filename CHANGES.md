# Changes in this fork

This fork extends YOLOv7 (base: upstream commit `a207844`) for fine-tuning small models (mainly `yolov7-tiny`) on
several camera datasets at once, with small objects, at each camera's native resolution. This document lists every
change: what it is for, how to use it, and how it is implemented. The multi-source data format has its own
reference: [`data/mixed/README.md`](data/mixed/README.md).

Unless stated otherwise, a feature is off by default: a stock data yaml (`train:` / `val:`) with no new flags trains as
upstream YOLOv7 does. This was checked: a seeded stock training run gives the same `results.txt`, value for value,
before and after the changes. The exceptions are listed in [Behaviour changes](#behaviour-changes-without-new-flags).

## Contents

| # | Feature | Enable with | Main files |
|---|---|---|---|
| 1 | [Compatibility with recent PyTorch / NumPy / Pillow / Python](#1-compatibility-with-recent-libraries) | always on | `utils/torch_utils.py`, `utils/metrics.py`, `utils/plots.py`, `utils/loss.py`, `models/experimental.py`, `train.py` |
| 2 | [Reproducible training](#2-reproducible-training) | `--seed [N]`, `--deterministic` | `utils/torch_utils.py`, `utils/general.py`, `utils/datasets.py`, `train.py` |
| 3 | [Configurable label folder](#3-configurable-label-folder) | `--label-folder-name` | `utils/datasets.py` |
| 4 | [Label caches: location, validity, clean-up](#4-label-caches) | `--train-cache-path`, `--test-cache-path`, `--keep-cache` | `utils/datasets.py`, `utils/mixed_data.py`, `train.py`, `test.py` |
| 5 | [AdamW and hyperparameter files](#5-adamw-and-hyperparameter-files) | `--adam`, `data/hyp.{scratch,finetune}.{sgd,adamw}.yaml` | `train.py`, `data/` |
| 6 | [Evaluation: per-area and per-frequency metrics, configurable fitness](#6-evaluation-and-fitness) | `--area-int`, `--freq-groups`, `--fitness-metric-weights`, `--fitness-area-weights` | `test.py`, `utils/metrics.py`, `train.py` |
| 7 | [Training recipe options](#7-training-recipe-options) | `--close-mosaic`, `--clip-grad`, `--patience`, `--unfreeze-epoch` | `train.py`, `utils/datasets.py` |
| 8 | [Multi-source (weighted) training](#8-multi-source-weighted-training) | `train_sources:` in the data yaml | `utils/mixed_data.py`, `train.py`, `test.py` |
| 9 | [Native-scale loading and augmentation](#9-native-scale-loading-and-augmentation) | `resize:` in the data yaml, `scale_min` / `scale_max` in the hyp | `utils/datasets.py`, `utils/autoanchor.py` |
| 10 | [Repeat-factor sampling for rare classes](#10-repeat-factor-sampling) | `repeat_factor_threshold:` in the data yaml | `utils/mixed_data.py`, `utils/datasets.py` |
| 11 | [Knowledge distillation](#11-knowledge-distillation) | `--teacher`, `--distill-weight` | `utils/distill.py`, `train.py`, `train_aux.py` |
| 12 | [IR polarity inversion and grayscale](#12-ir-polarity-inversion-and-grayscale) | `px_inversion_prob:` + `inversion_border:`, `to_gray:` in the data yaml; `test.py --invert` | `utils/datasets.py`, `utils/mixed_data.py`, `test.py` |
| 13 | [Tools, scripts and example configs](#13-tools-scripts-and-example-configs) | | `tools/class_audit.py`, `scripts/`, `data/mixed/` |

Also: [Behaviour changes without new flags](#behaviour-changes-without-new-flags), [What was tested](#what-was-tested),
[Known limitations](#known-limitations).

`train.py` has every feature. `train_aux.py` (W6 / E6 / D6 / E6E models with auxiliary heads) has 1-6, 11 and 12, not 4's
automatic cache deletion, 7, 8, 9 or 10.

---

## Behaviour changes without new flags

| Change | Upstream | Now | Why |
|---|---|---|---|
| Default label folder | `labels` | `labels_upright` | project convention; `--label-folder-name labels` restores upstream |
| Label caches | kept next to the labels, never checked | deleted when the run ends; rebuilt if made for other images or another label folder | stale caches silently fed old labels to new runs |
| Pooled `train:` lists | cache named after the first folder (overwriting that folder's own cache) | own cache file, `<name>_pooled_<hash>.cache` | a later single-folder run loaded every pooled image as that folder |
| No label files found | warning, then metrics of 0 | error naming the folder searched | a wrong label folder looked like a bad model |
| `--adam` | Adam | AdamW | decoupled weight decay |
| `test.test()` | returns 3 values | returns 4: `(results, maps, times, area_results)` | per-area metrics; update scripts that unpack it |
| `attempt_load` | failed on checkpoints whose weights require grad (`epoch_*.pt`) | works | fusion now runs under `torch.no_grad()` |
| `--multi-scale` | crashed on Python >= 3.12 | works | `random.randrange` needs ints |

---

## 1. Compatibility with recent libraries

**Purpose.** Upstream YOLOv7 fails on current PyTorch (2.6+), NumPy (2.x), Pillow (10+) and Python (3.12+). These
fixes keep older versions working.

| Problem | Fix | Where |
|---|---|---|
| `torch.load` defaults to `weights_only=True` (PyTorch 2.6), refusing pickled models and label caches | `torch_load()` passes `weights_only=False` only where the argument exists; every `torch.load` call uses it | `utils/torch_utils.py`, callers in `train*.py`, `detect.py`, `hubconf.py`, `models/experimental.py`, `utils/general.py`, `utils/datasets.py`, `utils/aws/resume.py` |
| `np.trapz` removed (NumPy 2) | `np.trapezoid` with `np.trapz` fallback | `utils/metrics.py` |
| `np.int` removed (NumPy 1.24) | `int` | `utils/datasets.py`, `deploy/triton-inference-server/processing.py` |
| `ImageFont.getsize` removed (Pillow 10) | `getbbox` with `getsize` fallback | `utils/plots.py` |
| Aux / bin OTA losses mix CPU and GPU tensors (PyTorch 1.13+) | layer indices created on the targets' device | `utils/loss.py` |
| Fusing a checkpoint with `requires_grad` weights fails | fuse under `torch.no_grad()` | `models/experimental.py: attempt_load` |
| `random.randrange` rejects floats (Python 3.12) | integer bounds in `--multi-scale` | `train.py`, `train_aux.py` |

---

## 2. Reproducible training

**Purpose.** Re-running a configuration gives the same data, augmentations and (on CPU) the same results, so
differences between experiments come from the change being tested.

**Usage.**
- `--seed` (42) or `--seed N`: seeds everything and turns on `--deterministic`.
- `--deterministic` without `--seed`: keeps the upstream seeds (`2 + rank`) with deterministic cuDNN.

**Implementation.**
- `utils/torch_utils.py: init_torch_seeds / set_deterministic`: `torch.manual_seed`, `torch.cuda.manual_seed_all`;
  deterministic mode sets `cudnn.benchmark = False`, `cudnn.deterministic = True`.
  `torch.use_deterministic_algorithms` is deliberately not used (it stops training on CUDA operations without a
  deterministic implementation).
- `utils/general.py: init_seeds`: Python, NumPy and PyTorch seeds; `train.py` seeds each DDP rank with `seed + rank`.
- Stock data loader (`utils/datasets.py: create_dataloader`): seeded `torch.Generator`, `seed_worker` re-seeding NumPy
  and `random` in each worker, `DistributedSampler(seed=seed)`.
- Multi-source loader (section 8): every sample gets its own augmentation seed derived from (seed, source, position in
  the source's stream), so the data does not depend on the number of workers, resuming or loader rebuilds.
- `--evolve` mutations are seeded with `seed + generation`.

**Notes.** GPU runs are not guaranteed bit-identical: some CUDA operations (e.g. upsampling backward) accumulate in a
non-deterministic order. Resumed runs differ slightly from uninterrupted ones because checkpoints store fp16 weights;
the data stream itself resumes exactly.

---

## 3. Configurable label folder

**Purpose.** Label sets live in folders next to `images/` with different names (e.g. `labels_upright`,
`labels_version1`).

**Usage.** `--label-folder-name NAME` (default `labels_upright`): `/dir/images/x.jpg` reads `/dir/NAME/x.txt`. In a
multi-source yaml, `label_folder:` sets it per source or for all sources.

**Implementation.** `utils/datasets.py: img2label_paths(paths, label_folder_name)`, used by `LoadImagesAndLabels`,
W&B dataset logging and the tools. Caches record the folder they were built from and are rebuilt for another one.
A dataset without any label file raises `FileNotFoundError` naming an example path searched and how to fix it.

---

## 4. Label caches

**Purpose.** YOLOv7 caches parsed labels in `.cache` files. Upstream writes them next to the labels and reuses them
without checking, so edited or replaced labels could be silently ignored, and pooled folder lists overwrote another
folder's cache.

**Usage.**
- `--train-cache-path`, `--test-cache-path`: a `.cache` file, or a directory (created if needed) that any number of
  datasets can share. With a multi-source yaml there is one cache per source: `cache/exp1.cache` becomes
  `cache/exp1.<source>.train.cache` (`.val.cache` for validation), a directory `cache/exp1` holds
  `<source>.train.cache`. Per-source `cache_path:` / `val_cache_path:` override the flags.
- Caches are **deleted when the run ends**: finished, failed, Ctrl+C, SIGTERM or SIGHUP. `--keep-cache` keeps them.
- Training prints every cache file it uses (`label caches: ...`).

**Implementation (`utils/datasets.py`).**
- `LoadImagesAndLabels`: cache path resolution (file, or directory + `labels_<hash of image paths>.cache`); pooled
  lists get `_pooled_<hash>`; a cache is rebuilt if its image list or label folder differs from the current ones.
- `delete_label_caches_at_exit()`: every cache read or written is recorded; an `atexit` handler deletes them (and
  directories left empty). SIGTERM / SIGHUP are turned into a normal exit. Only the main process cleans up (not
  dataloader workers, not other DDP ranks). Called by `train.py` and `test.py`.

**Notes.** A hard kill (SIGKILL, out-of-memory killer) cannot be caught: delete the listed caches by hand. Edited
label contents with the same images are not detected by the validity check (upstream's hash check is disabled), which
is why caches are deleted after each run.

---

## 5. AdamW and hyperparameter files

**Purpose.** AdamW for fine-tuning, and ready-made hyperparameters for `yolov7-tiny`.

**Usage.** `--adam` now uses `torch.optim.AdamW`. Hyp files based on `hyp.scratch.tiny.yaml` (tiny's lighter
augmentation and loss gains):

| File | lr0 | lrf | weight_decay | warmup_epochs | warmup_bias_lr |
|---|---|---|---|---|---|
| `hyp.scratch.sgd.yaml` | 0.01 | 0.01 | 0.0005 | 3 | 0.1 |
| `hyp.finetune.sgd.yaml` | 0.001 | 0.1 | 0.0005 | 1 | 0.01 |
| `hyp.scratch.adamw.yaml` | 0.001 | 0.01 | 0.05 | 3 | 0 |
| `hyp.finetune.adamw.yaml` | 1e-5 | 0.1 | 0.05 | 1 | 0 |

`train.py` warns if the hyp file name says AdamW but `--adam` is missing, or the reverse. The files contain
commented-out `scale_min` / `scale_max` (section 9).

---

## 6. Evaluation and fitness

### Per-area metrics: `--area-int`

**Purpose.** Precision, recall and mAP per object size, e.g. to see how small objects do.

**Usage.** `--area-int 300 650 1250` (in `train.py`, `train_aux.py`, `test.py`) reports `<300`, `300<=A<650`,
`650<=A<1250`, `>=1250`: box area in px² of the original image, each bin `lo <= area < hi`. Logged every epoch to
TensorBoard (`metrics_area/...`) and `area_results.txt`.

**Implementation.** `test.py` records each target's area and, per prediction, the area of the target it matched (or
its own area if unmatched). `utils/metrics.py: ap_per_area` assigns COCO-style: targets by their area, matched
predictions by their target's area, unmatched predictions by their own area, then runs `ap_per_class` per bin.
Checked against pycocotools COCOeval with the same area ranges (within 0.006 AP; the repo's AP interpolation reads
slightly higher than COCO's in general).

### Per class-frequency group: `--freq-groups`

**Purpose.** With many unbalanced classes, mAP averaged over all classes hides how rare classes do.

**Usage.** Training prints it automatically when there is more than one class; `test.py --freq-groups` computes it
from the training label files of the data yaml. Groups by the number of training frames containing the class (as
LVIS): rare 1-10, common 11-100, frequent >100, plus classes absent from training. Reported per group: classes, classes
with validation labels, labels, recall, mAP@.5, mAP@.5:.95.

**Implementation.** `utils/metrics.py: class_frames, ap_per_group`; `utils/mixed_data.py: train_class_frames`.

### Fitness: `--fitness-metric-weights`, `--fitness-area-weights`

**Purpose.** `best.pt`, `--patience` and `--evolve` select on "fitness", upstream fixed at
`0.1 x mAP@.5 + 0.9 x mAP@.5:.95`. For small objects, where the aim is finding them rather than tight boxes, mAP@.5
is usually the better criterion.

**Usage.**
- `--fitness-metric-weights P R mAP@.5 mAP@.5:.95`, e.g. `0 0 1 0`. Also used by `--evolve`.
- `--fitness-area-weights w1 ... wN` (one per `--area-int` bin): fitness = weighted mean of the bins' fitness; bins
  without labels are left out and the remaining weights renormalised. Not with `--evolve`.

**Implementation.** `utils/metrics.py: fitness(x, w), area_fitness`; `utils/general.py: print_mutation` takes the
weights.

---

## 7. Training recipe options

All off by default; in `train.py`.

| Option | Purpose | Implementation |
|---|---|---|
| `--close-mosaic N` | Last N epochs on clean images (no mosaic, mixup or paste-in), as in later YOLO versions | `LoadImagesAndLabels.close_mosaic()` / `MixedDataset.close_mosaic()`, then `utils/datasets.py: rebuild_loader` restarts the workers (they hold their own copy of the dataset) |
| `--clip-grad X` | Clip the gradient norm, against loss spikes | `scaler.unscale_` then `clip_grad_norm_` before the optimizer step |
| `--patience N` | Stop after N validated epochs without fitness improvement | best epoch tracked with `best_fitness`; DDP ranks stop together (broadcast); no effect with `--notest` |
| `--unfreeze-epoch N` | Train with `--freeze` layers frozen, then unfreeze them from epoch N | re-enables `requires_grad` (frozen weights are already in the optimizer); not in DDP mode |

---

## 8. Multi-source (weighted) training

**Purpose.** Fine-tune for a target camera with little data while also training on other cameras / datasets, to
bolster the target and avoid forgetting what the other data teaches, without letting the largest dataset dominate.

**Usage.** A `train_sources:` list in the data yaml (one `target: true`, per-source `weight`, optional
`weight_schedule`, `epoch_size`, per-source `val`). Full reference, with the data plan and logs:
[`data/mixed/README.md`](data/mixed/README.md); commented template: `data/mixed/template.yaml`. Dry run:
`python -m utils.mixed_data --data your.yaml --epochs N --img-size S [--measure 64]`.

**Key behaviours.**
- Weights set **training shares**: each source's share of the samples (training pixels and compute), exactly.
- Every sample is built from one source: mosaic, mixup and paste-in partners come from the sampled frame's source.
- `epoch_size: target` (default): `--epochs` = passes over the target, as when training on it alone; the weights decide
  how much other data is added. `total` gives a stock pooled run's budget.
- Validation and fitness use the target's val set; other sources' val sets are optional and only reported
  (`source_results.txt`). Autoanchor uses the target.
- Before training, the **data plan** (log and `data_plan.txt`) shows per source: training share, frames loaded per
  sample (sampled frame + partners) and their share, objects per sample and their share (measured on 64 samples per
  source), frames never loaded, and notes where frame or object shares differ from the training share by 1.5x.
- Every epoch, the **data log** (`data_usage.txt`) shows what was actually used: samples and objects per source
  (counted by the training loop: exact), frames per sample and objects shrunk below 2 px (counted by the workers).

**Implementation (`utils/mixed_data.py`).**
- `MixedConfig`: parses and validates the yaml; points `data['train']` / `data['val']` at the target for code that reads
  them.
- `WeightSchedule`: per-epoch weights (`step` or `linear` between points at `epoch` or `progress`).
- `MixedDataset`: the per-source `LoadImagesAndLabels` behind one index; each item is `(index, augmentation seed)`;
  re-seeds Python and NumPy before building the sample (restoring them when running in the main process).
- `MixedSourceSampler`: per epoch, exact quotas `round(w_k x epoch_size)` (largest remainder); each source is an endless
  stream of seeded shuffled passes; the draw for epoch e is a pure function of (seed, config, e), so resume and loader
  rebuilds need no sampler state; DDP: each rank takes every world_size-th item. `plan()` and `summary()` report it;
  `signature()` is stored in checkpoints and compared on `--resume`.
- `create_mixed_dataloader`, `create_val_dataloader`, `measure_sources`, `usage_totals` / `usage_report`.
- `train.py`: chooses the multi-source loader when `train_sources` is present; refuses `--image-weights` and `--rect`.
- `test.py`: tests the target's val set (or `--source NAME`) at its own `resize`.
- Usage counters: `LoadImagesAndLabels.stats` is a shared-memory tensor with one row per worker (no write races):
  samples, frames loaded, labels, labels shrunk below 2 px.

---

## 9. Native-scale loading and augmentation

**Purpose.** Upstream resizes every frame so its long side equals `--img-size`, so the same object gets different pixel
sizes on different cameras (a 640x480 camera is enlarged 2x at `--img-size 1280`, a 1280x720 one halved at 640).
When cameras see objects at the same pixel scale, that destroys the size information. Native scale keeps objects at
their real pixel size: `--img-size` becomes the training crop size, and validation / deployment run on full frames.

**Usage.** In the data yaml, `resize:` per source (or as the default): `fit` (upstream), `native` (never resized), or a
scale factor (e.g. `0.5`) for a source whose objects are at a different pixel scale. Optional hyp keys
`scale_min` / `scale_max` set the zoom augmentation. Deploy with `detect.py --img-size <camera long side>`.

**Implementation (`utils/datasets.py`).**
- `parse_resize`; `load_image` scales by the factor instead of fitting to `img_size`.
- `load_mosaic_native` (used for `native` / factor sources instead of `load_mosaic` / `load_mosaic9`):
  - frames are offset at random within their tile instead of anchored at the mosaic centre (which, with frames larger
    than the tiles, almost never showed objects near the frame centre: 60% empty samples in a synthetic test);
  - with probability `fg_crop_prob` (default 0.5) a frame is placed so one of its objects is in view;
  - **fill mosaic**: along an axis where frames are shorter than `img_size`, the canvas is cut into frame-sized cells
    laid out from the random centre (`_mosaic_cuts`, up to `mosaic_max_cells` per axis), so small frames do not leave
    the sample grey;
  - only cells that can reach the sample are loaded (`_mosaic_reach`: with no rotation / shear / perspective, at most
    `img_size x (0.5 / smallest zoom + translate)` from the centre); the sampled frame always goes to a cell
    overlapping the view.
- `load_crop`: without mosaic (e.g. `--close-mosaic` epochs), an `img_size` crop of the frame (object-aware), not a
  downscaled frame.
- Validation: each batch is padded to its largest frame, rounded up to the stride, without resizing (1280x720 ->
  1280x736, what `detect.py --img-size 1280` gives).
- `loaded_shapes()` gives autoanchor the object sizes as loaded (`utils/autoanchor.py`).
- `random_perspective(scale_range=...)`: with `scale_min` / `scale_max`, zoom is sampled log-uniformly from that range
  (stock: `uniform(1 - scale, 1.1 + scale)`); labels shrunk below 2 px are counted.
- `expected_images_per_sample`: expected frames per sample from the mosaic geometry, for the data plan.

**Measured** (synthetic JPEG frames, one CPU core, data loading only):

| Frames | `--img-size` | Zoom | Frames per sample | ms per sample | Empty samples |
|---|---|---|---|---|---|
| 1280x720 | 1280 | stock 0.5-1.6 | 10.0 | 81 | 5% |
| 1280x720 | 1280 | 0.8-2.0 | 7.6 | 70 | 15% |
| 640x480 | 1280 | stock 0.5-1.6 | 32.0 | 104 | 0% |
| 640x480 | 1280 | 0.8-2.0 | 18.8 | 73 | 0% |

---

## 10. Repeat-factor sampling

**Purpose.** With many unbalanced classes, frames with rare classes are trained on too rarely. Repeat-factor sampling
(LVIS, Gupta et al. 2019) shows them more often, gently and without dropping any frame.

**Usage.** `repeat_factor_threshold: t` in a multi-source data yaml (per source or as the default; off by default).
Each frame gets r = max(1, max over its classes of √(t / f_c)), f_c = fraction of the source's frames containing class
c. Classes in fewer than t x frames of the source are boosted: a class 100x below the threshold is repeated about 10x.
Choose t so that t x (frames in the source) is the frame count below which a class should be boosted (e.g. t = 0.01
with 2,000 frames: classes in fewer than 20 frames). Run `tools/class_audit.py` first to see the class frequencies,
and evaluate with `--freq-groups`.

**Implementation.**
- `utils/mixed_data.py: repeat_factors` computes r per frame from the source's labels.
- `MixedSourceSampler`: a source with repeat factors has a repeat-factor stream: each pass holds frame i floor(r_i)
  times plus a fixed number of extra copies drawn (seeded) in proportion to the fractional parts, so every pass has the
  same length and the stream stays deterministic.
- Mosaic, mixup and paste-in partners of that source are drawn in proportion to r as well
  (`utils/datasets.py: _partners`, `_partner`, via `partner_cum_weights`); otherwise the boost would only reach the
  sampled frame, a small part of each mosaic.
- With `epoch_size: target`, `--epochs` = passes over the target's repeat-factor stream. The data plan and the start-up
  log show the boost per source.

**Measured** (synthetic, 20 classes with Zipf-like frequencies, t = 0.1): the share of the rarest classes' objects in
training samples rose 1.5-3.2x, the most frequent class's fell to 0.9x.

---

## 11. Knowledge distillation

**Purpose.** Train `yolov7-tiny` (student) with guidance from a larger, more accurate YOLOv7 (teacher) trained on the
same classes.

**Usage.** `train.py --teacher teacher.pt [--distill-weight 1.0]` (also `train_aux.py` for auxiliary-head students).
Example: `scripts/train_distill.sh`. The teacher can be a P5 model (yolov7, yolov7x) or a P6 model (W6, E6, D6, E6E;
then `--img-size` must be a multiple of 64).

**Implementation (`utils/distill.py: Distiller`).**
- The teacher is loaded fused, in eval mode, without gradients, and run on every training batch.
- Each student output level is paired with the teacher level of the same stride, cell by cell and anchor by anchor;
  the student takes the teacher's anchors (`sync_anchors`), and autoanchor is skipped.
- Loss terms, scaled with the detection loss's gains and level balance: objectness KL divergence on every cell; class
  KL divergence and box squared error (centre offset in the cell, log width / height) weighted by teacher objectness.
  Total loss = detection loss + `--distill-weight` x distillation loss.
- Not 1 - IoU for boxes: IoU has a kink where boxes coincide, so its gradient did not vanish when student and teacher
  were identical (checked); the squared error's does.
- Logged per epoch: `distill loss (box, obj, cls)`, TensorBoard `train/distill_*`; objectness and class terms are KL
  divergences (0 = the student matches the teacher).

---

## 12. IR polarity inversion and grayscale

**Purpose.** IR cameras record white-hot or black-hot, and a dataset is often skewed to one polarity. Inverting
frames during training teaches both. RGB frames can supplement scarce IR data once colour, a signal IR does not have,
is removed.

**Usage** (data yaml: per source, or top-level as the default for every source; a stock `train:` / `val:` yaml takes
the top-level keys too):

| Key | Default | Meaning |
|---|---|---|
| `px_inversion_prob` | 0 | chance a training frame is inverted: 255 - value, inside the frame minus its `inversion_border` |
| `inversion_border` | none | `{top, bottom, left, right}` in pixels of the original frame, left un-inverted (an IR camera's black border, which is not exactly 0 and would turn white). **Required** when `px_inversion_prob` > 0 (all 0 if there is no border); missing or misspelt keys stop training with an error naming the source |
| `to_gray` | 0 | chance a training frame is converted to gray (3 equal channels); `1.0` also converts the source's val frames |

`test.py --invert` evaluates the other polarity: every val frame inverted inside the border. Example and advice:
`data/mixed/README.md`, "IR polarity and grayscale"; commented example in `data/mixed/template.yaml`.

**Implementation.**
- `utils/datasets.py: load_image` (the one place every training path loads a frame: mosaic, native-scale crops,
  mixup, paste-in, close-mosaic) converts to gray, then inverts, on a copy, so RAM-cached images stay as recorded.
  `_load_image` is the former body; image caching uses it.
- `invert_region` scales the border from original pixels to the loaded size (`resize: fit` or a factor).
  `parse_inversion` validates probability and border; `MixedConfig` calls it at start-up so a bad source fails before
  any data is loaded.
- Augmentation draws use Python's `random`, seeded like the other augmentations, so seeded runs stay
  reproducible; with both keys at 0 no random number is drawn and stock runs are unchanged.
- In validation (`augment=False`) a fraction is ignored: frames are gray only with `to_gray: 1` and inverted only with
  `--invert` (`invert_all`).

---

## 13. Tools, scripts and example configs

| Path | What |
|---|---|
| `tools/class_audit.py` | Class x size audit from label files and image headers: per class, frames containing it, frequency group, instances by smallest box side (px, as training sees them), per-source counts; optional coarse-class totals (`--hierarchy`), CSV output |
| `python -m utils.mixed_data` | Dry run of a multi-source config: the data plan without training (`--measure 64` also counts objects) |
| `data/mixed/template.yaml` | Commented multi-source template (target camC, supplementary camA / camB) |
| `data/mixed/1_...` to `6_...` | Experiment ladder, each step changing one thing: stock target-only, stock pooled, multi-source unweighted, native scale, weighted, repeat-factor sampling |
| `scripts/train_mixed_ladder.sh` | Trains and tests the ladder with identical settings (`STEPS="..."` for a subset) |
| `scripts/train_default.sh`, `train_weighted.sh`, `train_distill.sh` | Example training commands |

---

## What was tested

All on CPU with synthetic data (generated frames with known objects; real camera data and GPUs were not available):

- Stock pipeline unchanged: a seeded 2-epoch run reproduces the pre-change `results.txt` exactly (re-checked after each
  change to the data loading).
- Seeded runs are reproducible, with and without multi-source data, native scale, distillation and repeat-factor
  sampling; the multi-source data stream is identical with 0 or more workers and across resume and loader rebuilds.
- Multi-source quotas are exact every epoch and follow weight schedules; the per-epoch log matches them.
- Per-area AP matches pycocotools; frames per sample predicted by the data plan match measurements within ~0.5.
- Distillation gradients vanish when student and teacher are identical; distillation losses fall during training;
  P5 and P6 teachers pair correctly with the tiny student.
- Label caches: naming, validity check, deletion on normal end, Ctrl+C and SIGTERM.
- IR inversion: the border is untouched and the inside inverted, at native size and resized; ~50% of training frames
  inverted at `px_inversion_prob: 0.5`; val untouched except with `--invert`; missing / bad borders refused.
  `to_gray`: 27% of frames gray at 0.3, every train and val frame at 1.0; gray then inverted composes correctly.

## Known limitations

- **Not tested on GPU, in multi-GPU (DDP) mode or on real data.** Run a short smoke test first.
- `train_aux.py` does not support multi-source data, native scale, the training recipe options or cache deletion.
- Native-scale validation uses full frames: about 2.3x the memory per image of a 640 crop at 1280x720 (the validation
  batch is 2x `--batch-size`).
- Resumed runs are not bit-identical to uninterrupted ones (fp16 checkpoints).
- `--evolve` is untested with the new options, and `--fitness-area-weights` is refused with it.
- Repeat-factor sampling is only available through a multi-source data yaml (a single source works).

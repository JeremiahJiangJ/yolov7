# YOLOv7 for multi-camera, small-object fine-tuning

This fork of [YOLOv7](https://github.com/WongKinYiu/yolov7) (base: upstream commit `a207844`) is for fine-tuning small
models (mainly `yolov7-tiny`) on several camera datasets at once, with small objects, many unbalanced classes, IR
footage, and each camera's native resolution. This README lists every change: what it is for, how to use it, and how
it is implemented. For the original YOLOv7 documentation (pretrained weights and benchmarks, export to ONNX /
TensorRT, pose estimation, instance segmentation) see the
[upstream README](https://github.com/WongKinYiu/yolov7/blob/main/README.md).

Unless stated otherwise, a feature is off by default: a stock data yaml (`train:` / `val:`) with no new flags trains as
upstream YOLOv7 does. This was checked: a seeded stock training run gives the same `results.txt`, value for value,
before and after the changes. The exceptions are listed in [Behaviour changes](#behaviour-changes-without-new-flags).

## Quick start

**Install**: Python 3.8+, then `pip install -r requirements.txt` (works with current PyTorch, NumPy 2 and Pillow 10+;
see [section 1](#1-compatibility-with-recent-libraries)).

**Stock data yaml, with the most useful new options:**

```shell
python train.py --weights yolov7-tiny.pt --cfg cfg/training/yolov7-tiny.yaml --data data/your.yaml \
  --hyp data/hyp.finetune.sgd.yaml --epochs 100 --batch-size 32 --img-size 640 640 --device 0 \
  --seed 42 --label-folder-name labels \
  --fitness-metric-weights 0 0 1 0 --area-int 64 256 1024 \
  --close-mosaic 10 --patience 30 --train-cache-path ./cache/exp1 --test-cache-path ./cache/exp1
```

`--label-folder-name labels` is needed for upstream-style datasets: this fork's default label folder is
`labels_upright`.

**Several cameras / datasets at once** (one target camera, the others as extra data):

1. Copy [`data/mixed/template.yaml`](data/mixed/template.yaml) and set paths, `nc`, `names` and weights
   ([section 8](#8-multi-source-weighted-training)).
2. Check what each dataset will contribute, without training:
   `python -m utils.mixed_data --data your.yaml --epochs 150 --img-size 1280 --close-mosaic 10 --measure 64`
3. Train with the same `train.py` command, `--data your.yaml`, `--img-size` = the deployment frame's long side.
4. Test: `python test.py --data your.yaml --weights runs/train/exp/weights/best.pt --img-size 1280` (target camera),
   `--source camA` (another camera), `--freq-groups` (rare / common / frequent classes), `--invert` (other IR polarity).
5. Deploy at the native frame size: `python detect.py --weights best.pt --img-size 1280 ...`

**Common recipes:**

| Need | Use | Section |
|---|---|---|
| Reproducible experiments | `--seed 42` | [2](#2-reproducible-training) |
| Metrics per object size | `--area-int 64 256 1024` | [6](#6-evaluation-and-fitness) |
| Select `best.pt` on mAP@.5 | `--fitness-metric-weights 0 0 1 0` | [6](#6-evaluation-and-fitness) |
| Keep small objects at their pixel size | `resize: native` in the data yaml, `scale_min` / `scale_max` in the hyp | [9](#9-native-scale-loading-and-augmentation) |
| Rare classes | `tools/class_audit.py`, then `repeat_factor_threshold:` in the data yaml, `--freq-groups` to evaluate | [10](#10-repeat-factor-sampling), [13](#13-tools-scripts-and-example-configs) |
| A bigger model to guide tiny | `--teacher yolov7.pt` | [11](#11-knowledge-distillation) |
| IR white-hot and black-hot | `px_inversion_prob:` + `inversion_border:` in the data yaml, `test.py --invert` | [12](#12-ir-polarity-inversion-and-grayscale) |
| RGB data supplementing IR | `to_gray: 1.0` on the RGB source | [12](#12-ir-polarity-inversion-and-grayscale) |

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
[Known limitations](#known-limitations), [Citation](#citation).

`train.py` has every feature. `train_aux.py` (W6 / E6 / D6 / E6E models with auxiliary heads) has 1-6, 11 and 12 (with a
stock data yaml), not 4's automatic cache deletion, 7, 8, 9 or 10.

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

**Usage.** Pass `--train-cache-path` / `--test-cache-path` (`train.py`, `train_aux.py`; `test.py` takes
`--test-cache-path`) to keep every label cache of a run in one place. Training prints all cache files it uses
(`label caches: ...`). With a multi-source yaml there is one cache per source, named after the source:

| Flag value | Stock data yaml | `train_sources` data yaml |
|---|---|---|
| `cache/exp1.cache` (a `.cache` file) | exactly that file | `cache/exp1.<source>.train.cache` (`.val.cache` for `--test-cache-path`) |
| `cache/exp1` (a directory, created if needed) | `cache/exp1/labels_<hash>.cache` | `cache/exp1/<source>.train.cache` (`.val.cache`) |
| not given | next to the labels (upstream) | next to each source's labels |

Per-source `cache_path:` / `val_cache_path:` in a multi-source yaml override the flags.

**Caches are deleted when the run ends**, finished or stopped (error, Ctrl+C, SIGTERM / SIGHUP from a scheduler), so
every run reads the current label files; `--keep-cache` keeps them (also for `test.py`).

**Implementation (`utils/datasets.py`).**
- `LoadImagesAndLabels`: cache path resolution (file, or directory + `labels_<hash of image paths>.cache`); pooled
  lists get `_pooled_<hash>`; a cache is rebuilt if its image list or label folder differs from the current ones.
- `delete_label_caches_at_exit()`: every cache read or written is recorded; an `atexit` handler deletes them (and
  directories left empty). SIGTERM / SIGHUP are turned into a normal exit. Only the main process cleans up (not
  dataloader workers, not other DDP ranks). Called by `train.py` and `test.py`.

**Notes.** A hard kill (SIGKILL, out-of-memory killer) cannot be caught: delete the caches listed at startup by hand.
Edited label contents with the same images are not detected by the validity check (upstream's hash check is
disabled), which is why caches are deleted after each run.

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
commented-out `scale_min` / `scale_max` ([section 9](#scale-augmentation-scale_min--scale_max)).

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

### Other `test.py` options

- `--source NAME`: with a multi-source yaml, test that source's val set instead of the target's (section 8).
- `--resize fit|native|<factor>`: override the data yaml's `resize` for this test (section 9).
- `--invert`: test the other IR polarity (section 12).

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

Train on several datasets ("sources") at once, with one **target** source (the deployment camera) and per-source
weights. Enabled by a `train_sources:` list in the data yaml; a plain `train:` / `val:` data yaml keeps the stock
YOLOv7 pipeline unchanged. Commented template: [`data/mixed/template.yaml`](data/mixed/template.yaml).

### Terms

Used here and in the logs:

- **sample**: one training input, an `--img-size` x `--img-size` crop (usually a mosaic). Every sample has the same
  pixels and compute, and is built entirely from one source.
- **training share** of a source: its share of the samples, i.e. of the training pixels and compute. **This is what
  the weights set**, exactly.
- **frame**: one image of a source. A sample is built from several frames: the sampled frame plus its mosaic, mixup and
  paste-in partners from the same source (**frames loaded**; some may be only partly in view). Smaller frames fill a
  mosaic with more of them, so a source's share of the frames can differ a lot from its training share.
- **objects**: labelled boxes in the samples, i.e. the supervision. A source's share of the objects also depends on how
  many objects its frames contain.

### Data yaml

```yaml
sampling: weighted        # weighted (default) | proportional (every image once per epoch, weights ignored)
resize: native            # default for all sources: fit | native | <scale factor>
fg_crop_prob: 0.5         # default for all sources
label_folder: labels      # default for all sources (otherwise --label-folder-name)
train_sources:
  - {name: camC, path: /data/camC/images/train, val: /data/camC/images/val, target: true, weight: 50}
  - {name: camA, path: /data/camA/images/train, val: /data/camA/images/val, weight: 30}
  - {name: camB, path: /data/camB/images/train, weight: 20, resize: 0.5}
weight_schedule:          # optional
  mode: step              # step | linear
  points:
    - {progress: 0.6, weights: {camC: 60, camA: 25, camB: 15}}   # or epoch: <n>
epoch_size: target        # default: --epochs = passes over the target | total | <source name> | <int>
nc: 1
names: ['object']
```

Per-source keys (most can also be set at the top level as the default for every source):

| Key | Default | Meaning |
|---|---|---|
| `name` | `source<i>` | used in logs, `weight_schedule`, `epoch_size`, `test.py --source` |
| `path` | required | training images: a dir or list file, or a list of them, pooled into one source as stock YOLOv7 pools `train: [a, b]` |
| `weight` | required for `weighted` | relative training share (normalised, any scale): 60 / 40 = 60% / 40% of the samples |
| `target` | `false` | exactly one source: its `val` drives fitness / `best.pt` / `--patience`; autoanchor uses it |
| `val` | none | target: required (or a top-level `val:`). Other sources: opt-in validation, reported only. A list is pooled too |
| `resize` | top-level `resize`, else `fit` | `fit`: long side resized to `--img-size` (stock YOLOv7). `native`: never resized, objects keep their pixel size, `--img-size` is the training crop size. Number: fixed scale factor, for sources whose objects are at a different pixel scale (section 9) |
| `fg_crop_prob` | top-level, else 0.5 | `native` / factor only: chance a mosaic tile or crop is placed around an object (else at random) |
| `mosaic_max_cells` | top-level, else 6 | `native` / factor only: max mosaic cells per axis for frames smaller than the tiles (section 9). Lower it if data loading is the bottleneck |
| `repeat_factor_threshold` | top-level, else 0 (off) | repeat-factor sampling: frames with classes in fewer than this fraction of the source's frames are sampled more often (section 10) |
| `px_inversion_prob` | top-level, else 0 (off) | IR polarity: chance a training frame is inverted (255 - value: white-hot <-> black-hot) inside its `inversion_border` (section 12) |
| `inversion_border` | top-level, else none | `{top: .., bottom: .., left: .., right: ..}`: border width in pixels of the original frame that inversion leaves alone (an IR camera's black border; all 0 if there is none). Required when `px_inversion_prob` > 0 |
| `to_gray` | top-level, else 0 (off) | chance a training frame is converted to gray (3 equal channels), e.g. `1.0` for RGB frames supplementing IR. `1.0` also converts the source's val frames (section 12) |
| `label_folder` | top-level, else `--label-folder-name` | label folder next to `images` |
| `cache_images` | target: `--cache-images`, others: off | cache this source's images in RAM |
| `cache_path`, `val_cache_path` | from `--train-cache-path` / `--test-cache-path`, else next to the labels | this source's label cache file or directory (overrides the flags, section 4) |

### Several datasets in one source

A source is a group of frames that share the source keys (weight, resize, inversion, ...): list several folders
under one name and they are concatenated into one pooled dataset, exactly as stock YOLOv7 does with `train: [a, b]`.
This is also how to have "several targets": put them in the one target source.

```yaml
  - name: ir
    path: [/data/ir_site1/images/train, /data/ir_site2/images/train]
    val: [/data/ir_site1/images/val, /data/ir_site2/images/val]
    target: true
```

Within a source frames are sampled uniformly, so the larger folder dominates. If the folders need their own share,
give them their own source (with its own `weight`) instead.

### How it works

- **Sampling.** Epoch *e* has exactly `round(w_k(e) * epoch_size)` samples from source *k*, each built around one
  sampled frame. Each source is an endless stream of shuffled passes over its frames, so every frame is sampled once
  before any repeats. The mix is logged at the start and whenever it changes.
- **Samples stay within a source.** Mosaic, mixup and paste-in partners come from the source of the sampled frame, so
  a weight of 50% means 50% of the samples (training pixels and compute) come entirely from that source. It does
  **not** mean 50% of the frames or objects: those also depend on frame size (fill mosaic, section 9) and object
  density. The data plan and the per-epoch data log show all three shares.
- **Validation and fitness** use the target's val set; other sources' val sets are optional and only reported.
  Autoanchor uses the target.
- **Reproducibility.** With `--seed`, the data of every epoch (order and augmentations) depends only on the seed,
  config and epoch: not on `--workers`, resume or loader rebuilds (`--close-mosaic`). A source's stream is the same
  whatever the other sources' weights are, so two weightings are compared on the same random draws. (Resumed runs
  still differ slightly from uninterrupted ones because YOLOv7 checkpoints store fp16 weights.)

Not supported with `train_sources`: `--image-weights`, `--rect` (train), `train_aux.py`.

### Epoch size

With weighted sampling there is no natural "one pass over the data", so `epoch_size` sets the epoch length (fixed
for the run, as the training loop needs constant batches per epoch):

- **`target` (default): `--epochs` = passes over the target**, exactly as when training on the target alone, whatever
  the weights and schedule: each target frame is sampled `--epochs` times. (With mosaic on, it is also loaded as a
  partner in other target samples, as in stock YOLOv7; the plan's `loads/frame` counts both.) The weights only decide
  how much other data is added, i.e. how long training takes. (The epoch is the target size divided by its weight
  averaged over the run, so with a schedule raising the target weight, early epochs hold fewer target samples and
  late epochs more.)
- `total`: as many samples per epoch as frames in all sources, the same iterations per epoch as
  `2_default_pooled.yaml`, so the same `--epochs` is the same training budget (used in the ladder)
- a source name: like `target`, anchored on that source instead
- an int

Everything YOLOv7 counts in epochs scales with the epoch length: the LR schedule, `warmup_epochs` (at least 1000
iterations), validation frequency, `--close-mosaic`, `--patience` and checkpoint saving. Schedule points given as
`progress` scale with `--epochs`; points given as `epoch` do not. Check a config with the dry run below: it shows
the training samples and per-source views of the whole run.

### Data plan and data log

Before training starts, the log (and `data_plan.txt`) shows what each source will contribute over the whole run:

| Column | Meaning | Accuracy |
|---|---|---|
| `frames` | frames in the source | exact |
| `training share`, `samples` | samples from the source: training pixels and compute; set by the weights | exact (the draws are fixed in advance) |
| `frames/sample` | frames loaded per sample with mosaic on: the sampled frame + mosaic / mixup / paste-in partners (1 in `--close-mosaic` epochs) | expected value from the mosaic geometry at the source's frame size and the hyp probabilities; matches measurements within ~0.5 |
| `frames loaded`, `share`, `loads/frame` | frames loaded over the run, their share, loads per frame of the source | from `frames/sample` |
| `never loaded` | share of the source's frames never loaded | estimate (partners are drawn at random) |
| `objects/sample`, `objects`, `share` | labelled objects in the samples: the supervision | measured on 64 samples per source built before training (with mosaic on, and separately for `--close-mosaic` epochs) |

A `NOTE` names every source whose frame or object share differs from its training share by 1.5x or more, and a
`WARNING` every source with frames that are never loaded.

Check a config without training (`--hyp`, `--img-size` and `--close-mosaic` as for training; frame sizes are read
from the images; `--measure 64` also builds samples to count objects, which reads all labels; `--world-size N` for
an N-GPU run, which pads each epoch to a multiple of N):

```
python -m utils.mixed_data --data your.yaml --epochs 3 --img-size 1280 --close-mosaic 1 --measure 64
```
```
3 epochs x 40 samples/epoch = 120 training samples (each one --img-size crop: equal pixels and compute)
  source           frames  training share   samples  frames/sample  frames loaded   share  loads/frame  never loaded  objects/sample    objects   share
  camC                 24           60.0%        72            9.9            498   32.8%         20.8          0.0%             6.2        348   40.5%
  cam640              150           40.0%        48           31.4           1021   67.2%          6.8          0.1%            15.0        512   59.5%
  objects/sample: measured on samples built with mosaic on (and separately for the --close-mosaic epochs, included in objects)
  NOTE: frame / object shares differ from the training shares by 1.5x or more. Smaller frames fill a mosaic with more frames, and object density differs between sources: lower a source's weight if it should count less (the weights set training shares)
  NOTE: camC: 60% of the training, 33% of the frames loaded, 40% of the objects
  NOTE: cam640: 40% of the training, 67% of the frames loaded, 60% of the objects
```

Here the target (1280x720 frames) gets 60% of the training but only a third of the frames and 40% of the objects: at
`--img-size 1280` the 640x480 source fills each mosaic with three times as many frames. Lower its weight if it should
count less.

Every epoch, training also logs what it used (and appends it to `data_usage.txt`):

- `samples`, `training share`, `objects`, `share`, `objects/sample`: counted by the training loop on the batches it
  consumed: exact
- `frames/sample`, `frames loaded`, `share`: frames per sample counted by the data loader workers, times the samples
- `objects shrunk < 2 px`: share of objects augmentation shrank below 2 px, which YOLOv7 drops (counted by the
  workers)

In DDP the log covers the rank 0 process only (each GPU gets an equal share of the same mix).

### Validation and testing

`best.pt` and all fitness options (`--fitness-metric-weights`, `--area-int`, `--fitness-area-weights`) use the target
source's val set. Non-target sources with a `val` are validated every epoch at their own scale and logged to
`source_results.txt` and TensorBoard (`metrics_source/<name>/...`), to watch for forgetting.

`test.py` accepts the same yaml: it tests the target's val set by default, `--source <name>` another source's,
each at its own `resize` (and in gray with `to_gray: 1`; `--invert` for the other IR polarity, section 12).

### Implementation (`utils/mixed_data.py`)

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

**Usage.** In the data yaml, `resize:` per source (or as the default, also in a stock yaml): `fit` (upstream),
`native` (never resized), or a scale factor (e.g. `0.5`) for a source whose objects are at a different pixel scale
(estimate it from the median box size of comparable objects). Optional hyp keys `scale_min` / `scale_max` set the zoom
augmentation. Deploy with `detect.py --img-size <camera long side>`: frames are then padded to a multiple of 32 and
never resized, matching validation.

### How it works

- **Native-scale mosaic.** Training samples are `--img-size` crops: a 4-tile mosaic where each frame is offset at
  random within its tile, or with probability `fg_crop_prob` placed so one of its objects is in view (the stock mosaic
  anchors each frame's corner at the mosaic centre, which with large frames almost never shows objects near the frame
  centre: 60% empty samples in a synthetic test). Without mosaic (e.g. `--close-mosaic` epochs) a sample is an
  `--img-size` crop of the frame (object-aware), not a downscaled frame.
- **Validation** uses full frames: each batch is padded to its largest frame, rounded up to the stride, without
  resizing (1280x720 -> 1280x736), i.e. what `detect.py --img-size 1280` feeds the model. Autoanchor uses native
  object sizes.
- **Fill mosaic.** Along an axis where frames are shorter than `--img-size` (640x480 frames at 1280 on both axes; the
  720 px height of 1280x720 frames at 1280), a 4-tile mosaic would leave much of the sample grey. Along that axis the
  mosaic canvas is instead cut into cells the size of the frame, laid out from the random mosaic centre (up to
  `mosaic_max_cells` cells per axis). Only cells that can end up in the sample are loaded: with no rotation / shear /
  perspective, the sample reaches at most `img_size * (0.5 / smallest zoom + translate)` from the canvas centre. The
  sampled image always goes to a cell overlapping the centre of the view, so it is seen.

  Measured on synthetic JPEG frames (one CPU core, data loading only; `hyp.scratch.tiny.yaml`; zoom = scale
  augmentation, below):

  | Frames | `--img-size` | Zoom | Images per sample | ms per sample | Empty samples | Grey |
  |---|---|---|---|---|---|---|
  | 1280x720 | 640 | stock 0.5-1.6 | 4.4 | 26 | 9% | 0% |
  | 1280x720 | 1280 | stock 0.5-1.6 | 10.0 | 81 | 5% | 2% |
  | 1280x720 | 1280 | 0.8-2.0 | 7.6 | 70 | 15% | 0% |
  | 640x480 | 640 | stock 0.5-1.6 | 7.7 | 22 | 4% | 2% |
  | 640x480 | 1280 | stock 0.5-1.6 | 32.0 | 104 | 0% | 6% |
  | 640x480 | 1280 | 0.8-2.0 | 18.8 | 73 | 0% | 6% |

  For comparison, the stock 4-tile mosaic on images resized to `--img-size` uses ~5.5 images per sample. Zooming in
  shows less of the scene per sample, hence more empty samples with zoom up to 2x (raise `fg_crop_prob` if that
  matters). If data loading cannot keep up with the GPU, use more `--workers` or a lower `mosaic_max_cells`.

### Scale augmentation: `scale_min` / `scale_max`

The stock zoom is `uniform(1 - scale, 1.1 + scale)`, i.e. 0.5-1.6x with `scale: 0.5`, which can shrink objects of a
few pixels to almost nothing. At native scale, set the zoom range explicitly in the hyp yaml:

```yaml
scale_min: 0.8   # little zoom-out: small objects stay visible
scale_max: 2.0   # the largest zoom the deployment camera uses
```

The zoom is then sampled log-uniformly (zooming in and out by the same factor equally likely). Without these keys
the stock behaviour is unchanged. Labels that augmentation shrinks below 2 px are dropped by YOLOv7: the per-epoch
data log counts them per source (`objects shrunk < 2 px`).

### Implementation (`utils/datasets.py`)

- `parse_resize`; `load_image` scales by the factor instead of fitting to `img_size`.
- `load_mosaic_native` (used for `native` / factor sources instead of `load_mosaic` / `load_mosaic9`): random offsets,
  `fg_crop_prob` placement, fill mosaic (`_mosaic_cuts`), reach-based cell skipping (`_mosaic_reach`).
- `load_crop`: the no-mosaic crop.
- `loaded_shapes()` gives autoanchor the object sizes as loaded (`utils/autoanchor.py`).
- `random_perspective(scale_range=...)`: log-uniform zoom with `scale_min` / `scale_max`; labels shrunk below 2 px are
  counted.
- `expected_images_per_sample`: expected frames per sample from the mosaic geometry, for the data plan.

---

## 10. Repeat-factor sampling

**Purpose.** With many unbalanced classes, frames with rare classes are trained on too rarely. Repeat-factor sampling
(LVIS, Gupta et al. 2019) shows them more often, gently and without dropping any frame.

**Usage.** `repeat_factor_threshold: t` in a multi-source data yaml (per source or as the default; off by default).
Each frame gets r = max(1, max over its classes of √(t / f_c)), f_c = fraction of the source's frames containing class
c. Classes in fewer than t x frames of the source are boosted: a class 100x below the threshold is repeated about 10x.
Choose t so that t x (frames in the source) is the frame count below which a class should be boosted (e.g. t = 0.01
with 2,000 frames: classes in fewer than 20 frames). Run `tools/class_audit.py` first to see the class frequencies,
and evaluate with `--freq-groups`. With `epoch_size: target`, `--epochs` counts passes over the longer, boosted
stream.

**Implementation.**
- `utils/mixed_data.py: repeat_factors` computes r per frame from the source's labels.
- `MixedSourceSampler`: a source with repeat factors has a repeat-factor stream: each pass holds frame i floor(r_i)
  times plus a fixed number of extra copies drawn (seeded) in proportion to the fractional parts, so every pass has the
  same length and the stream stays deterministic.
- Mosaic, mixup and paste-in partners of that source are drawn in proportion to r as well
  (`utils/datasets.py: _partners`, `_partner`, via `partner_cum_weights`); otherwise the boost would only reach the
  sampled frame, a small part of each mosaic.
- The data plan and the start-up log show the boost per source.

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

```yaml
train_sources:
  - name: ir                 # target: IR camera, mostly white-hot
    path: /data/ir/images/train
    val: /data/ir/images/val
    target: true
    weight: 60
    px_inversion_prob: 0.5   # half the frames shown black-hot
    inversion_border: {top: 12, bottom: 12, left: 8, right: 8}
  - name: rgb                # RGB frames as extra data: colour is not available in IR
    path: /data/rgb/images/train
    weight: 40
    to_gray: 1.0
```

- Measure the border once per camera: the widest dark band on each side, rounded up a few pixels. A band left
  inverted becomes a bright frame edge the model can learn as a cue. The border is given in original pixels and
  scaled with `resize`.
- Both act per frame when it is loaded, before mosaic and the other augmentations, also during `--close-mosaic`; gray
  first, then inversion. The HSV augmentation keeps gray frames gray.
- Validation sees the frames as recorded (gray when `to_gray: 1`). `test.py --invert` tests the other polarity: every
  val frame inverted inside its border (the border must be set). Run `test.py` with and without `--invert` to check
  that both polarities are detected.
- Inversion is meant for IR; on a grayed RGB source it gives "negative" images that look like neither polarity, so
  leave it off there unless an experiment shows it helps.
- The start-up log shows each source's settings (`to_gray ..., px_inversion_prob ... (border ...)`).

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

**Checked** through the full training pipeline (mosaic, crops, zoom), every frame inverted, synthetic IR frames with
a noisy dark border: 0.00% of sample pixels came from an inverted border with `inversion_border` set (native, `fit`
and `0.5`), against 16-19% with the border set to 0.

---

## 13. Tools, scripts and example configs

| Path | What |
|---|---|
| `tools/class_audit.py` | Class x size audit from label files and image headers (see below) |
| `python -m utils.mixed_data` | Dry run of a multi-source config: the data plan without training (section 8) |
| `data/mixed/template.yaml` | Commented multi-source template (target camC, supplementary camA / camB, IR / RGB example) |
| `data/mixed/1_...` to `6_...` | Experiment ladder (below) |
| `scripts/train_mixed_ladder.sh` | Trains and tests the ladder with identical settings (`STEPS="1 5 6"` for a subset) |
| `scripts/train_default.sh`, `train_weighted.sh`, `train_distill.sh` | Example training commands |

### Class audit: `tools/class_audit.py`

Per class: frames containing it and its frequency group (rare / common / frequent), instances, instances by size of
the smallest box side in pixels (as training sees them: native size, or the source's resize factor), and instances per
source. Shows which classes have enough examples big enough to tell apart, which need repeat-factor sampling, and which
may be better merged into a parent class. Reads label files and image headers only.

```shell
python tools/class_audit.py --data data/mixed/template.yaml          # train_sources or train: yaml
python tools/class_audit.py --data data.yaml --label-folder-name labels --bins 7 12 16 --csv audit.csv
python tools/class_audit.py --data data.yaml --hierarchy hierarchy.yaml   # also totals per coarse class
```

Options: `--split train|val`, `--bins` (size bin edges, px; default 7 12 16), `--min-identifiable N` (flag classes
with fewer than N instances in the largest bin; default 10), `--hierarchy` (yaml mapping fine class names to coarse
class names, e.g. `{sedan: car, pickup: car}`), `--csv` (also write the table to a file).

### Experiment ladder

Each step changes one thing, so comparing neighbours isolates its effect:

| Config | Data loading | Compare with | Isolates |
|---|---|---|---|
| `1_default_target.yaml` | stock, target camera only | | |
| `2_default_pooled.yaml` | stock, all cameras pooled | 1 | adding the other cameras naively |
| `3_proportional_fit.yaml` | multi-source, no weighting, resized | 2 | source-local mosaic, shuffling/seeding, target anchors |
| `4_proportional_native.yaml` | multi-source, no weighting, native scale | 3 | native scale |
| `5_weighted_native.yaml` | multi-source, weighted, native scale | 4 | weighting |
| `6_weighted_native_rfs.yaml` | as 5, plus repeat-factor sampling | 5 | oversampling rare classes |

---

## What was tested

All on CPU with synthetic data (generated frames with known objects; real camera data and GPUs were not available):

- Stock pipeline unchanged: a seeded 2-epoch run reproduces the pre-change `results.txt` exactly (re-checked after each
  change to the data loading).
- Seeded runs are reproducible, with and without multi-source data, native scale, distillation, repeat-factor
  sampling, IR inversion and grayscale; the multi-source data stream is identical with 0 or more workers and across
  resume and loader rebuilds.
- Multi-source quotas are exact every epoch and follow weight schedules; the per-epoch log matches them. Sources with
  several paths are pooled (train and val).
- Per-area AP matches pycocotools; frames per sample predicted by the data plan match measurements within ~0.5.
- Distillation gradients vanish when student and teacher are identical; distillation losses fall during training;
  P5 and P6 teachers pair correctly with the tiny student.
- Label caches: naming, validity check, deletion on normal end, Ctrl+C and SIGTERM.
- IR inversion: the border is untouched and the inside inverted, at native size and resized, also through mosaic and
  zoom; ~50% of training frames inverted at `px_inversion_prob: 0.5`; val untouched except with `--invert`; missing /
  bad borders refused. `to_gray`: 27% of frames gray at 0.3, every train and val frame at 1.0; gray then inverted
  composes correctly.

## Known limitations

- **Not tested on GPU, in multi-GPU (DDP) mode or on real data.** Run a short smoke test first.
- `train_aux.py` does not support multi-source data, native scale, the training recipe options or cache deletion.
- Native-scale validation uses full frames: about 2.3x the memory per image of a 640 crop at 1280x720 (the validation
  batch is 2x `--batch-size`).
- Resumed runs are not bit-identical to uninterrupted ones (fp16 checkpoints).
- `--evolve` is untested with the new options, and `--fitness-area-weights` is refused with it.
- Repeat-factor sampling is only available through a multi-source data yaml (a single source works).

## Citation

YOLOv7 is by Chien-Yao Wang, Alexey Bochkovskiy and Hong-Yuan Mark Liao:

```
@inproceedings{wang2023yolov7,
  title={{YOLOv7}: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors},
  author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2023}
}
```

```
@article{wang2023designing,
  title={Designing Network Design Strategies Through Gradient Path Analysis},
  author={Wang, Chien-Yao and Liao, Hong-Yuan Mark and Yeh, I-Hau},
  journal={Journal of Information Science and Engineering},
  year={2023}
}
```

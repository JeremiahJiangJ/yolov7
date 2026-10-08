# Multi-source (weighted) training

Train on several datasets ("sources") at once, with one **target** source (the deployment camera) and per-source
weights. Enabled by a `train_sources:` list in the data yaml; a plain `train:` / `val:` data yaml keeps the stock
YOLOv7 pipeline unchanged.

Terms used below and in the logs:

- **sample**: one training input, an `--img-size` x `--img-size` crop (usually a mosaic). Every sample has the same
  pixels and compute, and is built entirely from one source.
- **training share** of a source: its share of the samples, i.e. of the training pixels and compute. **This is what
  the weights set**, exactly.
- **frame**: one image of a source. A sample is built from several frames: the sampled frame plus its mosaic, mixup and
  paste-in partners from the same source (**frames loaded**; some may be only partly in view). Smaller frames fill a
  mosaic with more of them, so a source's share of the frames can differ a lot from its training share.
- **objects**: labelled boxes in the samples, i.e. the supervision. A source's share of the objects also depends on how
  many objects its frames contain.

Example configs (an experiment ladder, each step changes one thing):

| Config | Data loading | Compare with | Isolates |
|---|---|---|---|
| `1_default_target.yaml` | stock, target camera only | | |
| `2_default_pooled.yaml` | stock, all cameras pooled | 1 | adding the other cameras naively |
| `3_proportional_fit.yaml` | multi-source, no weighting, resized | 2 | source-local mosaic, shuffling/seeding, target anchors |
| `4_proportional_native.yaml` | multi-source, no weighting, native scale | 3 | native scale |
| `5_weighted_native.yaml` | multi-source, weighted, native scale | 4 | weighting |
| `6_weighted_native_rfs.yaml` | as 5, plus repeat-factor sampling | 5 | oversampling rare classes |

`scripts/train_mixed_ladder.sh` trains and tests all six with identical settings (`STEPS="1 5 6"` for a subset).
All changes in this fork, beyond multi-source training: [`CHANGES.md`](../../CHANGES.md).

## Data yaml

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

Per-source keys:

| Key | Default | Meaning |
|---|---|---|
| `name` | `source<i>` | used in logs, `weight_schedule`, `epoch_size`, `test.py --source` |
| `path` | required | training images (dir, list file, or list of those) |
| `weight` | required for `weighted` | relative training share (normalised, any scale): 60 / 40 = 60% / 40% of the samples |
| `target` | `false` | exactly one source: its `val` drives fitness / `best.pt` / `--patience`; autoanchor uses it |
| `val` | none | target: required (or a top-level `val:`). Other sources: opt-in validation, reported only |
| `resize` | top-level `resize`, else `fit` | `fit`: long side resized to `--img-size` (stock YOLOv7). `native`: never resized, objects keep their pixel size, `--img-size` is the training crop size. Number: fixed scale factor, for sources whose objects are at a different pixel scale |
| `fg_crop_prob` | top-level, else 0.5 | `native` / factor only: chance a mosaic tile or crop is placed around an object (else at random) |
| `mosaic_max_cells` | top-level, else 6 | `native` / factor only: max mosaic cells per axis for frames smaller than the tiles (see below). Lower it if data loading is the bottleneck |
| `repeat_factor_threshold` | top-level, else 0 (off) | repeat-factor sampling: frames with classes in fewer than this fraction of the source's frames are sampled more often (see below) |
| `label_folder` | top-level, else `--label-folder-name` | label folder next to `images` |
| `cache_images` | target: `--cache-images`, others: off | cache this source's images in RAM |
| `cache_path`, `val_cache_path` | from `--train-cache-path` / `--test-cache-path`, else next to the labels | this source's label cache file or directory (overrides the flags) |

## How it works

- **Sampling.** Epoch *e* has exactly `round(w_k(e) * epoch_size)` samples from source *k*, each built around one
  sampled frame. Each source is an endless stream of shuffled passes over its frames, so every frame is sampled once
  before any repeats. The mix is logged at the start and whenever it changes.
- **Samples stay within a source.** Mosaic, mixup and paste-in partners come from the source of the sampled frame, so
  a weight of 50% means 50% of the samples (training pixels and compute) come entirely from that source. It does
  **not** mean 50% of the frames or objects: those also depend on frame size (fill mosaic, below) and object
  density. The data plan and the per-epoch data log show all three shares.
- **Native scale.** Images are never resized. Training samples are `--img-size` crops: a 4-tile mosaic where each
  frame is offset at random within its tile, or with probability `fg_crop_prob` placed so one of its objects is in
  view (the stock mosaic anchors each frame's corner at the mosaic centre, which with large frames almost never
  shows objects near the frame centre). Validation uses full frames padded to a multiple of 32 (1280x720 ->
  1280x736), i.e. what `detect.py --img-size 1280` feeds the model. Autoanchor uses native object sizes.
- **Fill mosaic.** Along an axis where frames are shorter than `--img-size` (640x480 frames at 1280 on both axes; the
  720 px height of 1280x720 frames at 1280), a 4-tile mosaic would leave much of the sample grey. Along that axis the
  mosaic canvas is instead cut into cells the size of the frame, laid out from the random mosaic centre (up to
  `mosaic_max_cells` cells per axis). Only cells that can end up in the sample are loaded: with no rotation / shear /
  perspective, the sample reaches at most `img_size * (0.5 / smallest zoom + translate)` from the canvas centre. The
  sampled image always goes to a cell overlapping the centre of the view, so it is seen.

  Measured on synthetic JPEG frames (one CPU core, data loading only; `hyp.scratch.tiny.yaml`; zoom = scale
  augmentation, see below):

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
- **Scale augmentation.** The stock zoom is `uniform(1 - scale, 1.1 + scale)`, i.e. 0.5-1.6x with `scale: 0.5`, which
  can shrink objects of a few pixels to almost nothing. At native scale, set the zoom range explicitly in the hyp yaml:

  ```yaml
  scale_min: 0.8   # little zoom-out: small objects stay visible
  scale_max: 2.0   # the largest zoom the deployment camera uses
  ```

  The zoom is then sampled log-uniformly (zooming in and out by the same factor equally likely). Without these keys
  the stock behaviour is unchanged. Labels that augmentation shrinks below 2 px are dropped by YOLOv7: the per-epoch
  data log counts them per source (`labels shrunk < 2px`).
- **Repeat-factor sampling.** With `repeat_factor_threshold: t`, frame i of the source gets a repeat factor
  r_i = max(1, max over its classes of √(t / f_c)), f_c = fraction of the source's frames containing class c (LVIS).
  Each pass over the source then holds frame i r_i times on average, and its mosaic / mixup / paste-in partners are
  drawn in proportion to r too. Rare classes are boosted gently (a class 100x below the threshold about 10x) and no
  frame is dropped. With `epoch_size: target`, `--epochs` counts passes over this longer stream. See
  `tools/class_audit.py` for class frequencies and `test.py --freq-groups` for rare / common / frequent mAP.
- **Reproducibility.** With `--seed`, the data of every epoch (order and augmentations) depends only on the seed,
  config and epoch: not on `--workers`, resume or loader rebuilds (`--close-mosaic`). A source's stream is the same
  whatever the other sources' weights are, so two weightings are compared on the same random draws. (Resumed runs
  still differ slightly from uninterrupted ones because YOLOv7 checkpoints store fp16 weights.)

Not supported with `train_sources`: `--image-weights`, `--rect` (train). `train_aux.py` does not support
`train_sources`.

## Label caches

Pass `--train-cache-path` / `--test-cache-path` to keep every label cache of a run in one place, then delete that
folder afterwards. Training prints all cache files it uses (`label caches: ...`). With a multi-source yaml there is
one cache per source, named after the source:

| Flag value | Stock data yaml | `train_sources` data yaml |
|---|---|---|
| `cache/exp1.cache` (a `.cache` file) | exactly that file | `cache/exp1.<source>.train.cache` (`.val.cache` for `--test-cache-path`) |
| `cache/exp1` (a directory, created if needed) | `cache/exp1/labels_<hash>.cache` | `cache/exp1/<source>.train.cache` (`.val.cache`) |
| not given | next to the labels (upstream) | next to each source's labels |

**Caches are deleted when the run ends**, finished or stopped (error, Ctrl+C, SIGTERM / SIGHUP from a scheduler), so
every run reads the current label files; `--keep-cache` keeps them (also for `test.py`). A hard kill (SIGKILL,
out-of-memory killer) cannot be caught: delete the caches listed at startup by hand after one. A cache built for a
different image list or label folder is rebuilt automatically, but edited labels with the same images are not detected.

## Data plan and data log

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
from the images; `--measure 64` also builds samples to count objects, which reads all labels):

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

## Epoch size

With weighted sampling there is no natural "one pass over the data", so `epoch_size` sets the epoch length (fixed
for the run, as the training loop needs constant batches per epoch):

- **`target` (default): `--epochs` = passes over the target**, exactly as when training on the target alone, whatever
  the weights and schedule: each target frame is sampled `--epochs` times. (With mosaic on, it is also loaded as a
  partner in other target samples, as in stock YOLOv7; the plan's `loads/frame` counts both.) The weights only decide
  how much other data is added, i.e. how long training takes. (The epoch is the target size divided by its weight
  averaged over the run, so with a schedule raising the target weight, early epochs hold fewer target samples and
  late epochs more.)
- `total`: as many samples per epoch as frames in all sources, the same iterations per epoch as `2_default_pooled.yaml`, so the same
  `--epochs` is the same training budget (used in the ladder)
- a source name: like `target`, anchored on that source instead
- an int

Everything YOLOv7 counts in epochs scales with the epoch length: the LR schedule, `warmup_epochs` (at least 1000
iterations), validation frequency, `--close-mosaic`, `--patience` and checkpoint saving. Schedule points given as
`progress` scale with `--epochs`; points given as `epoch` do not. Check a config with the dry run below: it shows
the training samples and per-source views of the whole run.

## Validation and testing

`best.pt` and all fitness options (`--fitness-metric-weights`, `--area-int`, `--fitness-area-weights`) use the target
source's val set. Non-target sources with a `val` are validated every epoch at their own scale and logged to
`source_results.txt` and TensorBoard (`metrics_source/<name>/...`), to watch for forgetting.

`test.py` accepts the same yaml: it tests the target's val set by default, `--source <name>` another source's,
each at its own `resize`.

## Inference at native scale

Set `detect.py --img-size` to the camera's long side (1280 for 1280x720): frames are then padded to a multiple of 32
and never resized, matching validation.

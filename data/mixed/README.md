# Multi-source (weighted) training

Train on several datasets ("sources") at once, with one **target** source (the deployment camera) and per-source
sampling weights. Enabled by a `train_sources:` list in the data yaml; a plain `train:` / `val:` data yaml keeps the
stock YOLOv7 pipeline unchanged.

Example configs (an experiment ladder, each step changes one thing):

| Config | Data loading | Compare with | Isolates |
|---|---|---|---|
| `1_default_target.yaml` | stock, target camera only | | |
| `2_default_pooled.yaml` | stock, all cameras pooled | 1 | adding the other cameras naively |
| `3_proportional_fit.yaml` | multi-source, no weighting, resized | 2 | source-local mosaic, shuffling/seeding, target anchors |
| `4_proportional_native.yaml` | multi-source, no weighting, native scale | 3 | native scale |
| `5_weighted_native.yaml` | multi-source, weighted, native scale | 4 | weighting |

`scripts/train_mixed_ladder.sh` trains all five with identical settings.

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
    - {epoch: 200, weights: {camC: 60, camA: 25, camB: 15}}
epoch_size: camC          # optional: int | source name | omitted (= total images)
nc: 1
names: ['object']
```

Per-source keys:

| Key | Default | Meaning |
|---|---|---|
| `name` | `source<i>` | used in logs, `weight_schedule`, `epoch_size`, `test.py --source` |
| `path` | required | training images (dir, list file, or list of those) |
| `weight` | required for `weighted` | relative sampling weight (normalised, any scale) |
| `target` | `false` | exactly one source: its `val` drives fitness / `best.pt` / `--patience`; autoanchor uses it |
| `val` | none | target: required (or a top-level `val:`). Other sources: opt-in validation, reported only |
| `resize` | top-level `resize`, else `fit` | `fit`: long side resized to `--img-size` (stock YOLOv7). `native`: never resized, objects keep their pixel size, `--img-size` is the training crop size. Number: fixed scale factor, for sources whose objects are at a different pixel scale |
| `fg_crop_prob` | top-level, else 0.5 | `native` / factor only: chance a mosaic tile or crop is placed around an object (else at random) |
| `mosaic_max_cells` | top-level, else 6 | `native` / factor only: max mosaic cells per axis for frames smaller than the tiles (see below). Lower it if data loading is the bottleneck |
| `label_folder` | top-level, else `--label-folder-name` | label folder next to `images` |
| `cache_images` | target: `--cache-images`, others: off | cache this source's images in RAM |
| `cache_path`, `val_cache_path` | from `--train-cache-path` / `--test-cache-path`, else next to the labels | this source's label cache file or directory (overrides the flags) |

## How it works

- **Sampling.** Epoch *e* draws exactly `round(w_k(e) * epoch_size)` images from source *k*. Each source is an endless
  stream of shuffled passes, so every image is seen once before any repeats. The mix is logged at the start and
  whenever it changes (images/epoch and passes/epoch per source).
- **Mosaic stays within a source.** Mosaic, mixup and paste-in partners come from the source of the sampled image,
  so a 50% weight really means 50% of the training images.
- **Native scale.** Images are never resized. Training samples are `--img-size` crops: a 4-tile mosaic where each
  frame is offset at random within its tile, or with probability `fg_crop_prob` placed so one of its objects is in
  view (the stock mosaic anchors each frame's corner at the mosaic centre, which with large frames almost never
  shows objects near the frame centre). Validation uses full frames padded to a multiple of 32 (1280x720 ->
  1280x736), i.e. what `detect.py --img-size 1280` feeds the model. Autoanchor uses native object sizes.
- **Fill mosaic.** Frames shorter than 1.5x `--img-size` along an axis (e.g. 640x480 frames at `--img-size 1280`)
  would leave much of a 4-tile mosaic grey. Along that axis the mosaic canvas is instead cut into cells the size of
  the frame, laid out from the random mosaic centre, so every cell is filled (up to `mosaic_max_cells` cells per axis,
  i.e. up to its square of images loaded per sample). Measured on synthetic 640x480 frames at `--img-size 1280`: 54%
  grey with 4 tiles, 6% with the fill mosaic (about 32 images loaded per sample, 1.5x the loading time).
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

## Exposure plan

Training logs, and writes to `data_plan.txt`, how often each source is seen over the whole run: training samples,
share, views per image and the share of images never drawn (exact: the draws are fixed in advance). Check a config
without training:

```
python -m utils.mixed_data --data data/mixed/5_weighted_native.yaml --epochs 100
```
```
100 epochs x 2857 images = 285700 training samples
  source             images    samples   share   views per image  never seen
  camC                 2000     200000   70.0%     100.0 (  100)          0%
  camA                35000      28600   10.0%       0.8 (  0-1)         18%
  camB                 7000      57100   20.0%       8.2 (  8-9)          0%
  WARNING: 18% of camA (6400 images) is never seen in this run
```

## Epoch size

With weighted sampling there is no natural "one pass over the data", so the epoch length is set by `epoch_size`
(fixed for the whole run):

- omitted: total images over all sources, the same number of iterations per epoch as `2_default_pooled.yaml`
- a source name, e.g. `camC`: `len(camC) / weight(camC at epoch 0)`, i.e. one pass over that source per epoch at the
  start (with 2k target images at weight 0.4: 5k images per epoch)
- an int

Everything YOLOv7 counts in epochs scales with it: `--epochs`, the LR schedule, `warmup_epochs` (at least 1000
iterations), validation frequency, `--close-mosaic`, `--patience`, `weight_schedule` points and checkpoint saving.
So:

- **Comparing against stock training: keep the training budget (epochs x images per epoch) equal.** Omitting
  `epoch_size` matches `2_default_pooled.yaml` at the same `--epochs`. Against `1_default_target.yaml` (epoch = one
  pass over the target), adjust `--epochs` so `epochs x epoch_size` matches.
- **With short epochs** (e.g. `epoch_size: camC`), scale the epoch-denominated settings with it, and expect
  validation to take a larger share of the run time (it runs every epoch, once per source with a `val`).

## Validation and testing

`best.pt` and all fitness options (`--fitness-metric-weights`, `--area-int`, `--fitness-area-weights`) use the target
source's val set. Non-target sources with a `val` are validated every epoch at their own scale and logged to
`source_results.txt` and TensorBoard (`metrics_source/<name>/...`), to watch for forgetting.

`test.py` accepts the same yaml: it tests the target's val set by default, `--source <name>` another source's,
each at its own `resize`.

## Inference at native scale

Set `detect.py --img-size` to the camera's long side (1280 for 1280x720): frames are then padded to a multiple of 32
and never resized, matching validation.

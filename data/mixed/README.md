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
| `label_folder` | top-level, else `--label-folder-name` | label folder next to `images` |
| `cache_images` | target: `--cache-images`, others: off | cache this source's images in RAM |
| `cache_path`, `val_cache_path` | next to the labels | label cache file or directory |

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
- **Reproducibility.** With `--seed`, the data of every epoch (order and augmentations) depends only on the seed,
  config and epoch: not on `--workers`, resume or loader rebuilds (`--close-mosaic`). A source's stream is the same
  whatever the other sources' weights are, so two weightings are compared on the same random draws. (Resumed runs
  still differ slightly from uninterrupted ones because YOLOv7 checkpoints store fp16 weights.)

Not supported with `train_sources`: `--image-weights`, `--rect` (train), `--train-cache-path` / `--test-cache-path`
(use `cache_path` / `val_cache_path`). `train_aux.py` does not support `train_sources`.

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

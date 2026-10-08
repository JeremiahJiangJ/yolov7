# Multi-source training data: weighted sampling over several datasets ("sources"), one of which is the target.
#
# Enabled by a `train_sources:` list in the data yaml (see data/mixed/README.md). A plain `train:` / `val:` data yaml
# keeps the stock YOLOv7 pipeline.
#
# Each source is its own LoadImagesAndLabels, so mosaic / mixup / paste-in partners come from the SAME source as the
# sampled image: every training sample is built from the source the sampler picked. (With one pooled dataset, 3 of the
# 4 mosaic tiles would be drawn uniformly from all images, pulling exposure back towards raw dataset sizes.)
#
# Sampling: epoch e draws exactly round(w_k(e) * epoch_size) images from source k, with weights w_k optionally
# following a schedule. By default the epoch size is set so that --epochs means passes over the target, as when
# training on the target alone: weights only decide how much other data is added (see MixedConfig.resolve_epoch_size). Each source is an endless stream of shuffled passes over its images, so every image is seen
# once before any repeats.
#
# Reproducibility: the draw for an epoch is a pure function of (seed, config, epoch): pass c over source k is a
# permutation seeded by hash(seed, k, c), the sample at stream position (c, p) is augmented with RNG seed
# hash(seed, k, 'aug', c, p), and each source's stream position at the start of epoch e is the sum of its earlier
# quotas. Resume, loader rebuilds and the number of workers therefore cannot change what is trained on, and there is
# no sampler state to checkpoint. A source's stream (order and augmentations) is also identical whatever the other
# sources' weights are, so two weightings are compared on the same random draws.

import hashlib
import logging
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from utils.datasets import STAT_SAMPLES, STAT_IMAGES, STAT_LABELS
from utils.datasets import LoadImagesAndLabels, InfiniteDataLoader, DEFAULT_LABEL_FOLDER, parse_resize, \
    create_dataloader
from utils.torch_utils import torch_distributed_zero_first

logger = logging.getLogger(__name__)


def is_mixed(data_dict):
    return 'train_sources' in data_dict


def stable_hash(*parts):
    # 63-bit hash, identical across processes and machines (python's hash() is salted per process)
    return int.from_bytes(hashlib.blake2b(repr(parts).encode(), digest_size=8).digest(), 'little') & (2 ** 63 - 1)


# ----------------------------------------------------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class Source:
    name: str
    path: object  # train images: dir, list file or list of those
    weight: float
    target: bool = False
    val: object = None  # val images; for non-target sources this opts in to per-source validation
    resize: object = 'fit'  # see utils.datasets.parse_resize()
    fg_crop_prob: float = 0.5
    mosaic_max_cells: int = 6
    repeat_factor_threshold: float = 0.0  # > 0: repeat-factor sampling of frames with rare classes (see repeat_factors())
    cache_images: Optional[bool] = None  # None: target follows --cache-images, other sources are not cached
    label_folder: Optional[str] = None  # None: --label-folder-name
    cache_path: Optional[str] = None  # train labels cache file or dir; None: next to the labels
    val_cache_path: Optional[str] = None


def _normalize(w, what):
    w = np.asarray(w, dtype=np.float64)
    assert np.isfinite(w).all() and (w >= 0).all() and w.sum() > 0, f'{what}: weights must be >= 0 and not all 0'
    return w / w.sum()


class WeightSchedule:
    """Per-epoch source weights. The train_sources weights apply from epoch 0; weight_schedule points change them:

        weight_schedule:
          mode: step      # step: switch at each point; linear: ramp from the previous point to this one
          points:
            - {progress: 0.6, weights: {target: 60, supplementary: 25, open_source: 10, synthetic: 5}}

    A point is placed at `progress` (fraction of --epochs, so the schedule scales with the run length) or at an absolute
    `epoch`. Every point lists every source (0 switches one off), so nothing is dropped silently."""

    def __init__(self, names, base_weights, spec=None, epochs=None):
        spec = spec or {}
        self.mode = spec.get('mode', 'step')
        assert self.mode in ('step', 'linear'), f'weight_schedule mode must be step or linear, got {self.mode}'
        self.points = [(0, _normalize(base_weights, 'train_sources'))]
        for p in spec.get('points', []):
            assert ('epoch' in p) != ('progress' in p), f'weight_schedule point needs epoch or progress: {p}'
            if 'progress' in p:
                assert epochs, 'weight_schedule progress points need the number of epochs'
                e = int(round(float(p['progress']) * epochs))
            else:
                e = int(p['epoch'])
            ws = p['weights']
            assert e > self.points[-1][0], 'weight_schedule points need strictly increasing epochs > 0'
            assert set(ws) == set(names), f'weight_schedule point at epoch {e} must list exactly {names}, got {list(ws)}'
            self.points.append((e, _normalize([ws[n] for n in names], f'weight_schedule epoch {e}')))

    def __call__(self, epoch):
        i = max(j for j, (e, _) in enumerate(self.points) if e <= epoch)
        if self.mode == 'step' or i == len(self.points) - 1:
            return self.points[i][1]
        (e0, w0), (e1, w1) = self.points[i], self.points[i + 1]
        return w0 + (w1 - w0) * (epoch - e0) / (e1 - e0)

    def ever_positive(self):
        return np.max([w for _, w in self.points], 0) > 0

    def signature(self):
        return self.mode, tuple((e, tuple(np.round(w, 12).tolist())) for e, w in self.points)


class MixedConfig:
    """Parsed `train_sources` data yaml. Also points data_dict['train'] / ['val'] at the target source, for code that
    reads them (check_dataset, W&B)."""
    KEYS = {'name', 'path', 'weight', 'target', 'val', 'resize', 'fg_crop_prob', 'mosaic_max_cells',
            'repeat_factor_threshold', 'cache_images',
            'label_folder', 'cache_path', 'val_cache_path'}

    def __init__(self, data_dict, label_folder_name=DEFAULT_LABEL_FOLDER):
        srcs = data_dict['train_sources']
        assert isinstance(srcs, list) and srcs, 'train_sources must be a non-empty list'
        self.sampling = data_dict.get('sampling', 'weighted')
        assert self.sampling in ('weighted', 'proportional'), \
            f"sampling must be 'weighted' or 'proportional', got {self.sampling!r}"
        # top-level defaults for the per-source keys of the same name
        default_resize = data_dict.get('resize', 'fit')
        default_fg = float(data_dict.get('fg_crop_prob', 0.5))
        default_cells = int(data_dict.get('mosaic_max_cells', 6))
        default_rfs = float(data_dict.get('repeat_factor_threshold', 0.0))
        default_label_folder = data_dict.get('label_folder') or label_folder_name
        self.sources = []
        for i, s in enumerate(srcs):
            unknown = set(s) - self.KEYS
            assert not unknown, f'train_sources[{i}]: unknown keys {sorted(unknown)}, expected {sorted(self.KEYS)}'
            assert 'path' in s, f'train_sources[{i}] needs a path'
            if self.sampling == 'weighted':
                assert 'weight' in s, f"train_sources[{i}] needs a weight (or use sampling: proportional)"
            self.sources.append(Source(
                name=str(s.get('name', f'source{i}')), path=s['path'], weight=float(s.get('weight', 1)),
                target=bool(s.get('target', False)), val=s.get('val'),
                resize=parse_resize(s.get('resize', default_resize)),
                fg_crop_prob=float(s.get('fg_crop_prob', default_fg)),
                mosaic_max_cells=int(s.get('mosaic_max_cells', default_cells)),
                repeat_factor_threshold=float(s.get('repeat_factor_threshold', default_rfs)),
                cache_images=s.get('cache_images'),
                label_folder=s.get('label_folder') or default_label_folder, cache_path=s.get('cache_path'),
                val_cache_path=s.get('val_cache_path')))
        self.names = [s.name for s in self.sources]
        assert len(set(self.names)) == len(self.names), f'train_sources names must be unique, got {self.names}'
        targets = [i for i, s in enumerate(self.sources) if s.target]
        assert len(targets) == 1, f'exactly one train_source needs target: true, got {[self.names[i] for i in targets]}'
        self.target_idx = targets[0]
        if self.target.val is None:  # allow a top-level val: for the target
            self.target.val = data_dict.get('val')
        assert self.target.val, f'target source "{self.target.name}" needs a val path'

        if self.sampling == 'proportional':
            ignored = [k for k in ('weight_schedule', 'epoch_size') if k in data_dict]
            if ignored:
                logger.warning(f'sampling: proportional ignores {ignored}')
            self.schedule_spec, self.epoch_size = None, None
        else:
            self.schedule_spec = data_dict.get('weight_schedule')
            self.epoch_size = data_dict.get('epoch_size', 'target')
            if self.epoch_size == 'target':
                self.epoch_size = self.target.name
            if isinstance(self.epoch_size, str):
                assert self.epoch_size in self.names + ['total'], \
                    f"epoch_size must be 'target', 'total', a source name or an int, got {self.epoch_size!r}"

        data_dict['train'], data_dict['val'] = self.target.path, self.target.val

    @property
    def target(self):
        return self.sources[self.target_idx]

    def schedule(self, sizes, epochs=None):
        # sizes: frames per source pass (the repeat-factor stream length where repeat factors are used)
        if self.sampling == 'proportional':  # every frame once per epoch (rfs: every stream position)
            return WeightSchedule(self.names, sizes)
        return WeightSchedule(self.names, [s.weight for s in self.sources], self.schedule_spec, epochs)

    def resolve_epoch_size(self, sizes, schedule, epochs, stream=None):
        """Samples per epoch, fixed for the whole run (the training loop needs constant batches per epoch).
        epoch_size: <source name> (default: the target): sized so that over `epochs` epochs every frame of that source
          is sampled `epochs` times, whatever the weights and schedule -- i.e. --epochs means passes over that source,
          like training on it alone, and the other sources are added on top. Epoch = len(source) / its weight averaged
          over the run, so with a schedule raising its weight it gets fewer samples than that early on, more late.
        'total': as many samples as frames in all sources (the epoch of a pooled stock run, for equal budgets).
        int: that many samples.
        stream: per-source pass lengths with repeat factors (default: sizes); a pass over the anchor source then
        includes its repeated frames."""
        es = self.epoch_size
        if es is None or es == 'total':
            return int(sum(sizes))
        if isinstance(es, str):
            k = self.names.index(es)
            mean_w = float(np.mean([schedule(e)[k] for e in range(max(epochs, 1))]))
            assert mean_w > 0, f'epoch_size source "{es}" has weight 0 throughout'
            return max(int(round((stream or sizes)[k] / mean_w)), 1)
        assert int(es) > 0, 'epoch_size must be > 0'
        return int(es)


def source_cache_path(user_path, name, kind):
    """Label cache of source `name` (kind: train / val) under --train-cache-path / --test-cache-path:
    a .cache file 'cache/exp1.cache' -> cache/exp1.<name>.<kind>.cache, a directory 'cache/exp1' ->
    cache/exp1/<name>.<kind>.cache. None if no path was given."""
    if not user_path:
        return None
    p = Path(user_path)
    if p.suffix == '.cache' and not p.is_dir():
        return str(p.with_name(f'{p.stem}.{name}.{kind}.cache'))
    return str(p / f'{name}.{kind}.cache')


def repeat_factors(frame_classes, t):
    """Repeat-factor sampling (LVIS, Gupta et al. 2019): frame i gets r_i = max(1, max over its classes c of sqrt(t / f_c)),
    f_c = fraction of the source's frames containing c; frames without labels get 1. A class in fewer than t x frames
    of its source is boosted (by the square root, so a class 100x below the threshold is repeated 10x, not 100x).
    frame_classes: per frame, an array of its label classes. t <= 0: all 1 (off)."""
    present = [np.unique(np.asarray(c, dtype=int)) for c in frame_classes]
    r = np.ones(len(present))
    if t <= 0 or not present:
        return r
    counts = np.bincount(np.concatenate(present + [np.zeros(0, dtype=int)]), minlength=1)
    boost = np.maximum(1.0, np.sqrt(t / np.maximum(counts / len(present), 1e-12)))
    return np.array([boost[p].max() if len(p) else 1.0 for p in present])


def repeat_summary(name, t, r):
    t = '' if t is None else f' (threshold {t:g})'
    return (f'repeat factors {name}{t}: {np.mean(r > 1):.1%} of frames boosted, mean {r.mean():.2f}x, '
            f'max {r.max():.1f}x, pass = {int(round(r.sum()))} samples for {len(r)} frames')


def largest_remainder(weights, total):
    # Integer counts summing exactly to `total`, as close as possible to weights * total
    raw = np.asarray(weights) * total
    counts = np.floor(raw).astype(int)
    for k in np.argsort(-(raw - counts), kind='stable')[:total - counts.sum()]:
        counts[k] += 1
    return counts


# ----------------------------------------------------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------------------------------------------------
class MixedDataset(Dataset):
    """Per-source LoadImagesAndLabels behind one index space. Indexed with (index, aug_seed) from MixedSourceSampler."""

    def __init__(self, datasets, names, target_idx):
        self.datasets, self.names, self.target_idx = datasets, names, target_idx
        self.sizes = [len(d) for d in datasets]
        self.offsets = np.cumsum([0] + self.sizes[:-1]).tolist()
        # Read by train.py: the class range check and class weights cover every source
        self.labels = [l for d in datasets for l in d.labels]
        self.shapes = np.concatenate([d.shapes for d in datasets], 0)
        self.img_files = [f for d in datasets for f in d.img_files]
        self.n = len(self.labels)

    def __len__(self):
        return self.n

    @property
    def target_dataset(self):  # autoanchor runs on the target only
        return self.datasets[self.target_idx]

    def _get(self, index):
        k = int(np.searchsorted(self.offsets, index, side='right') - 1)
        return self.datasets[k][index - self.offsets[k]]

    def __getitem__(self, item):
        if not isinstance(item, (tuple, list)):
            return self._get(item)
        index, aug_seed = item
        in_main_process = torch.utils.data.get_worker_info() is None  # --workers 0
        if in_main_process:  # keep the training loop's RNG streams independent of the data
            state = random.getstate(), np.random.get_state()
        # Everything random about the sample (mosaic partners, crops, flips, HSV, mixup) follows from aug_seed
        random.seed(aug_seed)
        np.random.seed(aug_seed % 2 ** 32)
        try:
            return self._get(index)
        finally:
            if in_main_process:
                random.setstate(state[0])
                np.random.set_state(state[1])

    @property
    def mosaic(self):
        return any(d.mosaic for d in self.datasets)

    def close_mosaic(self):  # see LoadImagesAndLabels.close_mosaic(); rebuild the loader afterwards
        for d in self.datasets:
            d.close_mosaic()

    collate_fn = staticmethod(LoadImagesAndLabels.collate_fn)
    collate_fn4 = staticmethod(LoadImagesAndLabels.collate_fn4)


# ----------------------------------------------------------------------------------------------------------------------
# Sampler
# ----------------------------------------------------------------------------------------------------------------------
class MixedSourceSampler(Sampler):
    """Per-epoch source quotas (optionally scheduled), DDP-aware. Stateless apart from the epoch counter: the draw for
    epoch e is computed from (seed, config, e). __iter__ is called once per epoch on every rank (InfiniteDataLoader),
    and advances the counter; rebuild_loader() and resume set it with rewind() / start_epoch."""

    def __init__(self, sizes, offsets, names, schedule, epoch_size, rank=-1, world_size=1, seed=0, start_epoch=0,
                 repeat=None):
        """repeat: per source, None or its frames' repeat factors (repeat_factors()): each pass over the source then
        holds frame i floor(r_i) times plus, for the fractional parts, a fixed number of extra copies drawn (seeded) in
        proportion to them, so every pass has length round(sum(r))."""
        self.sizes, self.offsets, self.names = list(sizes), list(offsets), list(names)
        self.repeat = list(repeat) if repeat is not None else [None] * len(self.sizes)
        self.stream_len = [n if r is None else int(round(float(np.sum(r)))) for n, r in zip(self.sizes, self.repeat)]
        self.schedule, self.seed = schedule, int(seed)
        for n, size, used in zip(self.names, self.sizes, schedule.ever_positive()):
            if used and size == 0:
                raise ValueError(f'source "{n}" has a positive weight but no images')
        self.rank, self.world_size = max(rank, 0), (world_size if rank != -1 else 1)
        self.num_samples = int(math.ceil(epoch_size / self.world_size))  # per rank
        self.total = self.num_samples * self.world_size  # padded to divide evenly between ranks
        self.next_epoch = start_epoch
        self._counts, self._perms, self._logged = {}, {}, None

    def counts(self, epoch):
        if epoch not in self._counts:
            self._counts[epoch] = largest_remainder(self.schedule(epoch), self.total)
        return self._counts[epoch]

    def consumed_before(self, epoch):
        # images drawn from each source in epochs [0, epoch)
        return sum((self.counts(e) for e in range(epoch)), np.zeros(len(self.sizes), dtype=np.int64))

    def _perm(self, k, cycle):
        if (k, cycle) not in self._perms:
            if len(self._perms) > 4 * len(self.sizes):
                self._perms.clear()
            g = torch.Generator().manual_seed(stable_hash(self.seed, self.names[k], 'perm', cycle))
            r = self.repeat[k]
            if r is None:
                self._perms[(k, cycle)] = torch.randperm(self.sizes[k], generator=g).tolist()
            else:  # repeat-factor pass: floor(r) copies of each frame + extra copies for the fractional parts
                base = np.floor(r).astype(int)
                frames = np.repeat(np.arange(len(r)), base)
                extra = self.stream_len[k] - len(frames)
                if extra > 0:
                    frac = torch.tensor(r - base, dtype=torch.float64)
                    frames = np.concatenate((frames, torch.multinomial(frac, extra, generator=g).numpy()))
                self._perms[(k, cycle)] = frames[torch.randperm(len(frames), generator=g).numpy()].tolist()
        return self._perms[(k, cycle)]

    def _take(self, k, start, count):
        # `count` (index, aug_seed) pairs from source k's stream, from stream position `start`
        out = []
        for pos in range(start, start + count):
            c, p = divmod(pos, self.stream_len[k])
            out.append((self._perm(k, c)[p] + self.offsets[k], stable_hash(self.seed, self.names[k], 'aug', c, p)))
        return out

    def draw(self, epoch):
        # the ordered (index, aug_seed) list of `epoch`, for all ranks
        start, count = self.consumed_before(epoch), self.counts(epoch)
        items = [x for k in range(len(self.sizes)) if count[k] for x in self._take(k, int(start[k]), int(count[k]))]
        g = torch.Generator().manual_seed(stable_hash(self.seed, 'mix', epoch))
        return [items[i] for i in torch.randperm(len(items), generator=g).tolist()]

    def __iter__(self):
        epoch = self.next_epoch
        self.next_epoch += 1
        count = tuple(self.counts(epoch))
        if self.rank == 0 and count != self._logged:  # log mix changes (runs slightly ahead because of prefetching)
            self._logged = count
            logger.info(f'mixed sources from epoch {epoch}: {self.summary(epoch)}')
        return iter(self.draw(epoch)[self.rank:self.total:self.world_size])

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):  # called by train.py in DDP mode; epochs are tracked internally
        pass

    def rewind(self, epoch):  # the next __iter__ draws `epoch`
        self.next_epoch = epoch

    def signature(self):  # stored in checkpoints: a different value on --resume means the data config changed
        return {'seed': self.seed, 'names': self.names, 'sizes': self.sizes, 'total': self.total,
                'stream_len': self.stream_len,
                'schedule': self.schedule.signature()}

    def plan(self, epochs, frames_per_sample=None, close_mosaic=0, objects_per_sample=None):
        """What each source contributes over a run of `epochs` epochs.
        - training share: share of the training samples. Every sample is one --img-size crop, so this is also the
          share of training pixels and compute. Exact (the draws are fixed in advance); this is what weights set.
        - frames loaded: frames (images of the source) loaded to build its samples: the sampled frame plus its mosaic
          / mixup / paste-in partners, all from the same source. frames_per_sample[k]: expected per sample while
          mosaic is on (utils.datasets.expected_images_per_sample()); 1 in the last close_mosaic epochs. A loaded
          frame may be only partly in view. Sources with smaller frames load more frames per sample.
        - never loaded: estimated share of the source's frames not loaded at all (sampled frames cycle through the
          source; partners are drawn at random)
        - objects: labelled objects in its training samples, i.e. its share of the supervision.
          objects_per_sample[k] = (with mosaic, after close_mosaic), measured (see measure_sources()).
        Without frames_per_sample / objects_per_sample those columns are left out."""
        draws = self.consumed_before(epochs).astype(float)
        n_clean = min(max(close_mosaic, 0), epochs)
        clean = (self.consumed_before(epochs) - self.consumed_before(epochs - n_clean)).astype(float)
        mos = draws - clean  # samples built with mosaic on
        total = max(draws.sum(), 1)
        lines = [f'{epochs} epochs x {self.total} samples/epoch = {int(total)} training samples '
                 f'(each one --img-size crop: equal pixels and compute)']
        head = f"  {'source':<14}{'frames':>9}{'training share':>16}{'samples':>10}"
        if frames_per_sample is not None:
            fps = np.asarray(frames_per_sample, dtype=float)
            loaded = mos * fps + clean
            head += f"{'frames/sample':>15}{'frames loaded':>15}{'share':>8}{'loads/frame':>13}{'never loaded':>14}"
        if objects_per_sample is not None:
            ops = np.asarray(objects_per_sample, dtype=float)  # (k, 2): with mosaic, after close_mosaic
            objects = mos * ops[:, 0] + clean * ops[:, 1]
            head += f"{'objects/sample':>16}{'objects':>11}{'share':>8}"
        lines.append(head)
        notes, warnings = [], []
        for k, (n, d, size) in enumerate(zip(self.names, draws, self.sizes)):
            size = max(size, 1)
            row = f'  {n:<14}{size:>9}{d / total:>16.1%}{int(d):>10}'
            shares = []
            if frames_per_sample is not None:
                u = loaded[k]
                rep = self.repeat[k]
                if rep is None:  # sampled frames cycle through the source
                    unsampled = max(0.0, 1 - d / size)
                else:  # a frame with r copies in a pass of L is missed by the first d draws w.p. ~(1 - d / L)^r
                    unsampled = float(np.mean((1 - min(d / self.stream_len[k], 1.0)) ** rep))
                # partners (uniform, or by repeat factor) miss a frame w.p. ~exp(-partners / frames)
                never = unsampled * math.exp(-(u - d) / size)
                row += f'{fps[k]:>15.1f}{int(u):>15}{u / loaded.sum():>8.1%}{u / size:>13.1f}{never:>14.1%}'
                shares.append(('frames loaded', u / loaded.sum()))
                if d > 0 and never >= 0.01:
                    warnings.append(f'~{never:.0%} of {n} (~{int(never * size)} frames) is never loaded in this run')
            if objects_per_sample is not None:
                row += f'{ops[k, 0]:>16.1f}{int(objects[k]):>11}{objects[k] / max(objects.sum(), 1):>8.1%}'
                shares.append(('objects', objects[k] / max(objects.sum(), 1)))
            lines.append(row)
            if d > 0 and any(not 1 / 1.5 < x / (d / total) < 1.5 for _, x in shares):
                notes.append(f'{n}: {d / total:.0%} of the training, ' + ', '.join(f'{x:.0%} of the {what}'
                                                                                   for what, x in shares))
        for n, rep in zip(self.names, self.repeat):
            if rep is not None:
                lines.append(f'  {repeat_summary(n, None, rep)}')
        if objects_per_sample is not None:
            lines.append('  objects/sample: measured on samples built with mosaic on (and separately for the '
                         '--close-mosaic epochs, included in objects)')
        if notes:
            notes.insert(0, 'frame / object shares differ from the training shares by 1.5x or more. Smaller frames '
                            'fill a mosaic with more frames, and object density differs between sources: lower a '
                            "source's weight if it should count less (the weights set training shares)")
        return '\n'.join(lines + [f'  NOTE: {x}' for x in notes] + [f'  WARNING: {w}' for w in warnings])

    def summary(self, epoch=0):
        count = self.counts(epoch)
        rows = [f'{n:<16}{c / self.total:>7.1%} of training{c:>9} samples/epoch ({c / max(size, 1):.2f} x its {size} frames '
                f'sampled)' for n, c, size in zip(self.names, count, self.sizes)]
        return (f'{self.total} samples/epoch (shares of training; frames and objects per source: see the data plan)\n' +
                '\n'.join('  ' + r for r in rows))


# ----------------------------------------------------------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------------------------------------------------------
def create_mixed_dataloader(cfg, imgsz, batch_size, stride, opt, hyp=None, cache=False, rank=-1, world_size=1,
                            workers=8, quad=False, prefix='', start_epoch=0, seed=0, epochs=300):
    """Training loader for a MixedConfig (replaces create_dataloader() for training)."""
    datasets = []
    for k, src in enumerate(cfg.sources):
        cache_images = src.cache_images if src.cache_images is not None else (cache if k == cfg.target_idx else False)
        with torch_distributed_zero_first(rank):
            datasets.append(LoadImagesAndLabels(src.path, imgsz, batch_size, augment=True, hyp=hyp, rect=False,
                                                cache_images=cache_images, single_cls=opt.single_cls,
                                                stride=int(stride), prefix=f'{prefix}[{src.name}] ',
                                                cache_path=src.cache_path or source_cache_path(
                                                    getattr(opt, 'train_cache_path', None), src.name, 'train'),
                                                label_folder_name=src.label_folder,
                                                resize=src.resize, fg_crop_prob=src.fg_crop_prob,
                                                mosaic_max_cells=src.mosaic_max_cells))
    dataset = MixedDataset(datasets, cfg.names, cfg.target_idx)
    # Repeat-factor sampling: sampled frames (sampler streams) and mosaic / mixup / paste-in partners (dataset) are
    # both drawn in proportion to the repeat factors
    repeat = [repeat_factors([l[:, 0] for l in d.labels], src.repeat_factor_threshold)
              if src.repeat_factor_threshold > 0 else None for d, src in zip(datasets, cfg.sources)]
    for d, r in zip(datasets, repeat):
        d.partner_cum_weights = None if r is None else np.cumsum(r).tolist()
    stream = [n if r is None else int(round(r.sum())) for n, r in zip(dataset.sizes, repeat)]
    schedule = cfg.schedule(stream, epochs)
    epoch_size = cfg.resolve_epoch_size(dataset.sizes, schedule, epochs, stream)
    sampler = MixedSourceSampler(dataset.sizes, dataset.offsets, cfg.names, schedule, epoch_size, rank=rank,
                                 world_size=world_size, seed=seed, start_epoch=start_epoch, repeat=repeat)
    if rank in [-1, 0]:
        s = f'sampling {cfg.sampling}, seed {sampler.seed}'
        if cfg.sampling == 'weighted' and len(schedule.points) > 1:
            s += f', weight schedule ({schedule.mode}) from epochs {[e for e, _ in schedule.points]}'
        s += f'\n  target: {cfg.target.name}'
        s += '\n  resize: ' + ', '.join(f'{x.name}={x.resize}' for x in cfg.sources)
        s += '\n  cached: ' + (', '.join(n for n, d in zip(cfg.names, datasets) if d.imgs[0] is not None) or 'none')
        for src, r in zip(cfg.sources, repeat):
            if r is not None:
                s += f'\n  {repeat_summary(src.name, src.repeat_factor_threshold, r)}'
        logger.info(f'{prefix}mixed sources: {s}')  # the sampler logs the per-source mix

    batch_size = min(batch_size, len(sampler))
    nw = min([(os.cpu_count() or 1) // world_size, batch_size if batch_size > 1 else 0, workers])
    loader = InfiniteDataLoader(dataset, batch_size=batch_size, num_workers=nw, sampler=sampler, pin_memory=True,
                                collate_fn=MixedDataset.collate_fn4 if quad else MixedDataset.collate_fn)
    return loader, dataset


def usage_totals(dataset):
    # cumulative usage counters per source (see LoadImagesAndLabels.stats), summed over this process's workers
    parts = zip(dataset.names, dataset.datasets) if isinstance(dataset, MixedDataset) else [('train', dataset)]
    return {n: d.stats_total().clone() for n, d in parts}


def usage_report(seen, now, before):
    """What training used from each source over an epoch.
    seen: {source: (samples, objects)} counted by the training loop on the batches it consumed: exact.
    now / before: usage_totals() snapshots of the data loader counters, for the per-sample ratios the training loop
    cannot see: frames loaded per sample (sampled frame + mosaic / mixup / paste-in partners) and the share of objects
    augmentation shrank below 2 px (dropped). These counters include batches prefetched across epoch boundaries,
    which does not affect the ratios. Frames loaded = samples x frames per sample."""
    c = {n: (now[n] - before[n]).tolist() if n in before else now[n].tolist() for n in now}
    rows = []
    for n, (samples, objects) in seen.items():
        cs, cf, co, cx = c.get(n, [0, 0, 0, 0])
        fps = cf / cs if cs else float('nan')
        rows.append((n, samples, objects, fps, samples * fps if cs else 0, cx / max(co + cx, 1)))
    ts, to, tf = (max(sum(r[i] for r in rows), 1) for i in (1, 2, 4))
    lines = [f"  {'source':<14}{'samples':>9}{'training share':>16}{'objects':>9}{'share':>8}{'objects/sample':>16}"
             f"{'frames/sample':>15}{'frames loaded':>15}{'share':>8}{'objects shrunk < 2 px':>23}"]
    for n, samples, objects, fps, frames, lost in rows:
        lines.append(f'  {n:<14}{samples:>9}{samples / ts:>16.1%}{objects:>9}{objects / to:>8.1%}'
                     f'{objects / max(samples, 1):>16.1f}{fps:>15.1f}{int(frames):>15}{frames / tf:>8.1%}{lost:>23.1%}')
    return '\n'.join(lines)


def measure_sources(datasets, n=64, seed=0):
    """Objects (labels) per training sample of each source, measured on n samples built with mosaic on and n after
    close_mosaic() (objects per sample depend on object density, frame size, zoom and fg_crop_prob, so they are
    measured rather than predicted). Leaves the global RNGs, the datasets and their usage counters as they were.
    Returns [(objects per sample with mosaic, after close_mosaic), ...]."""
    state = random.getstate(), np.random.get_state()
    out = []
    try:
        for k, d in enumerate(datasets):
            res = []
            for phase in ('mosaic', 'clean'):
                saved, counters = (d.mosaic, d.hyp), d.stats[0].clone()
                if phase == 'clean':
                    d.close_mosaic()
                objects = 0
                for j in range(n):
                    h = stable_hash(seed, 'measure', k, phase, j)
                    random.seed(h)
                    np.random.seed(h % 2 ** 32)
                    objects += len(d[h % len(d)][1])
                d.mosaic, d.hyp = saved
                d.stats[0] = counters  # measuring is not training use
                res.append(objects / n)
            out.append(tuple(res))
    finally:
        random.setstate(state[0])
        np.random.set_state(state[1])
    return out


def train_class_frames(data_dict, label_folder_name=DEFAULT_LABEL_FOLDER):
    """Training frames containing each class, read from the label files of a data yaml dict's training data
    (train: or train_sources:; each source's label folder)."""
    from utils.datasets import img2label_paths
    from utils.metrics import class_frames
    if is_mixed(data_dict):
        cfg = MixedConfig(dict(data_dict), label_folder_name)
        parts = [(src.path, src.label_folder) for src in cfg.sources]
    else:
        parts = [(data_dict['train'], label_folder_name)]
    labels = []
    for path, folder in parts:
        for lb in img2label_paths(list_images(path), folder):
            try:
                classes = [float(x.split()[0]) for x in Path(lb).read_text().splitlines() if x.strip()]
            except OSError:
                classes = []
            labels.append(np.array(classes).reshape(-1, 1))
    return class_frames(labels, int(data_dict['nc']))


def list_images(path):
    # Image files a LoadImagesAndLabels(path) would read (before dropping corrupt ones), without reading them
    from utils.datasets import img_formats
    files = []
    for p in path if isinstance(path, list) else [path]:
        p = Path(p)
        if p.is_dir():
            files += [str(x) for x in p.rglob('*.*')]
        elif p.is_file():
            parent = str(p.parent) + os.sep
            files += [x.replace('./', parent) if x.startswith('./') else x for x in p.read_text().strip().splitlines()]
        else:
            raise FileNotFoundError(f'{p} does not exist')
    return [f for f in files if f.split('.')[-1].lower() in img_formats]


def create_val_dataloader(src, imgsz, batch_size, stride, opt, hyp=None, cache=False, world_size=1, workers=8,
                          prefix=''):
    # Validation loader of one source, at its own resize mode (native sources: full frames, padded to the stride)
    return create_dataloader(src.val, imgsz, batch_size, stride, opt, hyp=hyp, cache=cache, rect=True, rank=-1,
                             world_size=world_size, workers=workers, pad=0.5, prefix=prefix,
                             cache_path=src.val_cache_path or source_cache_path(getattr(opt, 'test_cache_path', None),
                                                                                src.name, 'val'),
                             label_folder_name=src.label_folder, resize=src.resize)[0]


if __name__ == '__main__':
    # Dry run: exposure plan of a train_sources data yaml without training, e.g.
    #   python -m utils.mixed_data --data data/mixed/5_weighted_native.yaml --epochs 100
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description='Print the per-source exposure plan of a train_sources data yaml')
    parser.add_argument('--data', required=True, help='data yaml with train_sources')
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--world-size', type=int, default=1, help='number of GPUs (DDP pads epochs to a multiple)')
    parser.add_argument('--hyp', default='data/hyp.scratch.tiny.yaml', help='hyp yaml (mosaic, mixup, paste_in)')
    parser.add_argument('--img-size', type=int, default=640, help='training --img-size')
    parser.add_argument('--close-mosaic', type=int, default=0, help='training --close-mosaic')
    parser.add_argument('--measure', type=int, default=0,
                        help='also measure objects per sample on this many sample batches per source (reads labels)')
    opt = parser.parse_args()
    with open(opt.data) as f:
        cfg = MixedConfig(yaml.safe_load(f))
    with open(opt.hyp) as f:
        hyp = yaml.safe_load(f)
    from PIL import Image
    from utils.datasets import expected_images_per_sample
    files = [list_images(src.path) for src in cfg.sources]
    sizes = [len(f) for f in files]
    ips = []
    for src, f in zip(cfg.sources, files):  # frame size from up to 20 images (headers only)
        wh = []
        for x in f[:20]:
            try:
                with Image.open(x) as im:
                    wh.append(im.size)
            except OSError:
                pass
        if not wh and src.resize != 'fit':
            print(f'WARNING: could not read images of {src.name} to get its frame size, assuming img_size frames')
        wh = np.median(wh, 0) if wh else np.array([opt.img_size, opt.img_size])
        scale = 1.0 if src.resize in ('fit', 'native') else src.resize
        ips.append(expected_images_per_sample(hyp, src.resize, opt.img_size, wh * scale, src.mosaic_max_cells))
    repeat = []  # repeat factors from the label files
    for src, f in zip(cfg.sources, files):
        if src.repeat_factor_threshold <= 0:
            repeat.append(None)
            continue
        from utils.datasets import img2label_paths
        classes = []
        for lb in img2label_paths(f, src.label_folder):
            try:
                classes.append([int(float(x.split()[0])) for x in Path(lb).read_text().splitlines() if x.strip()])
            except OSError:
                classes.append([])
        repeat.append(repeat_factors(classes, src.repeat_factor_threshold))
    stream = [n if r is None else int(round(r.sum())) for n, r in zip(sizes, repeat)]
    schedule = cfg.schedule(stream, opt.epochs)
    sampler = MixedSourceSampler(sizes, np.cumsum([0] + sizes[:-1]).tolist(), cfg.names, schedule,
                                 cfg.resolve_epoch_size(sizes, schedule, opt.epochs, stream), repeat=repeat,
                                 rank=0 if opt.world_size > 1 else -1,
                                 world_size=opt.world_size)
    print(f'sampling {cfg.sampling}, target {cfg.target.name}')
    if len(schedule.points) > 1:
        print(f'weight schedule ({schedule.mode}): ' + '; '.join(
            f'epoch {e}: ' + ', '.join(f'{n} {x:.0%}' for n, x in zip(cfg.names, w)) for e, w in schedule.points))
    objects = None
    if opt.measure:  # build the datasets (scans labels; caches are deleted on exit) and measure objects per sample
        from utils.datasets import LoadImagesAndLabels, delete_label_caches_at_exit
        delete_label_caches_at_exit()
        datasets = [LoadImagesAndLabels(src.path, opt.img_size, 16, augment=True, hyp=hyp, prefix=f'[{src.name}] ',
                                        label_folder_name=src.label_folder, resize=src.resize,
                                        fg_crop_prob=src.fg_crop_prob, mosaic_max_cells=src.mosaic_max_cells)
                    for src in cfg.sources]
        for d, src in zip(datasets, cfg.sources):  # partners by repeat factor, as in training
            if src.repeat_factor_threshold > 0:
                d.partner_cum_weights = np.cumsum(repeat_factors([l[:, 0] for l in d.labels],
                                                                 src.repeat_factor_threshold)).tolist()
        objects = measure_sources(datasets, opt.measure)
    print(sampler.plan(opt.epochs, ips, opt.close_mosaic, objects))
    if not opt.measure:
        print('  (objects per source: add --measure 64 to build 64 samples per source and count their objects)')

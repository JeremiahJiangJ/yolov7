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
    KEYS = {'name', 'path', 'weight', 'target', 'val', 'resize', 'fg_crop_prob', 'mosaic_max_cells', 'cache_images',
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
                mosaic_max_cells=int(s.get('mosaic_max_cells', default_cells)), cache_images=s.get('cache_images'),
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
        if self.sampling == 'proportional':  # every image once per epoch, i.e. shares = dataset sizes
            return WeightSchedule(self.names, sizes)
        return WeightSchedule(self.names, [s.weight for s in self.sources], self.schedule_spec, epochs)

    def resolve_epoch_size(self, sizes, schedule, epochs):
        """Images per epoch, fixed for the whole run (the training loop needs constant batches per epoch).
        epoch_size: <source name> (default: the target): sized so that over `epochs` epochs every image of that source
          is seen `epochs` times, whatever the weights and schedule -- i.e. --epochs means passes over that source,
          like training on it alone, and the other sources are added on top. Epoch = len(source) / its weight averaged
          over the run, so with a schedule raising its weight it gets fewer images than that in early epochs, more late.
        'total': all images of all sources (the epoch of a pooled stock YOLOv7 run, for equal-budget comparisons).
        int: that many images."""
        es = self.epoch_size
        if es is None or es == 'total':
            return int(sum(sizes))
        if isinstance(es, str):
            k = self.names.index(es)
            mean_w = float(np.mean([schedule(e)[k] for e in range(max(epochs, 1))]))
            assert mean_w > 0, f'epoch_size source "{es}" has weight 0 throughout'
            return max(int(round(sizes[k] / mean_w)), 1)
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

    def __init__(self, sizes, offsets, names, schedule, epoch_size, rank=-1, world_size=1, seed=0, start_epoch=0):
        self.sizes, self.offsets, self.names = list(sizes), list(offsets), list(names)
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
            self._perms[(k, cycle)] = torch.randperm(self.sizes[k], generator=g).tolist()
        return self._perms[(k, cycle)]

    def _take(self, k, start, count):
        # `count` (index, aug_seed) pairs from source k's stream, from stream position `start`
        out = []
        for pos in range(start, start + count):
            c, p = divmod(pos, self.sizes[k])
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
                'schedule': self.schedule.signature()}

    def plan(self, epochs, images_per_sample=None, close_mosaic=0):
        """Exposure plan of a run of `epochs` epochs. Per source: training samples (one per sampled image; exact, the
        draws are fixed in advance) and their share; images used to build them (images_per_sample: expected per
        sample with mosaic on, see utils.datasets.expected_images_per_sample(); 1 in the last close_mosaic epochs),
        their share, uses per image, and the estimated share of images never used. Mosaic / mixup / paste-in partners
        come from the same source as the sampled image, so a source with small frames (more mosaic cells) gets a
        larger share of the images than of the samples. Without images_per_sample only sampled images are counted."""
        draws = self.consumed_before(epochs).astype(float)
        k = len(self.sizes)
        ips = np.ones(k) if images_per_sample is None else np.asarray(images_per_sample, dtype=float)
        n_clean = min(max(close_mosaic, 0), epochs)
        clean = self.consumed_before(epochs) - self.consumed_before(epochs - n_clean)  # samples after close_mosaic
        used = (draws - clean) * ips + clean  # images used, sampled images included
        total_s, total_u = max(draws.sum(), 1), max(used.sum(), 1)
        lines = [f'{epochs} epochs x {self.total} samples = {int(total_s)} training samples, '
                 f'~{int(total_u)} images used (sampled images + mosaic / mixup / paste-in partners)']
        lines.append(f"  {'source':<16}{'images':>9}{'samples':>11}{'share':>8}{'images/sample':>15}{'images used':>13}"
                     f"{'share':>8}{'uses per image':>16}{'never used':>12}")
        warnings = []
        for n, d, u, x, size in zip(self.names, draws, used, ips, self.sizes):
            size = max(size, 1)
            # sampled images cycle through the source (each drawn floor or ceil(d / size) times); partners are drawn
            # uniformly at random, missing a given image with probability ~exp(-partners / size)
            never = max(0.0, 1 - d / size) * math.exp(-(u - d) / size)
            lines.append(f'  {n:<16}{size:>9}{int(d):>11}{d / total_s:>8.1%}{x:>15.1f}{int(u):>13}{u / total_u:>8.1%}'
                         f'{u / size:>16.1f}{never:>12.1%}')
            if d > 0 and never >= 0.01:
                warnings.append(f'~{never:.0%} of {n} (~{int(never * size)} images) is never used in this run')
        if images_per_sample is None:
            lines.append('  (images/sample unknown: mosaic partners not counted)')
        return '\n'.join(lines + [f'  WARNING: {w}' for w in warnings])

    def summary(self, epoch=0):
        count = self.counts(epoch)
        rows = [f'{n:<16}{c / self.total:>7.1%}{size:>9} images{c:>9} samples/epoch{c / max(size, 1):>7.2f} passes/epoch'
                for n, c, size in zip(self.names, count, self.sizes)]
        return (f'{self.total} samples/epoch (sampled images; mosaic partners add more images, see the data plan)\n' +
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
    schedule = cfg.schedule(dataset.sizes, epochs)
    epoch_size = cfg.resolve_epoch_size(dataset.sizes, schedule, epochs)
    sampler = MixedSourceSampler(dataset.sizes, dataset.offsets, cfg.names, schedule, epoch_size, rank=rank,
                                 world_size=world_size, seed=seed, start_epoch=start_epoch)
    if rank in [-1, 0]:
        s = f'sampling {cfg.sampling}, seed {sampler.seed}'
        if cfg.sampling == 'weighted' and len(schedule.points) > 1:
            s += f', weight schedule ({schedule.mode}) from epochs {[e for e, _ in schedule.points]}'
        s += f'\n  target: {cfg.target.name}'
        s += '\n  resize: ' + ', '.join(f'{x.name}={x.resize}' for x in cfg.sources)
        s += '\n  cached: ' + (', '.join(n for n, d in zip(cfg.names, datasets) if d.imgs[0] is not None) or 'none')
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


def usage_report(now, before):
    """What the data loader actually used between two usage_totals() snapshots, per source: training samples, images
    used to build them (incl. mosaic / mixup / paste-in partners), labels per sample, and labels dropped because
    augmentation shrank them below 2 px. Counts what was loaded, including batches prefetched for the next epoch."""
    d = {n: (now[n] - before[n]).tolist() if n in before else now[n].tolist() for n in now}
    ts, ti = max(sum(x[0] for x in d.values()), 1), max(sum(x[1] for x in d.values()), 1)
    lines = [f"  {'source':<16}{'samples':>9}{'share':>8}{'images':>9}{'share':>8}{'images/sample':>15}"
             f"{'labels/sample':>15}{'labels shrunk < 2px':>21}"]
    for n, (samples, images, labels, small) in d.items():
        lost = f'{small} ({small / max(labels + small, 1):.1%})'
        lines.append(f'  {n:<16}{samples:>9}{samples / ts:>8.1%}{images:>9}{images / ti:>8.1%}'
                     f'{images / max(samples, 1):>15.1f}{labels / max(samples, 1):>15.1f}{lost:>21}')
    return '\n'.join(lines)


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
    schedule = cfg.schedule(sizes, opt.epochs)
    sampler = MixedSourceSampler(sizes, np.cumsum([0] + sizes[:-1]).tolist(), cfg.names, schedule,
                                 cfg.resolve_epoch_size(sizes, schedule, opt.epochs),
                                 rank=0 if opt.world_size > 1 else -1,
                                 world_size=opt.world_size)
    print(f'sampling {cfg.sampling}, target {cfg.target.name}')
    if len(schedule.points) > 1:
        print(f'weight schedule ({schedule.mode}): ' + '; '.join(
            f'epoch {e}: ' + ', '.join(f'{n} {x:.0%}' for n, x in zip(cfg.names, w)) for e, w in schedule.points))
    print(sampler.plan(opt.epochs, ips, opt.close_mosaic))

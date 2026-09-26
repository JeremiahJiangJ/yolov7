# Mixed-source training data for YOLOv7
#
# Each source (target, supplementary, open-source, synthetic, ...) is its own LoadImagesAndLabels,
# so mosaic / mixup / paste-in partners are drawn from the SAME source: every training sample is built
# entirely from the source the sampler selected. Otherwise 3 of 4 mosaic tiles would come uniformly
# from the pooled data and pull exposure back toward raw dataset sizes. (Exposure is controlled per
# sample; exact pixel shares still vary with mosaic crops and mixup blending.)
#
# Sampling: every epoch draws exactly round(w_k(epoch) * epoch_size) images from source k, where the
# weights may follow a schedule. Each source is an endless stream of shuffled passes ("cycles") over
# its images: every image is seen once before any repeats.
#
# Reproducibility: everything is a pure function of (seed, config, epoch):
#   - cycle c of source k is randperm seeded by hash(seed, name_k, c)
#   - the sample at (cycle c, position p) of source k gets augmentation seed hash(seed, name_k, c, p);
#     workers reseed python/numpy/torch RNGs with it right before building that sample
#   - where each source's stream stands at the start of epoch e is just the sum of its earlier quotas
#   - the per-epoch shuffle of the mixed indices is seeded by hash(seed, 'mix', e)
# So resume, loader rebuilds and worker count cannot change what is trained on, and there is no
# sampler state to checkpoint. It also gives "common random numbers" across experiments: a source's
# stream (order + augmentations) is identical whatever the other sources' weights are, so comparing
# two weightings compares the weightings, not two different random streams.

import hashlib
import logging
import math
import random

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from utils.datasets import LoadImagesAndLabels, InfiniteDataLoader
from utils.torch_utils import torch_distributed_zero_first

logger = logging.getLogger(__name__)


def stable_hash(*parts):
    """63-bit hash that is identical across processes/machines (python's hash() is salted per process)."""
    return int.from_bytes(hashlib.blake2b(repr(parts).encode(), digest_size=8).digest(), 'little') & (2 ** 63 - 1)


def seed_everything(s, torch_rng=True):
    random.seed(s)
    np.random.seed(s % 2 ** 32)
    if torch_rng:  # ~80us, so skipped per sample: the YOLOv7 augmentation pipeline only uses random / np.random
        torch.manual_seed(s)


# ---------------------------------------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------------------------------------
def _normalize(w, what):
    w = np.asarray(w, dtype=np.float64)
    assert np.isfinite(w).all() and (w >= 0).all() and w.sum() > 0, f'{what}: weights must be finite, >= 0, not all 0'
    return w / w.sum()


class WeightSchedule:
    """Per-epoch source weights. `train_sources[*].weight` applies from epoch 0; `weight_schedule` adds points:

        weight_schedule:
          mode: step            # step (default): switch at each epoch; linear: interpolate between points
          points:
            - {epoch: 100, weights: {target: 60, supplementary: 25, open_source: 10, synthetic: 5}}
            - {epoch: 150, weights: {target: 80, supplementary: 15, open_source: 5, synthetic: 0}}

    Every point must list every source (use 0 to switch one off), so nothing is dropped silently.
    """

    def __init__(self, names, base_weights, spec=None):
        self.names = names
        self.mode = (spec or {}).get('mode', 'step')
        assert self.mode in ('step', 'linear'), f'weight_schedule.mode must be step or linear, got {self.mode}'
        self.points = [(0, _normalize(base_weights, 'train_sources'))]
        for p in (spec or {}).get('points', []):
            e, ws = int(p['epoch']), p['weights']
            assert e > self.points[-1][0], 'weight_schedule points must have strictly increasing epochs > 0'
            assert set(ws) == set(names), f'weight_schedule point at epoch {e} must list exactly {names}, got {list(ws)}'
            self.points.append((e, _normalize([ws[n] for n in names], f'weight_schedule epoch {e}')))

    def __call__(self, epoch):
        pts = self.points
        i = max(j for j, (e, _) in enumerate(pts) if e <= epoch)
        if self.mode == 'step' or i == len(pts) - 1:
            return pts[i][1]
        (e0, w0), (e1, w1) = pts[i], pts[i + 1]
        return w0 + (w1 - w0) * (epoch - e0) / (e1 - e0)

    def ever_positive(self):
        return np.max([w for _, w in self.points], 0) > 0

    def signature(self):
        return (self.mode, tuple((e, tuple(np.round(w, 12))) for e, w in self.points))


def parse_sources(data_dict):
    """Read `train_sources` (+ optional `weight_schedule`, `epoch_size`) from the data yaml.
    Returns names, paths, schedule, epoch_size, target_idx, caches (per-source bool or None).
    With `sampling: proportional`, schedule and epoch_size are None: they are set from dataset sizes later."""
    srcs = data_dict['train_sources']
    proportional = data_dict.get('sampling', 'weighted') == 'proportional'
    assert isinstance(srcs, list) and len(srcs) > 0, 'train_sources must be a non-empty list'
    names = [s.get('name', f'src{i}') for i, s in enumerate(srcs)]
    assert len(set(names)) == len(names), f'train_sources names must be unique, got {names}'
    paths = [s['path'] for s in srcs]
    if proportional:
        ignored = [k for k in ('weight_schedule', 'epoch_size') if k in data_dict] + \
                  (['weight'] if any('weight' in s for s in srcs) else [])
        if ignored:
            logger.warning(f'sampling: proportional ignores {ignored}')
        schedule = None
    else:
        assert all('weight' in s for s in srcs), 'sampling: weighted needs a weight on every train_source'
        schedule = WeightSchedule(names, [float(s['weight']) for s in srcs], data_dict.get('weight_schedule'))
    targets = [i for i, s in enumerate(srcs) if s.get('target', False)]
    assert len(targets) == 1, \
        f'exactly one train_source must set target: true (the deployment domain), got {[names[i] for i in targets]}'
    never = [] if proportional else [n for n, p in zip(names, schedule.ever_positive()) if not p]
    if never:
        logger.warning(f'train_sources {never} have weight 0 throughout: still scanned/cached, consider removing them')
    caches = [s.get('cache', None) for s in srcs]
    es = None if proportional else data_dict.get('epoch_size', None)
    if isinstance(es, str):
        assert es in names, f'epoch_size "{es}" is not a train_sources name {names}'
    elif es is not None:
        assert int(es) > 0, 'epoch_size must be > 0'
    return names, paths, schedule, es, targets[0], caches


def resolve_epoch_size(epoch_size, sizes, schedule, names):
    """epoch_size: int; None (= total images); or a source name, meaning epoch = len(source) / its epoch-0 weight
    (that source is seen once per epoch at the START). It is fixed for the whole run -- batches per epoch must
    not change -- so with a schedule, that source's passes/epoch change as its weight changes."""
    if epoch_size is None:
        return int(sum(sizes))
    if isinstance(epoch_size, str):
        k = names.index(epoch_size)
        w0 = schedule(0)[k]
        assert w0 > 0, f'epoch_size anchor source "{epoch_size}" has weight 0 at epoch 0'
        return int(round(sizes[k] / w0))
    return int(epoch_size)


def largest_remainder(weights, total):
    """Integer counts that sum exactly to `total` and are as close as possible to weights * total."""
    raw = np.asarray(weights) * total
    counts = np.floor(raw).astype(int)
    for k in np.argsort(-(raw - counts), kind='stable')[:total - counts.sum()]:
        counts[k] += 1
    return counts


# ---------------------------------------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------------------------------------
def _close_mosaic(d):
    """Late-stage 'clean images' policy: no mosaic, mixup or paste-in. In upstream YOLOv7 mixup only runs
    inside the mosaic branch, so mosaic=False already disables it; zeroed explicitly so the policy doesn't
    depend on that. hyp is copied so the shared training hyp dict (saved to hyp.yaml) is untouched."""
    d.mosaic = False
    d.hyp = {**d.hyp, 'mixup': 0.0, 'paste_in': 0.0}


class MixedDataset(Dataset):
    """Concatenation of per-source LoadImagesAndLabels that still looks like one to train.py.
    Indexed with (global_index, aug_seed) tuples from MixedSourceSampler; a plain int also works (unseeded)."""

    def __init__(self, datasets, names, target_idx=None):
        self.datasets, self.names, self.target_idx = datasets, names, target_idx
        self.sizes = [len(d) for d in datasets]
        self.offsets = np.cumsum([0] + self.sizes[:-1]).tolist()
        # Attributes train.py / check_anchors / labels_to_class_weights read
        self.labels = [l for d in datasets for l in d.labels]
        self.shapes = np.concatenate([d.shapes for d in datasets], 0)
        self.img_files = [f for d in datasets for f in d.img_files]
        self.n = len(self.labels)

    def __len__(self):
        return self.n

    @property
    def target_dataset(self):
        """The deployment-domain source (for autoanchor). labels/shapes above stay pooled on purpose:
        train.py uses them for the class-range check, which must cover every source."""
        return self.datasets[self.target_idx] if self.target_idx is not None else self

    def __getitem__(self, item):
        if isinstance(item, (tuple, list)):
            index, aug_seed = item
            seed_everything(aug_seed, torch_rng=False)  # mosaic partners, crops, HSV, flips, mixup, paste-in all follow from this
        else:
            index = item
        k = int(np.searchsorted(self.offsets, index, side='right') - 1)
        return self.datasets[k][index - self.offsets[k]]

    @property
    def mosaic(self):
        return all(d.mosaic for d in self.datasets)

    def close_mosaic(self):
        # NOTE: only affects NEW worker processes. Rebuild the dataloader after calling this.
        for d in self.datasets:
            _close_mosaic(d)

    collate_fn = staticmethod(LoadImagesAndLabels.collate_fn)
    collate_fn4 = staticmethod(LoadImagesAndLabels.collate_fn4)


# ---------------------------------------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------------------------------------
class MixedSourceSampler(Sampler):
    """Stratified per-epoch sampler with exact (optionally scheduled) source quotas. DDP-aware.

    Stateless apart from `next_epoch`: the draw for epoch e is computed from scratch from (seed, config, e).
    `next_epoch` advances each time __iter__ is called, which with InfiniteDataLoader happens once per epoch
    on every rank; rebuild_loader() / resume just set it (see rewind())."""

    def __init__(self, sizes, offsets, names, schedule, epoch_size, rank=-1, world_size=1, seed=0, start_epoch=0):
        self.sizes, self.offsets, self.names = list(sizes), list(offsets), list(names)
        self.schedule, self.seed = schedule, int(seed)
        for n, sz, pos in zip(self.names, self.sizes, schedule.ever_positive()):
            if pos and sz == 0:
                raise ValueError(f'source "{n}" has a positive weight but contains no images')
        self.rank, self.world_size = max(rank, 0), (world_size if rank != -1 else 1)
        self.num_samples = int(math.ceil(epoch_size / self.world_size))  # per rank
        self.total = self.num_samples * self.world_size                   # padded to divide evenly
        self.next_epoch = start_epoch
        self._counts_cache, self._perm_cache = {}, {}
        self._logged = None

    # ---- deterministic building blocks ------------------------------------------------------------------
    def counts(self, epoch):
        if epoch not in self._counts_cache:
            self._counts_cache[epoch] = largest_remainder(self.schedule(epoch), self.total)
        return self._counts_cache[epoch]

    def consumed_before(self, epoch):
        """Images drawn from each source in epochs [0, epoch)."""
        return sum((self.counts(e) for e in range(epoch)), np.zeros(len(self.sizes), dtype=np.int64))

    def _perm(self, k, cycle):
        key = (k, cycle)
        if key not in self._perm_cache:
            if len(self._perm_cache) > 4 * len(self.sizes):
                self._perm_cache.clear()
            g = torch.Generator().manual_seed(stable_hash(self.seed, self.names[k], 'perm', cycle))
            self._perm_cache[key] = torch.randperm(self.sizes[k], generator=g).tolist()
        return self._perm_cache[key]

    def _take(self, k, start, count):
        """`count` (global_index, aug_seed) pairs from source k's stream, starting at stream position `start`."""
        n, out = self.sizes[k], []
        for pos in range(start, start + count):
            c, p = divmod(pos, n)
            out.append((self._perm(k, c)[p] + self.offsets[k], stable_hash(self.seed, self.names[k], 'aug', c, p)))
        return out

    def draw(self, epoch):
        """The full (all-rank) ordered list of (global_index, aug_seed) for `epoch`."""
        start, cnt = self.consumed_before(epoch), self.counts(epoch)
        items = [it for k in range(len(self.sizes)) if cnt[k] > 0 for it in self._take(k, int(start[k]), int(cnt[k]))]
        g = torch.Generator().manual_seed(stable_hash(self.seed, 'mix', epoch))
        return [items[i] for i in torch.randperm(len(items), generator=g).tolist()]

    # ---- Sampler API --------------------------------------------------------------------------------------
    def __iter__(self):
        epoch = self.next_epoch
        self.next_epoch += 1
        cnt = tuple(self.counts(epoch))
        if self.rank == 0 and cnt != self._logged:  # log when the mix changes (runs slightly ahead: prefetch)
            self._logged = cnt
            logger.info(f'mixed sources from epoch {epoch}: {self.summary(epoch)}')
        return iter(self.draw(epoch)[self.rank:self.total:self.world_size])

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):  # train.py calls this under DDP; epochs are tracked internally
        pass

    def rewind(self, epoch):
        """Next __iter__ draws `epoch` (discarding draws made for prefetched-but-unused batches)."""
        self.next_epoch = epoch

    def signature(self):
        """Stored in checkpoints; a mismatch on --resume means the data config changed."""
        return {'seed': self.seed, 'names': self.names, 'sizes': self.sizes, 'total': self.total,
                'schedule': self.schedule.signature()}

    # ---- reporting ----------------------------------------------------------------------------------------
    def exposure(self, epochs_done):
        """Cumulative passes through each source after `epochs_done` epochs."""
        return [float(c) / max(n, 1) for c, n in zip(self.consumed_before(epochs_done), self.sizes)]

    def summary(self, epoch=0):
        cnt = self.counts(epoch)
        rows = [f'{n:<16}{c / self.total:>7.1%}{sz:>10} imgs{c:>9}/epoch{c / max(sz, 1):>8.2f} passes/epoch'
                for n, c, sz in zip(self.names, cnt, self.sizes)]
        return f'{self.total} images/epoch\n' + '\n'.join('  ' + r for r in rows)


# ---------------------------------------------------------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------------------------------------------------------
def create_mixed_dataloader(data_dict, imgsz, batch_size, stride, opt, hyp=None, cache=False, rank=-1,
                            world_size=1, workers=8, quad=False, prefix='', start_epoch=0, dataset=None):
    """Drop-in replacement for create_dataloader() for training.
    cache: the global --cache-images flag. A source's own `cache:` key wins; otherwise the target source follows
    the global flag and auxiliary sources are NOT cached (they are usually large and only partly used per epoch)."""
    names, paths, schedule, epoch_size, target_idx, caches = parse_sources(data_dict)
    if dataset is None:
        dsets = []
        for k, (n, p) in enumerate(zip(names, paths)):
            c = caches[k] if caches[k] is not None else (cache if k == target_idx else False)
            with torch_distributed_zero_first(rank):
                dsets.append(LoadImagesAndLabels(p, imgsz, batch_size, augment=True, hyp=hyp, rect=False,
                                                 cache_images=bool(c), single_cls=opt.single_cls, stride=int(stride),
                                                 pad=0.0, image_weights=False, prefix=f'{prefix}[{n}] '))
        dataset = MixedDataset(dsets, names, target_idx)

    if schedule is None:  # sampling: proportional -- no source weighting: every image exactly once per epoch,
        # i.e. source shares = dataset sizes (what stock YOLOv7 does), with the same seeding, source-local mosaic
        # and target-only anchors as weighted mode, so weighted-vs-proportional isolates the effect of weighting
        schedule = WeightSchedule(names, dataset.sizes)
    epoch_size = resolve_epoch_size(epoch_size, dataset.sizes, schedule, names)
    sampler = MixedSourceSampler(dataset.sizes, dataset.offsets, names, schedule, epoch_size, rank=rank,
                                 world_size=world_size, seed=getattr(opt, 'seed', 0), start_epoch=start_epoch)
    if rank in [-1, 0]:
        mode = data_dict.get('sampling', 'weighted')
        s = f'sampling {mode}, seed {sampler.seed}' + ('' if mode == 'proportional' else
            f', schedule mode {schedule.mode}, change points at epochs {[e for e, _ in schedule.points]}')
        s += f'\n  target (autoanchor) source: {names[dataset.target_idx]}'
        s += '\n  cached: ' + (', '.join(n for n, d in zip(names, dataset.datasets) if d.imgs[0] is not None) or 'none')
        logger.info(f'{prefix}mixed sources: {s}')

    batch_size = min(batch_size, len(sampler))
    nw = min([(__import__('os').cpu_count() or 1) // world_size, batch_size if batch_size > 1 else 0, workers])
    loader = InfiniteDataLoader(dataset, batch_size=batch_size, num_workers=nw, sampler=sampler, pin_memory=True,
                                collate_fn=MixedDataset.collate_fn4 if quad else MixedDataset.collate_fn)
    return loader, dataset


def shutdown_loader(loader):
    """InfiniteDataLoader keeps a live iterator (and its worker processes) forever; stop it explicitly."""
    it = getattr(loader, 'iterator', None)
    if it is not None and hasattr(it, '_shutdown_workers'):
        it._shutdown_workers()


def rebuild_loader(loader, close_mosaic=False, epoch=None):
    """Recreate an (Infinite)DataLoader with fresh workers around the same dataset and sampler.
    Needed for any dataset attribute change mid-training: InfiniteDataLoader workers are persistent and
    hold their own pickled copy of the dataset, so changes in the main process never reach them.
    epoch: the epoch about to be trained. With a MixedSourceSampler, draws already made for prefetched batches
    of that epoch are discarded and redrawn identically, so nothing is skipped."""
    ds = loader.dataset
    if close_mosaic:
        if isinstance(ds, MixedDataset):
            ds.close_mosaic()
        else:
            _close_mosaic(ds)
    shutdown_loader(loader)  # must happen before rewind: stops the old iterator from requesting more indices
    if epoch is not None and hasattr(loader.sampler, 'rewind'):
        loader.sampler.rewind(epoch)
    return type(loader)(ds, batch_size=loader.batch_size, num_workers=loader.num_workers, sampler=loader.sampler,
                        pin_memory=loader.pin_memory, collate_fn=loader.collate_fn)

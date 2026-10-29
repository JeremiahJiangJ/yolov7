# Weighted multi-source training data (utils/mixed_data.py): the sampler, the epoch size, schedules, and the training
# loader exactly as train.py builds it (create_mixed_dataloader).

import copy
import os
import types

import numpy as np
import pytest
import torch

from conftest import SOURCES, sample_boxes, unlabelled_objects
from utils.datasets import rebuild_loader
from utils.mixed_data import (MixedConfig, MixedSourceSampler, WeightSchedule, create_mixed_dataloader,
                              repeat_factors)

EPOCHS, EPOCH_SIZE, SEED = 6, 60, 7
OPT = types.SimpleNamespace(single_cls=False, train_cache_path=None)


def mixed_yaml(sources, **top):
    d = {'nc': 1, 'names': ['o'], 'resize': 'native', 'epoch_size': EPOCH_SIZE,
         'train_sources': [{'name': 'A', 'path': sources['A'], 'val': sources['A'], 'target': True, 'weight': 50},
                           {'name': 'B', 'path': sources['B'], 'weight': 30},
                           {'name': 'C', 'path': sources['C'], 'weight': 20}],
         'weight_schedule': {'mode': 'linear', 'points': [{'epoch': 4, 'weights': {'A': 20, 'B': 30, 'C': 50}}]}}
    d.update(top)
    return d


def source_of(path):
    return os.path.basename(path)[0]


def frame_of(path):
    return int(os.path.basename(path).split('_')[1].split('.')[0])


def read_epochs(loader, n):
    # as train.py: one pass over the loader per epoch
    return [[(imgs, targets, paths) for imgs, targets, paths, _ in loader] for _ in range(n)]


def build(data, hyp, workers, start_epoch=0):
    return create_mixed_dataloader(MixedConfig(copy.deepcopy(data)), 640, 8, 32, OPT, hyp=hyp, workers=workers,
                                   seed=SEED, epochs=EPOCHS, start_epoch=start_epoch)


@pytest.fixture(scope='module')
def run(sources):
    """6 epochs of the training loader, with mixup and paste-in turned up to exercise every partner path."""
    import yaml
    from conftest import ROOT
    with open(ROOT / 'data' / 'hyp.scratch.tiny.yaml') as f:
        hyp = yaml.safe_load(f)
    hyp.update(hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, mixup=0.3, paste_in=0.3)
    data = mixed_yaml(sources)
    loader, dataset = build(data, hyp, workers=2)
    return types.SimpleNamespace(data=data, hyp=hyp, loader=loader, dataset=dataset,
                                 epochs=read_epochs(loader, EPOCHS))


def test_samples_per_source_follow_the_weights(run):
    for e, ep in enumerate(run.epochs):
        n = {k: 0 for k in SOURCES}
        for _, _, paths in ep:
            for p in paths:
                n[source_of(p)] += 1
        expected = dict(zip(SOURCES, (int(c) for c in run.loader.sampler.counts(e))))
        assert n == expected, f'epoch {e}'
        assert sum(n.values()) == EPOCH_SIZE
        w = np.array(run.loader.sampler.schedule(e))
        assert np.allclose(np.array(list(n.values())) / EPOCH_SIZE, w, atol=1 / EPOCH_SIZE)


def test_linear_schedule_interpolates_then_holds(run):
    s = run.loader.sampler.schedule
    assert np.allclose(s(0), [0.5, 0.3, 0.2])
    assert np.allclose(s(2), [0.35, 0.3, 0.35])  # halfway to epoch 4
    assert np.allclose(s(4), [0.2, 0.3, 0.5]) and np.allclose(s(5), s(4))


def test_partners_stay_within_the_source(run):
    colours = {k: np.array(c)[::-1] for k, (_, c, _) in SOURCES.items()}  # RGB in the tensors
    leaks = 0
    for ep in run.epochs:
        for imgs, _, paths in ep:
            for b, p in enumerate(paths):
                im = imgs[b].numpy().transpose(1, 2, 0).astype(int)
                for other, c in colours.items():
                    if other != source_of(p) and (np.abs(im - c).max(2) <= 2).sum() > 50:
                        leaks += 1
    assert leaks == 0


def test_every_visible_object_has_a_label(run):
    checked = missing = 0
    for ep in run.epochs:
        for imgs, targets, _ in ep:
            for b in range(len(imgs)):
                im = imgs[b].numpy().transpose(1, 2, 0)
                c, m = unlabelled_objects(im, sample_boxes(targets[targets[:, 0] == b], im.shape), thr=120)
                checked, missing = checked + c, missing + m
    assert checked > 500 and missing == 0, f'{missing} of {checked} objects without a label'


def test_frames_of_a_source_are_used_equally_often(run):
    for k, (_, _, n) in SOURCES.items():
        used = [frame_of(p) for ep in run.epochs for _, _, paths in ep for p in paths if source_of(p) == k]
        c = np.bincount(used, minlength=n)
        assert c.max() - c.min() <= 1, f'{k}: {c}'


def test_same_seed_same_batches_whatever_the_workers(run, sources):
    loader0, _ = build(run.data, run.hyp, workers=0)
    for a_ep, b_ep in zip(run.epochs, read_epochs(loader0, EPOCHS)):
        for (ia, ta, pa), (ib, tb, pb) in zip(a_ep, b_ep):
            assert pa == pb and torch.equal(ia, ib) and torch.equal(ta, tb)


def test_resume_draws_the_same_batches(run):
    loader, _ = build(run.data, run.hyp, workers=2, start_epoch=3)
    for a_ep, b_ep in zip(run.epochs[3:], read_epochs(loader, EPOCHS - 3)):
        for (ia, ta, pa), (ib, tb, pb) in zip(a_ep, b_ep):
            assert pa == pb and torch.equal(ia, ib) and torch.equal(ta, tb)


def test_close_mosaic_rebuild_continues_the_draw(sources, hyp):
    """--close-mosaic: dataset.close_mosaic() + rebuild_loader(epoch=e) restarts the workers; epoch e is drawn from its
    start (prefetched batches neither skipped nor repeated) and samples are single crops."""
    data = mixed_yaml(sources)
    loader, dataset = build(data, hyp, workers=2)
    ref = [[p for _, _, paths in ep for p in paths] for ep in read_epochs(build(data, hyp, workers=0)[0], 4)]
    got = [[p for _, _, paths in ep for p in paths] for ep in read_epochs(loader, 2)]
    dataset.close_mosaic()
    loader = rebuild_loader(loader, epoch=2)
    assert not dataset.mosaic
    got += [[p for _, _, paths in ep for p in paths] for ep in read_epochs(loader, 2)]
    assert got == ref
    imgs, targets, paths, _ = next(iter(loader))
    for b in range(len(imgs)):  # a single crop of the sampled frame: only its own background colour (and grey pad)
        im = imgs[b].numpy().transpose(1, 2, 0)
        assert unlabelled_objects(im, sample_boxes(targets[targets[:, 0] == b], im.shape))[1] == 0


def test_proportional_sampling_uses_every_frame_once_per_epoch(sources, hyp):
    data = mixed_yaml(sources, sampling='proportional', epoch_size='total')
    data.pop('weight_schedule')
    loader, _ = build(data, hyp, workers=0)
    n = sum(v[2] for v in SOURCES.values())
    for e in range(2):
        draw = loader.sampler.draw(e)
        assert len(draw) == n and len({i for i, _ in draw}) == n


# ----------------------------------------------------------------------------------------------------------------------
# Sampler and config units (no images)
# ----------------------------------------------------------------------------------------------------------------------
def cfg(**over):
    d = {'nc': 1, 'names': ['o'],
         'train_sources': [{'name': 'A', 'path': 'a', 'val': 'a', 'target': True, 'weight': 60},
                           {'name': 'B', 'path': 'b', 'weight': 40}]}
    d.update(over)
    return MixedConfig(d)


@pytest.mark.parametrize('epoch_size, expected', [('total', 1500), (None, 1000 / 0.6), ('A', 1000 / 0.6),
                                                   ('B', 500 / 0.4), (777, 777)])
def test_epoch_size(epoch_size, expected):
    c = cfg(**({} if epoch_size is None else {'epoch_size': epoch_size}))
    sizes = [1000, 500]
    assert c.resolve_epoch_size(sizes, c.schedule(sizes, 10), 10) == int(round(expected))


def test_epoch_size_target_uses_the_mean_weight_over_a_schedule():
    c = cfg(weight_schedule={'mode': 'step', 'points': [{'progress': 0.5, 'weights': {'A': 20, 'B': 80}}]})
    sizes = [1000, 500]
    sch = c.schedule(sizes, 10)
    assert np.allclose(sch(4), [0.6, 0.4]) and np.allclose(sch(5), [0.2, 0.8])  # step at 50% of 10 epochs
    assert c.resolve_epoch_size(sizes, sch, 10) == round(1000 / 0.4)  # mean target weight (0.6 + 0.2) / 2


def test_schedule_validation():
    with pytest.raises(AssertionError):  # a point must list every source
        WeightSchedule(['A', 'B'], [1, 1], {'points': [{'epoch': 2, 'weights': {'A': 1}}]}, 10)
    with pytest.raises(AssertionError):  # points must move forward
        WeightSchedule(['A', 'B'], [1, 1], {'points': [{'epoch': 0, 'weights': {'A': 1, 'B': 1}}]}, 10)


def sampler(rank=-1, world_size=1, repeat=None, sizes=(30, 20)):
    names = ['A', 'B']
    sch = WeightSchedule(names, [3, 2])
    offsets = [0, sizes[0]]
    return MixedSourceSampler(list(sizes), offsets, names, sch, 50, rank=rank, world_size=world_size, seed=1,
                              repeat=repeat)


def test_ddp_ranks_split_the_epoch_without_overlap():
    full = sampler().draw(0)
    r0, r1 = list(sampler(rank=0, world_size=2)), list(sampler(rank=1, world_size=2))
    assert len(r0) == len(r1) == 25
    assert sorted(r0 + r1) == sorted(full)  # together exactly the single-process epoch


def test_streams_do_not_depend_on_the_other_sources_weights():
    # the same seed gives source A the same frames and augmentation seeds whatever B's weight
    names = ['A', 'B']
    a = MixedSourceSampler([30, 20], [0, 30], names, WeightSchedule(names, [1, 1]), 40, seed=3).draw(0)
    b = MixedSourceSampler([30, 20], [0, 30], names, WeightSchedule(names, [1, 3]), 80, seed=3).draw(0)
    assert sorted(x for x in a if x[0] < 30) == sorted(x for x in b if x[0] < 30)


def test_repeat_factors():
    # class 1 in 1 of 100 frames, threshold 0.04: r = sqrt(0.04 / 0.01) = 2; frames without a rare class keep 1
    frames = [np.array([0])] * 99 + [np.array([0, 1])]
    r = repeat_factors(frames, 0.04)
    assert np.allclose(r[:99], 1) and np.isclose(r[99], 2)
    assert np.allclose(repeat_factors(frames, 0), 1)


def test_repeat_factor_stream_repeats_rare_frames():
    r = np.ones(30)
    r[0] = 3.0  # frame 0 three times per pass
    s = sampler(repeat=[r, None])
    a_frames = [i for i, _ in s.draw(0) + s.draw(1) + s.draw(2) if i < 30]
    c = np.bincount(a_frames, minlength=30)
    assert c[0] >= 2.5 * c[1:].mean()

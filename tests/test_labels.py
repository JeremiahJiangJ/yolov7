# Labels through loading and augmentation (utils/datasets.py): every visible object keeps its box with native-resolution
# crops and mosaics, mixed polygon / box label files, and the label cache's checks.

import random

import numpy as np
import pytest

from conftest import LABELS, make_source, sample_boxes, unlabelled_objects
from utils.datasets import LoadImagesAndLabels, delete_label_caches_at_exit


def samples(ds, n, seed=0, index=None):
    random.seed(seed)
    np.random.seed(seed)
    for k in range(n):
        img, labels, *_ = ds[k % len(ds) if index is None else index]
        im = img.numpy().transpose(1, 2, 0)
        yield im, sample_boxes(labels, im.shape)


def count_unlabelled(ds, n, **kw):
    checked = missing = 0
    for im, boxes in samples(ds, n, **kw):
        c, m = unlabelled_objects(im, boxes)
        checked, missing = checked + c, missing + m
    return checked, missing


@pytest.fixture(scope='module')
def mixed_sizes(tmp_path_factory):
    root = tmp_path_factory.mktemp('sizes')
    return [make_source(root, k, size, (60, 60, 60), 12, objects=3, seed=i, obj_px=(8, 60))
            for i, (k, size) in enumerate({'big': (1920, 1080), 'mid': (720, 576), 'small': (320, 320)}.items())]


@pytest.mark.parametrize('variant', ['mosaic', 'crop', 'zoom', 'mixup', 'paste_in'])
def test_native_samples_keep_every_visible_label(mixed_sizes, hyp, variant):
    hyp.update({'mosaic': {'mosaic': 1.0, 'crop': 0.0}.get(variant, 1.0),
                'mixup': 1.0 if variant == 'mixup' else 0.0, 'paste_in': 1.0 if variant == 'paste_in' else 0.0})
    if variant == 'zoom':
        hyp.update(scale_min=0.8, scale_max=2.0)
    ds = LoadImagesAndLabels(mixed_sizes, 640, 16, augment=True, hyp=hyp, resize='native', fg_crop_prob=0.5)
    checked, missing = count_unlabelled(ds, 120) if variant != 'mixup' else (0, 0)
    if variant == 'mixup':  # blended objects are half as bright
        for im, boxes in samples(ds, 120):
            c, m = unlabelled_objects(im, boxes, thr=120)
            checked, missing = checked + c, missing + m
    assert checked > 100 and missing == 0, f'{missing} of {checked} objects without a label'


def test_fit_resize_keeps_every_visible_label(mixed_sizes, hyp):
    ds = LoadImagesAndLabels(mixed_sizes, 640, 16, augment=True, hyp=hyp, resize='fit')
    checked, missing = count_unlabelled(ds, 80)
    assert checked > 50 and missing == 0


def test_fg_crop_puts_an_object_in_every_crop(mixed_sizes, hyp):
    hyp.update(mosaic=0.0, mixup=0.0, paste_in=0.0, translate=0.0, scale_min=1.0, scale_max=1.0)  # no zoom or shift
    ds = LoadImagesAndLabels(mixed_sizes, 640, 16, augment=True, hyp=hyp, resize='native', fg_crop_prob=1.0)
    big = [i for i, f in enumerate(ds.img_files) if '/big/' in f]
    for i in big[:4]:
        assert all(len(boxes) for _, boxes in samples(ds, 10, index=i))


def test_mosaic_mixing_polygon_and_box_labels(tmp_path, hyp):
    """A mosaic mixing polygon- and box-labelled frames used to give labels the wrong polygons and drop the rest."""
    poly = make_source(tmp_path, 'poly', (1920, 1080), (60, 60, 60), 6, polygon=True, seed=1)
    box = make_source(tmp_path, 'box', (1920, 1080), (60, 60, 60), 18, seed=2)
    hyp.update(mixup=0.0, paste_in=0.0)
    ds = LoadImagesAndLabels([poly, box], 640, 16, augment=True, hyp=hyp, resize='native', fg_crop_prob=1.0)
    assert sum(len(s) for s in ds.segments) == 12  # the polygon files' labels are polygons
    checked, missing = count_unlabelled(ds, 150)
    assert checked > 100 and missing == 0, f'{missing} of {checked} objects without a label'


def test_polygon_labels_give_their_enclosing_box(tmp_path):
    d = tmp_path / 'p'
    (d / 'images').mkdir(parents=True)
    (d / LABELS).mkdir()
    import cv2
    cv2.imwrite(str(d / 'images' / 'x.png'), np.zeros((100, 200, 3), np.uint8))
    # a rotated rectangle: 4 corners
    (d / LABELS / 'x.txt').write_text('0 0.18465 0.61458 0.21094 0.582465 0.2642 0.6475 0.23793 0.67969\n')
    ds = LoadImagesAndLabels(str(d / 'images'), 640, 1)
    x1, y1, x2, y2 = 0.18465, 0.582465, 0.2642, 0.67969
    assert np.allclose(ds.labels[0][0, 1:], [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], atol=1e-5)


# ----------------------------------------------------------------------------------------------------------------------
# Label cache
# ----------------------------------------------------------------------------------------------------------------------
@pytest.fixture
def three_folders(tmp_path):
    return [make_source(tmp_path, k, (320, 320), (60, 60, 60), 5, seed=i) for i, k in enumerate(['f1', 'f2', 'f3'])]


def test_pooled_folders_all_keep_their_labels(three_folders):
    ds = LoadImagesAndLabels(three_folders, 640, 4)
    for folder in ('/f1/', '/f2/', '/f3/'):
        idx = [i for i, f in enumerate(ds.img_files) if folder in f]
        assert len(idx) == 5 and all(len(ds.labels[i]) == 2 for i in idx)
    assert '_pooled_' in ds.cache_file.name  # never the first folder's own cache


def test_cache_is_rebuilt_when_labels_change(three_folders):
    ds = LoadImagesAndLabels(three_folders, 640, 4)
    assert ds.cache_file.exists()
    from pathlib import Path
    f2 = sorted((Path(three_folders[1]).parent / LABELS).glob('*.txt'))
    for f in f2:
        f.write_text('')  # e.g. a cache left by a killed run from before labels were added
    assert sum(len(l) for l in LoadImagesAndLabels(three_folders, 640, 4).labels) == 20
    for f in f2:
        f.write_text('0 0.5 0.5 0.1 0.1\n')
    assert sum(len(l) for l in LoadImagesAndLabels(three_folders, 640, 4).labels) == 25


def test_cache_is_rebuilt_for_another_label_folder(tmp_path):
    src = make_source(tmp_path, 's', (320, 320), (60, 60, 60), 3)
    other = tmp_path / 's' / 'labels_other'
    other.mkdir()
    for i in range(3):
        (other / f's_{i}.txt').write_text('0 0.5 0.5 0.2 0.2\n0 0.2 0.2 0.1 0.1\n0 0.8 0.8 0.1 0.1\n')
    assert sum(len(l) for l in LoadImagesAndLabels(src, 640, 1).labels) == 6
    assert sum(len(l) for l in LoadImagesAndLabels(src, 640, 1, label_folder_name='labels_other').labels) == 9


def test_missing_label_folder_is_an_error(tmp_path):
    src = make_source(tmp_path, 's', (320, 320), (60, 60, 60), 3)
    with pytest.raises(FileNotFoundError):
        LoadImagesAndLabels(src, 640, 1, label_folder_name='no_such_folder')


def test_missing_label_files_are_background(three_folders):
    from pathlib import Path
    for f in (Path(three_folders[2]).parent / LABELS).glob('*.txt'):
        f.unlink()
    ds = LoadImagesAndLabels(three_folders, 640, 4)
    assert sum(len(l) == 0 for l in ds.labels) == 5  # trained as background: the scan line reports 5 missing


def test_caches_are_deleted_at_exit_registration():
    # train.py / test.py call this; it registers an atexit handler (idempotent)
    delete_label_caches_at_exit()
    delete_label_caches_at_exit()

# Shared fixtures: small synthetic datasets (solid background, white square objects, YOLO labels) built once per test
# session. Every source has its own background colour, so each pixel of a training sample shows which source it is from.

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import utils.general  # noqa: E402,F401 (import order: utils.general before utils.metrics)

LABELS = 'labels_upright'  # this fork's default label folder


def make_source(root, name, size, colour, n, objects=2, polygon=False, seed=0, obj_px=(12, 40)):
    """n frames of size (w, h) with `objects` white squares each; labels as boxes, or as 4-corner polygons."""
    rng = np.random.default_rng(seed)
    w, h = size
    img_dir, lab_dir = Path(root) / name / 'images', Path(root) / name / LABELS
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        img = np.zeros((h, w, 3), np.uint8)
        img[:] = colour
        lines = []
        for _ in range(objects):
            s = int(rng.integers(*obj_px))
            x, y = int(rng.integers(0, w - s)), int(rng.integers(0, h - s))
            img[y:y + s, x:x + s] = 255
            if polygon:
                pts = np.array([[x, y], [x + s, y], [x + s, y + s], [x, y + s]]) / [w, h]
                lines.append('0 ' + ' '.join(f'{v:.6f}' for v in pts.ravel()))
            else:
                lines.append(f'0 {(x + s / 2) / w:.6f} {(y + s / 2) / h:.6f} {s / w:.6f} {s / h:.6f}')
        cv2.imwrite(str(img_dir / f'{name}_{i}.png'), img)
        (lab_dir / f'{name}_{i}.txt').write_text('\n'.join(lines) + '\n')
    return str(img_dir)


# name: (frame size, background colour BGR, frames)
SOURCES = {'A': ((1920, 1080), (40, 40, 160), 30), 'B': ((720, 576), (40, 160, 40), 20),
           'C': ((320, 320), (160, 40, 40), 10)}


@pytest.fixture(scope='session')
def sources(tmp_path_factory):
    root = tmp_path_factory.mktemp('sources')
    return {k: make_source(root, k, size, col, n, seed=i) for i, (k, (size, col, n)) in enumerate(SOURCES.items())}


@pytest.fixture
def hyp():
    with open(ROOT / 'data' / 'hyp.scratch.tiny.yaml') as f:
        h = yaml.safe_load(f)
    h.update(hsv_h=0.0, hsv_s=0.0, hsv_v=0.0)  # keep colours exact (source identification, white objects)
    return h


def unlabelled_objects(img_hwc, boxes_xywh_px, thr=200, min_px=8):
    """Fully visible white square objects whose pixels are not >= 90% covered by label boxes.
    Returns (objects checked, objects without a label)."""
    H, W = img_hwc.shape[:2]
    mask = (img_hwc.min(2) >= thr).astype(np.uint8)
    n, cc, st, _ = cv2.connectedComponentsWithStats(mask)
    covered = np.zeros((H, W), bool)
    for cx, cy, bw, bh in boxes_xywh_px:
        covered[max(int(cy - bh / 2), 0):int(np.ceil(cy + bh / 2)), max(int(cx - bw / 2), 0):int(np.ceil(cx + bw / 2))] \
            = True
    checked = missing = 0
    for c in range(1, n):
        x, y, w, h, _ = st[c]
        if w < min_px or h < min_px or x == 0 or y == 0 or x + w >= W or y + h >= H or max(w, h) > 1.3 * min(w, h):
            continue  # small, or cut by the sample or a tile edge (the objects are squares; a cut piece with less than
            # 10% of its box left is dropped on purpose, as in stock YOLOv7)
        checked += 1
        missing += covered[cc == c].mean() < 0.9
    return checked, missing


def sample_boxes(labels, shape):
    """LoadImagesAndLabels labels_out (n, 6: _, cls, x, y, w, h normalised) -> (n, 4) xywh pixels."""
    H, W = shape[:2]
    return np.asarray(labels)[:, 2:6] * [W, H, W, H]


def pytest_configure(config):
    config.addinivalue_line('markers', 'slow: end-to-end train.py / test.py runs (deselect with -m "not slow")')

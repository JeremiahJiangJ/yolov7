# Synthetic approach / recede / pan sequences from labelled still images, for tools/track_stability.py (README.md
# section 14), until real video of objects moving towards or away from the camera is available.
#
# For each image (and one of its objects, the target, by default the largest), three sequences of T frames at the
# image's size, made by cropping a window and resizing it to the full frame:
#   toward  the window shrinks around the target: it grows smoothly from 1x to --zoom (default 3x), log-linear in time
#   away    the same frames reversed: it shrinks from --zoom to 1x
#   across  the control: the window is fixed at --pan-zoom and slides sideways, so the target crosses the frame at
#           constant scale
# Every visible object keeps its track id (its row in the source label file) in a 6th label column, so
# track_stability.py follows each one; objects whose centre leaves the frame or that fall below 2 px are dropped for that
# frame. The target's motion is checked with classify_motion (toward / away / across); mismatches are reported.
# Zooming crops upsample the image (an approaching object gains no new detail): a surrogate, not real video.
#
# Usage:
#   python tools/zoom_sequences.py --images data/val/images --out data/zoom_seq --frames 30
#   python tools/zoom_sequences.py --images data/val/images --labels-folder labels --max-images 50 --zoom 3
# Output: <out>/images/<seq>/<frame:06d>.jpg, <out>/<labels-folder>/<seq>/<frame:06d>.txt (class x y w h track_id),
#         <out>/sequences.json (sequence -> source image, target track id, intended and classified motion).

import argparse
import json
import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
import utils.general  # noqa: F401,E402 (import order: utils.general before utils.datasets)
from utils.datasets import img2label_paths  # noqa: E402
from utils.stability import classify_motion  # noqa: E402

IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def read_labels(path, w, h):
    """YOLO label file -> (classes (N,), boxes (N, 4) xyxy pixels)."""
    rows = [r.split() for r in Path(path).read_text().strip().splitlines() if r.strip()] if Path(path).is_file() else []
    if not rows:
        return np.zeros(0, int), np.zeros((0, 4))
    a = np.array([[float(v) for v in r[:5]] for r in rows])
    cx, cy, bw, bh = a[:, 1] * w, a[:, 2] * h, a[:, 3] * w, a[:, 4] * h
    return a[:, 0].astype(int), np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)


def window(cx, cy, f, w, h):
    """Crop window (x0, y0, ww, wh) of zoom f centred on (cx, cy), kept inside the image."""
    ww, wh = w / f, h / f
    x0 = min(max(cx - ww / 2, 0), w - ww)
    y0 = min(max(cy - wh / 2, 0), h - wh)
    return x0, y0, ww, wh


def render(img, boxes, win):
    """The window resized to the image size, and the boxes mapped into it."""
    h, w = img.shape[:2]
    x0, y0, ww, wh = win
    sx, sy = w / ww, h / wh
    m = np.array([[sx, 0, -x0 * sx], [0, sy, -y0 * sy]], dtype=np.float64)
    frame = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    b = boxes.copy()
    b[:, [0, 2]] = (b[:, [0, 2]] - x0) * sx
    b[:, [1, 3]] = (b[:, [1, 3]] - y0) * sy
    return frame, b


def write_frame(out, seq, t, frame, cls, boxes, w, h, labels_folder):
    img_dir, lab_dir = out / 'images' / seq, out / labels_folder / seq
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(img_dir / f'{t:06d}.jpg'), frame)
    lines = []
    for i, (c, b) in enumerate(zip(cls, boxes)):
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        if not (0 <= cx < w and 0 <= cy < h):
            continue  # centre out of the frame
        x1, y1, x2, y2 = np.clip(b[0], 0, w), np.clip(b[1], 0, h), np.clip(b[2], 0, w), np.clip(b[3], 0, h)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        lines.append(f'{c} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f} {i}')
    (lab_dir / f'{t:06d}.txt').write_text('\n'.join(lines) + ('\n' if lines else ''))


def main(opt):
    random.seed(opt.seed)
    files = sorted(p for p in Path(opt.images).rglob('*') if p.suffix.lower() in IMG_EXT)
    random.shuffle(files)
    out = Path(opt.out)
    meta, made, mismatch = {}, 0, 0
    for p in files:
        if made >= opt.max_images:
            break
        lab = img2label_paths([str(p)], opt.labels_folder)[0]
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        cls, boxes = read_labels(lab, w, h)
        if not len(boxes):
            continue
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        k = int(np.argmax(area)) if opt.target == 'largest' else random.randrange(len(boxes))
        if (boxes[k, 2] - boxes[k, 0]) * opt.zoom > w or (boxes[k, 3] - boxes[k, 1]) * opt.zoom > h:
            k = int(np.argmin(area))  # the target must still fit the frame at full zoom
            if (boxes[k, 2] - boxes[k, 0]) * opt.zoom > w or (boxes[k, 3] - boxes[k, 1]) * opt.zoom > h:
                continue
        cx, cy = (boxes[k, 0] + boxes[k, 2]) / 2, (boxes[k, 1] + boxes[k, 3]) / 2
        T = opt.frames
        zooms = [math.exp(math.log(opt.zoom) * t / (T - 1)) for t in range(T)]
        f = opt.pan_zoom
        ww = w / f
        x_lo, x_hi = max(cx - ww * 0.9, 0), min(cx - ww * 0.1, w - ww)  # target from ~90% to ~10% of the frame width
        plans = {'toward': [window(cx, cy, z, w, h) for z in zooms]}
        plans['away'] = plans['toward'][::-1]
        if x_hi - x_lo > 0.3 * ww:
            y0 = min(max(cy - h / f / 2, 0), h - h / f)
            plans['across'] = [(x_lo + (x_hi - x_lo) * t / (T - 1), y0, ww, h / f) for t in range(T)]
        for kind, wins in plans.items():
            seq = f'{p.stem}_{kind}'
            track = []
            for t, win in enumerate(wins):
                frame, b = render(img, boxes, win)
                write_frame(out, seq, t, frame, cls, b, w, h, opt.labels_folder)
                track.append(b[k])
            mo = classify_motion(np.array(track), w)
            mismatch += mo['kind'] != kind
            meta[seq] = dict(source=str(p), target_track=k, intended=kind, classified=mo['kind'],
                             delta_log_scale=mo['delta_log_scale'], dx_fraction=mo['dx_fraction'])
        made += 1
    (out / 'sequences.json').parent.mkdir(parents=True, exist_ok=True)
    (out / 'sequences.json').write_text(json.dumps(meta, indent=1))
    print(f'{len(meta)} sequences from {made} images in {out} ({mismatch} whose target motion is not classified as '
          f'intended: see sequences.json)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='synthetic approach / recede / pan sequences from labelled images')
    parser.add_argument('--images', required=True, help='folder of labelled images (labels in ../<labels-folder>/)')
    parser.add_argument('--labels-folder', default='labels', help='label folder name next to images/')
    parser.add_argument('--out', required=True, help='output folder')
    parser.add_argument('--frames', type=int, default=30, help='frames per sequence')
    parser.add_argument('--zoom', type=float, default=3.0, help='largest zoom of toward / away (>= 1.65 to count as '
                                                                  'toward / away)')
    parser.add_argument('--pan-zoom', type=float, default=2.0, help='zoom of the across (pan) sequences')
    parser.add_argument('--target', choices=('largest', 'random'), default='largest', help='object zoomed into')
    parser.add_argument('--max-images', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    main(parser.parse_args())

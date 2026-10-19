# What can the LSE warp family express? Fits warp parameters directly (no canonicalizer, no training) to target local
# zooms and reports how closely each warp grid size can reach them. An upper bound on what a canonicalizer with that
# `grid` / `max_zoom` could do (README.md, section 14): if the warps cannot express it, no canonicalizer will learn it.
#
# Synthetic scenes (default): a perspective ramp (far rows magnified, near rows shrunk) and objects that each need
# their own zoom at a given spacing. Dataset mode (--data): real object layouts from label files, each object's target
# zoom being the size target's canonical form, target_size / sqrt(w * h) (clamped to max_zoom), as training sees the
# frame (long side resized to --img-size).
#
# Usage:
#   python tools/grid_capacity.py                                         # synthetic scenes, grids 4 8 16 32
#   python tools/grid_capacity.py --grids 8 16 --max-zoom 2.5 --objects 20:80 40:40 80:25
#   python tools/grid_capacity.py --data data.yaml --target-size 48 --frames 200 --csv capacity.csv
#
# Reading the table: "within 20%" is the share of object (or ramp) points whose achieved local zoom is within 20% of
# the target; rms is the root mean square error of log zoom. Warps map each region onto itself, so along every row
# (column) the horizontal (vertical) magnification averages 1x: a whole row cannot be magnified, which caps the
# perspective ramp whatever the grid.

import argparse
import csv
import math
import random
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
import utils.general  # noqa: F401 (import order: utils.general before utils.datasets)
from utils.datasets import DEFAULT_LABEL_FOLDER, img2label_paths
from utils.lse import Regions, make_grid
from utils.mixed_data import is_mixed, list_images


def fit(R, max_zoom, img, pts, logz, n_regions, steps, lr=0.05):
    """Warp parameters of n_regions regions (whole images) fitted so the local zoom at points pts (P, 2, normalised)
    of regions img (P,) matches logz (P,). Returns the per-point absolute log zoom error."""
    px = torch.zeros(n_regions, R, R - 1, requires_grad=True)
    py = torch.zeros(n_regions, R, R - 1, requires_grad=True)
    opt = torch.optim.Adam([px, py], lr=lr)
    base = Regions(torch.arange(n_regions), torch.tensor([[0., 0., 1., 1.]]).repeat(n_regions, 1))
    for _ in range(steps):
        z, _ = base.with_grid(make_grid(px, py, max_zoom)).zoom_at(img, pts[:, 0], pts[:, 1])
        loss = ((torch.log(z) - logz) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        z, _ = base.with_grid(make_grid(px, py, max_zoom)).zoom_at(img, pts[:, 0], pts[:, 1])
        return (torch.log(z) - logz).abs()


def object_points(boxes, n=5):
    """n x n points spread over each box (N, 4: cx, cy, w, h normalised), edges included: (N * n * n, 2) and the box
    index of each point."""
    t = torch.linspace(-0.5, 0.5, n)
    off = torch.stack(torch.meshgrid(t, t, indexing='xy'), -1).reshape(-1, 2)  # (n * n, 2)
    pts = boxes[:, None, :2] + off[None] * boxes[:, None, 2:4]
    return pts.reshape(-1, 2).clamp(0.001, 0.999), torch.arange(len(boxes)).repeat_interleave(n * n)


def synthetic_cases(img_px, obj_px, specs, seed):
    """[(name, points, log zoom targets)] for one img_px x img_px region."""
    g = torch.Generator().manual_seed(seed)
    cases = []
    ramp = torch.stack(torch.meshgrid(torch.linspace(.01, .99, 64), torch.linspace(.01, .99, 64), indexing='xy'),
                       -1).reshape(-1, 2)
    cases.append(('perspective ramp: 2x top -> 0.5x bottom', ramp, math.log(2) * (1 - 2 * ramp[:, 1])))
    for n, sep in specs:
        centres = []
        for _ in range(100000):  # objects at least sep px apart
            if len(centres) == n:
                break
            c = torch.rand(2, generator=g) * 0.9 + 0.05
            if all(((c - q) * img_px).norm() >= sep for q in centres):
                centres.append(c)
        if len(centres) < n:
            print(f'warning: only {len(centres)} of {n} objects fit {sep} px apart in {img_px} px')
        c = torch.stack(centres)
        boxes = torch.cat([c, torch.full_like(c, obj_px / img_px)], 1)
        pts, k = object_points(boxes)
        tz = (torch.rand(len(c), generator=g) * 2 - 1) * math.log(2)  # own zoom per object, 0.5x - 2x
        cases.append((f'{len(c)} objects {obj_px}px, >= {sep}px apart, 0.5-2x', pts, tz[k]))
    return cases


def dataset_frames(opt):
    """[(boxes (N, 4) normalised cx cy w h, scale to training pixels)] of up to opt.frames frames with labels."""
    with open(opt.data) as f:
        d = yaml.safe_load(f)
    assert not is_mixed(d), 'multi-source data yamls: point --data at a yaml with a plain train: / val: path'
    files = list_images(d[opt.split])
    random.Random(opt.seed).shuffle(files)
    frames = []
    for img, lab in zip(files, img2label_paths(files, opt.label_folder_name)):
        if len(frames) >= opt.frames:
            break
        p = Path(lab)
        if not p.is_file():
            continue
        rows = [r.split() for r in p.read_text().strip().splitlines() if r.strip()]
        if not rows:
            continue
        try:
            w, h = Image.open(img).size
        except OSError:
            continue
        boxes = torch.tensor([[float(v) for v in r[1:5]] for r in rows])
        frames.append((boxes, opt.img_size / max(w, h), (w, h)))
    assert frames, f'no labelled frames found for {opt.split} in {opt.data} (label folder {opt.label_folder_name})'
    return frames


def dataset_case(frames, target_size, max_zoom):
    """Points, region index and log zoom targets of every object of the frames: target zoom = target_size / object
    size (sqrt(w * h) in training pixels), clamped to [1 / max_zoom, max_zoom]."""
    pts, img, logz, n_obj = [], [], [], 0
    lim = math.log(max_zoom)
    for i, (boxes, scale, (w, h)) in enumerate(frames):
        size = torch.sqrt(boxes[:, 2] * w * scale * boxes[:, 3] * h * scale).clamp_min(1)
        p, k = object_points(boxes)
        pts.append(p)
        img.append(torch.full((len(p),), i, dtype=torch.long))
        logz.append(torch.log(target_size / size).clamp(-lim, lim)[k])
        n_obj += len(boxes)
    return torch.cat(pts), torch.cat(img), torch.cat(logz), n_obj


def main(opt):
    torch.manual_seed(opt.seed)
    torch.set_num_threads(opt.threads)
    rows = []
    if opt.data:
        frames = dataset_frames(opt)
        pts, img, logz, n_obj = dataset_case(frames, opt.target_size, opt.max_zoom)
        print(f'{len(frames)} frames, {n_obj} objects ({n_obj / len(frames):.1f} per frame), target_size '
              f'{opt.target_size} px at --img-size {opt.img_size}, max_zoom {opt.max_zoom}')
        cases = [(f'{Path(opt.data).name} ({opt.split}), size target', pts, logz, img, len(frames))]
    else:
        specs = [tuple(int(v) for v in s.split(':')) for s in opt.objects]
        cases = [(n, p, z, torch.zeros(len(p), dtype=torch.long), 1)
                 for n, p, z in synthetic_cases(opt.img_size, opt.obj_size, specs, opt.seed)]
        print(f'one {opt.img_size} x {opt.img_size} region, max_zoom {opt.max_zoom}')
    head = f'{"case":44s} ' + ' '.join(f'grid {R} ({opt.img_size // (R - 1)}px cells)'.ljust(27) for R in opt.grids)
    print(head)
    for name, pts, logz, img, n in cases:
        cells = []
        for R in opt.grids:
            err = fit(R, opt.max_zoom, img, pts, logz, n, opt.steps)
            rms, ok = math.sqrt((err ** 2).mean().item()), (err < math.log(1.2)).float().mean().item()
            cells.append(f'rms {rms:.2f}, {ok * 100:3.0f}% within 20%'.ljust(27))
            rows.append({'case': name, 'grid': R, 'cell_px': opt.img_size // (R - 1), 'max_zoom': opt.max_zoom,
                         'rms_log_zoom': round(rms, 4), 'within_20pct': round(ok, 4)})
        print(f'{name[:44]:44s} ' + ' '.join(cells))
    if opt.csv:
        with open(opt.csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f'wrote {opt.csv}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='How closely can LSE warps of each grid size reach target zooms?')
    parser.add_argument('--grids', nargs='+', type=int, default=[4, 8, 16, 32], help='warp grid sizes (knots per axis)')
    parser.add_argument('--max-zoom', type=float, default=4.0, help='LSE max_zoom')
    parser.add_argument('--img-size', type=int, default=640, help='region size in pixels (training --img-size)')
    parser.add_argument('--objects', nargs='+', default=['6:150', '20:80', '40:40'],
                        help='synthetic scenes, count:min spacing in px')
    parser.add_argument('--obj-size', type=int, default=32, help='synthetic object size in px')
    parser.add_argument('--data', default='', help='data yaml: fit real object layouts instead of synthetic scenes')
    parser.add_argument('--split', choices=('train', 'val'), default='val')
    parser.add_argument('--label-folder-name', default=DEFAULT_LABEL_FOLDER)
    parser.add_argument('--frames', type=int, default=100, help='frames sampled from the dataset (random)')
    parser.add_argument('--target-size', type=float, default=96, help='LSE target_size: objects\' sqrt(w * h) in px')
    parser.add_argument('--steps', type=int, default=600, help='optimisation steps per fit')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--csv', default='', help='also write the results to this csv file')
    main(parser.parse_args())

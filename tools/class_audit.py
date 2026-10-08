# Class x size audit of a dataset, from label files and image headers only (no training, no label caches).
#
# Per class: frames containing it and its frequency group (rare 1-10 / common 11-100 / frequent >100 frames, as LVIS),
# instances, instances by size of the smallest box side in pixels (as training sees them: native size, or the source's
# resize factor), and instances per source. Shows which classes have enough examples big enough to tell apart, which
# need repeat-factor sampling, and which may be better merged into a parent class.
#
# Usage:
#   python tools/class_audit.py --data data/mixed/template.yaml                     # train_sources or train: yaml
#   python tools/class_audit.py --data data.yaml --label-folder-name labels --bins 7 12 16 --csv audit.csv
#   python tools/class_audit.py --data data.yaml --hierarchy hierarchy.yaml        # also totals per coarse class
# hierarchy.yaml maps fine class names to coarse class names: {sedan: car, pickup: car, ...}

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
import utils.general  # noqa: F401 (import order: utils.general before utils.metrics)
from utils.datasets import DEFAULT_LABEL_FOLDER, img2label_paths
from utils.metrics import FREQ_GROUPS
from utils.mixed_data import MixedConfig, is_mixed, list_images


def group_of(frames):
    for name, lo, hi in FREQ_GROUPS:
        if lo <= frames <= hi:
            return name
    return 'absent'


def scan(path, label_folder, scale, split_name):
    """Yield (class, smallest side in px, frame id) for every label of the frames of `path`."""
    files = list_images(path)
    bad = 0
    for f, lb in zip(files, img2label_paths(files, label_folder)):
        try:
            rows = [x.split() for x in Path(lb).read_text().splitlines() if x.strip()]
        except OSError:
            continue  # no label file: background frame
        if not rows:
            continue
        if scale == 'fit':
            s = 1.0  # resized to --img-size in training: report native pixels
        else:
            s = 1.0 if scale == 'native' else float(scale)
        try:
            with Image.open(f) as im:
                w, h = im.size
        except OSError:
            bad += 1
            continue
        for r in rows:
            c, bw, bh = int(float(r[0])), float(r[3]), float(r[4])
            yield c, min(bw * w, bh * h) * s, f
    if bad:
        print(f'WARNING: {split_name}: {bad} images could not be read', file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description='Class x size audit of a YOLO dataset')
    parser.add_argument('--data', required=True, help='data yaml (train: / val:, or train_sources:)')
    parser.add_argument('--label-folder-name', default=DEFAULT_LABEL_FOLDER,
                        help='label folder (stock yaml; train_sources use their label_folder)')
    parser.add_argument('--split', choices=('train', 'val'), default='train')
    parser.add_argument('--bins', nargs='+', type=float, default=[7, 12, 16],
                        help='size bin edges, smallest box side in px (as training sees it)')
    parser.add_argument('--min-identifiable', type=int, default=10,
                        help='flag classes with fewer instances than this in the largest size bin')
    parser.add_argument('--hierarchy', default='', help='yaml mapping fine class names to coarse class names')
    parser.add_argument('--csv', default='', help='also write the table to this csv file')
    opt = parser.parse_args()

    with open(opt.data) as f:
        data = yaml.safe_load(f)
    names = data['names']
    if is_mixed(data):
        cfg = MixedConfig(data, opt.label_folder_name)
        parts = [(src.name, src.path if opt.split == 'train' else src.val, src.label_folder, src.resize)
                 for src in cfg.sources if (src.path if opt.split == 'train' else src.val)]
    else:
        parts = [(opt.split, data[opt.split], opt.label_folder_name, data.get('resize', 'fit'))]

    edges = sorted(opt.bins)
    bin_names = [f'<{edges[0]:g}'] + [f'{a:g}-{b:g}' for a, b in zip(edges[:-1], edges[1:])] + [f'>={edges[-1]:g}']
    inst = np.zeros((len(names), len(bin_names)), dtype=int)
    frames = defaultdict(set)
    per_source = defaultdict(lambda: np.zeros(len(names), dtype=int))
    n_frames = 0
    for src_name, path, folder, scale in parts:
        n_frames += len(list_images(path))
        for c, side, f in scan(path, folder, scale, src_name):
            if not 0 <= c < len(names):
                print(f'WARNING: class {c} out of range in {f}', file=sys.stderr)
                continue
            inst[c, int(np.searchsorted(edges, side, side='right'))] += 1
            frames[c].add(f)
            per_source[src_name][c] += 1

    rows = []
    for c, name in enumerate(names):
        nf = len(frames[c])
        rows.append(dict(cls=c, name=name, frames=nf, group=group_of(nf), instances=int(inst[c].sum()),
                         **{b: int(x) for b, x in zip(bin_names, inst[c])},
                         **{f'src:{s}': int(v[c]) for s, v in per_source.items()} if len(per_source) > 1 else {}))
    rows.sort(key=lambda x: -x['frames'])

    keys = list(rows[0].keys()) if rows else []
    widths = {k: max(len(k), *(len(str(r[k])) for r in rows)) + 2 for k in keys}
    print(f'{opt.split}: {n_frames} frames, {int(inst.sum())} instances, {len(names)} classes; sizes = smallest box side '
          f'in px as training sees it')
    print(''.join(f'{k:>{widths[k]}}' for k in keys))
    for r in rows:
        print(''.join(f'{r[k]:>{widths[k]}}' for k in keys))

    print('\nSummary')
    for name, lo, hi in FREQ_GROUPS + (('absent', 0, 0),):
        cls = [r for r in rows if r['group'] == name]
        rng = f'>{lo - 1}' if hi == float('inf') else f'{lo}-{hi:g}'
        print(f'  {name:>9} ({rng} frames): {len(cls):4d} classes, {sum(r["instances"] for r in cls):8d} instances')
    big = bin_names[-1]
    few = [r['name'] for r in rows if r['instances'] and r[big] < opt.min_identifiable]
    print(f'  {len(few)} classes have fewer than {opt.min_identifiable} instances {big} px (identifiable size)' +
          (f': {", ".join(map(str, few[:30]))}{" ..." if len(few) > 30 else ""}' if few else ''))
    tiny = inst[:, 0].sum() / max(inst.sum(), 1)
    print(f'  {tiny:.1%} of all instances are {bin_names[0]} px')

    if opt.hierarchy:  # totals per coarse class
        with open(opt.hierarchy) as f:
            parent = yaml.safe_load(f)
        coarse = defaultdict(lambda: np.zeros(len(bin_names), dtype=int))
        coarse_frames = defaultdict(set)
        for c, name in enumerate(names):
            p = parent.get(name, f'(unmapped) {name}')
            coarse[p] += inst[c]
            coarse_frames[p] |= frames[c]
        print('\nCoarse classes')
        print(f"{'coarse class':>24}{'frames':>9}{'group':>10}{'instances':>11}" + ''.join(f'{b:>9}' for b in bin_names))
        for p in sorted(coarse, key=lambda x: -len(coarse_frames[x])):
            nf = len(coarse_frames[p])
            print(f'{p:>24}{nf:>9}{group_of(nf):>10}{int(coarse[p].sum()):>11}' + ''.join(f'{x:>9}' for x in coarse[p]))

    if opt.csv:
        with open(opt.csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f'\nwrote {opt.csv}')


if __name__ == '__main__':
    main()

# Show how training loads one image: its label file, the labels training uses, its size, the frame as loaded with its
# labels drawn, and augmented training samples built around it (as in train_batch*.jpg). For checking an object that
# looks unlabelled in train_batch*.jpg.
#
# Usage:
#   python tools/inspect_sample.py --data data.yaml --image frame_0123 --hyp data/hyp.scratch.tiny.yaml --img-size 640
#   python tools/inspect_sample.py --data data.yaml --source camB --image frame_0123 --n 16 --fg-crop-prob 1
# --image: a substring of the image path (the first match is used; all matches are listed). Writes to --out.

import argparse
import copy
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
import utils.general  # noqa: F401 (import order: utils.general before utils.metrics)
from utils.datasets import DEFAULT_LABEL_FOLDER, LoadImagesAndLabels, _load_image, exif_size, img2label_paths
from utils.general import xywhn2xyxy
from utils.mixed_data import MixedConfig, measure_datasets
from utils.plots import plot_images


def main():
    parser = argparse.ArgumentParser(description='Show how training loads one image and its labels')
    parser.add_argument('--data', required=True, help='data yaml (train_sources or plain train:)')
    parser.add_argument('--image', required=True, help='substring of the image path')
    parser.add_argument('--source', default=None, help='train_sources: the source to search (default: all)')
    parser.add_argument('--hyp', default='data/hyp.scratch.tiny.yaml', help='training hyp yaml')
    parser.add_argument('--img-size', type=int, default=640, help='training --img-size')
    parser.add_argument('--label-folder-name', default=DEFAULT_LABEL_FOLDER,
                        help='plain data yaml: training --label-folder-name')
    parser.add_argument('--n', type=int, default=16, help='augmented samples to build around the image')
    parser.add_argument('--fg-crop-prob', type=float, default=None, help='override fg_crop_prob (e.g. 1)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', default='runs/inspect', help='output folder')
    opt = parser.parse_args()

    with open(opt.data) as f:
        data = yaml.safe_load(f)
    with open(opt.hyp) as f:
        hyp = yaml.safe_load(f)
    if data.get('train_sources'):
        cfg = MixedConfig(copy.deepcopy(data))
        datasets = measure_datasets(cfg, hyp, opt.img_size)
        names = [s.name for s in cfg.sources]
    else:
        datasets = [LoadImagesAndLabels(data['train'], opt.img_size, 16, augment=True, hyp=hyp,
                                        label_folder_name=opt.label_folder_name)]
        names = ['train']
    hits = [(k, i, p) for k, (d, nm) in enumerate(zip(datasets, names)) if opt.source in (None, nm)
            for i, p in enumerate(d.img_files) if opt.image in p]
    if not hits:
        sys.exit(f'error: no training image path contains {opt.image!r}')
    if len(hits) > 1:
        print(f'{len(hits)} matches (using the first):')
        for k, i, p in hits[:20]:
            print(f'  [{names[k]}] {p}')
    k, i, path = hits[0]
    d = datasets[k]
    if opt.fg_crop_prob is not None:
        d.fg_crop_prob = opt.fg_crop_prob
    out = Path(opt.out)
    out.mkdir(parents=True, exist_ok=True)

    # labels: the file vs what training uses (the label cache)
    lp = img2label_paths([path], d.label_folder_name)[0]
    print(f'source {names[k]}, index {i}: {path}')
    print(f'label file: {lp}')
    print('  ' + (Path(lp).read_text().strip().replace('\n', '\n  ') if Path(lp).exists() else '(missing)'))
    print(f'labels training uses (class, x, y, w, h normalised; {len(d.labels[i])}):')
    print('  ' + str(np.round(d.labels[i], 6)).replace('\n', '\n  '))

    # size: the cache's (EXIF-corrected) size vs the frame as loaded
    with Image.open(path) as im:
        raw, ex = im.size, exif_size(im)
        orient = (im.getexif() or {}).get(274)
    img, (h0, w0), _ = _load_image(d, i)
    print(f'size: file {raw[0]}x{raw[1]}, EXIF-corrected {ex[0]}x{ex[1]} (orientation tag {orient}), '
          f'cache {int(d.shapes[i][0])}x{int(d.shapes[i][1])}, loaded {img.shape[1]}x{img.shape[0]} '
          f'(resize {d.resize})')
    if (img.shape[1], img.shape[0]) != tuple(int(x * (d.scale or 1)) for x in d.shapes[i]) and d.resize != 'fit':
        print('WARNING: the loaded size differs from the cached size: labels will be misplaced')

    # the frame as loaded, labels drawn
    vis = img.copy()
    for c, *b in d.labels[i]:
        x1, y1, x2, y2 = xywhn2xyxy(np.array([b]), img.shape[1], img.shape[0])[0].round().astype(int)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(vis, str(int(c)), (x1, max(y1 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imwrite(str(out / 'frame_labels.jpg'), vis)

    # augmented training samples with this image as the sampled frame (mosaic partners are random)
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    batch = [d[i] for _ in range(opt.n)]
    imgs, targets, paths, _ = LoadImagesAndLabels.collate_fn(batch)
    plot_images(imgs, targets, paths, str(out / 'samples.jpg'), max_size=opt.img_size * 4)
    print(f'labels per sample: {[len(b[1]) for b in batch]}')
    print(f'wrote {out / "frame_labels.jpg"} (the frame as loaded, with its labels) and {out / "samples.jpg"} '
          f'({opt.n} training samples built around it, as train_batch*.jpg)')


if __name__ == '__main__':
    main()

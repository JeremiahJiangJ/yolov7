# How steadily does a detector follow objects that move towards / away from the camera? Track stability (S1 scale
# jitter, S2 centre jitter, S3 continuity, S4 worst IoU drop: utils/stability.py) against ground-truth tracks over
# sequences of frames, per motion class (toward / away / across / mixed), plus frame-to-frame consistency (no ground
# truth needed); and a comparison of runs over sequences (README.md section 14). Works for plain and DEC (LSE) models:
# DEC models canonicalize the letterbox content and map boxes back after NMS, as detect.py does.
#
# Data: <source>/images/<sequence>/<frame>.jpg with YOLO labels in <source>/<labels>/<sequence>/<frame>.txt; frames are
# ordered by the last number in their file name (never by listing order). A 6th label column is the object's track id
# (tools/zoom_sequences.py writes it); without it, a sequence must have at most one object per frame.
#
# Association of the detector's boxes with ground-truth tracks:
#   --assoc conf   highest-confidence detection of the track's class per frame: ground truth not used, only valid with one
#                  object per sequence (equivdet's choice)
#   --assoc gt     Hungarian matching of the frame's detections to its ground-truth boxes (same class, IoU >= --assoc-iou):
#                  for several objects per frame; S3 then partly measures matching, so compare runs with the same setting
# Detections at --conf-thres 0.25 (not test.py's 0.001: low-confidence boxes would make the metrics measure noise).
#
# Usage:
#   python tools/track_stability.py --weights runs/x/weights/best.pt --source data/zoom_seq --img-size 1280 --out a.json
#   python tools/track_stability.py --compare base.json lse.json            # arm B (lse) against arm A (base)
#   python tools/track_stability.py --compare base_s0.json,base_s1.json lse_s0.json,lse_s1.json   # seeds averaged
# A difference counts only if it exceeds 2x the pooled sd over sequences ("OUTSIDE"); also check it against the A/A
# floor: the difference between two seeds of the same configuration.

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
import utils.general  # noqa: F401,E402 (import order: utils.general before utils.datasets)
from utils.stability import (classify_motion, compare_arms, interframe_metrics, match_detections,  # noqa: E402
                             pool_track_stability, track_stability)

IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
METRICS = ('s1', 's2', 's3', 's4')


def frame_key(p):
    n = re.findall(r'\d+', p.stem)
    return (int(n[-1]) if n else -1, p.name)


def sequences(source, labels_folder):
    """{sequence: (image paths in frame order, label paths)}."""
    from utils.datasets import img2label_paths
    out = {}
    for d in sorted(p for p in (Path(source) / 'images').iterdir() if p.is_dir()):
        files = sorted((p for p in d.iterdir() if p.suffix.lower() in IMG_EXT), key=frame_key)
        if files:
            out[d.name] = (files, [Path(x) for x in img2label_paths([str(f) for f in files], labels_folder)])
    return out


def read_tracks(label_files, w, h):
    """Ground-truth tracks of a sequence: {track id: (class, gt (T, 4) xyxy pixels, NaN where absent)}, or None when
    a frame has several objects without track ids."""
    T = len(label_files)
    tracks = {}
    for t, lf in enumerate(label_files):
        rows = [r.split() for r in lf.read_text().strip().splitlines() if r.strip()] if lf.is_file() else []
        if len(rows) > 1 and any(len(r) < 6 for r in rows):
            return None
        for r in rows:
            tid = int(float(r[5])) if len(r) >= 6 else 0
            c, x, y, bw, bh = int(float(r[0])), float(r[1]) * w, float(r[2]) * h, float(r[3]) * w, float(r[4]) * h
            if tid not in tracks:
                tracks[tid] = (c, np.full((T, 4), np.nan))
            tracks[tid][1][t] = [x - bw / 2, y - bh / 2, x + bw / 2, y + bh / 2]
    return tracks


def detect_sequence(model, files, img_size, conf, iou, device, half):
    """Detections per frame, (N, 6) [x1, y1, x2, y2, score, cls] in original-frame pixels."""
    import cv2
    import torch
    from models.dec import map_detections
    from utils.datasets import letterbox
    from utils.general import non_max_suppression, scale_coords
    from utils.lse import letterbox_region
    stride = int(model.stride.max())
    dec = getattr(model, 'dec_ctx', None) is not None
    frame_dec = dec and any(getattr(m, 'frame', False) for m in model.model)
    out = []
    for f in files:
        im0 = cv2.imread(str(f))
        img = letterbox(im0, img_size, stride=stride)[0][:, :, ::-1].transpose(2, 0, 1)
        x = torch.from_numpy(np.ascontiguousarray(img)).to(device)
        x = (x.half() if half else x.float())[None] / 255.0
        kw = {}
        if dec:  # canonicalize the letterbox content, map boxes back after NMS (as training / detect.py)
            kw['lse_tiles'] = [letterbox_region(*im0.shape[:2], *x.shape[2:])]
            if frame_dec:
                kw['lse_defer_map'] = True
        with torch.no_grad():
            det = non_max_suppression(model(x, **kw)[0], conf, iou)[0]
            det = map_detections(model, [det])[0]
        det = det.float().clone()
        det[:, :4] = scale_coords(x.shape[2:], det[:, :4], im0.shape)
        out.append(det.cpu().numpy())
    return out, im0.shape[:2]


def associate(tracks, dets, assoc, assoc_iou):
    """The detector's box per track and frame: {track id: pred (T, 4)}."""
    T = len(dets)
    pred = {tid: np.full((T, 4), np.nan) for tid in tracks}
    if assoc == 'conf':
        (tid, (c, _)), = tracks.items()
        for t, d in enumerate(dets):
            d = d[d[:, 5] == c]
            if len(d):
                pred[tid][t] = d[np.argmax(d[:, 4]), :4]
        return pred
    for t, d in enumerate(dets):
        ids = [tid for tid, (c, g) in tracks.items() if np.isfinite(g[t]).all()]
        if not ids or not len(d):
            continue
        gt = np.array([[*tracks[tid][1][t], 1.0, tracks[tid][0]] for tid in ids])
        pairs, _ = match_detections(gt, d, assoc_iou, same_class=True)
        for i, j in pairs:
            pred[ids[i]][t] = d[j, :4]
    return pred


def evaluate(opt):
    import torch
    from models.experimental import attempt_load
    from utils.torch_utils import select_device
    device = select_device(opt.device)
    half = device.type != 'cpu' and not opt.no_half
    model = attempt_load(opt.weights, map_location=device).eval()
    if half:
        model.half()
    seqs = sequences(opt.source, opt.labels)
    assert seqs, f'no sequences in {opt.source}/images/<sequence>/'
    rows, per_seq, inter = [], {}, {}
    for name, (files, labs) in seqs.items():
        dets, (h, w) = detect_sequence(model, files, opt.img_size, opt.conf_thres, opt.iou_thres, device, half)
        tracks = read_tracks(labs, w, h)
        if tracks is None:
            print(f'skip {name}: several objects per frame without track ids (6th label column)')
            continue
        if opt.assoc == 'conf' and len(tracks) > 1:
            print(f'skip {name}: {len(tracks)} tracks; --assoc conf needs one object per sequence (use --assoc gt)')
            continue
        inter[name] = {'native': interframe_metrics(dets), 'every_3rd': interframe_metrics(dets[::3])}
        pred = associate(tracks, dets, opt.assoc, opt.assoc_iou)
        per_seq[name] = []
        for tid, (c, g) in tracks.items():
            mo, st = classify_motion(g, w), track_stability(g, pred[tid])
            rows.append((mo, st))
            per_seq[name].append(dict(track=tid, cls=c, motion=mo, **st))
    pooled = pool_track_stability(rows)
    nanmean = lambda xs: float(np.mean([x for x in xs if np.isfinite(x)])) if any(np.isfinite(xs)) else float('nan')
    inter_mean = {k: nanmean([v['native'][k] for v in inter.values()]) for k in
                  ('survival', 'centre_jitter', 'mean_iou', 'score_jitter')} if inter else {}
    res = dict(weights=opt.weights, source=opt.source, img_size=opt.img_size, conf_thres=opt.conf_thres,
               assoc=opt.assoc, pooled=pooled, interframe_mean=inter_mean, sequences=per_seq, interframe=inter)
    if opt.out:
        Path(opt.out).write_text(json.dumps(res, indent=1, default=float))
    print(f'{opt.weights}: {len(per_seq)} sequences, {len(rows)} tracks (assoc {opt.assoc})')
    print(f'{"group":8s} {"tracks":>6s} {"defined":>7s}  {"S1 scale":>9s} {"S2 centre":>9s} {"S3 cont.":>9s} {"S4 drop":>9s}')
    for g, p in pooled.items():
        print(f'{g:8s} {p["n_tracks"]:6d} {p["n_defined"]:7d}  {p["s1"]:9.4f} {p["s2"]:9.4f} {p["s3"]:9.4f} {p["s4"]:9.4f}')
    if inter_mean:
        print('frame to frame (mean over sequences): ' + ', '.join(f'{k} {v:.4f}' for k, v in inter_mean.items()))
    print('lower is better for S1, S2, S4; higher for S3. Report next to mAP.')
    return res


def per_sequence(paths):
    """{group: {metric: {sequence: value}}} from one or more result files of an arm (seeds averaged per sequence)."""
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for p in paths:
        res = json.loads(Path(p).read_text())
        for seq, tracks in res['sequences'].items():
            for group in ('all', 'toward', 'away', 'across', 'mixed'):
                sel = [tr for tr in tracks if group == 'all' or tr['motion']['kind'] == group]
                for m in METRICS:
                    vals = [tr[m] for tr in sel if tr[m] is not None and math.isfinite(tr[m])]
                    if vals:
                        acc[group][m][seq].append(float(np.mean(vals)))
    return {g: {m: {s: float(np.mean(v)) for s, v in d.items()} for m, d in md.items()} for g, md in acc.items()}


def compare(opt):
    a, b = (per_sequence(x.split(',')) for x in opt.compare)
    print(f'B ({opt.compare[1]}) - A ({opt.compare[0]}), over sequences; OUTSIDE = |delta| > 2 x pooled sd')
    print(f'{"group":8s} {"metric":6s} {"n":>3s} {"delta":>9s} {"bar":>9s} {"verdict":>9s} {"paired mean":>12s} {"sd":>8s}')
    out = {}
    for g in ('all', 'toward', 'away', 'across', 'mixed'):
        for m in METRICS:
            if g in a and g in b and m in a[g] and m in b[g]:
                r = compare_arms(a[g][m], b[g][m])
                out[f'{g}/{m}'] = r
                print(f'{g:8s} {m:6s} {r["n"]:3d} {r["delta"]:9.4f} {r["bar"]:9.4f} {r["verdict"]:>9s} '
                      f'{r["paired_mean"]:12.4f} {r["paired_sd"]:8.4f}')
    print('S1, S2, S4: negative delta = B steadier; S3: positive delta = B more continuous.')
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='track stability of a detector over sequences, or a comparison of runs')
    parser.add_argument('--weights', help='model checkpoint (plain or DEC)')
    parser.add_argument('--source', help='folder with images/<sequence>/ and <labels>/<sequence>/')
    parser.add_argument('--labels', default='labels', help='label folder name next to images/')
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--conf-thres', type=float, default=0.25)
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--assoc', choices=('conf', 'gt'), default='gt')
    parser.add_argument('--assoc-iou', type=float, default=0.1, help='--assoc gt: minimum IoU to match a detection')
    parser.add_argument('--device', default='')
    parser.add_argument('--no-half', action='store_true', help='keep fp32 on GPU')
    parser.add_argument('--out', default='', help='write the results (json) here, for --compare')
    parser.add_argument('--compare', nargs=2, metavar=('A', 'B'), help='compare result files (comma-separated seeds)')
    opt = parser.parse_args()
    if opt.compare:
        compare(opt)
    else:
        assert opt.weights and opt.source, '--weights and --source (or --compare A B)'
        evaluate(opt)

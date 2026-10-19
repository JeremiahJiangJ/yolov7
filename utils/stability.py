# Detection stability over sequences of frames (after equivdet's track-stability and interframe metrics, reimplemented
# here): how steadily a detector follows an object whose apparent size changes as it moves towards or away from the
# camera. Numpy only (scipy optional, for Hungarian matching). Used by tools/track_stability.py; README.md section 14.
#
# Per object track (ground truth gt[T, 4] and the detector's box for that object pred[T, 4], xyxy pixels, NaN rows where
# absent), on the frames where the prediction hits the ground truth (IoU >= 0.5):
#   S1 scale jitter   std over frames of log sqrt(area) residuals (prediction - ground truth), constant bias removed
#   S2 centre jitter  std of centre residuals / ground-truth diagonal, constant bias removed
#   S3 continuity     share of ground-truth frames hit
#   S4 worst drop     largest frame-to-frame fall of IoU with the ground truth (a miss counts as IoU 0)
# Every metric is a residual against the ground-truth trajectory, never the change between consecutive predictions: a
# frozen box would otherwise look perfectly steady. Tracks with fewer than 2 hit frames have S1 / S2 undefined (NaN,
# counted, never 0). Tracks are classed by their ground-truth motion: toward (grows >= 1.65x, monotone), away (shrinks
# as much), across (constant scale, crosses > 30% of the width: the control), mixed.

import math

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # greedy matching instead
    linear_sum_assignment = None


def iou_matrix(a, b):
    """IoU of boxes a (N, 4) and b (M, 4), xyxy; rows with NaN give 0. (N, M)."""
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x1, y1 = np.maximum(a[:, None, 0], b[None, :, 0]), np.maximum(a[:, None, 1], b[None, :, 1])
    x2, y2 = np.minimum(a[:, None, 2], b[None, :, 2]), np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    iou = inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-12)
    return np.nan_to_num(iou, nan=0.0)


def match_detections(a, b, iou_thr=0.5, same_class=True):
    """One-to-one matching of detections a (N, >=4) and b (M, >=4) ([x1, y1, x2, y2, score, cls]): pairs with IoU >= iou_thr
    (and the same class) maximising total IoU (Hungarian; greedy without scipy). Returns (pairs (K, 2), ious (K,))."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a.reshape(len(a), -1) if a.size else np.zeros((0, 6)), b.reshape(len(b), -1) if b.size else np.zeros((0, 6))
    iou = iou_matrix(a[:, :4], b[:, :4]) if len(a) and len(b) else np.zeros((len(a), len(b)))
    ok = iou >= iou_thr
    if same_class and len(a) and len(b) and a.shape[1] > 5 and b.shape[1] > 5:
        ok &= a[:, None, 5] == b[None, :, 5]
    if not ok.any():
        return np.zeros((0, 2), int), np.zeros(0)
    if linear_sum_assignment is not None:
        r, c = linear_sum_assignment(np.where(ok, 1 - iou, 4.0))
        keep = ok[r, c]
        pairs = np.stack([r[keep], c[keep]], 1)
    else:
        pairs, used_a, used_b = [], set(), set()
        for i, j in sorted(zip(*np.nonzero(ok)), key=lambda p: -iou[p]):
            if i not in used_a and j not in used_b:
                pairs.append((i, j)), used_a.add(i), used_b.add(j)
        pairs = np.array(pairs, int).reshape(-1, 2)
    return pairs, iou[pairs[:, 0], pairs[:, 1]]


def _log_scale(b):
    return 0.5 * np.log(np.clip((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]), 1e-12, None))


def classify_motion(gt, width, scale_thr=0.5, across_scale_thr=0.2, across_dx=0.3, monotone=0.8):
    """Motion of a ground-truth track gt (T, 4) in an image of this width: dict(kind=toward | away | across | mixed,
    delta_log_scale, dx_fraction, monotone_fraction)."""
    gt = np.asarray(gt, float)
    ok = np.isfinite(gt).all(1)
    g = gt[ok]
    if len(g) < 2:
        return dict(kind='mixed', delta_log_scale=float('nan'), dx_fraction=float('nan'), monotone_fraction=float('nan'))
    s = _log_scale(g)
    ds = s[-1] - s[0]
    steps = np.diff(s)
    mono = float((np.sign(steps) == np.sign(ds)).mean()) if ds != 0 else 0.0
    dx = float(((g[-1, 0] + g[-1, 2]) - (g[0, 0] + g[0, 2])) / 2 / width)
    if ds > scale_thr and mono >= monotone:
        kind = 'toward'
    elif ds < -scale_thr and mono >= monotone:
        kind = 'away'
    elif abs(ds) < across_scale_thr and abs(dx) > across_dx:
        kind = 'across'
    else:
        kind = 'mixed'
    return dict(kind=kind, delta_log_scale=float(ds), dx_fraction=dx, monotone_fraction=mono)


def track_stability(gt, pred, iou_thr=0.5):
    """S1 - S4 of one track: ground truth gt (T, 4) and the detector's box for it pred (T, 4), frame aligned, NaN rows
    where absent. dict(n_frames, n_scored, s1, s2, s3, s4)."""
    gt, pred = np.asarray(gt, float), np.asarray(pred, float)
    has_gt, has_pred = np.isfinite(gt).all(1), np.isfinite(pred).all(1)
    iou = np.zeros(len(gt))
    both = has_gt & has_pred
    if both.any():
        iou[both] = np.diag(iou_matrix(gt[both], pred[both]))
    hit = has_gt & (iou >= iou_thr)
    n_gt = int(has_gt.sum())
    s3 = float(hit.sum() / n_gt) if n_gt else float('nan')
    s4 = float(np.max(np.clip(iou[:-1] - iou[1:], 0, None))) if len(iou) > 1 else float('nan')
    s1 = s2 = float('nan')
    if hit.sum() >= 2:
        g, p = gt[hit], pred[hit]
        r = _log_scale(p) - _log_scale(g)
        s1 = float(np.sqrt(np.mean((r - r.mean()) ** 2)))
        diag = np.sqrt((g[:, 2] - g[:, 0]) ** 2 + (g[:, 3] - g[:, 1]) ** 2)[:, None]
        d = ((p[:, :2] + p[:, 2:]) - (g[:, :2] + g[:, 2:])) / 2 / diag
        s2 = float(np.sqrt(np.mean(np.sum((d - d.mean(0)) ** 2, 1))))
    return dict(n_frames=n_gt, n_scored=int(hit.sum()), s1=s1, s2=s2, s3=s3, s4=s4)


def pool_track_stability(rows):
    """rows: [(motion dict, stability dict)] -> {group: {n_tracks, n_defined, n_excluded, s1, s2, s3, s4}} for 'all' and
    each motion kind: unweighted means over tracks (S1 / S2 over tracks where they are defined)."""
    out = {}
    for group in ('all', 'toward', 'away', 'across', 'mixed'):
        sel = [st for mo, st in rows if group == 'all' or mo['kind'] == group]
        if not sel:
            continue
        defined = [st for st in sel if math.isfinite(st['s1'])]
        mean = lambda xs: float(np.mean(xs)) if xs else float('nan')
        out[group] = dict(n_tracks=len(sel), n_defined=len(defined), n_excluded=len(sel) - len(defined),
                          s1=mean([st['s1'] for st in defined]), s2=mean([st['s2'] for st in defined]),
                          s3=mean([st['s3'] for st in sel if math.isfinite(st['s3'])]),
                          s4=mean([st['s4'] for st in sel if math.isfinite(st['s4'])]))
    return out


def interframe_metrics(frames, iou_thr=0.5):
    """Consistency of detections between adjacent frames, no ground truth needed. frames: list of (N, 6) arrays
    [x1, y1, x2, y2, score, cls] of one sequence. dict(survival, centre_jitter, mean_iou, score_jitter, tracks,
    mean_run, max_run). Report it next to accuracy: a consistent but wrong detector also scores well."""
    n_src = n_match = 0
    jit, ious, dscore = [], [], []
    run_of = {}  # detection index in the current frame -> length of the chain ending there
    runs = []
    prev = None
    for f in frames:
        f = np.asarray(f, float).reshape(-1, 6) if len(f) else np.zeros((0, 6))
        cur = {j: 1 for j in range(len(f))}
        if prev is not None:
            n_src += len(prev)
            pairs, iou = match_detections(prev, f, iou_thr)
            n_match += len(pairs)
            for (i, j), v in zip(pairs, iou):
                a, b = prev[i], f[j]
                wa, ha = a[2] - a[0], a[3] - a[1]
                jit.append(np.hypot((b[0] + b[2] - a[0] - a[2]) / 2, (b[1] + b[3] - a[1] - a[3]) / 2) /
                           math.sqrt(max(wa * ha, 1e-12)))
                ious.append(v)
                dscore.append(abs(b[4] - a[4]))
                cur[j] = run_of.get(i, 1) + 1
            matched_prev = set(pairs[:, 0].tolist())
            runs += [run_of.get(i, 1) for i in range(len(prev)) if i not in matched_prev]  # chains that end here
        run_of, prev = cur, f
    runs += list(run_of.values())
    mean = lambda xs: float(np.mean(xs)) if len(xs) else float('nan')
    return dict(survival=n_match / n_src if n_src else float('nan'), centre_jitter=mean(jit), mean_iou=mean(ious),
                score_jitter=mean(dscore), tracks=len(runs), mean_run=mean(runs), max_run=max(runs) if runs else 0)


def compare_arms(a, b, k=2.0):
    """Compare a metric between two runs (arms) over sequences: a, b {sequence: value} (average seeds per sequence
    first). The difference counts only if |mean(b) - mean(a)| > k x pooled sd over sequences; also the paired mean
    and sd of the per-sequence differences. dict(delta, pooled_sd, bar, verdict, paired_mean, paired_sd, n)."""
    keys = sorted(k_ for k_ in set(a) & set(b) if math.isfinite(a[k_]) and math.isfinite(b[k_]))
    if len(keys) < 2:
        return dict(delta=float('nan'), pooled_sd=float('nan'), bar=float('nan'), verdict='UNDEFINED',
                    paired_mean=float('nan'), paired_sd=float('nan'), n=len(keys))
    va, vb = np.array([a[x] for x in keys]), np.array([b[x] for x in keys])
    delta = float(vb.mean() - va.mean())
    sd = float(math.sqrt((va.var(ddof=1) + vb.var(ddof=1)) / 2))
    d = vb - va
    return dict(delta=delta, pooled_sd=sd, bar=k * sd, verdict='OUTSIDE' if abs(delta) > k * sd else 'inside',
                paired_mean=float(d.mean()), paired_sd=float(d.std(ddof=1)), n=len(keys))

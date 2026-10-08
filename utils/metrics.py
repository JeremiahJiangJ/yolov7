# Model validation metrics

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from . import general

# np.trapz was renamed np.trapezoid in numpy 2.0 (and later removed)
trapezoid = getattr(np, 'trapezoid', None) or np.trapz


DEFAULT_FITNESS_WEIGHTS = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]


def fitness(x, w=None):
    # Model fitness as a weighted combination of metrics
    w = DEFAULT_FITNESS_WEIGHTS if w is None else w  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x[:, :4] * w).sum(1)


def area_fitness(results, area_results, w=None, area_w=None):
    # Fitness from per-area metrics, weighted by area_w (one weight per area bin). Bins without labels have no
    # defined AP and are dropped, renormalising the remaining weights. Falls back to fitness(results) if unusable
    if area_w is not None and area_results:
        fi = [(wb, fitness(np.array([[b['p'], b['r'], b['map50'], b['map']]]), w)) for wb, b in
              zip(area_w, area_results) if b['nt'] > 0]
        total = sum(wb for wb, _ in fi)
        if total > 0:
            return sum(wb * f for wb, f in fi) / total
    return fitness(np.array(results).reshape(1, -1), w)


def area_bins(area_int):
    # Area cut points [a, b, ...] -> [(lo, hi, name), ...] covering [0, inf), each bin lo <= area < hi
    edges = [0.0] + sorted({float(a) for a in area_int}) + [float("inf")]
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        name = f'<{hi:g}' if lo == 0 else f'>={lo:g}' if hi == float('inf') else f'{lo:g}<=A<{hi:g}'
        bins.append((lo, hi, name))
    return bins


# Class frequency groups (LVIS): by the number of training frames containing the class
FREQ_GROUPS = (('rare', 1, 10), ('common', 11, 100), ('frequent', 101, float('inf')))


def class_frames(labels, nc):
    # number of frames containing each class; labels: per frame, an array whose first column is the class
    return np.bincount(np.concatenate([np.unique(np.asarray(l)[:, 0].astype(int)) for l in labels if len(l)] +
                                      [np.zeros(0, dtype=int)]), minlength=nc)[:nc]


def ap_per_group(ap_class, ap50, ap, r, nt, frames):
    """mAP and recall per class-frequency group (FREQ_GROUPS, plus classes absent from training). ap_class, ap50, ap,
    r: per evaluated class (classes with validation labels), as from ap_per_class(); nt: validation labels per class;
    frames: training frames containing each class (class_frames()). Returns a list of dicts."""
    frames = np.asarray(frames)
    groups = [('not in training', 0, 0)] + list(FREQ_GROUPS)
    out = []
    for name, lo, hi in groups:
        in_group = (frames >= lo) & (frames <= hi)
        idx = [i for i, c in enumerate(ap_class) if c < len(frames) and in_group[c]]
        res = dict(name=name if hi == 0 else f'{name} ({lo}-{hi:g} frames)' if hi < float('inf') else f'{name} (>{lo - 1})',
                   classes=int(in_group.sum()), evaluated=len(idx), labels=int(sum(nt[ap_class[i]] for i in idx)))
        if idx:
            res.update(map50=float(np.mean(ap50[idx])), map=float(np.mean(ap[idx])), r=float(np.mean(r[idx])))
        out.append(res)
    return out


def ap_per_area(tp, conf, pred_cls, target_cls, pred_area, target_area, area_int, v5_metric=False):
    """ P, R, mAP@0.5 and mAP@0.5:0.95 per object-area bin, COCO style.
    # Arguments
        tp, conf, pred_cls, target_cls:  as ap_per_class()
        pred_area:  area of the matched target for matched predictions, own area for unmatched ones (nparray)
        target_area:  target areas (nparray), in the same units as area_int (px^2 of the original image)
        area_int:  area cut points
    A target belongs to the bin of its area. A matched prediction belongs to the bin of its target, so it is ignored
    by the other bins, and an unmatched prediction (false positive) to the bin of its own area.
    # Returns
        list of dicts with keys name, nt (number of targets), p, r, map50, map (nan for bins without targets)
    """
    results = []
    for lo, hi, name in area_bins(area_int):
        ti = (target_area >= lo) & (target_area < hi)
        pi = (pred_area >= lo) & (pred_area < hi)
        res = dict(name=name, nt=int(ti.sum()), p=np.nan, r=np.nan, map50=np.nan, map=np.nan)
        if ti.any():
            if pi.any():
                p, r, ap, _, _ = ap_per_class(tp[pi], conf[pi], pred_cls[pi], target_cls[ti], v5_metric=v5_metric)
                res.update(p=p.mean(), r=r.mean(), map50=ap[:, 0].mean(), map=ap.mean())
            else:  # labels but no predictions: everything missed
                res.update(p=0.0, r=0.0, map50=0.0, map=0.0)
        results.append(res)
    return results


def ap_per_class(tp, conf, pred_cls, target_cls, v5_metric=False, plot=False, save_dir='.', names=()):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = (target_cls == c).sum()  # number of labels
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + 1e-16)  # recall curve
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j], v5_metric=v5_metric)
                if plot and j == 0:
                    py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png', names)
        plot_mc_curve(px, f1, Path(save_dir) / 'F1_curve.png', names, ylabel='F1')
        plot_mc_curve(px, p, Path(save_dir) / 'P_curve.png', names, ylabel='Precision')
        plot_mc_curve(px, r, Path(save_dir) / 'R_curve.png', names, ylabel='Recall')

    i = f1.mean(0).argmax()  # max F1 index
    return p[:, i], r[:, i], ap, f1[:, i], unique_classes.astype('int32')


def compute_ap(recall, precision, v5_metric=False):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
        v5_metric: Assume maximum recall to be 1.0, as in YOLOv5, MMDetetion etc.
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    if v5_metric:  # New YOLOv5 metric, same as MMDetection and Detectron2 repositories
        mrec = np.concatenate(([0.], recall, [1.0]))
    else:  # Old YOLOv5 metric, i.e. default YOLOv7 metric
        mrec = np.concatenate(([0.], recall, [recall[-1] + 0.01]))
    mpre = np.concatenate(([1.], precision, [0.]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = trapezoid(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


class ConfusionMatrix:
    # Updated version of https://github.com/kaanakan/object_detection_confusion_matrix
    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        self.matrix = np.zeros((nc + 1, nc + 1))
        self.nc = nc  # number of classes
        self.conf = conf
        self.iou_thres = iou_thres

    def process_batch(self, detections, labels):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        """
        detections = detections[detections[:, 4] > self.conf]
        gt_classes = labels[:, 0].int()
        detection_classes = detections[:, 5].int()
        iou = general.box_iou(labels[:, 1:], detections[:, :4])

        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(np.int16)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[gc, detection_classes[m1[j]]] += 1  # correct
            else:
                self.matrix[self.nc, gc] += 1  # background FP

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1  # background FN

    def matrix(self):
        return self.matrix

    def plot(self, save_dir='', names=()):
        try:
            import seaborn as sn

            array = self.matrix / (self.matrix.sum(0).reshape(1, self.nc + 1) + 1E-6)  # normalize
            array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

            fig = plt.figure(figsize=(12, 9), tight_layout=True)
            sn.set(font_scale=1.0 if self.nc < 50 else 0.8)  # for label size
            labels = (0 < len(names) < 99) and len(names) == self.nc  # apply names to ticklabels
            sn.heatmap(array, annot=self.nc < 30, annot_kws={"size": 8}, cmap='Blues', fmt='.2f', square=True,
                       xticklabels=names + ['background FP'] if labels else "auto",
                       yticklabels=names + ['background FN'] if labels else "auto").set_facecolor((1, 1, 1))
            fig.axes[0].set_xlabel('True')
            fig.axes[0].set_ylabel('Predicted')
            fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)
        except Exception as e:
            pass

    def print(self):
        for i in range(self.nc + 1):
            print(' '.join(map(str, self.matrix[i])))


# Plots ----------------------------------------------------------------------------------------------------------------

def plot_pr_curve(px, py, ap, save_dir='pr_curve.png', names=()):
    # Precision-recall curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f}')  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color='grey')  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)


def plot_mc_curve(px, py, save_dir='mc_curve.png', names=(), xlabel='Confidence', ylabel='Metric'):
    # Metric-confidence curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f'{names[i]}')  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color='grey')  # plot(confidence, metric)

    y = py.mean(0)
    ax.plot(px, y, linewidth=3, color='blue', label=f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)

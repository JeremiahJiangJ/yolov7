# Frames, camera motion, the template tracker and the (optional) YOLO detector for the annotator.

from __future__ import annotations

import collections
import threading
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')


class Frames:
    """A sequence of frames: a folder of images (sorted by name, as CVAT's default lexicographical order) or a video,
    extracted once to a folder of JPEGs next to it."""

    def __init__(self, path):
        path = Path(path)
        if path.is_file():
            path = extract_video(path)
        self.dir = path
        self.files = sorted(f for f in path.iterdir() if f.suffix.lower() in IMG_EXT)
        if not self.files:
            raise FileNotFoundError(f'no images in {path}')
        h, w = self.read(0).shape[:2]
        self.width, self.height = w, h
        self._gray = collections.OrderedDict()
        self._lock = threading.Lock()

    def __len__(self):
        return len(self.files)

    @property
    def names(self):
        return [f.name for f in self.files]

    def read(self, i):
        img = cv2.imread(str(self.files[i]), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f'cannot read {self.files[i]}')
        if img.dtype != np.uint8:  # e.g. 16-bit IR: stretch to 8 bits for display and tracking
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    def gray(self, i):
        with self._lock:
            if i in self._gray:
                self._gray.move_to_end(i)
                return self._gray[i]
        g = cv2.cvtColor(self.read(i), cv2.COLOR_BGR2GRAY)
        with self._lock:
            self._gray[i] = g
            while len(self._gray) > 64:
                self._gray.popitem(last=False)
        return g

    def encoded(self, i):
        """(bytes, content type) for the browser: the file itself when the browser can show it."""
        f = self.files[i]
        ext = f.suffix.lower()
        if ext in ('.jpg', '.jpeg', '.png', '.webp', '.bmp'):
            img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
            if img is not None and img.dtype == np.uint8:
                return f.read_bytes(), {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png',
                                        '.webp': 'image/webp', '.bmp': 'image/bmp'}[ext]
        ok, buf = cv2.imencode('.png', self.read(i))
        return buf.tobytes(), 'image/png'


def extract_video(video):
    out = video.with_name(video.stem + '_frames')
    if out.is_dir() and any(out.iterdir()):
        return out
    out.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(str(out / f'{i:06d}.jpg'), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        i += 1
    cap.release()
    if i == 0:
        raise IOError(f'no frames read from {video}')
    return out


# ----------------------------------------------------------------------------------------------------------------------
# Camera motion
# ----------------------------------------------------------------------------------------------------------------------
def pair_motion(g_prev, g_cur, max_side=640):
    """2x3 similarity transform mapping frame `cur` coordinates to frame `prev` coordinates, from background features
    (corners tracked with optical flow; RANSAC ignores moving objects). Identity when it cannot be estimated."""
    s = min(1.0, max_side / max(g_prev.shape))
    a = cv2.resize(g_prev, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else g_prev
    b = cv2.resize(g_cur, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else g_cur
    pts = cv2.goodFeaturesToTrack(a, maxCorners=500, qualityLevel=0.01, minDistance=8)
    if pts is None or len(pts) < 8:
        return np.array([[1, 0, 0], [0, 1, 0]], np.float64)
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(a, b, pts, None, winSize=(21, 21), maxLevel=3)
    ok = st.ravel() == 1
    if ok.sum() < 8:
        return np.array([[1, 0, 0], [0, 1, 0]], np.float64)
    M, inl = cv2.estimateAffinePartial2D(nxt[ok], pts[ok], method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if M is None:
        return np.array([[1, 0, 0], [0, 1, 0]], np.float64)
    M = M.astype(np.float64)
    M[:, 2] /= s  # back to full-resolution pixels
    return M


def estimate_motion(frames, progress=None):
    """Per-frame 2x3 transforms from frame f to frame 0 coordinates (cumulative pair motion)."""
    W = [np.eye(3)]
    prev = frames.gray(0)
    for i in range(1, len(frames)):
        cur = frames.gray(i)
        M = np.vstack([pair_motion(prev, cur), [0, 0, 1]])
        W.append(W[-1] @ M)
        prev = cur
        if progress:
            progress(i, len(frames))
    return [w[:2].tolist() for w in W]


# ----------------------------------------------------------------------------------------------------------------------
# Template tracker
# ----------------------------------------------------------------------------------------------------------------------
def track_template(frames, box, start, stop, move=None, min_score=0.35, update=0.15, search=2.0):
    """Follow `box` from frame `start` to `stop` (exclusive) by normalised cross-correlation in a window around the
    predicted position. Suited to small objects: the template includes some context and the box size is kept.
    move(box, f_from, f_to): camera motion, used for the prediction. Returns ([(frame, box, score)], lost_at), with
    lost_at the frame where the score fell below min_score (None if it reached `stop`)."""
    move = move or (lambda b, f0, f1: list(b))
    x1, y1, x2, y2 = box
    w, h = max(x2 - x1, 2.0), max(y2 - y1, 2.0)
    pad = max(4.0, 0.25 * max(w, h))
    g = frames.gray(start).astype(np.float32)
    tmpl = _crop(g, x1 - pad, y1 - pad, x2 + pad, y2 + pad)
    if tmpl is None:
        return [], start + 1
    out, cur, vel = [], [x1, y1, x2, y2], np.zeros(2)
    for f in range(start + 1, stop):
        pred = move(cur, f - 1, f)
        pred = [pred[0] + vel[0], pred[1] + vel[1], pred[2] + vel[0], pred[3] + vel[1]]
        r = max(search * max(w, h), 16.0)
        gx = frames.gray(f).astype(np.float32)
        H, W = gx.shape
        sx1, sy1 = int(max(0, pred[0] - pad - r)), int(max(0, pred[1] - pad - r))
        sx2, sy2 = int(min(W, pred[2] + pad + r)), int(min(H, pred[3] + pad + r))
        win = gx[sy1:sy2, sx1:sx2]
        th, tw = tmpl.shape
        if win.shape[0] < th or win.shape[1] < tw:
            return out, f
        res = cv2.matchTemplate(win, tmpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        if not np.isfinite(score) or score < min_score:
            return out, f
        nx1, ny1 = sx1 + loc[0] + pad, sy1 + loc[1] + pad
        new = [nx1, ny1, nx1 + w, ny1 + h]
        moved = move(cur, f - 1, f)
        vel = 0.5 * vel + 0.5 * np.array([new[0] - moved[0], new[1] - moved[1]])
        cur = new
        out.append((f, [float(v) for v in new], float(score)))
        patch = _crop(gx, new[0] - pad, new[1] - pad, new[2] + pad, new[3] + pad)
        if patch is not None and patch.shape == tmpl.shape:
            tmpl = (1 - update) * tmpl + update * patch
    return out, None


def _crop(g, x1, y1, x2, y2):
    x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
    H, W = g.shape
    if x1 < 0 or y1 < 0 or x2 > W or y2 > H or x2 - x1 < 3 or y2 - y1 < 3:
        # near the border: clamp (the template shifts by at most the clamped amount)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
        if x2 - x1 < 3 or y2 - y1 < 3:
            return None
    return g[y1:y2, x1:x2].copy()


# ----------------------------------------------------------------------------------------------------------------------
# Detector (optional: needs torch and a YOLOv7 checkpoint from this fork)
# ----------------------------------------------------------------------------------------------------------------------
class Detector:
    def __init__(self, weights, device='', img_size=None):
        import torch
        from models.experimental import attempt_load
        from utils.torch_utils import select_device
        self.torch = torch
        self.device = select_device(device)
        self.model = attempt_load(weights, map_location=self.device).eval()
        self.half = self.device.type != 'cpu'
        if self.half:
            self.model.half()
        self.stride = int(self.model.stride.max())
        self.names = list(self.model.module.names if hasattr(self.model, 'module') else self.model.names)
        self.img_size = img_size  # None: native resolution (padded to the stride), as test.py with resize: native

    def __call__(self, img, conf=0.1, iou=0.5):
        from utils.datasets import letterbox
        from utils.general import non_max_suppression, scale_coords
        torch = self.torch
        h, w = img.shape[:2]
        if self.img_size:
            size = self.img_size
        else:
            size = (int(np.ceil(h / self.stride) * self.stride), int(np.ceil(w / self.stride) * self.stride))
        x = letterbox(img, size, stride=self.stride, auto=False, scaleup=bool(self.img_size))[0]
        x = torch.from_numpy(np.ascontiguousarray(x[:, :, ::-1].transpose(2, 0, 1))).to(self.device)
        x = (x.half() if self.half else x.float()) / 255.0
        with torch.no_grad():
            pred = self.model(x[None])[0]
        det = non_max_suppression(pred, conf, iou)[0]
        if not len(det):
            return []
        det[:, :4] = scale_coords(x.shape[1:], det[:, :4], img.shape).round()
        return [[float(v) for v in d[:4]] + [float(d[4]), int(d[5])] for d in det.cpu().numpy()]

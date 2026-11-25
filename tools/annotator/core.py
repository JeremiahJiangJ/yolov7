# Annotation model for video: tracks of boxes, with manual keyframes and automatic (tracker / detector) boxes kept apart.
#
# A track holds explicit shapes on some frames; every other frame is interpolated between its neighbours (or held after
# the last one), as in CVAT. Each explicit shape is either
#   'key'  -- placed or approved by a person. Corrections never move it.
#   'auto' -- from the tracker, the detector or a range edit. Corrections spread over these.
# Fixing an auto box spreads the same correction over the neighbouring auto boxes ("drift correction"), tapering to zero
# at the previous keyframe, so one fix repairs a whole drifted stretch instead of one box per frame.
#
# Optional camera motion (`motion`: per-frame 2x3 transforms from frame to world = frame 0 coordinates) makes
# interpolation and holding happen in world coordinates, so boxes follow the scene when the camera pans or zooms.

from __future__ import annotations

import bisect
import copy
import json
import os
from pathlib import Path

import numpy as np

PALETTE = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4',
           '#469990', '#dcbeff', '#9a6324', '#800000', '#aaffc3', '#808000', '#000075']
SPREADS = ('frame', 'drift', 'shift')


class AnnotationError(ValueError):
    pass


def _box(b):
    b = [float(v) for v in b]
    if len(b) != 4:
        raise AnnotationError(f'a box needs 4 numbers, got {b}')
    x1, y1, x2, y2 = b
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


class Project:
    def __init__(self, n_frames, width, height, labels=(), tags=(), frames=None):
        self.n_frames, self.width, self.height = int(n_frames), int(width), int(height)
        self.frames = list(frames) if frames is not None else None  # frame file names, for exports
        self.labels = []
        for name in labels:
            self.add_label(name)
        self.tags = []
        for name in tags:
            self.add_tag(name)
        self.tracks = {}  # id -> {'id', 'label', 'shapes': {frame: shape}}
        self.next_id = 0
        self.motion = None  # list of 2x3 frame->world transforms, or None
        self._undo = []
        self.dirty = False

    # ------------------------------------------------------------------------------------------------------------------
    # Labels and tags
    # ------------------------------------------------------------------------------------------------------------------
    def label_names(self):
        return [l['name'] for l in self.labels]

    def add_label(self, name, color=None):
        name = str(name).strip()
        if not name:
            raise AnnotationError('label name is empty')
        if name in self.label_names():
            return
        self.labels.append({'name': name, 'color': color or PALETTE[len(self.labels) % len(PALETTE)]})
        self.dirty = True

    def add_tag(self, name):
        name = str(name).strip()
        if not name:
            raise AnnotationError('tag name is empty')
        if name in ('occluded', 'outside', 'keyframe'):
            raise AnnotationError(f'"{name}" is a built-in property, not a tag')
        if name not in self.tags:
            self.tags.append(name)
            self.dirty = True

    def remove_tag(self, name):
        self._push_undo('remove tag', list(self.tracks), meta=True)
        if name in self.tags:
            self.tags.remove(name)
        for t in self.tracks.values():
            for s in t['shapes'].values():
                if name in s['tags']:
                    s['tags'].remove(name)
        self.dirty = True

    def rename_tag(self, old, new):
        new = str(new).strip()
        if old not in self.tags:
            raise AnnotationError(f'no tag "{old}"')
        if not new or new in self.tags:
            raise AnnotationError(f'cannot rename to "{new}"')
        self._push_undo('rename tag', list(self.tracks), meta=True)
        self.tags[self.tags.index(old)] = new
        for t in self.tracks.values():
            for s in t['shapes'].values():
                s['tags'] = [new if x == old else x for x in s['tags']]
        self.dirty = True

    # ------------------------------------------------------------------------------------------------------------------
    # Camera motion: frame <-> world coordinates
    # ------------------------------------------------------------------------------------------------------------------
    def _A(self, f):
        if self.motion is None:
            return None
        return np.asarray(self.motion[f], dtype=np.float64)

    def to_world(self, box, f):
        A = self._A(f)
        return box if A is None else _map_box(box, A)

    def from_world(self, box, f):
        A = self._A(f)
        return box if A is None else _map_box(box, _inv(A))

    def move_box(self, box, f_from, f_to):
        """The box at frame f_from, at the same scene position in frame f_to (unchanged without camera motion)."""
        if self.motion is None or f_from == f_to:
            return list(box)
        return self.from_world(self.to_world(box, f_from), f_to)

    # ------------------------------------------------------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------------------------------------------------------
    def track(self, tid):
        try:
            return self.tracks[int(tid)]
        except (KeyError, ValueError, TypeError):
            raise AnnotationError(f'no track {tid}')

    @staticmethod
    def keys(t):
        return sorted(t['shapes'])

    def resolve(self, t, f, ignore_outside=False):
        """The track's box at frame f: {'box', 'kind': key|auto|interp, 'outside', 'occluded', 'tags', 'score'}, or None
        before the track starts or while it is outside (unless ignore_outside: then the last visible box, held)."""
        keys = self.keys(t)
        i = bisect.bisect_right(keys, f) - 1
        if i < 0:
            return None
        if ignore_outside:
            while i >= 0 and t['shapes'][keys[i]]['outside']:
                i -= 1
            if i < 0:
                return None
        p = keys[i]
        sp = t['shapes'][p]
        if sp['outside']:
            return None
        if p == f:
            return copy.deepcopy(sp)
        nxt = keys[i + 1] if i + 1 < len(keys) else None  # with ignore_outside this can be an outside shape <= f
        out = {'kind': 'interp', 'outside': False, 'occluded': sp['occluded'], 'tags': list(sp['tags']),
               'score': None}
        sn = t['shapes'][nxt] if nxt is not None else None
        if sn is None or sn['outside'] or nxt <= f:
            out['box'] = self.move_box(sp['box'], p, f)  # hold (at the same scene position)
        else:
            a, b = self.to_world(sp['box'], p), self.to_world(sn['box'], nxt)
            r = (f - p) / (nxt - p)
            out['box'] = self.from_world([u + (v - u) * r for u, v in zip(a, b)], f)
        return out

    def annotations(self, f):
        out = []
        for tid in sorted(self.tracks):
            t = self.tracks[tid]
            r = self.resolve(t, f)
            if r is not None and not r['outside']:
                out.append({'id': tid, 'label': t['label'], **r})
        return out

    def span(self, t):
        """(first, last) frame the track is visible on, inclusive; last is n_frames - 1 for a track that never ends."""
        keys = self.keys(t)
        if not keys:
            return None
        last = self.n_frames - 1
        if t['shapes'][keys[-1]]['outside']:
            last = keys[-1] - 1
        return keys[0], last

    def summary(self):
        out = []
        for tid in sorted(self.tracks):
            t = self.tracks[tid]
            sp = self.span(t)
            kinds = [s['kind'] for s in t['shapes'].values() if not s['outside']]
            out.append({'id': tid, 'label': t['label'], 'first': sp[0] if sp else None,
                        'last': sp[1] if sp else None, 'keys': kinds.count('key'), 'auto': kinds.count('auto')})
        return out

    def timeline(self, tid):
        t = self.track(tid)
        return {'id': t['id'], 'label': t['label'], 'span': self.span(t),
                'marks': [[f, s['kind'], s['outside']] for f, s in sorted(t['shapes'].items())]}

    # ------------------------------------------------------------------------------------------------------------------
    # Undo
    # ------------------------------------------------------------------------------------------------------------------
    def _push_undo(self, what, tids, meta=False):
        snap = {int(i): copy.deepcopy(self.tracks.get(int(i))) for i in tids}
        entry = {'what': what, 'tracks': snap, 'next_id': self.next_id}
        if meta:
            entry['meta'] = {'labels': copy.deepcopy(self.labels), 'tags': list(self.tags)}
        self._undo.append(entry)
        del self._undo[:-300]

    def undo(self):
        if not self._undo:
            return None
        e = self._undo.pop()
        for tid, t in e['tracks'].items():
            if t is None:
                self.tracks.pop(tid, None)
            else:
                self.tracks[tid] = t
        self.next_id = max(self.next_id, e['next_id'])
        if 'meta' in e:
            self.labels, self.tags = e['meta']['labels'], e['meta']['tags']
        self.dirty = True
        return e['what']

    # ------------------------------------------------------------------------------------------------------------------
    # Editing
    # ------------------------------------------------------------------------------------------------------------------
    def _check_frame(self, f):
        f = int(f)
        if not 0 <= f < self.n_frames:
            raise AnnotationError(f'frame {f} is outside 0..{self.n_frames - 1}')
        return f

    def _shape(self, box, kind='key', outside=False, occluded=False, tags=(), score=None):
        assert kind in ('key', 'auto')
        tags = [x for x in tags if x in self.tags]
        return {'box': _box(box), 'kind': kind, 'outside': bool(outside), 'occluded': bool(occluded), 'tags': tags,
                'score': None if score is None else float(score)}

    def _materialize(self, t, f, kind='auto'):
        """Make frame f explicit, at its current resolved box (so it no longer depends on its neighbours)."""
        if f in t['shapes'] or not 0 <= f < self.n_frames:
            return
        r = self.resolve(t, f)
        if r is not None and not r['outside']:
            t['shapes'][f] = self._shape(r['box'], kind, False, r['occluded'], r['tags'])

    def create_track(self, frame, box, label, kind='key', tags=(), occluded=False, score=None, end=None, undo=True):
        f = self._check_frame(frame)
        if label not in self.label_names():
            self.add_label(label)
        tid = self.next_id
        if undo:
            self._push_undo('create track', [tid])
        self.next_id += 1
        t = {'id': tid, 'label': label, 'shapes': {f: self._shape(box, kind, False, occluded, tags, score)}}
        if end is not None and end < self.n_frames:  # ends (goes outside) at frame `end`
            t['shapes'][int(end)] = self._shape(box, kind, outside=True)
        self.tracks[tid] = t
        self.dirty = True
        return tid

    def set_label(self, tids, label):
        if label not in self.label_names():
            self.add_label(label)
        tids = [int(i) for i in _as_list(tids)]
        self._push_undo('set label', tids)
        for tid in tids:
            self.track(tid)['label'] = label
        self.dirty = True

    def delete_tracks(self, tids):
        tids = [int(i) for i in _as_list(tids)]
        self._push_undo('delete track', tids)
        for tid in tids:
            self.tracks.pop(tid, None)
        self.dirty = True

    def set_boxes(self, frame, items, spread='drift'):
        """items: [(track id, new box)] on `frame`. Each becomes a keyframe; with spread 'drift' or 'shift' the same
        correction also moves the neighbouring auto boxes (see _spread). One undo step for the group."""
        f = self._check_frame(frame)
        if spread not in SPREADS:
            raise AnnotationError(f'spread must be one of {SPREADS}')
        items = [(int(tid), _box(b)) for tid, b in items]
        self._push_undo('move' if len(items) == 1 else f'move {len(items)} boxes', [i for i, _ in items])
        for tid, box in items:
            t = self.track(tid)
            r = self.resolve(t, f)
            if r is None or r['outside']:
                r = self.resolve(t, f, ignore_outside=True)
                if r is None:
                    raise AnnotationError(f'track {tid} has no box on frame {f}')
            delta = [b - a for a, b in zip(r['box'], box)]
            if spread != 'frame' and any(abs(d) > 1e-9 for d in delta):
                self._spread(t, f, self.to_world(r['box'], f), self.to_world(box, f), spread)
            t['shapes'][f] = self._shape(box, 'key', False, r['occluded'], r['tags'], None)
        self.dirty = True

    def _spread(self, t, f, old_w, new_w, mode):
        """Move the auto boxes around frame f by the correction old -> new (in world coordinates). 'shift': the full
        correction on the run of auto boxes between the keyframes around f. 'drift': tapered to zero at the previous
        keyframe and at the next keyframe; full from f onwards when there is no next keyframe (the drift persists)."""
        delta = [b - a for a, b in zip(old_w, new_w)]
        keys = self.keys(t)
        prev_key = next_key = None
        stop_back = stop_fwd = None  # an outside shape bounds the run
        for k in reversed([k for k in keys if k < f]):
            s = t['shapes'][k]
            if s['outside']:
                stop_back = k
                break
            if s['kind'] == 'key':
                prev_key = k
                break
        for k in [k for k in keys if k > f]:
            s = t['shapes'][k]
            if s['outside']:
                stop_fwd = k
                break
            if s['kind'] == 'key':
                next_key = k
                break
        lo = prev_key if prev_key is not None else stop_back
        hi = next_key if next_key is not None else stop_fwd
        for k in keys:
            if k == f or (lo is not None and k <= lo) or (hi is not None and k >= hi):
                continue
            s = t['shapes'][k]
            if s['kind'] != 'auto' or s['outside']:
                continue
            if mode == 'shift':
                w = 1.0
            elif k < f:
                w = 0.0 if prev_key is None else (k - prev_key) / (f - prev_key)
            else:
                w = 1.0 if next_key is None else (next_key - k) / (next_key - f)
            if w <= 0:
                continue
            bw = self.to_world(s['box'], k)
            s['box'] = _box(self.from_world([v + d * w for v, d in zip(bw, delta)], k))

    def set_attrs(self, tids, frame, a=None, b=None, outside=None, occluded=None, tags=None):
        """Set outside / occluded / tags ({name: bool}) on `frame` -- from there on, until the track's next box, as
        CVAT does -- or on exactly the frames [a, b] (the frame after the range keeps its previous values).
        outside=False on a frame where the track is hidden brings it back there."""
        tids = [int(i) for i in _as_list(tids)]
        f = self._check_frame(frame)
        is_range = a is not None
        a0 = f if a is None else self._check_frame(a)
        b0 = a0 if b is None else self._check_frame(b)
        a0, b0 = min(a0, b0), max(a0, b0)
        tags = {k: bool(v) for k, v in (tags or {}).items()}
        for k in tags:
            if k not in self.tags:
                raise AnnotationError(f'no tag "{k}": add it first')
        self._push_undo('set attributes', tids)
        for tid in tids:
            t = self.track(tid)
            a, b = a0, b0
            if not is_range:  # a single frame: up to the next keyframe, so a tracked run of auto boxes follows
                b = next((k - 1 for k in self.keys(t) if k > a and t['shapes'][k]['kind'] == 'key'),
                         self.n_frames - 1)
            after = b + 1
            had_after = is_range and after < self.n_frames and after not in t['shapes']
            r_after = self.resolve(t, after) if had_after else None
            if outside is True:
                r = self.resolve(t, a, ignore_outside=True)
                if r is None:
                    continue
                for k in [k for k in t['shapes'] if a < k <= b]:
                    del t['shapes'][k]
                t['shapes'][a] = self._shape(r['box'], t['shapes'].get(a, {}).get('kind', 'key'), True,
                                             r['occluded'], r['tags'])
            else:
                if outside is False:
                    for k in range(a, b + 1):
                        r = self.resolve(t, k)
                        if r is None or r['outside']:
                            h = self.resolve(t, k, ignore_outside=True)
                            if h is None:  # before the track starts: start it here, at the first box
                                first = self.keys(t)[0]
                                h = {**self.resolve(t, first), 'box': self.move_box(t['shapes'][first]['box'], first, k)}
                            t['shapes'][k] = self._shape(h['box'], 'key', False, h['occluded'], h['tags'])
                            break
                self._materialize(t, a, 'key')
                for k, s in t['shapes'].items():
                    if a <= k <= b and not s['outside']:
                        if occluded is not None:
                            s['occluded'] = bool(occluded)
                        for name, on in tags.items():
                            if on and name not in s['tags']:
                                s['tags'].append(name)
                            elif not on and name in s['tags']:
                                s['tags'].remove(name)
            if r_after is not None and not r_after['outside'] and after not in t['shapes']:
                t['shapes'][after] = self._shape(r_after['box'], 'auto', False, r_after['occluded'], r_after['tags'])
        self.dirty = True

    def delete_shape(self, tid, frame):
        t = self.track(tid)
        f = int(frame)
        if f not in t['shapes']:
            raise AnnotationError(f'frame {f} has no explicit box on track {tid} (it is interpolated)')
        self._push_undo('delete box', [tid])
        del t['shapes'][f]
        if not any(not s['outside'] for s in t['shapes'].values()):
            del self.tracks[int(tid)]
        self.dirty = True

    def _range_op(self, what, tid, a, b, fn):
        t = self.track(tid)
        a, b = sorted((self._check_frame(a), self._check_frame(b)))
        self._push_undo(what, [tid])
        fn(t, a, b)
        if not any(not s['outside'] for s in t['shapes'].values()):
            del self.tracks[int(tid)]
        self.dirty = True

    def clear_auto(self, tid, a, b):
        """Delete the auto boxes in [a, b]: those frames go back to plain interpolation between the keyframes."""
        def fn(t, a, b):
            for k in [k for k, s in t['shapes'].items() if a <= k <= b and s['kind'] == 'auto' and not s['outside']]:
                del t['shapes'][k]
        self._range_op('re-interpolate', tid, a, b, fn)

    def delete_range(self, tid, a, b):
        """Delete every explicit box in [a, b] (keyframes too)."""
        def fn(t, a, b):
            for k in [k for k in t['shapes'] if a <= k <= b]:
                del t['shapes'][k]
        self._range_op('delete range', tid, a, b, fn)

    def set_kind(self, tid, a, b, kind):
        """Turn the boxes in [a, b] into keyframes ('key', locked) or auto boxes ('auto', moved by corrections). Use
        'auto' on tracker output imported from CVAT, where every frame is a keyframe."""
        if kind not in ('key', 'auto'):
            raise AnnotationError("kind must be 'key' or 'auto'")

        def fn(t, a, b):
            for k, s in t['shapes'].items():
                if a <= k <= b and not s['outside']:
                    s['kind'] = kind
        self._range_op('lock' if kind == 'key' else 'unlock', tid, a, b, fn)

    def shift_range(self, tids, a, b, delta):
        """Move every box of the tracks in [a, b] by delta = [dx1, dy1, dx2, dy2] (keyframes included)."""
        tids = [int(i) for i in _as_list(tids)]
        a, b = sorted((self._check_frame(a), self._check_frame(b)))
        d = _box_delta(delta)
        self._push_undo('shift range', tids)
        for tid in tids:
            t = self.track(tid)
            self._materialize(t, a)
            self._materialize(t, b)
            for k, s in t['shapes'].items():
                if a <= k <= b and not s['outside']:
                    s['box'] = _box([v + x for v, x in zip(s['box'], d)])
        self.dirty = True

    def write_track(self, tid, results, start, reached_key=None, lost_at=None):
        """Store tracker results [(frame, box, score)] as auto boxes after `start`, replacing the auto boxes up to the
        next keyframe. reached_key: (frame, tracked box) at the next keyframe -- the tracker's error there is spread back
        over the run (tapered from 0 at `start`). lost_at: the frame where the tracker lost the object; with no later
        keyframe the track ends there."""
        t = self.track(tid)
        self._push_undo('track', [tid])
        self._materialize(t, start, 'key')  # tracking starts from a keyframe
        nxt = next((k for k in self.keys(t) if k > start and t['shapes'][k]['kind'] == 'key'), None)
        end = nxt if nxt is not None else self.n_frames
        for k in [k for k, s in t['shapes'].items() if start < k < end and s['kind'] == 'auto']:
            del t['shapes'][k]
        if nxt is None:  # tracking replaces whatever followed (e.g. an old end)
            for k in [k for k in t['shapes'] if k > start]:
                del t['shapes'][k]
        src = t['shapes'][start]
        boxes = [(int(f), _box(b), s) for f, b, s in results if start < f < end]
        if reached_key is not None and boxes:
            kf, tracked = reached_key
            err = [b - a for a, b in zip(self.to_world(_box(tracked), kf), self.to_world(t['shapes'][kf]['box'], kf))]
            fixed = []
            for f, b, s in boxes:
                w = (f - start) / (kf - start)
                fixed.append((f, _box(self.from_world([v + e * w for v, e in
                                                         zip(self.to_world(b, f), err)], f)), s))
            boxes = fixed
        for f, b, s in boxes:
            t['shapes'][f] = self._shape(b, 'auto', False, src['occluded'], src['tags'], s)
        if lost_at is not None and nxt is None and lost_at < self.n_frames:
            last = boxes[-1][1] if boxes else src['box']
            t['shapes'][int(lost_at)] = self._shape(last, 'auto', outside=True)
        self.dirty = True
        return len(boxes)

    def add_tracks(self, tracks, what='pre-label'):
        """tracks: [{'label', 'boxes': [(frame, box, score)]}] from the detector; auto boxes, ending after the last."""
        new = list(range(self.next_id, self.next_id + len(tracks)))
        self._push_undo(what, new)
        for tr in tracks:
            boxes = sorted(tr['boxes'], key=lambda x: x[0])
            if not boxes:
                continue
            if tr['label'] not in self.label_names():
                self.add_label(tr['label'])
            tid = self.next_id
            self.next_id += 1
            t = {'id': tid, 'label': tr['label'], 'shapes': {}}
            for f, b, s in boxes:
                t['shapes'][int(f)] = self._shape(b, 'auto', score=s)
            end = boxes[-1][0] + 1
            if end < self.n_frames:
                t['shapes'][end] = self._shape(boxes[-1][1], 'auto', outside=True)
            self.tracks[tid] = t
        self.dirty = True
        return new

    # ------------------------------------------------------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------------------------------------------------------
    def to_dict(self):
        return {'format': 'annotator-1', 'n_frames': self.n_frames, 'width': self.width, 'height': self.height,
                'frames': self.frames, 'labels': self.labels, 'tags': self.tags, 'next_id': self.next_id,
                'motion': self.motion,
                'tracks': [{'id': t['id'], 'label': t['label'],
                            'shapes': {str(f): s for f, s in sorted(t['shapes'].items())}}
                           for _, t in sorted(self.tracks.items())]}

    @classmethod
    def from_dict(cls, d):
        p = cls(d['n_frames'], d['width'], d['height'], frames=d.get('frames'))
        p.labels = d.get('labels', [])
        p.tags = d.get('tags', [])
        p.motion = d.get('motion')
        for t in d.get('tracks', []):
            p.tracks[int(t['id'])] = {'id': int(t['id']), 'label': t['label'],
                                      'shapes': {int(f): s for f, s in t['shapes'].items()}}
        p.next_id = max([d.get('next_id', 0)] + [i + 1 for i in p.tracks])
        return p

    def save(self, path):
        path = Path(path)
        tmp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
        tmp.write_text(json.dumps(self.to_dict()))
        os.replace(tmp, path)  # atomic: a crash mid-write never leaves a broken project
        self.dirty = False

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))


# ----------------------------------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------------------------------
def _as_list(x):
    return list(x) if isinstance(x, (list, tuple)) else [x]


def _box_delta(d):
    d = [float(v) for v in d]
    if len(d) == 2:  # a plain (dx, dy) move
        d = d + d
    if len(d) != 4:
        raise AnnotationError('delta needs 2 or 4 numbers')
    return d


def _inv(A):
    M = np.vstack([A, [0, 0, 1]])
    return np.linalg.inv(M)[:2]


def _map_box(box, A):
    """Map an axis-aligned box through a similarity transform: centre mapped, size scaled (rotation ignored)."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    s = float(np.sqrt(abs(np.linalg.det(A[:, :2]))))
    nx, ny = A @ np.array([cx, cy, 1.0])
    hw, hh = (x2 - x1) / 2 * s, (y2 - y1) / 2 * s
    return [float(nx - hw), float(ny - hh), float(nx + hw), float(ny + hh)]


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def link_detections(dets, labels, iou_thr=0.1, dist=1.5, max_gap=5, min_len=3, min_score=0.0, move=None):
    """Link per-frame detections into tracks (tracking by detection).
    dets: {frame: [(x1, y1, x2, y2, score, class)]}. A detection joins the track whose last box (moved to this frame
    by `move(box, f_from, f_to)`, for camera motion) overlaps it by IoU >= iou_thr or whose centre is within
    dist * the box size (small objects rarely overlap between frames); same class; gaps of up to max_gap frames.
    Tracks shorter than min_len detections or with a mean score below min_score are dropped."""
    move = move or (lambda b, f0, f1: b)
    active, done = [], []
    for f in sorted(dets):
        cands = [d for d in dets[f]]
        still = []
        for tr in active:
            if f - tr['boxes'][-1][0] > max_gap + 1:
                done.append(tr)
            else:
                still.append(tr)
        active = still
        pairs = []
        for ti, tr in enumerate(active):
            lf, lb, _ = tr['boxes'][-1]
            pb = move(lb, lf, f)
            pc = ((pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2)
            size = max(pb[2] - pb[0], pb[3] - pb[1], 8.0)
            for di, d in enumerate(cands):
                if int(d[5]) != tr['cls']:
                    continue
                o = iou(pb, d[:4])
                c = ((d[0] + d[2]) / 2 - pc[0], (d[1] + d[3]) / 2 - pc[1])
                r = (c[0] ** 2 + c[1] ** 2) ** 0.5 / size
                if o >= iou_thr or r <= dist:
                    pairs.append((-o + r * 0.01, ti, di))
        used_t, used_d = set(), set()
        for _, ti, di in sorted(pairs):
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            d = cands[di]
            active[ti]['boxes'].append((f, [float(v) for v in d[:4]], float(d[4])))
        for di, d in enumerate(cands):
            if di not in used_d:
                active.append({'cls': int(d[5]), 'boxes': [(f, [float(v) for v in d[:4]], float(d[4]))]})
    done += active
    out = []
    for tr in done:
        scores = [s for _, _, s in tr['boxes']]
        if len(tr['boxes']) >= min_len and np.mean(scores) >= min_score:
            name = labels[tr['cls']] if tr['cls'] < len(labels) else f'class{tr["cls"]}'
            out.append({'label': name, 'boxes': tr['boxes']})
    return out

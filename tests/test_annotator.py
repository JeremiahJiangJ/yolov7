# The video annotator (tools/annotator): interpolation, drift correction over auto boxes, group and range edits, tags,
# CVAT for video round trip, YOLO export, the template tracker, camera motion, detection linking and the HTTP API.

import json
import threading
import urllib.request
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pytest

from tools.annotator.core import AnnotationError, Project, link_detections
from tools.annotator.cvat import export_cvat, export_yolo, import_cvat
from tools.annotator.vision import Frames, estimate_motion, track_template


def close(a, b, tol=1e-6):
    return np.allclose(a, b, atol=tol)


def proj(n=50):
    return Project(n, 640, 480, labels=['vehicle', 'aircraft'], tags=['truncated'],
                   frames=[f'{i:04d}.png' for i in range(n)])


def tracked(p, n_auto=19, drift=1.0):
    """A track with a keyframe at 0 and tracker boxes on 1..n_auto that drift right by `drift` px per frame."""
    tid = p.create_track(0, [100, 100, 110, 110], 'vehicle')
    p.write_track(tid, [(f, [100 + drift * f, 100, 110 + drift * f, 110], 0.9) for f in range(1, n_auto + 1)], 0)
    return tid


# ----------------------------------------------------------------------------------------------------------------------
# Interpolation
# ----------------------------------------------------------------------------------------------------------------------
def test_interpolates_between_keyframes_and_holds_after_the_last():
    p = proj()
    tid = p.create_track(0, [0, 0, 10, 10], 'vehicle')
    p.set_boxes(10, [(tid, [100, 50, 120, 70])], 'frame')
    r = p.resolve(p.track(tid), 5)
    assert r['kind'] == 'interp' and close(r['box'], [50, 25, 65, 40])
    assert p.resolve(p.track(tid), 10)['kind'] == 'key'
    assert close(p.resolve(p.track(tid), 40)['box'], [100, 50, 120, 70])  # held, as CVAT does
    assert p.span(p.track(tid)) == (0, 49)


def test_outside_hides_until_the_track_comes_back():
    p = proj()
    tid = p.create_track(0, [0, 0, 10, 10], 'vehicle')
    p.set_attrs(tid, 5, outside=True)
    assert p.resolve(p.track(tid), 4) is not None and p.resolve(p.track(tid), 5) is None
    assert all(o['id'] != tid for o in p.annotations(20))
    p.set_attrs(tid, 20, outside=False)  # O on a hidden frame brings it back there
    assert close(p.resolve(p.track(tid), 20)['box'], [0, 0, 10, 10]) and p.resolve(p.track(tid), 12) is None


# ----------------------------------------------------------------------------------------------------------------------
# Corrections spread over auto boxes
# ----------------------------------------------------------------------------------------------------------------------
def test_drift_correction_tapers_back_to_the_keyframe_and_carries_forward():
    p = proj()
    tid = tracked(p)
    p.set_boxes(10, [(tid, [100, 100, 110, 110])], 'drift')  # the tracker drifted +10 px by frame 10
    t = p.track(tid)
    assert t['shapes'][10]['kind'] == 'key'
    assert close(t['shapes'][5]['box'], [100, 100, 110, 110])  # drift 5 corrected by half of -10
    assert close(t['shapes'][1]['box'], [100, 100, 110, 110])
    assert close(t['shapes'][15]['box'], [105, 100, 115, 110])  # after the fix: the full -10, no next keyframe
    assert close(t['shapes'][0]['box'], [100, 100, 110, 110])  # keyframes never move


def test_drift_correction_tapers_to_the_next_keyframe():
    p = proj()
    tid = tracked(p)
    p.set_boxes(19, [(tid, [119, 100, 129, 110])], 'frame')  # keyframe at 19 where the tracker was right
    p.set_boxes(10, [(tid, [100, 100, 110, 110])], 'drift')
    t = p.track(tid)
    assert close(t['shapes'][15]['box'], [115 - 10 * 4 / 9, 100, 125 - 10 * 4 / 9, 110])
    assert close(t['shapes'][19]['box'], [119, 100, 129, 110])


def test_shift_and_frame_only_corrections():
    p = proj()
    tid = tracked(p)
    p.set_boxes(10, [(tid, [100, 100, 110, 110])], 'shift')
    t = p.track(tid)
    assert close(t['shapes'][3]['box'], [93, 100, 103, 110]) and close(t['shapes'][15]['box'], [105, 100, 115, 110])
    p.undo()
    p.set_boxes(10, [(tid, [100, 100, 110, 110])], 'frame')
    assert close(p.track(tid)['shapes'][5]['box'], [105, 100, 115, 110])


def test_group_move_is_one_undo_step():
    p = proj()
    a = p.create_track(0, [0, 0, 10, 10], 'vehicle')
    b = p.create_track(0, [50, 50, 60, 60], 'aircraft')
    p.set_boxes(0, [(a, [5, 5, 15, 15]), (b, [55, 55, 65, 65])])
    assert p.undo() == 'move 2 boxes'
    assert close(p.resolve(p.track(a), 0)['box'], [0, 0, 10, 10])
    assert close(p.resolve(p.track(b), 0)['box'], [50, 50, 60, 60])


def test_unlock_makes_imported_keyframes_correctable():
    p = proj()
    tid = p.create_track(0, [100, 100, 110, 110], 'vehicle')
    for f in range(1, 20):  # CVAT tracker output: a keyframe on every frame
        p.set_boxes(f, [(tid, [100 + f, 100, 110 + f, 110])], 'frame')
    p.set_boxes(10, [(tid, [100, 100, 110, 110])], 'drift')
    assert close(p.track(tid)['shapes'][15]['box'], [115, 100, 125, 110])  # locked: nothing else moved
    p.undo()
    p.set_kind(tid, 1, 19, 'auto')
    p.set_boxes(10, [(tid, [100, 100, 110, 110])], 'drift')
    assert close(p.track(tid)['shapes'][15]['box'], [105, 100, 115, 110])


def test_shift_range_moves_every_box_in_the_range():
    p = proj()
    tid = p.create_track(0, [0, 0, 10, 10], 'vehicle')
    p.set_boxes(20, [(tid, [20, 0, 30, 10])], 'frame')
    p.shift_range([tid], 5, 15, [0, 3])
    t = p.track(tid)
    assert close(p.resolve(t, 5)['box'], [5, 3, 15, 13]) and close(p.resolve(t, 15)['box'], [15, 3, 25, 13])
    assert close(p.resolve(t, 10)['box'], [10, 3, 20, 13]) and close(p.resolve(t, 20)['box'], [20, 0, 30, 10])


def test_clear_auto_returns_to_interpolation():
    p = proj()
    tid = tracked(p)
    p.set_boxes(19, [(tid, [119, 100, 129, 110])], 'frame')
    p.clear_auto(tid, 1, 18)
    t = p.track(tid)
    assert sorted(t['shapes']) == [0, 19] and p.resolve(t, 7)['kind'] == 'interp'


# ----------------------------------------------------------------------------------------------------------------------
# Tags and attributes
# ----------------------------------------------------------------------------------------------------------------------
def test_tags_on_a_range_and_the_frame_after_keeps_its_value():
    p = proj()
    tid = p.create_track(0, [0, 0, 10, 10], 'vehicle')
    p.add_tag('glare')
    p.set_attrs(tid, 0, a=5, b=9, tags={'glare': True}, occluded=True)
    t = p.track(tid)
    assert p.resolve(t, 4)['tags'] == [] and not p.resolve(t, 4)['occluded']
    assert p.resolve(t, 5)['tags'] == ['glare'] and p.resolve(t, 9)['occluded']
    assert p.resolve(t, 10)['tags'] == [] and not p.resolve(t, 30)['occluded']
    with pytest.raises(AnnotationError):
        p.set_attrs(tid, 0, tags={'nope': True})


def test_single_frame_changes_carry_over_a_tracked_run_up_to_the_next_keyframe():
    p = proj()
    tid = tracked(p)  # auto boxes on 1..19
    p.set_boxes(15, [(tid, [115, 100, 125, 110])], 'frame')  # a keyframe at 15
    p.set_attrs(tid, 5, tags={'truncated': True})
    t = p.track(tid)
    assert p.resolve(t, 5)['tags'] == p.resolve(t, 14)['tags'] == ['truncated'] and p.resolve(t, 15)['tags'] == []
    p.set_attrs(tid, 8, outside=True)  # O: the object leaves at 8 and is back at the keyframe
    assert p.resolve(t, 7) is not None and p.resolve(t, 8) is None and p.resolve(t, 14) is None
    assert p.resolve(t, 15) is not None


def test_rename_and_remove_tags():
    p = proj()
    tid = p.create_track(0, [0, 0, 10, 10], 'vehicle', tags=['truncated'])
    p.rename_tag('truncated', 'cut_off')
    assert p.resolve(p.track(tid), 3)['tags'] == ['cut_off'] and p.tags == ['cut_off']
    p.remove_tag('cut_off')
    assert p.resolve(p.track(tid), 3)['tags'] == [] and p.tags == []
    p.undo()
    assert p.tags == ['cut_off'] and p.resolve(p.track(tid), 3)['tags'] == ['cut_off']


# ----------------------------------------------------------------------------------------------------------------------
# CVAT for video and YOLO
# ----------------------------------------------------------------------------------------------------------------------
def test_cvat_export_writes_every_frame_and_round_trips(tmp_path):
    p = proj(30)
    a = tracked(p, n_auto=9)
    p.set_attrs(a, 0, a=3, b=4, tags={'truncated': True})
    p.set_attrs(a, 12, outside=True)
    b = p.create_track(5, [10, 10, 30, 40], 'aircraft')
    p.set_boxes(15, [(b, [20, 10, 40, 40])], 'frame')
    x = export_cvat(p, tmp_path / 'a.xml')
    root = ET.parse(x).getroot()
    assert root.findtext('version') == '1.1' and root.findtext('meta/task/mode') == 'interpolation'
    labels = root.findall('meta/task/labels/label')
    assert [l.findtext('name') for l in labels] == ['vehicle', 'aircraft']
    assert labels[0].findtext('attributes/attribute/input_type') == 'checkbox'
    ta = root.find("track[@id='%d']" % a)
    boxes = ta.findall('box')
    assert [int(e.get('frame')) for e in boxes] == list(range(0, 13))  # 0..11 visible, then the outside box at 12
    assert boxes[-1].get('outside') == '1' and boxes[10].get('keyframe') == '0'  # frame 10: interpolated (held)
    assert boxes[3].find("attribute[@name='truncated']").text == 'true'
    tb = root.find("track[@id='%d']" % b)
    assert len(tb.findall('box')) == 25  # 5..29, never ends

    q = import_cvat(x, 30, 640, 480)
    assert q.tags == ['truncated']
    for f in range(30):
        for tid in (a, b):
            r1, r2 = p.resolve(p.track(tid), f), q.resolve(q.track(tid), f)
            assert (r1 is None or r1['outside']) == (r2 is None or r2['outside'])
            if r1 is not None and not r1['outside']:
                assert close(r1['box'], r2['box'], 0.01) and r1['tags'] == r2['tags']


def test_import_as_auto_and_yolo_export(tmp_path):
    p = proj(10)
    tid = p.create_track(0, [64, 48, 128, 96], 'aircraft')
    q = import_cvat(export_cvat(p, tmp_path / 'a.xml'), 10, 640, 480, keyframes_as='auto',
                    frames=[f'{i:04d}.png' for i in range(10)])
    assert q.track(tid)['shapes'][0]['kind'] == 'auto'
    out = export_yolo(q, tmp_path / 'labels')
    assert (out / '0003.txt').read_text().split() == ['1', '0.150000', '0.150000', '0.100000', '0.100000']


# ----------------------------------------------------------------------------------------------------------------------
# Tracker, camera motion, detection linking
# ----------------------------------------------------------------------------------------------------------------------
def textured(h, w, seed=0):
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, (h // 8 + 2, w // 8 + 2), dtype=np.uint8)
    return cv2.GaussianBlur(cv2.resize(small, (small.shape[1] * 8, small.shape[0] * 8),
                                       interpolation=cv2.INTER_NEAREST), (5, 5), 0)


def make_video(tmp_path, pan, obj, n=30, size=(320, 240)):
    """Frames of a textured scene seen by a camera at offset pan(f), with a bright 8 px object at obj(f) (scene
    coordinates). Returns (Frames, true object box in frame coordinates per frame)."""
    W, H = size
    scene = textured(H + 200, W + 200)
    d = tmp_path / 'frames'
    d.mkdir()
    truth = []
    for f in range(n):
        cx, cy = pan(f)
        img = scene[100 + cy:100 + cy + H, 100 + cx:100 + cx + W].copy()
        ox, oy = obj(f)
        fx, fy = ox - cx, oy - cy
        img[fy:fy + 8, fx:fx + 8] = 255
        img[fy + 2:fy + 6, fx + 2:fx + 6] = 40  # some structure inside, so the template locks on
        truth.append([fx, fy, fx + 8, fy + 8])
        cv2.imwrite(str(d / f'{f:04d}.png'), cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    return Frames(d), truth


def test_template_tracker_follows_a_small_object(tmp_path):
    frames, truth = make_video(tmp_path, lambda f: (0, 0), lambda f: (60 + 3 * f, 80 + f))
    res, lost = track_template(frames, truth[0], 0, 30)
    assert lost is None and len(res) == 29
    assert max(abs(b[0] - truth[f][0]) + abs(b[1] - truth[f][1]) for f, b, _ in res) <= 1


def test_camera_motion_and_world_interpolation(tmp_path):
    pan = lambda f: (int(round(40 * np.sin(np.pi * f / 20))), 0)  # pans right and back: 0 at frames 0 and 20
    frames, truth = make_video(tmp_path, pan, lambda f: (150, 100), n=21)
    motion = estimate_motion(frames)
    assert abs(motion[10][0][2] - pan(10)[0]) < 1.0  # frame 10 -> frame 0: shift by the pan
    p = Project(21, frames.width, frames.height, labels=['vehicle'])
    tid = p.create_track(0, truth[0], 'vehicle')
    p.set_boxes(20, [(tid, truth[20])], 'frame')
    assert abs(p.resolve(p.track(tid), 10)['box'][0] - truth[10][0]) > 30  # plain interpolation misses
    p.motion = motion
    assert abs(p.resolve(p.track(tid), 10)['box'][0] - truth[10][0]) < 1.5  # follows the camera


def test_link_detections_builds_tracks_across_gaps():
    dets = {}
    for f in range(20):
        dets[f] = []
        if f not in (7, 8):  # missed twice
            dets[f].append([10 + 4 * f, 10, 18 + 4 * f, 18, 0.9, 0])
        dets[f].append([300, 200 - 3 * f, 310, 210 - 3 * f, 0.8, 1])
    dets[5].append([500, 400, 505, 405, 0.3, 0])  # a one-frame false alarm
    tracks = link_detections(dets, ['vehicle', 'aircraft'], max_gap=3, min_len=3)
    assert sorted((t['label'], len(t['boxes'])) for t in tracks) == [('aircraft', 20), ('vehicle', 18)]
    p = proj(25)
    p.add_tracks(tracks)
    v = [t for t in p.tracks.values() if t['label'] == 'vehicle'][0]
    assert p.resolve(v, 7)['kind'] == 'interp' and p.resolve(v, 20) is None  # gap filled; ends after the last


# ----------------------------------------------------------------------------------------------------------------------
# Server
# ----------------------------------------------------------------------------------------------------------------------
def test_server_api_end_to_end(tmp_path):
    from http.server import ThreadingHTTPServer
    from tools.annotator.server import build_app, make_handler, parse_opt
    frames, truth = make_video(tmp_path, lambda f: (0, 0), lambda f: (60 + 2 * f, 80), n=15)
    app = build_app(parse_opt(['--frames', str(frames.dir), '--labels', 'vehicle', '--tags', 'truncated']))
    srv = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(app))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f'http://127.0.0.1:{srv.server_address[1]}'

    def get(path):
        with urllib.request.urlopen(base + path) as r:
            return r.read(), r.headers['Content-Type']

    def post(body):
        req = urllib.request.Request(base + '/api/op', json.dumps(body).encode(), {'Content-Type': 'application/json'})
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read())

    try:
        meta = json.loads(get('/api/meta')[0])
        assert meta['n_frames'] == 15 and meta['tags'] == ['truncated'] and meta['labels'][0]['name'] == 'vehicle'
        assert get('/api/frame?i=3')[1] == 'image/png' and b'Annotator' in get('/')[0]
        tid = post({'op': 'create', 'frame': 0, 'box': truth[0], 'label': 'vehicle'})['id']
        r = post({'op': 'track', 'ids': [tid], 'frame': 0, 'max': 100})
        assert r[str(tid)]['frames'] == 14
        objs = json.loads(get('/api/ann?i=10')[0])['objects']
        assert objs[0]['kind'] == 'auto' and abs(objs[0]['box'][0] - truth[10][0]) <= 1
        post({'op': 'attrs', 'ids': [tid], 'frame': 4, 'tags': {'truncated': True}})
        post({'op': 'add_tag', 'name': 'glare'})
        out = post({'op': 'export'})
        assert ET.parse(out['cvat']).getroot().find('track') is not None
        with pytest.raises(urllib.error.HTTPError):
            post({'op': 'attrs', 'ids': [tid], 'frame': 99})
        app.save()
        saved = Project.load(app.path)
        assert saved.tags == ['truncated', 'glare'] and len(saved.tracks) == 1
    finally:
        srv.shutdown()


def test_prelabel_with_a_yolov7_checkpoint(tmp_path):
    """The detector path: a (randomly initialised) yolov7-tiny checkpoint, native-resolution inference, linking, and
    tracks added as auto boxes with scores."""
    import torch
    from conftest import ROOT
    from models.yolo import Model
    from tools.annotator.server import build_app, parse_opt
    m = Model(str(ROOT / 'cfg/training/yolov7-tiny.yaml'), ch=3, nc=2)
    m.names = ['vehicle', 'aircraft']
    ckpt = tmp_path / 'tiny.pt'
    torch.save({'model': m.half()}, ckpt)
    frames, _ = make_video(tmp_path, lambda f: (0, 0), lambda f: (60 + 2 * f, 80), n=4)
    app = build_app(parse_opt(['--frames', str(frames.dir), '--weights', str(ckpt), '--device', 'cpu']))
    det = app.detector(frames.read(0), conf=0.0)
    assert det and all(len(d) == 6 and 0 <= d[0] <= d[2] <= frames.width + 1 and d[5] in (0, 1) for d in det)
    assert app.p.label_names()[:2] == ['vehicle', 'aircraft']
    msg = app.job_prelabel(0, 3, conf=0.0, min_len=1, max_gap=1, progress=lambda *a: None)
    assert 'tracks' in msg and app.p.tracks
    s = next(iter(app.p.tracks.values()))['shapes']
    assert all(x['kind'] == 'auto' for x in s.values()) and any(x['score'] is not None for x in s.values())

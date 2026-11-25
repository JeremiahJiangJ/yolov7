# Video annotation tool: a local web app (standard library server + one page, no internet needed).
#
#   python tools/annotator/server.py --frames /data/flight3/images --labels vehicle aircraft
#   python tools/annotator/server.py --video flight3.mp4 --weights best.pt          # pre-labelling with your model
#   python tools/annotator/server.py --frames ... --import-cvat annotations.xml      # continue a CVAT task
#
# then open http://127.0.0.1:8765. The project autosaves to <frames>.annotations.json; export writes CVAT for video 1.1
# XML (and optionally YOLO labels). See README section 14.

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.annotator.core import AnnotationError, Project, link_detections  # noqa: E402
from tools.annotator.cvat import export_cvat, export_yolo, import_cvat  # noqa: E402
from tools.annotator.vision import Frames, estimate_motion, track_template  # noqa: E402

STATIC = Path(__file__).parent / 'static'


class App:
    def __init__(self, frames, project_path, project, detector=None, export_path=None):
        self.frames, self.path, self.p, self.detector = frames, Path(project_path), project, detector
        self.export_path = Path(export_path) if export_path else self.path.with_suffix('.xml')
        self.lock = threading.RLock()
        self.job = {'name': None, 'done': 0, 'total': 0, 'message': ''}
        self._stop = threading.Event()
        threading.Thread(target=self._autosave, daemon=True).start()

    def _autosave(self):
        while not self._stop.wait(1.0):
            self.save()

    def save(self):
        with self.lock:
            if self.p.dirty:
                self.p.save(self.path)

    # ------------------------------------------------------------------------------------------------------------------
    def meta(self):
        p = self.p
        return {'n_frames': p.n_frames, 'width': p.width, 'height': p.height, 'labels': p.labels, 'tags': p.tags,
                'detector': self.detector is not None, 'motion': p.motion is not None, 'project': str(self.path),
                'export': str(self.export_path)}

    def op(self, d):
        p, name = self.p, d.get('op')
        if name == 'create':
            return {'id': p.create_track(d['frame'], d['box'], d['label'], tags=d.get('tags', []))}
        if name == 'set_boxes':
            p.set_boxes(d['frame'], [(i['id'], i['box']) for i in d['items']], d.get('spread', 'drift'))
        elif name == 'attrs':
            p.set_attrs(d['ids'], d['frame'], d.get('a'), d.get('b'), d.get('outside'), d.get('occluded'),
                        d.get('tags'))
        elif name == 'label':
            p.set_label(d['ids'], d['label'])
        elif name == 'delete_tracks':
            p.delete_tracks(d['ids'])
        elif name == 'delete_shape':
            p.delete_shape(d['id'], d['frame'])
        elif name == 'delete_range':
            p.delete_range(d['id'], d['a'], d['b'])
        elif name == 'clear_auto':
            p.clear_auto(d['id'], d['a'], d['b'])
        elif name == 'set_kind':
            p.set_kind(d['id'], d['a'], d['b'], d['kind'])
        elif name == 'shift_range':
            p.shift_range(d['ids'], d['a'], d['b'], d['delta'])
        elif name == 'add_label':
            p.add_label(d['name'], d.get('color'))
        elif name == 'add_tag':
            p.add_tag(d['name'])
        elif name == 'remove_tag':
            p.remove_tag(d['name'])
        elif name == 'rename_tag':
            p.rename_tag(d['old'], d['new'])
        elif name == 'undo':
            return {'undone': p.undo()}
        elif name == 'track':
            return self.track(d['ids'], d['frame'], d.get('max', 300))
        elif name == 'export':
            out = {'cvat': str(export_cvat(p, d.get('path') or self.export_path, self.frames.dir.name))}
            if d.get('yolo'):
                out['yolo'] = str(export_yolo(p, d['yolo']))
            return out
        elif name == 'save':
            self.save()
        else:
            raise AnnotationError(f'unknown op {name!r}')
        return {}

    def track(self, ids, frame, max_frames):
        """Run the tracker on each track from `frame`, up to its next keyframe (whose position then corrects the run)
        or max_frames, outside the lock for the image work."""
        p, f = self.p, int(frame)
        out = {}
        for tid in ids if isinstance(ids, list) else [ids]:
            with self.lock:
                t = p.track(tid)
                r = p.resolve(t, f)
                if r is None or r['outside']:
                    raise AnnotationError(f'track {tid} is not visible on frame {f}')
                nxt = next((k for k in p.keys(t) if k > f and t['shapes'][k]['kind'] == 'key'), None)
                outside_next = next((k for k in p.keys(t) if k > f and t['shapes'][k]['outside']), None)
                stop = min(f + 1 + int(max_frames), p.n_frames)
                if nxt is not None and (outside_next is None or nxt < outside_next):
                    stop = min(stop, nxt + 1)
                box = list(r['box'])
            res, lost = track_template(self.frames, box, f, stop, move=p.move_box)
            reached = None
            if nxt is not None and res and res[-1][0] == nxt:
                reached = (nxt, res[-1][1])
            with self.lock:
                n = p.write_track(tid, [x for x in res if x[0] != nxt], f, reached, lost)
            out[str(tid)] = {'frames': n, 'lost_at': lost, 'reached_key': nxt if reached else None}
        return out

    # ------------------------------------------------------------------------------------------------------------------
    # Background jobs
    # ------------------------------------------------------------------------------------------------------------------
    def start_job(self, name, fn):
        with self.lock:
            if self.job['name']:
                raise AnnotationError(f'"{self.job["name"]}" is still running')
            self.job = {'name': name, 'done': 0, 'total': 0, 'message': '', 'error': None}

        def run():
            try:
                self.job['message'] = fn(self._progress) or 'done'
            except Exception as e:  # reported to the page
                traceback.print_exc()
                self.job['error'] = str(e)
            finally:
                self.job['name'] = None
        threading.Thread(target=run, daemon=True).start()

    def _progress(self, done, total):
        self.job['done'], self.job['total'] = done, total

    def job_motion(self, progress):
        motion = estimate_motion(self.frames, progress)
        with self.lock:
            self.p.motion = motion
            self.p.dirty = True
        return f'camera motion estimated for {len(motion)} frames'

    def job_prelabel(self, a, b, conf, min_len, max_gap, progress):
        if self.detector is None:
            raise AnnotationError('no detector: start the server with --weights')
        a, b = sorted((int(a), int(b)))
        dets = {}
        for i, f in enumerate(range(a, b + 1)):
            dets[f] = self.detector(self.frames.read(f), conf=conf)
            progress(i + 1, b - a + 1)
        with self.lock:
            tracks = link_detections(dets, self.detector.names, max_gap=max_gap, min_len=min_len,
                                     move=self.p.move_box)
            new = self.p.add_tracks(tracks)
        return f'{len(new)} tracks from {sum(len(v) for v in dets.values())} detections'


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype='application/json'):
            if isinstance(body, (dict, list)):
                body = json.dumps(body).encode()
            elif isinstance(body, str):
                body = body.encode()
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store' if ctype == 'application/json' else 'max-age=3600')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path in ('/', '/index.html'):
                    return self._send(200, (STATIC / 'index.html').read_bytes(), 'text/html; charset=utf-8')
                if u.path.startswith('/static/'):
                    f = (STATIC / u.path[len('/static/'):]).resolve()
                    if STATIC.resolve() not in f.parents or not f.is_file():
                        return self._send(404, {'error': 'not found'})
                    ctype = {'.js': 'text/javascript', '.css': 'text/css'}.get(f.suffix, 'application/octet-stream')
                    return self._send(200, f.read_bytes(), ctype + '; charset=utf-8')
                if u.path == '/api/meta':
                    with app.lock:
                        return self._send(200, app.meta())
                if u.path == '/api/frame':
                    data, ctype = app.frames.encoded(int(q['i']))
                    return self._send(200, data, ctype)
                if u.path == '/api/ann':
                    i = int(q['i'])
                    with app.lock:
                        return self._send(200, {'frame': i, 'name': app.frames.names[i],
                                                'objects': app.p.annotations(i)})
                if u.path == '/api/track':
                    with app.lock:
                        return self._send(200, app.p.timeline(q['id']))
                if u.path == '/api/tracks':
                    with app.lock:
                        return self._send(200, app.p.summary())
                if u.path == '/api/job':
                    return self._send(200, app.job)
                return self._send(404, {'error': 'not found'})
            except (AnnotationError, KeyError, ValueError, IndexError) as e:
                return self._send(400, {'error': str(e)})

        def do_POST(self):
            try:
                n = int(self.headers.get('Content-Length', 0))
                d = json.loads(self.rfile.read(n) or b'{}')
                if self.path == '/api/op':
                    if d.get('op') == 'track':
                        return self._send(200, {'ok': True, **app.op(d)})
                    with app.lock:
                        return self._send(200, {'ok': True, **app.op(d)})
                if self.path == '/api/job':
                    if d['job'] == 'motion':
                        app.start_job('camera motion', app.job_motion)
                    elif d['job'] == 'prelabel':
                        app.start_job('pre-label', lambda pr: app.job_prelabel(
                            d['a'], d['b'], float(d.get('conf', 0.2)), int(d.get('min_len', 3)),
                            int(d.get('max_gap', 5)), pr))
                    else:
                        raise AnnotationError(f'unknown job {d["job"]!r}')
                    return self._send(200, {'ok': True})
                return self._send(404, {'error': 'not found'})
            except (AnnotationError, KeyError, ValueError, IndexError, TypeError) as e:
                return self._send(400, {'error': str(e)})
            except Exception as e:
                traceback.print_exc()
                return self._send(500, {'error': f'{type(e).__name__}: {e}'})

    return Handler


def build_app(opt):
    frames = Frames(opt.frames or opt.video)
    path = Path(opt.project) if opt.project else frames.dir.with_name(frames.dir.name + '.annotations.json')
    if path.exists() and not opt.import_cvat:
        p = Project.load(path)
        if p.n_frames != len(frames):
            raise SystemExit(f'{path} has {p.n_frames} frames, the folder has {len(frames)}: wrong project file?')
        p.frames = frames.names
    elif opt.import_cvat:
        if path.exists():
            raise SystemExit(f'{path} exists: move it away or pass another --project to import into')
        p = import_cvat(opt.import_cvat, len(frames), frames.width, frames.height, frames.names,
                        keyframes_as='auto' if opt.import_as_auto else 'key')
    else:
        p = Project(len(frames), frames.width, frames.height, frames=frames.names)
    for name in opt.labels or []:
        p.add_label(name)
    for name in opt.tags or []:
        p.add_tag(name)
    detector = None
    if opt.weights:
        from tools.annotator.vision import Detector
        detector = Detector(opt.weights, opt.device, opt.img_size)
        for name in detector.names:
            p.add_label(name)
    if not p.labels:
        p.add_label('object')
    p.dirty = True
    return App(frames, path, p, detector, opt.export)


def parse_opt(argv=None):
    ap = argparse.ArgumentParser(description='Video annotation tool (CVAT for video export)')
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--frames', help='folder of frames (sorted by file name)')
    src.add_argument('--video', help='video file (extracted once to <name>_frames/)')
    ap.add_argument('--project', help='project file (default: <frames folder>.annotations.json)')
    ap.add_argument('--export', help='CVAT XML path (default: the project path with .xml)')
    ap.add_argument('--labels', nargs='*', help='labels to start with (more can be added in the page)')
    ap.add_argument('--tags', nargs='*', help='tags to start with (more can be added in the page)')
    ap.add_argument('--import-cvat', help='start from a CVAT for video 1.1 XML file')
    ap.add_argument('--import-as-auto', action='store_true',
                    help='import every CVAT box as an auto box (CVAT tracker output), so corrections spread over it')
    ap.add_argument('--weights', help='YOLOv7 checkpoint for pre-labelling (needs torch)')
    ap.add_argument('--device', default='', help='cuda device, e.g. 0, or cpu')
    ap.add_argument('--img-size', type=int, default=None, help='detector input size (default: native resolution)')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8765)
    return ap.parse_args(argv)


def main(argv=None):
    opt = parse_opt(argv)
    app = build_app(opt)
    srv = ThreadingHTTPServer((opt.host, opt.port), make_handler(app))
    print(f'{len(app.frames)} frames {app.frames.width}x{app.frames.height} from {app.frames.dir}')
    print(f'project: {app.path}\nopen http://{opt.host}:{opt.port}  (Ctrl+C to stop; the project autosaves)')
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.save()
        print('saved', app.path)


if __name__ == '__main__':
    main()

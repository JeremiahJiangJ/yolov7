# CVAT for video 1.1 (XML) import / export, and YOLO label export.
#
# Export follows CVAT's own exports: every frame of a track is written, from its first box to the frame it goes outside.
# Keyframes and auto boxes are written with keyframe="1" (so CVAT keeps them exactly), interpolated frames with
# keyframe="0". Tags become checkbox attributes on every label (mutable, default false).
# Import reads keyframe="1" boxes as keyframes; CVAT tracker output therefore arrives as keyframes on every frame --
# use "unlock" (Project.set_kind(..., 'auto')) on such a range so corrections spread over it.

from __future__ import annotations

import datetime
import xml.etree.ElementTree as ET
from pathlib import Path

from .core import Project

TRUE = ('true', '1', 'yes')


def _fmt(v):
    return f'{v:.2f}'


def export_cvat(p: Project, path, task_name='annotator'):
    root = ET.Element('annotations')
    ET.SubElement(root, 'version').text = '1.1'
    meta = ET.SubElement(root, 'meta')
    task = ET.SubElement(meta, 'task')
    for k, v in (('id', '0'), ('name', task_name), ('size', str(p.n_frames)), ('mode', 'interpolation'),
                 ('overlap', '0'), ('start_frame', '0'), ('stop_frame', str(p.n_frames - 1))):
        ET.SubElement(task, k).text = v
    labels = ET.SubElement(task, 'labels')
    for l in p.labels:
        le = ET.SubElement(labels, 'label')
        ET.SubElement(le, 'name').text = l['name']
        ET.SubElement(le, 'color').text = l['color']
        ET.SubElement(le, 'type').text = 'rectangle'
        attrs = ET.SubElement(le, 'attributes')
        for tag in p.tags:
            a = ET.SubElement(attrs, 'attribute')
            for k, v in (('name', tag), ('mutable', 'True'), ('input_type', 'checkbox'), ('default_value', 'false'),
                         ('values', 'false')):
                ET.SubElement(a, k).text = v
    size = ET.SubElement(task, 'original_size')
    ET.SubElement(size, 'width').text = str(p.width)
    ET.SubElement(size, 'height').text = str(p.height)
    ET.SubElement(meta, 'dumped').text = datetime.datetime.now().isoformat()

    for tid in sorted(p.tracks):
        t = p.tracks[tid]
        sp = p.span(t)
        if sp is None:
            continue
        src = 'manual' if any(s['kind'] == 'key' for s in t['shapes'].values()) else 'auto'
        te = ET.SubElement(root, 'track', id=str(tid), label=t['label'], source=src)
        was_visible = False
        for f in range(sp[0], p.n_frames):
            r = p.resolve(t, f)
            if r is None or r['outside']:
                if was_visible:  # the frame it leaves: one outside box, as CVAT writes it
                    last = p.resolve(t, f, ignore_outside=True)
                    _box_el(te, f, last['box'], True, True, last['occluded'], last['tags'], p.tags)
                was_visible = False
                if f > max(t['shapes']):
                    break
                continue
            _box_el(te, f, r['box'], r['kind'] != 'interp', False, r['occluded'], r['tags'], p.tags)
            was_visible = True
    ET.indent(root) if hasattr(ET, 'indent') else None
    Path(path).write_text('<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(root, encoding='unicode'))
    return path


def _box_el(parent, f, box, keyframe, outside, occluded, tags, all_tags):
    x1, y1, x2, y2 = box
    e = ET.SubElement(parent, 'box', frame=str(f), keyframe=str(int(keyframe)), outside=str(int(outside)),
                      occluded=str(int(occluded)), xtl=_fmt(x1), ytl=_fmt(y1), xbr=_fmt(x2), ybr=_fmt(y2), z_order='0')
    for tag in all_tags:
        ET.SubElement(e, 'attribute', name=tag).text = 'true' if tag in tags else 'false'


def import_cvat(path, n_frames=None, width=None, height=None, frames=None, keyframes_as='key'):
    """A Project from a CVAT for video 1.1 file. Boolean attributes become tags. keyframes_as='auto' imports every box
    as an auto box (e.g. CVAT tracker output, so corrections spread over it)."""
    root = ET.parse(path).getroot()
    task = root.find('meta/task')
    if task is None:
        task = root.find('meta/job')
    size = int(task.findtext('size')) if task is not None and task.findtext('size') else None
    w = task.findtext('original_size/width') if task is not None else None
    h = task.findtext('original_size/height') if task is not None else None
    p = Project(n_frames or size or 1, width or int(w or 0), height or int(h or 0), frames=frames)
    if task is not None:
        for le in task.findall('labels/label'):
            p.add_label(le.findtext('name'), le.findtext('color'))
            for a in le.findall('attributes/attribute'):
                if a.findtext('input_type') == 'checkbox':
                    p.add_tag(a.findtext('name'))
    for te in root.findall('track'):
        label = te.get('label')
        if label not in p.label_names():
            p.add_label(label)
        tid = int(te.get('id', p.next_id))
        if tid in p.tracks:
            tid = p.next_id
        t = {'id': tid, 'label': label, 'shapes': {}}
        for b in te.findall('box'):
            outside = b.get('outside') == '1'
            if b.get('keyframe', '1') != '1' and not outside:
                continue  # interpolated frame: recomputed from the keyframes
            tags = []
            for a in b.findall('attribute'):
                if (a.text or '').strip().lower() in TRUE:
                    if a.get('name') not in p.tags:
                        p.add_tag(a.get('name'))
                    tags.append(a.get('name'))
            box = [float(b.get(k)) for k in ('xtl', 'ytl', 'xbr', 'ybr')]
            t['shapes'][int(b.get('frame'))] = p._shape(box, keyframes_as, outside, b.get('occluded') == '1', tags)
        if t['shapes']:
            p.tracks[tid] = t
            p.next_id = max(p.next_id, tid + 1)
    p.dirty = False
    return p


def export_yolo(p: Project, out_dir):
    """One YOLO label file per frame (class cx cy w h, normalised), classes in the project's label order. Frames with
    no boxes get an empty file (background). Needs the frame names (frames from a folder or a video)."""
    if not p.frames:
        raise ValueError('the project has no frame names to name the label files after')
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = p.label_names()
    W, H = p.width, p.height
    for f, name in enumerate(p.frames):
        lines = []
        for o in p.annotations(f):
            x1, y1, x2, y2 = (max(0.0, min(v, lim)) for v, lim in zip(o['box'], (W, H, W, H)))
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            lines.append(f'{names.index(o["label"])} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} '
                         f'{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}')
        (out / f'{Path(name).stem}.txt').write_text('\n'.join(lines) + ('\n' if lines else ''))
    return out

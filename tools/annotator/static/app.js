'use strict';
// Video annotator page: one canvas, a side panel and a timeline. All editing logic runs on the server (core.py);
// the page draws, collects gestures and sends one operation per edit.

const $ = (id) => document.getElementById(id);
const view = $('view'), ctx = view.getContext('2d');
const tl = $('timeline'), tctx = tl.getContext('2d');

const S = {
  meta: null, frame: 0, objects: [], name: '',
  sel: new Set(), drawLabel: null, mode: 'select', hide: false, loupe: false,
  scale: 1, ox: 0, oy: 0, img: null, cache: new Map(),
  drag: null, mouse: null, range: null, timeline: null, playing: null,
};

// --------------------------------------------------------------------------------------------------------------------
// Server
// --------------------------------------------------------------------------------------------------------------------
async function get(path) {
  const r = await fetch(path);
  const d = await r.json();
  if (!r.ok) throw new Error(d.error || r.statusText);
  return d;
}

async function op(body, quiet) {
  const r = await fetch('/api/op', {method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body)});
  const d = await r.json();
  if (!r.ok) { toast(d.error || r.statusText, true); throw new Error(d.error); }
  await refresh();
  if (!quiet && d.undone) toast('Undid: ' + d.undone);
  return d;
}

function toast(msg, err) {
  const t = $('toast');
  t.textContent = msg; t.className = err ? 'err' : ''; t.style.display = 'block';
  clearTimeout(toast.h);
  toast.h = setTimeout(() => { t.style.display = 'none'; }, err ? 5000 : 2500);
}

// --------------------------------------------------------------------------------------------------------------------
// Frames
// --------------------------------------------------------------------------------------------------------------------
function image(i) {
  if (S.cache.has(i)) return S.cache.get(i);
  const im = new Image();
  im.src = '/api/frame?i=' + i;
  S.cache.set(i, im);
  if (S.cache.size > 40) S.cache.delete(S.cache.keys().next().value);
  return im;
}

async function goto(i) {
  if (!S.meta) return;
  i = Math.max(0, Math.min(S.meta.n_frames - 1, i | 0));
  S.frame = i;
  const im = image(i);
  for (let k = 1; k <= 3; k++) if (i + k < S.meta.n_frames) image(i + k);
  if (!im.complete) await new Promise((res) => { im.onload = res; im.onerror = res; });
  if (S.frame !== i) return;
  S.img = im;
  await loadAnn();
}

async function loadAnn() {
  const d = await get('/api/ann?i=' + S.frame);
  if (d.frame !== S.frame) return;
  S.objects = d.objects; S.name = d.name;
  render(); panel();
}

async function refresh() {
  S.meta = await get('/api/meta');
  await loadAnn();
  await loadTimeline();
  renderLabels(); renderTags();
}

async function loadTimeline() {
  const id = first();
  S.timeline = id === null ? null : await get('/api/track?id=' + id).catch(() => null);
  drawTimeline();
}

function first() {
  for (const id of S.sel) return id;
  return null;
}

// --------------------------------------------------------------------------------------------------------------------
// Canvas
// --------------------------------------------------------------------------------------------------------------------
function fit() {
  if (!S.meta) return;
  const w = view.clientWidth, h = view.clientHeight;
  S.scale = Math.min(w / S.meta.width, h / S.meta.height);
  S.ox = (w - S.meta.width * S.scale) / 2; S.oy = (h - S.meta.height * S.scale) / 2;
  render();
}
const toImg = (x, y) => [(x - S.ox) / S.scale, (y - S.oy) / S.scale];
const toScr = (x, y) => [x * S.scale + S.ox, y * S.scale + S.oy];
const color = (label) => (S.meta.labels.find((l) => l.name === label) || {color: '#fff'}).color;

function shownBox(o) {
  // the box as dragged, while a drag is in progress
  const d = S.drag;
  if (d && d.boxes && d.boxes.has(o.id)) return d.boxes.get(o.id);
  return o.box;
}

function render() {
  const dpr = window.devicePixelRatio || 1;
  const w = view.clientWidth, h = view.clientHeight;
  if (view.width !== w * dpr || view.height !== h * dpr) { view.width = w * dpr; view.height = h * dpr; }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = '#0b0d10'; ctx.fillRect(0, 0, w, h);
  if (!S.img || !S.meta) return;
  ctx.imageSmoothingEnabled = S.scale < 2;
  ctx.drawImage(S.img, S.ox, S.oy, S.meta.width * S.scale, S.meta.height * S.scale);
  if (!S.hide) {
    for (const o of S.objects) drawBox(o, shownBox(o), S.sel.has(o.id));
  }
  if (S.drag && S.drag.kind === 'draw' && S.drag.now) {
    const [a, b] = [S.drag.start, S.drag.now];
    const [x1, y1] = toScr(Math.min(a[0], b[0]), Math.min(a[1], b[1]));
    const [x2, y2] = toScr(Math.max(a[0], b[0]), Math.max(a[1], b[1]));
    ctx.strokeStyle = color(S.drawLabel); ctx.lineWidth = 1.5; ctx.setLineDash([]);
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  }
  if (S.mode === 'draw' && S.mouse) {  // crosshair helps place small boxes
    ctx.strokeStyle = '#ffffff55'; ctx.lineWidth = 1; ctx.setLineDash([]);
    ctx.beginPath(); ctx.moveTo(S.mouse[0], 0); ctx.lineTo(S.mouse[0], h);
    ctx.moveTo(0, S.mouse[1]); ctx.lineTo(w, S.mouse[1]); ctx.stroke();
  }
  if (S.loupe && S.mouse) drawLoupe();
  const m = $('mode');
  m.style.display = S.mode === 'draw' || S.hide ? 'block' : 'none';
  m.textContent = (S.mode === 'draw' ? 'DRAW: ' + S.drawLabel : '') + (S.hide ? '  boxes hidden (H)' : '');
}

function drawBox(o, b, selected) {
  const [x1, y1] = toScr(b[0], b[1]), [x2, y2] = toScr(b[2], b[3]);
  const c = color(o.label);
  ctx.setLineDash(o.kind === 'interp' ? [4, 3] : []);
  if (selected) { ctx.strokeStyle = '#fff'; ctx.lineWidth = 4; ctx.strokeRect(x1, y1, x2 - x1, y2 - y1); }
  ctx.strokeStyle = c; ctx.lineWidth = o.kind === 'key' ? 2 : 1.2;
  ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  ctx.setLineDash([]);
  if (selected && resizable(x1, y1, x2, y2)) {
    ctx.fillStyle = '#fff';
    for (const [hx, hy] of handles(x1, y1, x2, y2)) ctx.fillRect(hx - 3, hy - 3, 6, 6);
  }
  const tags = (o.occluded ? ['occ'] : []).concat(o.tags);
  const txt = '#' + o.id + (selected || S.scale > 1.5 ? ' ' + o.label : '') + (tags.length ? ' [' + tags.join(',') + ']' : '');
  ctx.font = '11px system-ui, sans-serif';
  const tw = ctx.measureText(txt).width;
  ctx.fillStyle = '#000b'; ctx.fillRect(x1, y1 - 14, tw + 6, 13);
  ctx.fillStyle = c; ctx.fillText(txt, x1 + 3, y1 - 4);
}

const resizable = (x1, y1, x2, y2) => x2 - x1 >= 20 && y2 - y1 >= 20;

function handles(x1, y1, x2, y2) {
  const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
  return [[x1, y1], [mx, y1], [x2, y1], [x2, my], [x2, y2], [mx, y2], [x1, y2], [x1, my]];
}

function drawLoupe() {
  const [mx, my] = S.mouse, R = 90, Z = 4;
  const [ix, iy] = toImg(mx, my);
  const src = R / (S.scale * Z);
  ctx.save();
  ctx.beginPath(); ctx.arc(mx + R + 20, my - R - 20, R, 0, Math.PI * 2); ctx.clip();
  ctx.imageSmoothingEnabled = false;
  const k = S.scale * Z;
  const cx = mx + R + 20, cy = my - R - 20;
  ctx.drawImage(S.img, ix - src, iy - src, src * 2, src * 2, cx - R, cy - R, R * 2, R * 2);
  for (const o of S.objects) {
    const b = shownBox(o);
    ctx.strokeStyle = color(o.label); ctx.lineWidth = 1;
    ctx.strokeRect(cx + (b[0] - ix) * k, cy + (b[1] - iy) * k, (b[2] - b[0]) * k, (b[3] - b[1]) * k);
  }
  ctx.restore();
  ctx.strokeStyle = '#fff'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.arc(mx + R + 20, my - R - 20, R, 0, Math.PI * 2); ctx.stroke();
}

// Hit testing: the handle of a selected box first, then the smallest box under the cursor
function hit(sx, sy) {
  for (const o of S.objects) {
    if (!S.sel.has(o.id)) continue;
    const b = o.box;
    const [x1, y1] = toScr(b[0], b[1]), [x2, y2] = toScr(b[2], b[3]);
    if (!resizable(x1, y1, x2, y2)) continue;  // small on screen: a drag moves it (zoom in to resize)
    const hs = handles(x1, y1, x2, y2);
    for (let k = 0; k < 8; k++) {
      if (Math.abs(sx - hs[k][0]) <= 6 && Math.abs(sy - hs[k][1]) <= 6) return {o, handle: k};
    }
  }
  let best = null, area = Infinity;
  for (const o of S.objects) {
    const b = o.box;
    const [x1, y1] = toScr(b[0], b[1]), [x2, y2] = toScr(b[2], b[3]);
    if (sx >= x1 - 4 && sx <= x2 + 4 && sy >= y1 - 4 && sy <= y2 + 4) {
      const a = (x2 - x1) * (y2 - y1);
      if (a < area) { area = a; best = o; }
    }
  }
  return best ? {o: best, handle: -1} : null;
}

function applyHandle(b, k, dx, dy) {
  let [x1, y1, x2, y2] = b;
  if (k === 0 || k === 6 || k === 7) x1 += dx;
  if (k === 2 || k === 3 || k === 4) x2 += dx;
  if (k === 0 || k === 1 || k === 2) y1 += dy;
  if (k === 4 || k === 5 || k === 6) y2 += dy;
  return [Math.min(x1, x2), Math.min(y1, y2), Math.max(x1, x2), Math.max(y1, y2)];
}

view.addEventListener('contextmenu', (e) => e.preventDefault());
view.addEventListener('mousedown', (e) => {
  const r = view.getBoundingClientRect(), sx = e.clientX - r.left, sy = e.clientY - r.top;
  if (e.button === 1 || e.button === 2) { S.drag = {kind: 'pan', sx, sy, ox: S.ox, oy: S.oy}; return; }
  const p = toImg(sx, sy);
  if (S.mode === 'draw') { S.drag = {kind: 'draw', start: p, now: null}; return; }
  const h = hit(sx, sy);
  if (!h) { if (!e.shiftKey) { S.sel.clear(); selChanged(); } return; }
  if (e.shiftKey && h.handle < 0) {
    if (S.sel.has(h.o.id)) S.sel.delete(h.o.id); else S.sel.add(h.o.id);
    selChanged(); return;
  }
  if (!S.sel.has(h.o.id)) { S.sel.clear(); S.sel.add(h.o.id); selChanged(); }
  const ids = h.handle >= 0 ? [h.o.id] : [...S.sel].filter((id) => S.objects.some((o) => o.id === id));
  S.drag = {kind: 'edit', handle: h.handle, start: p, ids, alt: e.altKey, moved: false,
    orig: new Map(ids.map((id) => [id, S.objects.find((o) => o.id === id).box])), boxes: new Map()};
});

window.addEventListener('mousemove', (e) => {
  const r = view.getBoundingClientRect(), sx = e.clientX - r.left, sy = e.clientY - r.top;
  S.mouse = sx >= 0 && sy >= 0 && sx <= r.width && sy <= r.height ? [sx, sy] : null;
  const d = S.drag;
  if (d && d.kind === 'pan') { S.ox = d.ox + sx - d.sx; S.oy = d.oy + sy - d.sy; }
  else if (d && d.kind === 'draw') d.now = toImg(sx, sy);
  else if (d && d.kind === 'edit') {
    const p = toImg(sx, sy), dx = p[0] - d.start[0], dy = p[1] - d.start[1];
    if (Math.abs(dx) * S.scale + Math.abs(dy) * S.scale > 1) d.moved = true;
    for (const [id, b] of d.orig) {
      d.boxes.set(id, d.handle >= 0 ? applyHandle(b, d.handle, dx, dy) : [b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy]);
    }
  }
  if (d || S.mode === 'draw' || S.loupe) render();
});

window.addEventListener('mouseup', async () => {
  const d = S.drag; S.drag = null;
  if (!d) return;
  if (d.kind === 'draw' && d.now) {
    const b = [d.start[0], d.start[1], d.now[0], d.now[1]];
    if (Math.abs(b[2] - b[0]) >= 1 && Math.abs(b[3] - b[1]) >= 1) {
      const r = await op({op: 'create', frame: S.frame, box: b, label: S.drawLabel});
      S.sel = new Set([r.id]); selChanged();
    }
  } else if (d.kind === 'edit' && d.moved) {
    if (d.alt) {
      if (!S.range) { toast('Alt+drag moves a range: set one first with [ and ]', true); render(); return; }
      const [id0] = d.ids, a = d.orig.get(id0), b = d.boxes.get(id0);
      await op({op: 'shift_range', ids: d.ids, a: S.range[0], b: S.range[1], delta: b.map((v, k) => v - a[k])});
    } else {
      await op({op: 'set_boxes', frame: S.frame, spread: spread(),
        items: d.ids.map((id) => ({id, box: d.boxes.get(id)}))});
    }
  }
  render();
});

view.addEventListener('wheel', (e) => {
  e.preventDefault();
  const r = view.getBoundingClientRect(), sx = e.clientX - r.left, sy = e.clientY - r.top;
  const [ix, iy] = toImg(sx, sy);
  S.scale = Math.max(0.05, Math.min(64, S.scale * Math.exp(-e.deltaY * 0.0015)));
  S.ox = sx - ix * S.scale; S.oy = sy - iy * S.scale;
  render();
}, {passive: false});

window.addEventListener('resize', () => render());

// --------------------------------------------------------------------------------------------------------------------
// Panel
// --------------------------------------------------------------------------------------------------------------------
const spread = () => document.querySelector('input[name=spread]:checked').value;

function selChanged() {
  // drop ids that no longer exist anywhere in this frame only when they are gone from the project (timeline 404s)
  render(); panel(); loadTimeline();
}

function renderLabels() {
  const box = $('labels'); box.innerHTML = '';
  S.meta.labels.forEach((l, i) => {
    const c = document.createElement('span');
    c.className = 'chip' + (l.name === S.drawLabel ? ' on' : '');
    c.innerHTML = `<span class="sw" style="background:${l.color}"></span>${i < 9 ? i + 1 + ' ' : ''}`;
    c.append(l.name);
    c.onclick = () => { S.drawLabel = l.name; renderLabels(); render(); };
    box.append(c);
  });
  if (!S.drawLabel || !S.meta.labels.some((l) => l.name === S.drawLabel)) {
    S.drawLabel = S.meta.labels.length ? S.meta.labels[0].name : null;
  }
}

function renderTags() {
  const box = $('tags'); box.innerHTML = '';
  if (!S.meta.tags.length) { box.innerHTML = '<span class="muted">No tags yet.</span>'; return; }
  S.meta.tags.forEach((t, i) => {
    const c = document.createElement('span');
    c.className = 'chip';
    c.title = 'double-click to rename';
    c.append((i < 9 ? 'Alt+' + (i + 1) + ' ' : '') + t);
    const x = document.createElement('span'); x.className = 'x'; x.textContent = '×';
    x.onclick = async (e) => {
      e.stopPropagation();
      if (confirm(`Remove the tag "${t}" from every box?`)) await op({op: 'remove_tag', name: t});
    };
    c.ondblclick = async () => {
      const n = prompt('Rename tag', t);
      if (n && n !== t) await op({op: 'rename_tag', old: t, new: n});
    };
    c.append(x); box.append(c);
  });
}

function panel() {
  if (!S.meta) return;
  $('frameinfo').textContent = `Frame ${S.frame} / ${S.meta.n_frames - 1}  ${S.name}`;
  $('slider').max = S.meta.n_frames - 1; $('slider').value = S.frame; $('frameno').value = S.frame;
  $('range').textContent = S.range ? `range ${S.range[0]}–${S.range[1]}` : 'no range ([ and ] set it)';
  $('motionstate').textContent = S.meta.motion ? 'on: interpolation follows the camera' : '';
  $('prelabel').hidden = !S.meta.detector;
  // objects on this frame
  const ob = $('objects'); ob.innerHTML = '';
  if (!S.objects.length) ob.innerHTML = '<span class="muted">None.</span>';
  for (const o of S.objects) {
    const row = document.createElement('div');
    row.className = 'obj' + (S.sel.has(o.id) ? ' sel' : '');
    row.innerHTML = `<span class="sw" style="background:${color(o.label)}"></span>#${o.id} ${o.label}` +
      `<span class="kind">${o.kind}${o.score != null ? ' ' + o.score.toFixed(2) : ''}</span>`;
    row.onclick = (e) => {
      if (!e.shiftKey) S.sel.clear();
      if (S.sel.has(o.id) && e.shiftKey) S.sel.delete(o.id); else S.sel.add(o.id);
      selChanged();
    };
    ob.append(row);
  }
  // selection
  const sel = [...S.sel], here = S.objects.filter((o) => S.sel.has(o.id));
  $('selcount').textContent = sel.length ? `${sel.length} track${sel.length > 1 ? 's' : ''}` : '';
  const sb = $('selection');
  if (!sel.length) { sb.className = 'muted'; sb.textContent = 'Click a box (Shift+click adds; Ctrl+A selects all).'; return; }
  sb.className = '';
  const o = here[0];
  const tagsOn = (t) => here.length && here.every((x) => x.tags.includes(t));
  let html = '';
  if (sel.length === 1) {
    html += `<div class="row">Track #${sel[0]} ` +
      `<select id="sellabel">${S.meta.labels.map((l) => `<option${o && o.label === l.name ? ' selected' : ''}>${l.name}</option>`).join('')}</select>` +
      `<span class="muted">${o ? o.kind : 'not visible here'}</span></div>`;
  }
  if (here.length) {
    html += `<div class="row"><label><input type="checkbox" id="selocc"${here.every((x) => x.occluded) ? ' checked' : ''}> occluded (C)</label></div>`;
    html += '<div class="row">' + S.meta.tags.map((t) =>
      `<label><input type="checkbox" data-tag="${t}"${tagsOn(t) ? ' checked' : ''}> ${t}</label>`).join(' ') + '</div>';
  }
  html += `<div class="row muted">${inRange() ? `Changes apply to the range ${S.range[0]}–${S.range[1]}` :
    'Changes apply from this frame to the next keyframe'}</div>`;
  html += '<div class="btns">' +
    '<button data-a="track" title="T">Track ▶</button>' +
    '<button data-a="key" title="K">Keyframe</button>' +
    '<button data-a="outside" title="O">Outside</button>' +
    '<button data-a="reinterp" title="R">Re-interpolate range</button>' +
    '<button data-a="unlock" title="U">Unlock range</button>' +
    '<button data-a="lock" title="L">Lock range</button>' +
    '<button data-a="delrange">Delete range</button>' +
    '<button data-a="delbox" title="Delete">Delete box here</button>' +
    '<button data-a="deltrack" title="Shift+Delete">Delete track</button></div>';
  sb.innerHTML = html;
  const lab = $('sellabel');
  if (lab) lab.onchange = () => op({op: 'label', ids: sel, label: lab.value});
  const occ = $('selocc');
  if (occ) occ.onchange = () => attrs({occluded: occ.checked});
  sb.querySelectorAll('input[data-tag]').forEach((cb) => {
    cb.onchange = () => attrs({tags: {[cb.dataset.tag]: cb.checked}});
  });
  sb.querySelectorAll('button[data-a]').forEach((b) => { b.onclick = () => action(b.dataset.a); });
}

// inside the selected range, attribute changes (tags, occluded, outside) cover the whole range
const inRange = () => S.range && S.frame >= S.range[0] && S.frame <= S.range[1];
function rangeArgs() {
  return inRange() ? {a: S.range[0], b: S.range[1]} : {};
}

async function attrs(a) {
  if (!S.sel.size) return;
  await op({op: 'attrs', ids: [...S.sel], frame: S.frame, ...rangeArgs(), ...a});
}

function needRange() {
  if (!S.range) { toast('Set a range first: [ and ] at the start and end frames', true); return false; }
  return true;
}

async function action(a) {
  const ids = [...S.sel];
  if (!ids.length) { toast('Select a box first', true); return; }
  const here = S.objects.filter((o) => S.sel.has(o.id));
  if (a === 'track') {
    toast('Tracking…');
    const r = await op({op: 'track', ids: here.map((o) => o.id), frame: S.frame, max: 300});
    const msg = Object.entries(r).filter(([k]) => k !== 'ok').map(([id, v]) =>
      `#${id}: ${v.frames} frames` + (v.reached_key != null ? ', corrected to keyframe ' + v.reached_key :
        v.lost_at != null ? ', lost at ' + v.lost_at : '')).join('; ');
    toast(msg || 'Nothing to track');
  } else if (a === 'key') {
    await op({op: 'set_boxes', frame: S.frame, spread: 'frame', items: here.map((o) => ({id: o.id, box: o.box}))});
  } else if (a === 'outside') {
    const visible = here.length > 0;
    await op({op: 'attrs', ids, frame: S.frame, ...rangeArgs(), outside: visible ? true : false});
  } else if (a === 'reinterp' || a === 'unlock' || a === 'lock' || a === 'delrange') {
    if (!needRange()) return;
    for (const id of ids) {
      if (a === 'reinterp') await op({op: 'clear_auto', id, a: S.range[0], b: S.range[1]}, true);
      if (a === 'unlock') await op({op: 'set_kind', id, a: S.range[0], b: S.range[1], kind: 'auto'}, true);
      if (a === 'lock') await op({op: 'set_kind', id, a: S.range[0], b: S.range[1], kind: 'key'}, true);
      if (a === 'delrange') await op({op: 'delete_range', id, a: S.range[0], b: S.range[1]}, true);
    }
    toast({reinterp: 'Re-interpolated', unlock: 'Unlocked: corrections now spread over this range',
      lock: 'Locked as keyframes', delrange: 'Deleted the boxes in the range'}[a]);
  } else if (a === 'delbox') {
    for (const o of here) {
      if (o.kind === 'interp') { toast(`#${o.id} is interpolated here: nothing to delete`, true); continue; }
      await op({op: 'delete_shape', id: o.id, frame: S.frame}, true);
    }
  } else if (a === 'deltrack') {
    if (!confirm(`Delete ${ids.length} track(s) on every frame?`)) return;
    await op({op: 'delete_tracks', ids});
    S.sel.clear(); selChanged();
  }
}

// --------------------------------------------------------------------------------------------------------------------
// Timeline
// --------------------------------------------------------------------------------------------------------------------
function drawTimeline() {
  const dpr = window.devicePixelRatio || 1, w = tl.clientWidth, h = tl.clientHeight;
  if (tl.width !== w * dpr || tl.height !== h * dpr) { tl.width = w * dpr; tl.height = h * dpr; }
  tctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  tctx.fillStyle = '#101317'; tctx.fillRect(0, 0, w, h);
  if (!S.meta) return;
  const n = S.meta.n_frames, X = (f) => (f + 0.5) / n * w;
  if (S.range) {
    tctx.fillStyle = '#4f8cff33';
    tctx.fillRect(X(S.range[0]) - 0.5 * w / n, 0, (S.range[1] - S.range[0] + 1) * w / n, h);
  }
  const t = S.timeline;
  if (t && t.span) {
    tctx.fillStyle = '#ffffff14';
    tctx.fillRect(X(t.span[0]), h / 2 - 4, X(t.span[1]) - X(t.span[0]) + 1, 8);
    for (const [f, kind, outside] of t.marks) {
      const x = X(f);
      if (outside) { tctx.fillStyle = '#ff6b6b'; tctx.fillRect(x - 1, 2, 2, h - 4); }
      else if (kind === 'key') { tctx.fillStyle = '#fff'; tctx.fillRect(x - 1, 3, 2, h - 6); }
      else { tctx.fillStyle = '#4f8cff'; tctx.fillRect(x - 0.5, h / 2 - 6, 1, 12); }
    }
    tctx.fillStyle = '#8b93a1'; tctx.font = '11px system-ui'; tctx.fillText('#' + t.id + ' ' + t.label, 4, 12);
  } else {
    tctx.fillStyle = '#8b93a1'; tctx.font = '11px system-ui';
    tctx.fillText('Select a track to see its keyframes; click to jump, Shift+drag to set a range', 4, 20);
  }
  tctx.fillStyle = '#ffd24f'; tctx.fillRect(X(S.frame) - 1, 0, 2, h);
}

let tlDrag = null;
tl.addEventListener('mousedown', (e) => {
  const f = Math.floor(e.offsetX / tl.clientWidth * S.meta.n_frames);
  if (e.shiftKey) { tlDrag = f; S.range = [f, f]; drawTimeline(); panel(); } else goto(f).then(drawTimeline);
});
tl.addEventListener('mousemove', (e) => {
  if (tlDrag === null) return;
  const f = Math.max(0, Math.min(S.meta.n_frames - 1, Math.floor(e.offsetX / tl.clientWidth * S.meta.n_frames)));
  S.range = [Math.min(tlDrag, f), Math.max(tlDrag, f)]; drawTimeline(); panel();
});
window.addEventListener('mouseup', () => { tlDrag = null; });

// leave the keyboard to the shortcuts after a jump
$('slider').addEventListener('input', (e) => goto(+e.target.value).then(drawTimeline));
$('slider').addEventListener('change', (e) => e.target.blur());
$('frameno').addEventListener('change', (e) => { e.target.blur(); goto(+e.target.value).then(drawTimeline); });

// --------------------------------------------------------------------------------------------------------------------
// Keyboard
// --------------------------------------------------------------------------------------------------------------------
async function step(df) { await goto(S.frame + df); drawTimeline(); panel(); }

async function jumpKey(dir) {
  const t = S.timeline;
  if (!t) return;
  const ks = t.marks.filter((m) => m[1] === 'key' && !m[2]).map((m) => m[0]);
  const f = dir > 0 ? ks.find((k) => k > S.frame) : ks.reverse().find((k) => k < S.frame);
  if (f !== undefined) await step(f - S.frame);
}

window.addEventListener('keydown', async (e) => {
  if (e.target.tagName === 'INPUT' && e.target.type !== 'checkbox' && e.target.type !== 'radio' ||
      e.target.tagName === 'SELECT') return;
  const k = e.key.toLowerCase(), digit = e.code.startsWith('Digit') ? +e.code.slice(5) : 0;
  if (e.ctrlKey || e.metaKey) {
    if (k === 'z') { e.preventDefault(); await op({op: 'undo'}); }
    else if (k === 's') { e.preventDefault(); await op({op: 'save'}, true); toast('Saved'); }
    else if (k === 'a') { e.preventDefault(); S.sel = new Set(S.objects.map((o) => o.id)); selChanged(); }
    return;
  }
  if (e.altKey && digit) {
    e.preventDefault();
    const t = S.meta.tags[digit - 1];
    if (!t) return;
    const here = S.objects.filter((o) => S.sel.has(o.id));
    await attrs({tags: {[t]: !(here.length && here.every((o) => o.tags.includes(t)))}});
    return;
  }
  if (digit) {
    const l = S.meta.labels[digit - 1];
    if (!l) return;
    if (e.shiftKey && S.sel.size) await op({op: 'label', ids: [...S.sel], label: l.name});
    else { S.drawLabel = l.name; renderLabels(); render(); }
    return;
  }
  const n = e.shiftKey ? 10 : 1;
  switch (k) {
    case 'arrowright': case 'f': e.preventDefault(); await step(n); break;
    case 'arrowleft': case 'd': e.preventDefault(); await step(-n); break;
    case 'home': await step(-S.frame); break;
    case 'end': await step(S.meta.n_frames - 1 - S.frame); break;
    case ',': await jumpKey(-1); break;
    case '.': await jumpKey(1); break;
    case ' ': e.preventDefault(); togglePlay(); break;
    case 'n': S.mode = S.mode === 'draw' ? 'select' : 'draw'; view.style.cursor = S.mode === 'draw' ? 'crosshair' : ''; render(); break;
    case 'escape': S.mode = 'select'; view.style.cursor = ''; S.sel.clear(); selChanged(); break;
    case 'tab': {
      e.preventDefault();
      if (!S.objects.length) break;
      const ids = S.objects.map((o) => o.id), cur = first();
      const i = cur === null ? -1 : ids.indexOf(cur);
      S.sel = new Set([ids[(i + (e.shiftKey ? ids.length - 1 : 1)) % ids.length]]); selChanged(); break;
    }
    case 't': await action('track'); break;
    case 'k': await action('key'); break;
    case 'o': await action('outside'); break;
    case 'c': {
      const here = S.objects.filter((o) => S.sel.has(o.id));
      await attrs({occluded: !(here.length && here.every((o) => o.occluded))}); break;
    }
    case '[': S.range = [S.frame, S.range ? Math.max(S.frame, S.range[1]) : S.frame]; panel(); drawTimeline(); break;
    case ']': S.range = [S.range ? Math.min(S.frame, S.range[0]) : S.frame, S.frame]; panel(); drawTimeline(); break;
    case '\\': S.range = null; panel(); drawTimeline(); break;
    case 'r': await action('reinterp'); break;
    case 'u': await action('unlock'); break;
    case 'l': await action('lock'); break;
    case 'delete': case 'backspace': e.preventDefault(); await action(e.shiftKey ? 'deltrack' : 'delbox'); break;
    case 's': {
      const modes = ['frame', 'drift', 'shift'], i = modes.indexOf(spread());
      document.querySelector(`input[name=spread][value=${modes[(i + 1) % 3]}]`).checked = true;
      toast('Correction: ' + modes[(i + 1) % 3]); break;
    }
    case 'h': S.hide = !S.hide; render(); break;
    case 'm': S.loupe = !S.loupe; render(); break;
    case '0': fit(); break;
    case '?': $('help').hidden = false; break;
  }
});

function togglePlay() {
  if (S.playing) { clearInterval(S.playing); S.playing = null; return; }
  S.playing = setInterval(async () => {
    if (S.frame >= S.meta.n_frames - 1) { togglePlay(); return; }
    await step(1);
  }, 100);
}

// --------------------------------------------------------------------------------------------------------------------
// Buttons and jobs
// --------------------------------------------------------------------------------------------------------------------
$('undo').onclick = () => op({op: 'undo'});
$('helpbtn').onclick = () => { $('help').hidden = false; };
$('helpclose').onclick = () => { $('help').hidden = true; };
$('export').onclick = async () => {
  const yolo = $('exportyolo').checked ? $('yolodir').value.trim() : '';
  if ($('exportyolo').checked && !yolo) { toast('Enter a folder for the YOLO labels', true); return; }
  const r = await op({op: 'export', yolo: yolo || undefined}, true);
  toast('Exported ' + r.cvat + (r.yolo ? ' and YOLO labels to ' + r.yolo : ''));
};
$('addlabel').onsubmit = async (e) => {
  e.preventDefault();
  const n = $('newlabel').value.trim();
  if (n) { await op({op: 'add_label', name: n}); $('newlabel').value = ''; S.drawLabel = n; renderLabels(); }
};
$('addtag').onsubmit = async (e) => {
  e.preventDefault();
  const n = $('newtag').value.trim();
  if (n) { await op({op: 'add_tag', name: n}); $('newtag').value = ''; panel(); }
};
document.querySelectorAll('input[name=spread]').forEach((r) => { r.onchange = () => r.blur(); });

async function job(body) {
  const r = await fetch('/api/job', {method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body)});
  const d = await r.json();
  if (!r.ok) { toast(d.error, true); return; }
  const poll = setInterval(async () => {
    const j = await get('/api/job');
    if (j.name) { $('job').textContent = `${j.name}: ${j.done}/${j.total || '?'}`; return; }
    clearInterval(poll); $('job').textContent = '';
    if (j.error) toast(j.error, true); else toast(j.message);
    await refresh();
  }, 500);
}
$('motion').onclick = () => job({job: 'motion'});
$('plgo').onclick = () => {
  const a = $('pla').value === '' ? (S.range ? S.range[0] : 0) : +$('pla').value;
  const b = $('plb').value === '' ? (S.range ? S.range[1] : S.meta.n_frames - 1) : +$('plb').value;
  job({job: 'prelabel', a, b, conf: +$('plconf').value, min_len: +$('plmin').value});
};

(async function init() {
  await refresh();
  fit();
  await goto(0);
  drawTimeline();
})();

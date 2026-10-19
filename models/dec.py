# Deep equilibrium canonicalizers (DECs) as model layers: yolov7-DEC architectures. See README.md, section 14.
#
# A DEC model is an ordinary model yaml with these layers (warp geometry and canonicalizer: utils/lse.py):
#   DECInput  [-1, 1, DECInput, []]                  layer 0: the input image, so that later layers can read it
#   DEC       [img, 1, DEC, [name, grid, max_zoom, canon_size, channels, iters, grad_iters]]
#             a canonicalizer: looks at image layer img, outputs its warps (one per region, see below)
#   DECWarp   [[x, dec], 1, DECWarp, []]             x (image or feature map) warped into DEC dec's canonical frame
#   DECUnwarp [[x, dec], 1, DECUnwarp, []]           x warped back from DEC dec's canonical frame to the image frame
#   DECScale  [[x, dec, ...], 1, DECScale, []]       x + a learned function (zero-initialised 1x1 conv) of the local
#                                                    scale of the listed DECs: tells a head what was rescaled
# A DEC that no DECUnwarp reads is a frame DEC: the predictions are made in its canonical frame, so the training labels
# are warped into it (by the DEC layer) and the predicted boxes are mapped back to the image (by the Model, in reverse
# layer order). The other DECs are closed: whatever they warp is warped back.
#
# The training settings (losses, warp augmentation, learning rate, ...) are the yaml's top-level `dec:` dict (keys of
# utils/lse.py DEFAULTS; the architecture keys there are ignored, the layers have their own). dec_yaml() writes such a
# yaml from a plain one and an LSE config (input, P1..P6, P1+..P6+, [first, last]): what train.py --lse does, and
#   python -m models.dec --cfg cfg/training/yolov7-tiny.yaml --lse data/lse/open.yaml --out yolov7-tiny-dec.yaml
#
# Per forward pass the DEC layers share a DECContext (set up by Model.forward_once): the regions (frame tiles of mosaic
# samples, else whole images), each DEC's warps, the labels in the current frame, the auxiliary losses and statistics,
# and cached sampling grids.

import argparse
import json
import logging
import math
import re
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.lse import (DEFAULTS, Canonicalizer, Regions, dec_name, load_lse_cfg, make_grid, random_phi, stage_spans,
                       unwarp_predictions, validate_spans, warp_stats, warp_targets)

ARCH_KEYS = ('grid', 'max_zoom', 'canon_size', 'channels', 'iters', 'grad_iters')  # per DEC layer, not in `dec:`


def train_cfg(d):
    """The `dec:` training settings of a model yaml, with defaults filled in."""
    d = dict(d or {})
    unknown = set(d) - set(DEFAULTS)
    assert not unknown, f'unknown dec: keys {sorted(unknown)}, expected {sorted(DEFAULTS)}'
    return {**DEFAULTS, **d}


class DECContext:
    """State shared by a model's DEC layers during one forward pass (not saved: a checkpoint keeps the settings)."""

    def __init__(self, cfg):
        self.cfg, self.epoch, self.keep = cfg, 0, False
        self.clear()

    def clear(self):
        self.base, self.regs, self.cache, self.seen, self.n = None, {}, {}, {}, 0
        self.targets, self.loss, self.stats, self.training = None, None, None, False
        self.deferred = None  # (frame DECs' regions, last warp first; image size) for map_detections()

    def __getstate__(self):  # pickling / deepcopy: settings only
        return {'cfg': self.cfg, 'epoch': self.epoch}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.keep = False
        self.clear()

    def begin(self, x, tiles=None, targets=None, training=False):
        """Start a forward pass: regions, warp augmentation (training), labels. Returns the (augmented) input."""
        cfg, n, (h, w) = self.cfg, x.shape[0], x.shape[-2:]
        self.clear()
        self.training, self.n = training, n
        self.base = Regions.make(n, tiles if (training and cfg['per_tile']) or self.keep else None, x.device)
        if training and cfg['warp_aug'] > 0:  # rescale some regions at random (labels follow)
            K = len(self.base.b)
            tx, ty = random_phi(K, cfg['grid'], cfg['warp_aug_strength'], x.device)
            on = (torch.rand(K, 1, 1, device=x.device) < cfg['warp_aug']).float()
            aug = self.base.with_grid(make_grid(tx * on, ty * on, cfg['max_zoom']))
            with torch.no_grad(), torch.autocast(device_type=x.device.type, enabled=False):
                x = aug.warp(x.float()).to(x.dtype)
                if targets is not None:
                    targets = warp_targets(targets, aug, h, w)
        if training:
            self.loss, self.stats = torch.zeros((), device=x.device), {'regions/img': len(self.base.b) / n}
            if targets is not None and len(targets):  # median object size the detector sees (for target_size)
                self.stats['obj_px'] = torch.sqrt(targets[:, 4].float() * w * targets[:, 5].float() * h).median()
        self.targets = targets
        return x

    def grid(self, dec, h, w, inverse):
        """Cached sampling grid of DEC layer dec for an h x w map: canonical -> image (inverse: also log magnification)."""
        key = (dec, inverse, h, w)
        if key not in self.cache:
            reg = self.regs[dec]
            self.cache[key] = reg.inverse_grid(self.n, h, w) if inverse else reg.sampling_grid(self.n, h, w,
                                                                                                torch.float32)
        return self.cache[key]


class _DECLayer(nn.Module):
    ctx = None  # the model's DECContext (set by Model)
    srcs = ()  # DEC layers read (absolute indices, set by Model)


class DECInput(nn.Identity):
    """Layer 0 of a DEC model: the input image, readable by later layers."""


class DEC(_DECLayer):
    """Deep equilibrium canonicalizer: from the image it reads, the warp grids (K, 2, R, R) of its regions."""

    def __init__(self, c1, name='dec', grid=8, max_zoom=4.0, canon_size=224, channels=(16, 32, 64), iters=4,
                 grad_iters=2):
        super().__init__()
        assert c1 == 3, f'DEC {name} must read an image (3 channels), got {c1} channels'
        self.name, self.frame = name, False  # frame: set by Model (no DECUnwarp reads this DEC)
        self.canon = Canonicalizer(grid, list(channels), max_zoom, canon_size, iters, grad_iters)

    def forward(self, x):
        ctx, c = self.ctx, self.canon
        with torch.autocast(device_type=x.device.type, enabled=False):
            xf = x.float()
            h, w = xf.shape[-2:]
            s = ctx.base.crops(xf, c.canon_size)
            G, phi_x, phi_y = c.solve(s)
            reg = ctx.base.with_grid(G)
            if ctx.training:
                loss, stats = self._aux(s, G, phi_x, phi_y, reg, ctx.targets, h, w)
                ctx.loss = ctx.loss + loss
                ctx.stats.update({f'{self.name}/{k}': v for k, v in stats.items()})
            if ctx.training and ctx.epoch < ctx.cfg['det_grad_epochs']:  # warm-up: no detection gradient yet
                reg = ctx.base.with_grid(G.detach())
            ctx.regs[self.i] = reg
            if ctx.keep:
                ctx.seen[self.i] = xf
            if self.frame and ctx.targets is not None:  # with gradients: content and its boxes move together
                ctx.targets = warp_targets(ctx.targets, reg, h, w)
        return reg.G

    def _aux(self, s, G, phi_x, phi_y, reg, targets, h, w):
        """Consistency (skipped, with its second solve, when eq_weight is 0), prior and size losses, and statistics
        (0-dim tensors: no host sync per step)."""
        cfg, c = self.ctx.cfg, self.canon
        loss, stats = torch.zeros((), device=s.device), {}
        if cfg['eq_weight'] > 0:  # the canonical frame of a randomly rescaled crop must be that of the crop
            if s.shape[0] > cfg['eq_regions']:  # a random subset of the regions: the second solve is the costly part
                idx = torch.randperm(s.shape[0], device=s.device)[:cfg['eq_regions']]
                s_eq, G_eq = s[idx], G[idx]
            else:
                s_eq, G_eq = s, G
            tx, ty = random_phi(s_eq.shape[0], c.grid, cfg['eq_strength'], s.device)
            s_t = c.warp_crops(s_eq, make_grid(tx, ty, c.max_zoom))
            G_t, _, _ = c.solve(s_t)
            eq = F.mse_loss(c.warp_crops(s_t, G_t), c.warp_crops(s_eq, G_eq))
            loss, stats['eq'] = cfg['eq_weight'] * eq, eq.detach()
        prior = (torch.tanh(phi_x) ** 2).mean() + (torch.tanh(phi_y) ** 2).mean()
        loss = loss + cfg['prior_weight'] * prior
        stats['prior'] = prior.detach()
        if targets is not None and len(targets):
            t = targets.float()
            z, m = reg.zoom_at(t[:, 0].long(), t[:, 2], t[:, 3])
            if m.any():
                size = torch.sqrt(t[m, 4] * w * t[m, 5] * h).clamp_min(1)
                stats['obj_px'] = (size * z.detach()).median()  # median object size in the canonical frame
                if cfg['size_weight'] > 0:
                    want = torch.log(cfg['target_size'] / size).clamp(-math.log(c.max_zoom), math.log(c.max_zoom))
                    size_loss = ((torch.log(z) - want) ** 2).mean()
                    loss = loss + cfg['size_weight'] * size_loss
                    stats['size'] = size_loss.detach()
        stats.update(warp_stats(G.detach()))
        return loss, stats


class DECWarp(_DECLayer):
    """[x, dec]: x warped into the canonical frame of DEC layer dec."""

    def forward(self, x):
        t = x[0]
        with torch.autocast(device_type=t.device.type, enabled=False):
            grid = self.ctx.grid(self.srcs[0], *t.shape[-2:], inverse=False)
            return F.grid_sample(t.float(), grid, mode='bilinear', padding_mode='border',
                                 align_corners=False).to(t.dtype)


class DECUnwarp(_DECLayer):
    """[x, dec]: x warped back from the canonical frame of DEC layer dec to the image frame."""

    def forward(self, x):
        t = x[0]
        with torch.autocast(device_type=t.device.type, enabled=False):
            grid, _ = self.ctx.grid(self.srcs[0], *t.shape[-2:], inverse=True)
            return F.grid_sample(t.float(), grid, mode='bilinear', padding_mode='border',
                                 align_corners=False).to(t.dtype)


class DECScale(_DECLayer):
    """[x, dec1, dec2, ...]: x + a 1x1 conv (zero-initialised: no effect at start) of the local log magnification
    (x, y) of each listed DEC at x's resolution."""

    def __init__(self, c1, n):
        super().__init__()
        self.conv = nn.Conv2d(2 * n, c1, 1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        t = x[0]
        h, w = t.shape[-2:]
        with torch.autocast(device_type=t.device.type, enabled=False):
            maps = torch.cat([self.ctx.grid(d, h, w, inverse=True)[1] for d in self.srcs], 1)
        return t + self.conv(maps.to(self.conv.weight.dtype)).to(t.dtype)


DEC_LAYERS = (DECInput, DEC, DECWarp, DECUnwarp, DECScale)


# ----------------------------------------------------------------------------------------------------------------------
# Model support (used by models/yolo.py Model)
# ----------------------------------------------------------------------------------------------------------------------
def setup(model):
    """Link a model's DEC layers: shared context, the DEC layers each layer reads, frame / closed DECs. Returns the
    context (None without DEC layers)."""
    layers = list(model.model)
    decs = [m for m in layers if isinstance(m, DEC)]
    if not decs:
        assert not any(isinstance(m, DEC_LAYERS[2:]) for m in layers), 'DECWarp / DECUnwarp / DECScale without a DEC'
        return None
    ctx = DECContext(train_cfg(model.yaml.get('dec')))
    unwarped = set()
    for m in layers:
        if isinstance(m, DEC_LAYERS[2:]):
            fs = [m.f] if isinstance(m.f, int) else list(m.f)
            srcs = [f if f >= 0 else m.i + f for f in fs[1:]]
            assert srcs and all(isinstance(layers[s], DEC) for s in srcs), \
                f'layer {m.i} ({m.type}): from must be [input, DEC layer(s)], got {m.f}'
            assert len(srcs) == 1 or isinstance(m, DECScale), f'layer {m.i} ({m.type}) reads one DEC, got {m.f}'
            m.srcs = srcs
            if isinstance(m, DECUnwarp):
                unwarped.add(srcs[0])
        if isinstance(m, _DECLayer):
            m.ctx = ctx
    for m in decs:
        m.frame = m.i not in unwarped
    return ctx


def finish(model, ctx, x, img_hw, return_extras, defer_map=False):
    """End a DEC model's forward pass: map decoded boxes back from the frame DECs' canonical frames (inference) and
    return the extras (targets in the frame of the predictions, auxiliary loss, statistics) if asked. defer_map: leave
    the decoded boxes in the canonical frame; NMS runs there and map_detections() maps the kept boxes afterwards."""
    frame = [m for m in model.model if isinstance(m, DEC) and m.frame]
    deferred = None
    if frame and not model.training and isinstance(x, tuple) and torch.is_tensor(x[0]) and x[0].dim() == 3:
        if defer_map:
            deferred = ([ctx.regs[m.i] for m in reversed(frame)], tuple(img_hw))
        else:
            z = x[0]
            for m in reversed(frame):  # canonical -> image pixels, the last warp first
                z = unwarp_predictions(z, ctx.regs[m.i], *img_hw, ctx.cfg['min_conf'])
            x = (z,) + tuple(x[1:])
    out = (x, ctx.targets, ctx.loss, ctx.stats) if return_extras else x
    if not ctx.keep:
        ctx.clear()
    ctx.deferred = deferred
    return out


def map_detections(model, dets):
    """Detections of a forward pass with lse_defer_map=True, after NMS (non_max_suppression's list of (n, 6) [xyxy, conf,
    cls] per image, canonical-frame pixels) -> image pixels. NMS in the canonical frame keeps its decisions independent
    of the warp (a non-uniform warp changes IoUs), and only the kept boxes are mapped. A no-op for other models / passes.
    """
    ctx = getattr(model, 'dec_ctx', None)
    if ctx is None or ctx.deferred is None:
        return dets
    regs, (h, w) = ctx.deferred
    ctx.deferred = None
    n = [len(d) for d in dets]
    if not sum(n):
        return dets
    d = torch.cat(dets)
    img = torch.cat([torch.full((k,), i, device=d.device, dtype=torch.long) for i, k in enumerate(n)])
    scale = torch.tensor([w, h, w, h], device=d.device, dtype=torch.float32)
    b = d[:, :4].float() / scale
    for reg in regs:  # canonical -> image pixels, the last warp first
        b = reg.map_boxes(img, b, to_canonical=False)
    d = torch.cat([(b * scale).to(d.dtype), d[:, 4:]], 1)
    return list(d.split(n))


def input_dec(model):
    """The DEC layer whose canonical frame the detector input is (a DECWarp of the DECInput layer reads it), or None."""
    for m in model.model:
        if isinstance(m, DECWarp) and m.f[0] in (0, -m.i) and isinstance(model.model[0], DECInput):
            return model.model[m.srcs[0]]
    return None


@torch.no_grad()
def preview(model, imgs, targets, tiles=None):
    """What the detector sees for a training batch (input DEC, without warp augmentation) and its targets, for plots;
    None without an input DEC."""
    m = input_dec(model)
    if m is None:
        return None
    ctx, training = model.dec_ctx, model.training
    model.eval()
    ctx.keep = True
    try:
        x = imgs.float()
        model(x, lse_tiles=tiles if ctx.cfg['per_tile'] else None)
        reg = ctx.regs[m.i]
        return reg.warp(x), warp_targets(targets, reg, *x.shape[-2:])
    finally:
        ctx.keep = False
        ctx.clear()
        model.train(training)


def is_dec_model(model):
    return any(isinstance(m, DEC) for m in model.model)


def dec_parameters(model):
    """Parameters of the DEC layers (canonicalizers and head scale convs): their own optimizer group."""
    return [p for m in model.model if isinstance(m, _DECLayer) for p in m.parameters()]


def map_plain_weights(state_dict, src_model, model):
    """Weights of a plain checkpoint renamed for a DEC model built from the same architecture: the model's non-DEC
    layers, in order, are the plain model's layers."""
    if is_dec_model(src_model) or not is_dec_model(model):
        return state_dict
    plain = [m.i for m in model.model if not isinstance(m, DEC_LAYERS)]
    if len(plain) != len(src_model.model):
        logging.getLogger(__name__).warning(f'DEC model: {len(plain)} non-DEC layers, the checkpoint has '
                                            f'{len(src_model.model)}: weights matched by position may not fit')
    out = {}
    for k, v in state_dict.items():
        p = k.split('.')
        if p[0] == 'model' and p[1].isdigit() and int(p[1]) < len(plain):
            p[1] = str(plain[int(p[1])])
            out['.'.join(p)] = v
    return out


# ----------------------------------------------------------------------------------------------------------------------
# yaml generator: a plain model yaml + an LSE config -> a DEC model yaml
# ----------------------------------------------------------------------------------------------------------------------
def dec_yaml(base, lse_cfg, notes=None):
    """DEC model yaml (dict) from a plain model yaml (path or dict) and an LSE config (path or dict; README.md section
    14). Stage spans are found from the built plain model. notes: if a list, filled with a comment per layer."""
    from models.yolo import Model, Detect, IDetect
    if isinstance(base, str):
        import yaml
        with open(base) as f:
            base = yaml.safe_load(f)
    base = deepcopy(base)
    cfg = load_lse_cfg(lse_cfg)
    layers = base['backbone'] + base['head']
    assert not any(m in [c.__name__ for c in DEC_LAYERS] for _, _, m, _ in layers), \
        'this model yaml already has DEC layers: edit it (or its dec: settings) instead of adding --lse'
    names = [dec_name(d) for d in cfg['decs']]
    spans = {}
    if any(n != 'input' for n in names):
        log = logging.getLogger('models.yolo')
        level = log.level
        log.setLevel(logging.WARNING)  # the plain model is only built to find the stages
        try:
            plain = Model(deepcopy(base))
        finally:
            log.setLevel(level)
        stages = stage_spans(plain)
        for d, n in zip(cfg['decs'], names):
            if n == 'input':
                continue
            if isinstance(d, str):
                k = d.rstrip('+')
                assert k in stages, f'LSE dec {d}: this model has backbone stages {sorted(stages)}'
                spans[n] = [stages[k][0], None] if d.endswith('+') else stages[k]
            else:
                spans[n] = d
        layer_from = [[m.f] if isinstance(m.f, int) else list(m.f) for m in plain.model]
        validate_spans([v for v in spans.values() if v[1] is not None], layer_from)
        closed = [n for n in spans if spans[n][1] is not None]
        assert not closed or isinstance(plain.model[-1], (Detect, IDetect)), 'closed stage DECs need Detect / IDetect'
    closed = [n for n in names if n in spans and spans[n][1] is not None]

    n_layers, nb = len(layers), len(base['backbone'])
    out = []

    notes = notes if notes is not None else []

    def add(f, m, args, r=1, note=''):
        out.append([f, r, m, args])
        notes.append(note)
        return len(out) - 1

    arch = [cfg[k] for k in ARCH_KEYS]
    img = add(-1, 'DECInput', [], note='the input image')
    dec, det_in = {}, img
    if 'input' in names:
        dec['input'] = add(img, 'DEC', ['input', *arch], note='canonicalizer input: warps of the image')
        det_in = add([det_in, dec['input']], 'DECWarp', [], note='the canonical image: the detector input')
    for n in names:
        if n != 'input':
            dec[n] = add(det_in, 'DEC', [n, *arch], note=f'canonicalizer {n}: looks at the detector input')
    span_of = {}
    for n, (a, b) in spans.items():
        for i in range(a, (b if b is not None else n_layers - 1) + 1):
            span_of[i] = n
    new, warped, end_backbone = {}, {}, None
    for i, (f, r, m, args) in enumerate(layers):
        d = span_of.get(i)
        src = []
        for fj in ([f] if isinstance(f, int) else list(f)):
            j = i - 1 if fj == -1 else (fj if fj >= 0 else i + fj)
            s = det_in if j < 0 else new[j]
            if d is not None and j < spans[d][0]:  # entering DEC d's span: warp the input once per (DEC, source)
                if (d, s) not in warped:
                    warped[(d, s)] = add([s, dec[d]], 'DECWarp', [], note=f'into {d}: layer {s} warped')
                s = warped[(d, s)]
            src.append(s)
        if i == n_layers - 1 and closed:  # detection head: told the local scales of the closed stages
            src = [add([s] + [dec[n] for n in closed], 'DECScale', [], note=f'head input {s} + local scale of {closed}')
                   for s in src]
        k = add(src[0] if isinstance(f, int) else src, m, args, r,
                note=f'plain layer {i}' + (f' (in {d})' if d is not None else ''))
        if d is not None and spans[d][1] == i:  # end of a closed stage: back to the image frame
            k = add([k, dec[d]], 'DECUnwarp', [], note=f'end of {d}: layer {k} warped back')
        new[i] = k
        if i == nb - 1:
            end_backbone = len(out)
    y = {k: v for k, v in base.items() if k not in ('backbone', 'head', 'ch')}
    y['dec'] = {k: v for k, v in cfg.items() if k not in ARCH_KEYS}
    y['backbone'], y['head'] = out[:end_backbone], out[end_backbone:]
    return y


def write_yaml(y, path, header='', notes=None):
    """A model yaml with one layer per line (flow style, as the yolov7 cfgs), each with its index and note."""
    plain = re.compile(r'^([A-Za-z_][\w.]*(\([\d.]*\))?|P\d\+?)$')  # module names, None, nn.X(0.1), P3+: unquoted

    def line(v):
        if isinstance(v, list):
            return '[' + ', '.join(line(x) for x in v) + ']'
        if v is None:
            return 'None'
        if isinstance(v, str):
            return v if plain.match(v) and v not in ('true', 'false', 'null', 'yes', 'no') else repr(v)
        return json.dumps(v)

    with open(path, 'w') as f:
        f.write(header)
        for k, v in y.items():
            if k in ('backbone', 'head'):
                f.write(f'\n{k}:\n')
                for i, layer in enumerate(v):
                    j = i + (len(y['backbone']) if k == 'head' else 0)
                    text = ('  [' if i == 0 else '   ') + line(layer) + (']' if i == len(v) - 1 else ',')
                    note = notes[j] if notes and j < len(notes) else ''
                    f.write(f'{text}  # {j}{": " + note if note else ""}\n')
            elif k == 'dec':
                f.write('\ndec:  # training settings of the DEC layers (utils/lse.py DEFAULTS, data/lse/input.yaml)\n')
                for kk, vv in v.items():
                    f.write(f'  {kk}: {json.dumps(vv)}\n')
            else:
                f.write(f'{k}: {json.dumps(v)}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='write a DEC model yaml from a plain model yaml and an LSE config')
    parser.add_argument('--cfg', required=True, help='plain model yaml, e.g. cfg/training/yolov7-tiny.yaml')
    parser.add_argument('--lse', required=True, help='LSE config, e.g. data/lse/input.yaml')
    parser.add_argument('--out', required=True, help='output model yaml')
    opt = parser.parse_args()
    notes = []
    y = dec_yaml(opt.cfg, opt.lse, notes)
    write_yaml(y, opt.out, f'# DEC model (README.md, section 14): {opt.cfg} with deep equilibrium canonicalizers from '
                           f'{opt.lse}\n# (python -m models.dec --cfg {opt.cfg} --lse {opt.lse} --out {opt.out})\n\n',
               notes)
    print(f'wrote {opt.out}')

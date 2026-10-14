# Local scale equivariance (LSE) for YOLOv7, after "Local Scale Equivariance with Latent Deep Equilibrium
# Canonicalizer" (Rahman et al., ICCV 2025).
#
# A small canonicalizer network picks, per image, a smooth monotone warp that rescales parts of the image differently
# (e.g. magnifies distant, small objects). Two modes (README.md, section 14):
#   input:  the detector runs on the warped ("canonical") image; training labels are warped into it and predicted
#           boxes are mapped back exactly. The detector itself is unchanged.
#   latent: the image is not warped; feature maps are warped around whole backbone stages (`spans`) and un-warped
#           before they leave the stage, and the head is told the local scale (zero-initialised 1x1 convs), so box
#           sizes stay in image pixels.
#
# Warp family. Coordinates are normalised to [0, 1] on pixel centres (pixel i of n is at i / (n - 1), as
# grid_sample(align_corners=True)). A warp is a coarse grid G (B, 2, R, R) of monotone knots: along each column c,
# G[:, 1, :, c] are increasing y knots (0 to 1), along each input row r, G[:, 0, r, :] increasing x knots; cell widths
# are softmax of bounded parameters. A canonical point (u, v) samples the input point
#     y = gy(u, v)  (y knots interpolated across columns at u, then along v)
#     x = gx(u, y)  (x knots interpolated across rows at the input y, then along u)
# Evaluating gx at the input y (rather than at v) makes the Jacobian determinant gx_u * gy_v > 0: the warp never folds,
# and its inverse is two exact 1-D inversions. A cell of width w (identity: 1 / (R - 1)) magnifies by 1 / (w (R - 1)).
# Parameters phi_x, phi_y have shape (B, R, R - 1); phi = 0 is the identity.
#
# Canonicalizer. A fixed point of phi <- phi + delta(warp(image, phi)): delta only sees the canonical image, so if
# phi* canonicalizes x, the parameters that map a rescaled x to the same canonical image are a fixed point too
# (equivariance by construction, up to the warp family being closed under composition). Trained by the detection loss
# through the warp, a consistency loss (canonical image of a randomly warped input = canonical image of the input) and a
# small prior towards the identity. The output layer starts at zero: at initialisation the model is exactly the
# detector without LSE.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULTS = dict(
    mode='input',        # input | latent
    grid=8,              # knots per axis of the coarse warp grid (R): R - 1 cells per axis
    max_zoom=4.0,        # largest ratio between the most magnified and the most shrunk cell of a row / column
    canon_size=320,      # long side of the (downscaled) image the canonicalizer looks at
    channels=[16, 32, 64],  # canonicalizer conv widths (stride 2 each)
    iters=4,             # fixed-point iterations
    grad_iters=2,        # of which the last grad_iters are backpropagated (truncated fixed-point gradient)
    eq_weight=1.0,       # consistency loss weight
    eq_strength=1.0,     # strength of the random warps used by the consistency loss (in units of phi)
    prior_weight=0.01,   # pull of phi towards the identity
    spans=[],            # latent mode: [[first, last], ...] layer indices of the warped stages
    min_conf=1e-4,       # input mode inference: only boxes with objectness above this are mapped back
)


def load_lse_cfg(cfg):
    """LSE config dict from a yaml path or dict, with defaults filled in and checked."""
    if isinstance(cfg, str):
        import yaml
        with open(cfg) as f:
            cfg = yaml.safe_load(f) or {}
    unknown = set(cfg) - set(DEFAULTS)
    assert not unknown, f'unknown LSE config keys {sorted(unknown)}, expected {sorted(DEFAULTS)}'
    cfg = {**DEFAULTS, **cfg}
    assert cfg['mode'] in ('input', 'latent'), f"LSE mode must be input or latent, got {cfg['mode']!r}"
    assert cfg['grid'] >= 2 and cfg['max_zoom'] >= 1 and 0 < cfg['grad_iters'] <= cfg['iters'], cfg
    assert cfg['mode'] == 'input' or cfg['spans'], 'LSE latent mode needs spans: [[first, last], ...]'
    cfg['spans'] = [list(map(int, s)) for s in cfg['spans']]
    return cfg


# ----------------------------------------------------------------------------------------------------------------------
# Warp geometry
# ----------------------------------------------------------------------------------------------------------------------
def make_grid(phi_x, phi_y, max_zoom):
    """Coarse warp grid (B, 2, R, R) from parameters (B, R, R - 1). Cell widths are softmax(a tanh(phi)), so the ratio
    between the widest and narrowest cell of a row (column) is at most exp(2a) = max_zoom."""
    a = 0.5 * math.log(max_zoom)

    def knots(phi):  # (B, R, R - 1) -> (B, R, R): 0, cumulative widths, last exactly 1
        w = torch.softmax(a * torch.tanh(phi), dim=-1)
        k = torch.cumsum(w, dim=-1)
        return torch.cat([torch.zeros_like(k[..., :1]), k[..., :-1], torch.ones_like(k[..., :1])], dim=-1)

    gx = knots(phi_x)  # [b, r, c]: x of knot c in input row r
    gy = knots(phi_y).transpose(1, 2)  # phi_y[b, c, r'] -> [b, r, c]: y of knot r in column c
    return torch.stack([gx, gy], dim=1)


def identity_phi(n, grid, device, dtype=torch.float32):
    return torch.zeros(n, grid, grid - 1, device=device, dtype=dtype), \
        torch.zeros(n, grid, grid - 1, device=device, dtype=dtype)


def dense_grid(G, h, w):
    """grid_sample grid (B, h, w, 2) in [-1, 1]: for each canonical pixel, the input location it samples."""
    R = G.shape[-1]
    gy = F.interpolate(G[:, 1:], size=(h, w), mode='bilinear', align_corners=True)[:, 0]  # y = gy(u, v)
    rows = F.interpolate(G[:, :1], size=(R, w), mode='bilinear', align_corners=True)[:, 0]  # x knots of each row, at u
    f = gy * (R - 1)
    r0 = f.detach().floor().clamp(0, R - 2).long()
    t = f - r0
    gx = rows.gather(1, r0) * (1 - t) + rows.gather(1, r0 + 1) * t  # x = gx(u, y): rows interpolated at the input y
    return torch.stack([gx, gy], -1) * 2 - 1


def warp(x, G):
    """Canonical image / feature map: x sampled at g(u, v). x: (B, C, h, w); G: (B, 2, R, R)."""
    grid = dense_grid(G.float(), *x.shape[-2:]).to(x.dtype)
    return F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=True)


def _bilinear_rows(G, ch, b, pos):
    """G[b, ch] interpolated along rows (dim 2) at normalised pos: (P, R) knot values."""
    R = G.shape[-1]
    f = pos.clamp(0, 1) * (R - 1)
    i0 = f.floor().clamp(max=R - 2).long()
    t = (f - i0).unsqueeze(-1)
    return G[b, ch, i0] * (1 - t) + G[b, ch, i0 + 1] * t


def _bilinear_cols(G, ch, b, pos):
    """G[b, ch] interpolated along columns (dim 3) at normalised pos: (P, R) knot values (one per row)."""
    R = G.shape[-1]
    f = pos.clamp(0, 1) * (R - 1)
    i0 = f.floor().clamp(max=R - 2).long()
    t = (f - i0).unsqueeze(-1)
    return G[b, ch, :, i0] * (1 - t) + G[b, ch, :, i0 + 1] * t


def _interp_1d(knots, q):
    """Piecewise-linear function through (i / (R - 1), knots[:, i]) evaluated at q: values and slopes. Outside [0, 1]
    the end segments are extended (boxes may extend past the image)."""
    R = knots.shape[-1]
    f = q * (R - 1)
    i0 = f.floor().clamp(0, R - 2).long().unsqueeze(-1)
    k0, k1 = knots.gather(1, i0).squeeze(1), knots.gather(1, i0 + 1).squeeze(1)
    t = f - i0.squeeze(1)
    return k0 + (k1 - k0) * t, (k1 - k0) * (R - 1)


def _invert_1d(knots, x):
    """Inverse of the increasing piecewise-linear function through (i / (R - 1), knots[:, i]): q with f(q) = x, and
    the slope df/dq there. Outside [0, 1] the end segments are extended."""
    R = knots.shape[-1]
    idx = torch.searchsorted(knots.contiguous(), x.unsqueeze(-1).contiguous()).clamp(1, R - 1)
    k0, k1 = knots.gather(1, idx - 1).squeeze(1), knots.gather(1, idx).squeeze(1)
    d = (k1 - k0).clamp_min(1e-9)
    q = ((idx.squeeze(1) - 1) + (x - k0) / d) / (R - 1)
    return q, d * (R - 1)


def forward_points(G, b, u, v):
    """Canonical points (u, v) of images b -> input points (x, y). All (P,) tensors, normalised coordinates."""
    y, _ = _interp_1d(_bilinear_cols(G, 1, b, u), v)  # y knots interpolated across columns at u, along v
    x, _ = _interp_1d(_bilinear_rows(G, 0, b, y), u)  # x knots interpolated across rows at the input y, along u
    return x, y


def inverse_points(G, b, x, y):
    """Input points (x, y) of images b -> canonical points (u, v): exact, x = gx(u, y) is inverted along the input
    row y, then y = gy(u, v) along the column u. Also returns the local slopes dgx/du, dgy/dv (1 = no scaling,
    < 1 = magnified)."""
    u, sx = _invert_1d(_bilinear_rows(G, 0, b, y), x)
    v, sy = _invert_1d(_bilinear_cols(G, 1, b, u), y)
    return u, v, sx, sy


def _lattice(n, h, w, device):
    v, u = torch.meshgrid(torch.linspace(0, 1, h, device=device), torch.linspace(0, 1, w, device=device),
                          indexing='ij')
    b = torch.arange(n, device=device).view(n, 1, 1).expand(n, h, w).reshape(-1)
    return b, u.reshape(1, -1).expand(n, -1).reshape(-1), v.reshape(1, -1).expand(n, -1).reshape(-1)


def inverse_grid(G, h, w):
    """grid_sample grid (B, h, w, 2) un-warping a canonical map back to the input frame, and the log magnification
    (B, 2, h, w) at each input location (x, y; > 0 where the canonical frame magnifies)."""
    n = G.shape[0]
    b, x, y = _lattice(n, h, w, G.device)
    u, v, sx, sy = inverse_points(G, b, x, y)
    grid = torch.stack([u, v], -1).view(n, h, w, 2) * 2 - 1
    logm = -torch.log(torch.stack([sx, sy], 0).clamp_min(1e-6)).view(2, n, h, w).transpose(0, 1)
    return grid, logm


def map_boxes(G, b, xyxy, w, h, to_canonical):
    """Boxes (N, 4) in pixel-edge coordinates of a w x h image -> boxes in the canonical frame (to_canonical) or back
    in the input frame. Each side is placed where the warp moves that side's midpoint (left / right midpoints give x1 /
    x2, top / bottom midpoints y1 / y2). Unlike the bounding box of the warped outline, this does not grow a box at
    every mapping: mapping a box to the canonical frame and back returns it (up to the warp's curvature)."""
    x1, y1, x2, y2 = xyxy.unbind(-1)
    xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
    px = torch.stack([x1, x2, xm, xm], -1)  # left, right, top, bottom midpoints
    py = torch.stack([ym, ym, y1, y2], -1)
    px = ((px - 0.5) / max(w - 1, 1)).reshape(-1)  # to pixel-centre normalised coordinates
    py = ((py - 0.5) / max(h - 1, 1)).reshape(-1)
    bb = b.view(-1, 1).expand(-1, 4).reshape(-1)
    if to_canonical:
        qx, qy, _, _ = inverse_points(G, bb, px, py)
    else:
        qx, qy = forward_points(G, bb, px, py)
    qx = (qx * (w - 1) + 0.5).view(-1, 4)
    qy = (qy * (h - 1) + 0.5).view(-1, 4)
    return torch.stack([torch.minimum(qx[:, 0], qx[:, 1]), torch.minimum(qy[:, 2], qy[:, 3]),
                        torch.maximum(qx[:, 0], qx[:, 1]), torch.maximum(qy[:, 2], qy[:, 3])], -1)


def warp_targets(targets, G, h, w, min_px=2.0):
    """YOLO targets (n, 6) [image, class, x, y, w, h] normalised to a w x h image -> the same boxes in the canonical
    frame (boxes smaller than min_px on a side are dropped, as YOLOv7's augmentation does)."""
    if not len(targets):
        return targets
    t = targets.float()
    xy, wh = t[:, 2:4] * t.new_tensor([w, h]), t[:, 4:6] * t.new_tensor([w, h])
    xyxy = torch.cat([xy - wh / 2, xy + wh / 2], -1)
    q = map_boxes(G.float(), t[:, 0].long(), xyxy, w, h, to_canonical=True)
    q[:, 0::2], q[:, 1::2] = q[:, 0::2].clamp(0, w), q[:, 1::2].clamp(0, h)
    out = t.clone()
    out[:, 2:4] = (q[:, :2] + q[:, 2:]) / 2 / t.new_tensor([w, h])
    out[:, 4:6] = (q[:, 2:] - q[:, :2]) / t.new_tensor([w, h])
    keep = ((q[:, 2] - q[:, 0]) >= min_px) & ((q[:, 3] - q[:, 1]) >= min_px)
    return out[keep].to(targets.dtype)


def unwarp_predictions(z, G, h, w, min_conf=1e-4):
    """Decoded predictions (B, N, no) [x, y, w, h, obj, ...] in canonical pixels -> input pixels. Only boxes with
    objectness above min_conf are mapped (NMS discards the others at any usual threshold)."""
    z = z.clone()
    keep = z[..., 4] > min_conf
    if keep.any():
        b = torch.nonzero(keep)[:, 0]
        p = z[keep][:, :4].float()
        xyxy = torch.cat([p[:, :2] - p[:, 2:] / 2, p[:, :2] + p[:, 2:] / 2], -1)
        q = map_boxes(G.float(), b, xyxy, w, h, to_canonical=False)
        z[keep] = torch.cat([((q[:, :2] + q[:, 2:]) / 2), q[:, 2:] - q[:, :2], z[keep][:, 4:].float()], -1).to(z.dtype)
    return z


def warp_stats(G):
    """Mean displacement from the identity (fraction of the image) and the strongest local magnification and
    shrinking of a batch of warps."""
    R = G.shape[-1]
    ident = torch.linspace(0, 1, R, device=G.device)
    disp = torch.cat([(G[:, 0] - ident.view(1, 1, R)).abs().flatten(), (G[:, 1] - ident.view(1, R, 1)).abs().flatten()])
    sx = (G[:, 0, :, 1:] - G[:, 0, :, :-1]) * (R - 1)  # slopes: < 1 magnifies
    sy = (G[:, 1, 1:, :] - G[:, 1, :-1, :]) * (R - 1)
    s = torch.cat([sx.flatten(), sy.flatten()])
    return {'disp': disp.mean().item(), 'zoom_in': (1 / s.min()).item(), 'zoom_out': s.max().item()}


# ----------------------------------------------------------------------------------------------------------------------
# Canonicalizer and the LSE module attached to a Model
# ----------------------------------------------------------------------------------------------------------------------
class Canonicalizer(nn.Module):
    def __init__(self, grid=8, channels=(16, 32, 64), max_zoom=4.0, canon_size=320, iters=4, grad_iters=2):
        super().__init__()
        self.grid, self.max_zoom, self.canon_size = grid, max_zoom, canon_size
        self.iters, self.grad_iters = iters, grad_iters
        layers, c_in = [], 3
        for c in channels:
            layers += [nn.Conv2d(c_in, c, 3, 2, 1), nn.GroupNorm(min(8, c), c), nn.SiLU()]
            c_in = c
        self.features = nn.Sequential(*layers)
        self.head = nn.Conv2d(c_in, 2, 1)
        nn.init.zeros_(self.head.weight)  # delta = 0: identity warp, the model starts as the plain detector
        nn.init.zeros_(self.head.bias)

    def small(self, x):
        """Downscaled copy of the input images for the canonicalizer (never upscaled)."""
        r = min(1.0, self.canon_size / max(x.shape[-2:]))
        if r < 1:
            x = F.interpolate(x, scale_factor=r, mode='bilinear', align_corners=False, recompute_scale_factor=False)
        return x

    def delta(self, c):
        R = self.grid
        f = self.head(self.features(c.to(self.head.weight.dtype))).float()
        dx = F.adaptive_avg_pool2d(f[:, :1], (R, R - 1))[:, 0]  # [b, row knot, x cell]
        dy = F.adaptive_avg_pool2d(f[:, 1:], (R - 1, R))[:, 0].transpose(1, 2)  # [b, column knot, y cell]
        # a constant added to a row's (column's) parameters does not change the warp (softmax): remove it, so phi does
        # not drift into the flat part of tanh, where the canonicalizer would lose its gradient
        return dx - dx.mean(-1, keepdim=True), dy - dy.mean(-1, keepdim=True)

    def solve(self, s):
        """Fixed point phi = phi + delta(warp(s, phi)) for downscaled images s: returns the warp grid and the canonical
        images. Only the last grad_iters iterations are backpropagated."""
        phi_x, phi_y = identity_phi(s.shape[0], self.grid, s.device)
        grad = torch.is_grad_enabled()
        for k in range(self.iters):
            track = grad and k >= self.iters - self.grad_iters
            if track and k == self.iters - self.grad_iters:
                phi_x, phi_y = phi_x.detach(), phi_y.detach()
            with torch.set_grad_enabled(track):
                dx, dy = self.delta(warp(s, make_grid(phi_x, phi_y, self.max_zoom)))
                phi_x, phi_y = phi_x + dx, phi_y + dy
        G = make_grid(phi_x, phi_y, self.max_zoom)
        return G, phi_x, phi_y


class LSE(nn.Module):
    """Local scale equivariance attached to a Model as model.lse (see Model.attach_lse)."""

    def __init__(self, cfg, head_channels=(), layer_from=None):
        super().__init__()
        self.cfg = cfg = load_lse_cfg(cfg)
        self.mode = cfg['mode']
        self.canon = Canonicalizer(cfg['grid'], cfg['channels'], cfg['max_zoom'], cfg['canon_size'], cfg['iters'],
                                   cfg['grad_iters'])
        self.starts = {a: i for i, (a, _) in enumerate(cfg['spans'])} if self.mode == 'latent' else {}
        self.ends = {b: i for i, (_, b) in enumerate(cfg['spans'])} if self.mode == 'latent' else {}
        if self.mode == 'latent':
            validate_spans(cfg['spans'], layer_from)
            self.cond = nn.ModuleList(nn.Conv2d(2, c, 1) for c in head_channels)  # local scale -> head features
            for m in self.cond:
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

    def canonicalize(self, x):
        """Warp grids for images x (B, 3, H, W), and in training mode the auxiliary loss (consistency + prior) and
        statistics. Runs in fp32 outside autocast."""
        with torch.autocast(device_type=x.device.type, enabled=False):
            s = self.canon.small(x.float())
            G, phi_x, phi_y = self.canon.solve(s)
            if not self.training:
                return G, None, None
            cfg, n, R = self.cfg, x.shape[0], self.canon.grid
            # consistency: the canonical image of a randomly rescaled input must equal that of the input
            tx = (torch.rand(n, R, R - 1, device=x.device) * 2 - 1) * cfg['eq_strength']
            ty = (torch.rand(n, R, R - 1, device=x.device) * 2 - 1) * cfg['eq_strength']
            s_t = warp(s, make_grid(tx, ty, self.canon.max_zoom))
            G_t, _, _ = self.canon.solve(s_t)
            eq = F.mse_loss(warp(s_t, G_t), warp(s, G))
            prior = (torch.tanh(phi_x) ** 2).mean() + (torch.tanh(phi_y) ** 2).mean()
            loss = cfg['eq_weight'] * eq + cfg['prior_weight'] * prior
            stats = {'eq': eq.item(), 'prior': prior.item(), **warp_stats(G.detach())}
            return G, loss, stats

    def warp_feat(self, x, G):
        return [warp(t, G) for t in x] if isinstance(x, list) else warp(x, G)

    def unwarp_feat(self, x, G, cache):
        def one(t):
            key = tuple(t.shape[-2:])
            if key not in cache:
                cache[key] = inverse_grid(G.float(), *key)
            return F.grid_sample(t, cache[key][0].to(t.dtype), mode='bilinear', padding_mode='border',
                                 align_corners=True)
        return [one(t) for t in x] if isinstance(x, list) else one(x)

    def condition(self, x, G, cache):
        """Head inputs (list of (B, C_i, h_i, w_i)) + a learned function of the local log magnification."""
        out = []
        for t, conv in zip(x, self.cond):
            key = tuple(t.shape[-2:])
            if key not in cache:
                cache[key] = inverse_grid(G.float(), *key)
            out.append(t + conv(cache[key][1].to(conv.weight.dtype)).to(t.dtype))
        return out + list(x[len(self.cond):])

    @torch.no_grad()
    def preview(self, imgs, targets):
        """Canonical images and targets of a batch (input mode), for plots."""
        G, _, _ = self.canonicalize_eval(imgs)
        return warp(imgs.float(), G), warp_targets(targets, G, imgs.shape[2], imgs.shape[3])

    def canonicalize_eval(self, x):
        training = self.training
        self.train(False)
        try:
            return self.canonicalize(x)
        finally:
            self.train(training)


def validate_spans(spans, layer_from):
    """Latent spans [first, last] must be self-contained: inside a span, layers only read the span's input or
    earlier layers of the span; after it, no layer reads a layer of the span other than its last."""
    assert layer_from is not None, 'latent LSE needs the model layer graph'
    n = len(layer_from)
    covered = set()
    for a, b in spans:
        assert 0 <= a <= b < n - 1, f'LSE span {[a, b]}: layers must be in [0, {n - 2}] (not the detection head)'
        assert not covered & set(range(a, b + 1)), f'LSE span {[a, b]} overlaps another span'
        covered |= set(range(a, b + 1))
        for j in range(n):
            srcs = [j - 1 if f == -1 else (f if f >= 0 else j + f) for f in layer_from[j]]
            if a < j <= b:
                bad = [s for s in srcs if not a <= s < j]
                assert not bad, (f'LSE span {[a, b]}: layer {j} reads layer(s) {bad} outside the span (a span may '
                                 f'only read its input at layer {a})')
            elif j > b:
                bad = [s for s in srcs if a <= s < b]
                assert not bad, (f'LSE span {[a, b]}: layer {j} reads layer(s) {bad} inside the span (only the '
                                 f'span output, layer {b}, may be read after it)')
            elif j == a:
                pass  # the span input: anything

# Local scale equivariance (LSE) for YOLOv7, after "Local Scale Equivariance with Latent Deep Equilibrium
# Canonicalizer" (Rahman et al., ICCV 2025). See README.md, section 14. This file: the LSE config, the warp geometry,
# the canonicalizer network and plots; the model layers that use them (DEC models) are in models/dec.py.
#
# Deep equilibrium canonicalizers (DECs) pick, per frame, a smooth monotone warp that rescales parts of the frame
# differently, so objects whose apparent size varies (e.g. with distance) are seen at a more uniform, "canonical" scale.
# Each DEC has its own canonicalizer network; `decs` lists where models/dec.py dec_yaml() inserts them:
#   input:  the detector runs on the warped (canonical) image; labels are warped into it for training and predicted
#           boxes are mapped back. The detector itself is unchanged.
#   P1..P6: the backbone stage starting at the stride-2^k downsampling layer runs in a canonical frame: its inputs are
#           warped, its output is warped back before it is used. The head is told every stage's local scale
#           (zero-initialised 1x1 convs), so boxes stay in image pixels.
#   P1+..P6+: from the start of that stage on, everything (later stages, neck, head) runs in the canonical frame: inputs
#           are warped on entry and never warped back; labels are warped in and predicted boxes mapped back, as for
#           input (exact box sizes; the stem before the stage sees the image). At most one, combinable with input.
#   [a, b]: the same as Pk for an explicit span of layers a..b.
# Each latent DEC looks at the detector's input image (after the input DEC, if any), as in the paper.

# Regions. A training sample can be a mosaic of several frames: each frame rectangle (tracked by the data loader, see
# LoadImagesAndLabels._tiles) is a region with its own warp, computed from a crop of that frame alone, and applied
# inside its rectangle only (a warp maps its rectangle onto itself, so neighbouring regions still meet at the seams).
# Without tiles (inference, or per_tile: false) each image is one region.
#
# Warp family. Region coordinates are in [0, 1], continuous (0 = left / top edge of the region, 1 = right / bottom).
# A warp is a coarse grid G (2, R, R) of monotone knots: along each column c, G[1, :, c] are increasing y knots (0 to 1),
# along each input row r, G[0, r, :] increasing x knots; cell widths are softmax of bounded parameters. A canonical point
# (u, v) samples the input point
#     y = gy(u, v)  (y knots interpolated across columns at u, then along v)
#     x = gx(u, y)  (x knots interpolated across rows at the input y, then along u)
# Evaluating gx at the input y makes the Jacobian determinant gx_u * gy_v > 0: the warp never folds, and its inverse is
# two exact 1-D inversions. A cell of width w (identity: 1 / (R - 1)) magnifies by 1 / (w (R - 1)). Parameters
# phi_x, phi_y have shape (R, R - 1) per region; phi = 0 is the identity.
#
# Canonicalizer. A fixed point of phi <- phi + delta(warp(frame, phi)): delta only sees the canonical frame, so if phi*
# canonicalizes x, the parameters that map a rescaled x to the same canonical frame are a fixed point too. Its output
# layer starts at zero: an LSE model starts as exactly the detector without LSE. Losses: the detection loss through the
# warp (labels are warped with gradients, so moving content together with its boxes costs nothing), consistency (the
# canonical frame of a randomly rescaled frame = that of the frame), optionally a size target (objects at target_size
# pixels in the canonical frame) and a prior towards the identity.

import math

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULTS = dict(
    decs=['input'],        # canonicalizers: input, P1..P6 (backbone stages) and / or [first, last] layer spans
    grid=8,                # knots per axis of the coarse warp grid (R): R - 1 cells per axis
    max_zoom=4.0,          # largest ratio between the most magnified and the most shrunk cell of a row / column
    canon_size=224,        # a canonicalizer sees each region resized to canon_size x canon_size
    channels=[16, 32, 64],  # canonicalizer conv widths (stride 2 each)
    iters=4,               # fixed-point iterations
    grad_iters=2,          # of which the last grad_iters are backpropagated (truncated fixed-point gradient)
    per_tile=True,         # mosaic samples: one warp per frame tile (False: one per sample)
    eq_weight=1.0,         # consistency loss weight
    eq_strength=1.0,       # strength of the random rescalings used by the consistency loss (in units of phi)
    eq_regions=32,         # at most this many regions per batch (random) in the consistency loss (bounds its cost)
    size_weight=0.0,       # size target loss weight (0: off)
    target_size=96,        # size target: objects' sqrt(w * h) in pixels in the canonical frame
    prior_weight=0.0,      # pull of the warps towards the identity
    warp_aug=0.0,          # chance a region of the detector input is randomly rescaled first (labels follow)
    warp_aug_strength=1.0,  # strength of those rescalings (in units of phi)
    det_grad_epochs=0,     # first epochs in which the canonicalizers learn from their own losses only (not detection)
    lr_scale=1.0,          # canonicalizer learning rate = lr0 x lr_scale
    min_conf=1e-3,         # inference: only boxes with objectness above this are mapped back (test.py conf_thres)
)


def dec_name(d):
    return d if isinstance(d, str) else f'L{d[0]}-{d[1]}'


def load_lse_cfg(cfg):
    """LSE config dict from a yaml path or dict, with defaults filled in and checked."""
    if isinstance(cfg, str):
        import yaml
        with open(cfg) as f:
            cfg = yaml.safe_load(f) or {}
    unknown = set(cfg) - set(DEFAULTS)
    assert not unknown, f'unknown LSE config keys {sorted(unknown)}, expected {sorted(DEFAULTS)}'
    cfg = {**DEFAULTS, **cfg}
    assert cfg['grid'] >= 2 and cfg['max_zoom'] >= 1 and 0 < cfg['grad_iters'] <= cfg['iters'], cfg
    decs = []
    for d in cfg['decs']:
        if isinstance(d, str):
            assert d == 'input' or (d[0] == 'P' and d[1:].rstrip('+').isdigit() and d.count('+') <= 1), \
                f'LSE decs: input, P1..P6, P1+..P6+ or [first, last], got {d}'
            decs.append(d)
        else:
            assert len(d) == 2 and int(d[0]) <= int(d[1]), f'LSE decs: a layer span is [first, last], got {d}'
            decs.append([int(d[0]), int(d[1])])
    assert decs, 'LSE decs is empty'
    names = [dec_name(d) for d in decs]
    assert len(set(names)) == len(names), f'LSE decs repeated: {names}'
    opened = [n for n in names if n.endswith('+')]
    assert len(opened) <= 1, f'LSE decs: at most one open-ended stage (Pk+), got {opened}'
    assert not opened or all(n == 'input' or n.endswith('+') for n in names), \
        f'LSE decs: an open-ended stage ({opened[0]}) combines with input only, not with closed stages, got {names}'
    cfg['decs'] = decs
    return cfg


# ----------------------------------------------------------------------------------------------------------------------
# Warp geometry (region coordinates)
# ----------------------------------------------------------------------------------------------------------------------
def make_grid(phi_x, phi_y, max_zoom):
    """Coarse warp grids (K, 2, R, R) from parameters (K, R, R - 1). Cell widths are softmax(a tanh(phi)), so the ratio
    between the widest and narrowest cell of a row (column) is at most exp(2a) = max_zoom."""
    a = 0.5 * math.log(max_zoom)

    def knots(phi):  # (K, R, R - 1) -> (K, R, R): 0, cumulative widths, last exactly 1
        w = torch.softmax(a * torch.tanh(phi), dim=-1)
        k = torch.cumsum(w, dim=-1)
        return torch.cat([torch.zeros_like(k[..., :1]), k[..., :-1], torch.ones_like(k[..., :1])], dim=-1)

    gx = knots(phi_x)  # [k, r, c]: x of knot c in input row r
    gy = knots(phi_y).transpose(1, 2)  # phi_y[k, c, r'] -> [k, r, c]: y of knot r in column c
    return torch.stack([gx, gy], dim=1)


def random_phi(n, grid, strength, device):
    return [(torch.rand(n, grid, grid - 1, device=device) * 2 - 1) * strength for _ in range(2)]


_BASE_GRIDS = {}  # (S, R, device, dtype) -> warp_crops' base sampling points (built once, not per call)


def _dense_base(us, vs, R):
    """grid_sample points of dense_local for region grids us x vs: (u, v) points and (u, row knot) points. us (w,) and
    vs (h,) are shared by every region (shapes (1, h, w, 2), (1, R, w, 2)); us (K, w) and vs (K, h) are per region."""
    gu, gv = us * 2 - 1, vs * 2 - 1
    kv = torch.linspace(-1, 1, R, device=us.device, dtype=us.dtype)
    if us.dim() == 1:
        pts = torch.stack(torch.meshgrid(gv, gu, indexing='ij')[::-1], -1).unsqueeze(0)
        rp = torch.stack(torch.meshgrid(kv, gu, indexing='ij')[::-1], -1).unsqueeze(0)
    else:
        K, h, w = us.shape[0], vs.shape[1], us.shape[1]
        pts = torch.stack([gu[:, None, :].expand(K, h, w), gv[:, :, None].expand(K, h, w)], -1)
        rp = torch.stack([gu[:, None, :].expand(K, R, w), kv[None, :, None].expand(K, R, w)], -1)
    return pts, rp


def dense_local(G, us, vs, base=None):
    """Warps G (K, 2, R, R) evaluated on the region grid us x vs (shared (w,), (h,) or per region (K, w), (K, h)):
    input coordinates gx, gy (K, h, w). base: _dense_base(us, vs, R), if already built."""
    K, R = G.shape[0], G.shape[-1]
    pts, rp = base if base is not None else _dense_base(us, vs, R)
    gy = F.grid_sample(G[:, 1:], pts.expand(K, -1, -1, -1), mode='bilinear', align_corners=True)[:, 0]  # y = gy(u, v)
    rows = F.grid_sample(G[:, :1], rp.expand(K, -1, -1, -1), mode='bilinear', align_corners=True)[:, 0]  # (K, R, w)
    f = gy * (R - 1)
    r0 = f.detach().floor().clamp(0, R - 2).long()
    t = f - r0
    gx = rows.gather(1, r0) * (1 - t) + rows.gather(1, r0 + 1) * t  # x = gx(u, y): rows interpolated at the input y
    return gx, gy


def _bilinear_rows(G, ch, k, pos):
    """G[k, ch] interpolated along rows (dim 2) at pos: (P, R) knot values."""
    R = G.shape[-1]
    f = pos.clamp(0, 1) * (R - 1)
    i0 = f.floor().clamp(max=R - 2).long()
    t = (f - i0).unsqueeze(-1)
    return G[k, ch, i0] * (1 - t) + G[k, ch, i0 + 1] * t


def _bilinear_cols(G, ch, k, pos):
    """G[k, ch] interpolated along columns (dim 3) at pos: (P, R) knot values (one per row)."""
    R = G.shape[-1]
    f = pos.clamp(0, 1) * (R - 1)
    i0 = f.floor().clamp(max=R - 2).long()
    t = (f - i0).unsqueeze(-1)
    return G[k, ch, :, i0] * (1 - t) + G[k, ch, :, i0 + 1] * t


def _interp_1d(knots, q):
    """Piecewise-linear function through (i / (R - 1), knots[:, i]) at q: values and slopes. Outside [0, 1] the end
    segments are extended (boxes may extend past their region)."""
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
    idx = torch.searchsorted(knots.detach().contiguous(), x.detach().unsqueeze(-1).contiguous()).clamp(1, R - 1)
    k0, k1 = knots.gather(1, idx - 1).squeeze(1), knots.gather(1, idx).squeeze(1)
    d = (k1 - k0).clamp_min(1e-9)
    q = ((idx.squeeze(1) - 1) + (x - k0) / d) / (R - 1)
    return q, d * (R - 1)


def forward_points(G, k, u, v):
    """Canonical points (u, v) of regions k -> input points (x, y), region coordinates. All (P,) tensors."""
    y, _ = _interp_1d(_bilinear_cols(G, 1, k, u), v)
    x, _ = _interp_1d(_bilinear_rows(G, 0, k, y), u)
    return x, y


def inverse_points(G, k, x, y):
    """Input points (x, y) of regions k -> canonical points (u, v), exact; also the local slopes dgx/du, dgy/dv
    (1 = no scaling, < 1 = magnified)."""
    u, sx = _invert_1d(_bilinear_rows(G, 0, k, y), x)
    v, sy = _invert_1d(_bilinear_cols(G, 1, k, u), y)
    return u, v, sx, sy


# ----------------------------------------------------------------------------------------------------------------------
# Regions: one warp per frame tile (or per image)
# ----------------------------------------------------------------------------------------------------------------------
class Regions:
    """Warps of regions: G (K, 2, R, R), image index b (K,), rectangle rect (K, 4) normalised xyxy of the image."""

    def __init__(self, b, rect, G=None):
        self.b, self.rect, self.G = b, rect, G

    @staticmethod
    def make(n, tiles, device):
        b, rects = [], []
        for i in range(n):
            t = tiles[i] if tiles is not None and i < len(tiles) else None
            t = t.to(device).float().reshape(-1, 4) if torch.is_tensor(t) and len(t) else \
                torch.tensor([[0., 0., 1., 1.]], device=device)
            b.append(torch.full((len(t),), i, device=device, dtype=torch.long))
            rects.append(t)
        return Regions(torch.cat(b), torch.cat(rects))

    def with_grid(self, G):
        return Regions(self.b, self.rect, G)

    def crops(self, x, size):
        """Each region of images x (B, C, H, W) resized to size x size: (K, C, size, size), in region order."""
        r = self.rect
        theta = torch.zeros(len(r), 2, 3, device=x.device, dtype=x.dtype)
        theta[:, 0, 0], theta[:, 0, 2] = r[:, 2] - r[:, 0], r[:, 0] + r[:, 2] - 1
        theta[:, 1, 1], theta[:, 1, 2] = r[:, 3] - r[:, 1], r[:, 1] + r[:, 3] - 1
        grid = F.affine_grid(theta, (len(r), x.shape[1], size, size), align_corners=False)
        src = x if len(r) == x.shape[0] else x.index_select(0, self.b)  # one region per image: region k is image k
        return F.grid_sample(src, grid, mode='bilinear', padding_mode='border', align_corners=False)

    def _blocks(self, h, w):
        """The pixels of every region on an h x w map (those whose centre is inside its rectangle), padded to a
        (K, Hm, Wm) layout: flat indices into (n, h, w), a mask of the real pixels, and region coordinates us (K, Wm),
        vs (K, Hm). None if no region has a pixel. One host sync (the padded size), whatever the number of regions."""
        r = self.rect.double()  # float64: pixel ranges exact at the rectangle edges
        j0, j1 = torch.ceil(r[:, 0] * w - 0.5).clamp(min=0).long(), torch.ceil(r[:, 2] * w - 0.5).clamp(max=w).long()
        i0, i1 = torch.ceil(r[:, 1] * h - 0.5).clamp(min=0).long(), torch.ceil(r[:, 3] * h - 0.5).clamp(max=h).long()
        nh, nw = (i1 - i0).clamp(min=0), (j1 - j0).clamp(min=0)
        Hm, Wm = torch.stack([nh.max(), nw.max()]).tolist()
        if Hm <= 0 or Wm <= 0:
            return None
        dev = self.rect.device
        a, c = torch.arange(Hm, device=dev), torch.arange(Wm, device=dev)
        valid = (a[None, :, None] < nh[:, None, None]) & (c[None, None, :] < nw[:, None, None])
        rows, cols = (i0[:, None] + a).clamp(max=h - 1), (j0[:, None] + c).clamp(max=w - 1)
        flat = (self.b[:, None, None] * h + rows[:, :, None]) * w + cols[:, None, :]
        rf = self.rect
        us = ((cols + 0.5) / w - rf[:, :1]) / (r[:, 2] - r[:, 0]).clamp(min=1e-9).float()[:, None]
        vs = ((rows + 0.5) / h - rf[:, 1:2]) / (r[:, 3] - r[:, 1]).clamp(min=1e-9).float()[:, None]
        return flat, valid, us, vs

    def sampling_grid(self, n, h, w, dtype):
        """grid_sample grid (n, h, w, 2) (align_corners=False) warping every region: canonical pixel -> input."""
        dev = self.rect.device
        ys, xs = (torch.arange(h, device=dev) + 0.5) / h, (torch.arange(w, device=dev) + 0.5) / w
        grid = torch.stack(torch.meshgrid(ys, xs, indexing='ij')[::-1], -1) * 2 - 1
        grid = grid.unsqueeze(0).repeat(n, 1, 1, 1).to(self.G.dtype)
        blocks = self._blocks(h, w) if len(self.b) else None
        if blocks is not None:  # every region at once (regions do not overlap)
            flat, valid, us, vs = blocks
            gx, gy = dense_local(self.G, us.to(self.G.dtype), vs.to(self.G.dtype))
            x1, y1, x2, y2 = (v[:, None, None] for v in self.rect.unbind(1))
            vals = torch.stack([x1 + gx * (x2 - x1), y1 + gy * (y2 - y1)], -1) * 2 - 1
            grid.view(-1, 2)[flat[valid]] = vals[valid]
        return grid.to(dtype)

    def inverse_grid(self, n, h, w):
        """grid_sample grid (n, h, w, 2) un-warping canonical maps back to the image, and the log magnification
        (n, 2, h, w) at each image location (x, y; > 0 where the canonical frame magnifies)."""
        dev = self.rect.device
        idx = torch.full((n, h, w), -1, device=dev, dtype=torch.long)
        blocks = self._blocks(h, w) if len(self.b) else None
        if blocks is not None:
            flat, valid = blocks[:2]
            idx.view(-1)[flat[valid]] = torch.arange(len(self.b), device=dev)[:, None, None].expand_as(flat)[valid]
        ys, xs = (torch.arange(h, device=dev) + 0.5) / h, (torch.arange(w, device=dev) + 0.5) / w
        py, px = [t.reshape(1, h, w).expand(n, h, w) for t in torch.meshgrid(ys, xs, indexing='ij')]
        grid = torch.stack([px, py], -1).clone().to(self.G.dtype)
        logm = torch.zeros(n, 2, h, w, device=dev, dtype=self.G.dtype)
        m = idx >= 0
        if blocks is not None and m.any():
            k = idx[m]
            r = self.rect[k].to(self.G.dtype)
            tx, ty = (px[m] - r[:, 0]) / (r[:, 2] - r[:, 0]), (py[m] - r[:, 1]) / (r[:, 3] - r[:, 1])
            u, v, sx, sy = inverse_points(self.G, k, tx, ty)
            grid[m] = torch.stack([r[:, 0] + u * (r[:, 2] - r[:, 0]), r[:, 1] + v * (r[:, 3] - r[:, 1])], -1)
            lm = -torch.log(torch.stack([sx, sy], -1).clamp_min(1e-6))
            logm.permute(0, 2, 3, 1)[m] = lm
        return grid * 2 - 1, logm

    def warp(self, x):
        grid = self.sampling_grid(x.shape[0], *x.shape[-2:], x.dtype)
        return F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=False)

    def assign(self, img, cx, cy):
        """Region of each point (image index img, normalised centre cx, cy): the region of that image containing it, else
        -1."""
        if not len(img):
            return img
        r = self.rect
        inside = (img.view(-1, 1) == self.b.view(1, -1)) & (cx.view(-1, 1) >= r[:, 0]) & (cx.view(-1, 1) < r[:, 2]) & \
                 (cy.view(-1, 1) >= r[:, 1]) & (cy.view(-1, 1) < r[:, 3])
        k = inside.float().argmax(1)
        return torch.where(inside.any(1), k, torch.full_like(k, -1))

    def _sides(self, k, b, to_canonical):
        """Edge-midpoint mapping of boxes b (N, 4, region coordinates) of regions k: each side goes where the warp moves
        its midpoint."""
        x1, y1, x2, y2 = b.unbind(-1)
        xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
        px, py = torch.stack([x1, x2, xm, xm], -1).reshape(-1), torch.stack([ym, ym, y1, y2], -1).reshape(-1)
        kk = k.view(-1, 1).expand(-1, 4).reshape(-1)
        if to_canonical:
            qx, qy, _, _ = inverse_points(self.G.to(b.dtype), kk, px, py)
        else:
            qx, qy = forward_points(self.G.to(b.dtype), kk, px, py)
        qx, qy = qx.view(-1, 4), qy.view(-1, 4)
        return torch.stack([torch.minimum(qx[:, 0], qx[:, 1]), torch.minimum(qy[:, 2], qy[:, 3]),
                            torch.maximum(qx[:, 0], qx[:, 1]), torch.maximum(qy[:, 2], qy[:, 3])], -1)

    def _outline(self, k, b, n=8):
        """Bounding box, in the canonical frame, of the warped outline of boxes b (region coordinates): the extent of
        the warped object (edges bend under a non-uniform warp)."""
        x1, y1, x2, y2 = b.unbind(-1)
        t = torch.linspace(0, 1, n + 1, device=b.device, dtype=b.dtype).view(1, -1)
        xs = torch.cat([x1[:, None] + (x2 - x1)[:, None] * t, x2[:, None].expand(-1, n + 1),
                        x1[:, None] + (x2 - x1)[:, None] * t, x1[:, None].expand(-1, n + 1)], 1)
        ys = torch.cat([y1[:, None].expand(-1, n + 1), y1[:, None] + (y2 - y1)[:, None] * t,
                        y2[:, None].expand(-1, n + 1), y1[:, None] + (y2 - y1)[:, None] * t], 1)
        kk = k.view(-1, 1).expand_as(xs).reshape(-1)
        qx, qy, _, _ = inverse_points(self.G.to(b.dtype), kk, xs.reshape(-1), ys.reshape(-1))
        qx, qy = qx.view(len(b), -1), qy.view(len(b), -1)
        return torch.stack([qx.amin(1), qy.amin(1), qx.amax(1), qy.amax(1)], -1)

    def _centre_scale(self, k, b):
        """Boxes b of the canonical frame (region coordinates) -> the input box with the forward-mapped centre and the
        size divided by the local magnification there."""
        G = self.G.to(b.dtype)
        cx, cy = forward_points(G, k, (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2)
        _, _, sx, sy = inverse_points(G, k, cx, cy)
        w, h = (b[:, 2] - b[:, 0]) * sx, (b[:, 3] - b[:, 1]) * sy
        return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)

    def _solve_outline(self, k, b, q, iters):
        """Newton's method for outline(q) = b (finite-difference 4 x 4 Jacobian per box) from q, halving the step of a
        box whose error grows and keeping each box's best estimate. Corners are kept ordered: a box with x1 and x2 (or
        y1 and y2) swapped has the same outline. Returns the boxes and their remaining error."""
        err = (self._outline(k, q) - b).abs().amax(1)
        step = torch.ones_like(err)
        eps = 1e-3
        eye = torch.eye(4, device=b.device, dtype=b.dtype)
        for _ in range(iters):
            f = self._outline(k, q)
            qs = (q[:, None, :] + eps * eye[None]).reshape(-1, 4)  # one coordinate nudged at a time
            J = ((self._outline(k.repeat_interleave(4), qs).view(-1, 4, 4) - f[:, None, :]) / eps).transpose(1, 2)
            dq = torch.linalg.solve(J + 1e-6 * eye, (b - f).unsqueeze(-1)).squeeze(-1)
            cand = q + step[:, None] * dq.clamp(-0.5, 0.5)
            cand = torch.cat([torch.minimum(cand[:, :2], cand[:, 2:]), torch.maximum(cand[:, :2], cand[:, 2:])], 1)
            cerr = (self._outline(k, cand) - b).abs().amax(1)
            better = cerr < err
            q = torch.where(better[:, None], cand, q)
            err = torch.where(better, cerr, err)
            step = torch.where(better, torch.ones_like(step), step / 2)
        return q, err

    def map_boxes(self, img, xyxy, to_canonical, iters=12):
        """Boxes (N, 4) normalised xyxy of images img -> canonical frame (to_canonical) or back. Into the canonical frame
        a box becomes the bounding box of its warped outline (the warped object's extent). Back, the box whose warped
        outline gives the canonical box is solved for, from two starts (edge midpoints mapped back; centre mapped back
        with the size divided by the local magnification), keeping the better. Under strongly sheared warps several
        boxes can share one outline bounding box; see README.md, section 14 for the measured accuracy. Boxes whose centre
        is in no region are unchanged."""
        k = self.assign(img, (xyxy[:, 0] + xyxy[:, 2]) / 2, (xyxy[:, 1] + xyxy[:, 3]) / 2)
        m = k >= 0
        if not m.any():
            return xyxy
        k, b = k[m], xyxy[m]
        r = self.rect[k].to(b.dtype)
        rw, rh = r[:, 2] - r[:, 0], r[:, 3] - r[:, 1]
        scale, off = torch.stack([rw, rh, rw, rh], -1), r[:, [0, 1, 0, 1]]
        b = (b - off) / scale  # region coordinates
        if to_canonical:
            q = self._outline(k, b)
        else:
            n = len(b)
            q, err = self._solve_outline(k.repeat(2), b.repeat(2, 1),
                                         torch.cat([self._sides(k, b, False), self._centre_scale(k, b)]), iters)
            q = torch.where((err[:n] <= err[n:])[:, None], q[:n], q[n:])
        out = xyxy.clone()
        out[m] = q * scale + off
        return out

    def zoom_at(self, img, cx, cy):
        """Local magnification (geometric mean of x and y) of the warp at points (normalised image coordinates), and a
        mask of the points that are in a region."""
        k = self.assign(img, cx, cy)
        m = k >= 0
        k = k[m]
        r = self.rect[k]
        _, _, sx, sy = inverse_points(self.G, k, (cx[m] - r[:, 0]) / (r[:, 2] - r[:, 0]),
                                      (cy[m] - r[:, 1]) / (r[:, 3] - r[:, 1]))
        return 1 / torch.sqrt(sx * sy).clamp_min(1e-6), m


def warp_targets(targets, regions, h, w, min_px=2.0):
    """YOLO targets (n, 6) [image, class, x, y, w, h] normalised -> the same boxes in the canonical frame (boxes smaller
    than min_px on a side are dropped). Differentiable in the warp."""
    if not len(targets):
        return targets
    t = targets.float()
    xyxy = torch.cat([t[:, 2:4] - t[:, 4:6] / 2, t[:, 2:4] + t[:, 4:6] / 2], -1)
    q = regions.map_boxes(t[:, 0].long(), xyxy, to_canonical=True).clamp(0, 1)
    keep = (((q[:, 2] - q[:, 0]) * w) >= min_px) & (((q[:, 3] - q[:, 1]) * h) >= min_px)
    out = torch.cat([t[:, :2], (q[:, :2] + q[:, 2:]) / 2, q[:, 2:] - q[:, :2]], -1)
    return out[keep].to(targets.dtype)


def unwarp_predictions(z, regions, h, w, min_conf=1e-3):
    """Decoded predictions (B, N, no) [x, y, w, h, obj, ...] in canonical pixels -> image pixels. Only boxes with
    objectness above min_conf are mapped (NMS discards the others at any usual threshold)."""
    z = z.clone()
    keep = z[..., 4] > min_conf
    if keep.any():
        b = torch.nonzero(keep)[:, 0]
        p = z[keep][:, :4].float() / torch.tensor([w, h, w, h], device=z.device, dtype=torch.float32)
        q = regions.map_boxes(b, torch.cat([p[:, :2] - p[:, 2:] / 2, p[:, :2] + p[:, 2:] / 2], -1), to_canonical=False)
        q = q * torch.tensor([w, h, w, h], device=z.device, dtype=torch.float32)
        z[keep] = torch.cat([(q[:, :2] + q[:, 2:]) / 2, q[:, 2:] - q[:, :2], z[keep][:, 4:].float()], -1).to(z.dtype)
    return z


def warp_stats(G):
    """Mean displacement from the identity (fraction of the region) and the strongest local magnification and
    shrinking of a set of warps (0-dim tensors)."""
    R = G.shape[-1]
    ident = torch.linspace(0, 1, R, device=G.device)
    disp = torch.cat([(G[:, 0] - ident.view(1, 1, R)).abs().flatten(), (G[:, 1] - ident.view(1, R, 1)).abs().flatten()])
    sx = (G[:, 0, :, 1:] - G[:, 0, :, :-1]) * (R - 1)  # slopes: < 1 magnifies
    sy = (G[:, 1, 1:, :] - G[:, 1, :-1, :]) * (R - 1)
    s = torch.cat([sx.flatten(), sy.flatten()])
    return {'disp': disp.mean(), 'zoom_in': 1 / s.min(), 'zoom_out': s.max()}  # tensors: no host sync per step


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

    def delta(self, c):
        R = self.grid
        f = self.head(self.features(c.to(self.head.weight.dtype))).float()
        dx = F.adaptive_avg_pool2d(f[:, :1], (R, R - 1))[:, 0]  # [k, row knot, x cell]
        dy = F.adaptive_avg_pool2d(f[:, 1:], (R - 1, R))[:, 0].transpose(1, 2)  # [k, column knot, y cell]
        # a constant added to a row's (column's) parameters does not change the warp (softmax): remove it, so phi does
        # not drift into the flat part of tanh, where the canonicalizer would lose its gradient
        return dx - dx.mean(-1, keepdim=True), dy - dy.mean(-1, keepdim=True)

    def warp_crops(self, s, G):
        """Warp whole crops s (K, C, S, S) by G (K, 2, R, R)."""
        S, R = s.shape[-1], G.shape[-1]
        key = (S, R, str(s.device), G.dtype)
        if key not in _BASE_GRIDS:  # the same points every call: build them once
            c = (torch.arange(S, device=s.device, dtype=G.dtype) + 0.5) / S
            _BASE_GRIDS[key] = _dense_base(c, c, R)
        gx, gy = dense_local(G, None, None, _BASE_GRIDS[key])
        return F.grid_sample(s, torch.stack([gx, gy], -1) * 2 - 1, mode='bilinear', padding_mode='border',
                             align_corners=False)

    def solve(self, s):
        """Fixed point phi = phi + delta(warp(s, phi)) for region crops s: the warp grids. Only the last grad_iters
        iterations are backpropagated."""
        R = self.grid
        phi_x = torch.zeros(s.shape[0], R, R - 1, device=s.device)
        phi_y = torch.zeros_like(phi_x)
        grad = torch.is_grad_enabled()
        for k in range(self.iters):
            track = grad and k >= self.iters - self.grad_iters
            if track and k == self.iters - self.grad_iters:
                phi_x, phi_y = phi_x.detach(), phi_y.detach()
            with torch.set_grad_enabled(track):
                dx, dy = self.delta(self.warp_crops(s, make_grid(phi_x, phi_y, self.max_zoom)))
                phi_x, phi_y = phi_x + dx, phi_y + dy
        return make_grid(phi_x, phi_y, self.max_zoom), phi_x, phi_y


def stage_spans(model):
    """Backbone stages of a Model: {'P1': [first, last], ...}. Stage k starts at the first layer whose output stride
    is 2^k and ends before the next stage starts (the last stage at the end of the backbone)."""
    n_backbone = len(model.yaml['backbone'])
    shapes = []
    hooks = [m.register_forward_hook(lambda mod, i, o: shapes.append(o.shape[-1] if torch.is_tensor(o) else None))
             for m in model.model[:n_backbone]]
    try:
        p = next(model.parameters())
        with torch.no_grad():
            training = model.training
            model.eval()
            model.forward_once(torch.zeros(1, model.yaml.get('ch', 3), 256, 256, device=p.device, dtype=p.dtype))
            model.train(training)
    finally:
        for h in hooks:
            h.remove()
    starts = {}
    for i, wdt in enumerate(shapes[:n_backbone]):
        if wdt:
            k = round(math.log2(256 / wdt))
            if k >= 1 and f'P{k}' not in starts:
                starts[f'P{k}'] = i
    keys = sorted(starts, key=lambda k: starts[k])
    return {k: [starts[k], (starts[keys[j + 1]] - 1) if j + 1 < len(keys) else n_backbone - 1]
            for j, k in enumerate(keys)}


def validate_spans(spans, layer_from):
    """Latent spans [first, last] may read any earlier layers (warped on entry), but after a span no layer may read a
    layer of the span other than its last (whose output is warped back). Spans must not overlap."""
    if not spans:
        return
    assert layer_from is not None, 'latent LSE needs the model layer graph'
    n = len(layer_from)
    covered = set()
    for a, b in spans:
        assert 0 <= a <= b < n - 1, f'LSE span {[a, b]}: layers must be in [0, {n - 2}] (not the detection head)'
        assert not covered & set(range(a, b + 1)), f'LSE span {[a, b]} overlaps another span'
        covered |= set(range(a, b + 1))
        for j in range(b + 1, n):
            srcs = [f if f >= 0 else j + f for f in layer_from[j]]
            bad = [s for s in srcs if a <= s < b]
            assert not bad, (f'LSE span {[a, b]}: layer {j} reads layer(s) {bad} inside the span (only the span output, '
                             f'layer {b}, may be read after it)')


# ----------------------------------------------------------------------------------------------------------------------
# Visualisation
# ----------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def plot_warps(model, imgs, tiles=None, lines=8):
    """Per DEC layer of a DEC model (models/dec.py), a panel with one row per image: the image the DEC sees with its
    canonical grid drawn on it (where each canonical cell samples from), the local zoom (red: magnified, blue: shrunk,
    full scale = max_zoom) and the canonical image. imgs: (n, 3, H, W) in [0, 1] (RGB); tiles: per image the content
    rectangle(s) (e.g. without letterbox bars). Returns {dec name: (panel, grid overlay of the first image)}, uint8 BGR."""
    from models.dec import DEC
    ctx, training = model.dec_ctx, model.training
    model.eval()
    ctx.keep = True  # keep the warps and the images the DECs see
    try:
        x = imgs.float()
        n, _, H, W = x.shape
        model(x, lse_tiles=tiles)
        base = ctx.base
        t = torch.linspace(0, 1, 200, device=x.device)
        out = {}
        for m in model.model:
            if not isinstance(m, DEC):
                continue
            reg, seen, lim = ctx.regs[m.i], ctx.seen[m.i], math.log(m.canon.max_zoom)
            G = reg.G
            canon = reg.warp(seen)
            _, logm = reg.inverse_grid(n, max(H // 4, 1), max(W // 4, 1))
            zoom = logm.mean(1)  # log geometric magnification at each image location
            rows, first = [], None
            for i in range(n):
                im = np.ascontiguousarray((seen[i].permute(1, 2, 0).cpu().numpy()[:, :, ::-1] * 255).clip(0, 255)
                                          .astype(np.uint8))
                ov = im.copy()
                for k in torch.nonzero(base.b == i)[:, 0].tolist():  # each region of the image
                    kk = torch.full_like(t, k, dtype=torch.long)
                    x1, y1, x2, y2 = base.rect[k].tolist()
                    for c in torch.linspace(0, 1, lines + 1, device=x.device):
                        for u, v in ((t, torch.full_like(t, c)), (torch.full_like(t, c), t)):  # canonical rows / columns
                            px, py = forward_points(G, kk, u, v)
                            pts = torch.stack([(x1 + px * (x2 - x1)) * W, (y1 + py * (y2 - y1)) * H], -1)
                            cv2.polylines(ov, [pts.cpu().numpy().round().astype(np.int32)], False, (0, 255, 255), 1,
                                          cv2.LINE_AA)
                z = ((zoom[i].cpu().numpy() / lim).clip(-1, 1) * 127 + 128).astype(np.uint8)
                heat = cv2.resize(cv2.applyColorMap(z, cv2.COLORMAP_JET), (W, H), interpolation=cv2.INTER_LINEAR)
                heat = cv2.addWeighted(im, 0.4, heat, 0.6, 0)
                cv = np.ascontiguousarray((canon[i].permute(1, 2, 0).cpu().numpy()[:, :, ::-1] * 255).clip(0, 255)
                                          .astype(np.uint8))
                for p, label in ((ov, f'{m.name}: warp grid'), (heat, 'local zoom'), (cv, 'canonical')):
                    cv2.putText(p, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                rows.append(np.hstack([ov, heat, cv]))
                first = ov if first is None else first
            out[m.name] = (np.vstack(rows), first)
        return out
    finally:
        ctx.keep = False
        ctx.clear()
        model.train(training)

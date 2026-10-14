# Local scale equivariance (LSE) for YOLOv7, after "Local Scale Equivariance with Latent Deep Equilibrium
# Canonicalizer" (Rahman et al., ICCV 2025). See README.md, section 14.
#
# Deep equilibrium canonicalizers (DECs) pick, per frame, a smooth monotone warp that rescales parts of the frame
# differently, so objects whose apparent size varies (e.g. with distance) are seen at a more uniform, "canonical" scale.
# Each DEC has its own canonicalizer network; `decs` lists where they are inserted:
#   input:  the detector runs on the warped (canonical) image; labels are warped into it for training and predicted
#           boxes are mapped back. The detector itself is unchanged.
#   P1..P6: the backbone stage starting at the stride-2^k downsampling layer runs in a canonical frame: its inputs are
#           warped, its output is warped back before it is used. The head is told every stage's local scale
#           (zero-initialised 1x1 convs), so boxes stay in image pixels.
#   [a, b]: the same for an explicit span of layers a..b.
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
    min_conf=1e-4,         # input DEC inference: only boxes with objectness above this are mapped back
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
            assert d == 'input' or (d[0] == 'P' and d[1:].isdigit()), f'LSE decs: input, P1..P6 or [first, last], got {d}'
            decs.append(d)
        else:
            assert len(d) == 2 and int(d[0]) <= int(d[1]), f'LSE decs: a layer span is [first, last], got {d}'
            decs.append([int(d[0]), int(d[1])])
    assert decs, 'LSE decs is empty'
    names = [dec_name(d) for d in decs]
    assert len(set(names)) == len(names), f'LSE decs repeated: {names}'
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


def dense_local(G, us, vs):
    """Warps G (K, 2, R, R) evaluated on the region grid us (w,) x vs (h,): input coordinates gx, gy (K, h, w)."""
    K, R = G.shape[0], G.shape[-1]
    gu, gv = us * 2 - 1, vs * 2 - 1
    pts = torch.stack(torch.meshgrid(gv, gu, indexing='ij')[::-1], -1).unsqueeze(0).expand(K, -1, -1, -1)
    gy = F.grid_sample(G[:, 1:], pts, mode='bilinear', align_corners=True)[:, 0]  # y = gy(u, v)
    kv = torch.linspace(-1, 1, R, device=G.device, dtype=G.dtype)
    rp = torch.stack(torch.meshgrid(kv, gu, indexing='ij')[::-1], -1).unsqueeze(0).expand(K, -1, -1, -1)
    rows = F.grid_sample(G[:, :1], rp, mode='bilinear', align_corners=True)[:, 0]  # (K, R, w): row x knots at u
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
        """Each region of images x (B, C, H, W) resized to size x size: (K, C, size, size)."""
        out = []
        for i in range(x.shape[0]):
            m = self.b == i
            if not m.any():
                continue
            r = self.rect[m]
            theta = torch.zeros(len(r), 2, 3, device=x.device, dtype=x.dtype)
            theta[:, 0, 0], theta[:, 0, 2] = r[:, 2] - r[:, 0], r[:, 0] + r[:, 2] - 1
            theta[:, 1, 1], theta[:, 1, 2] = r[:, 3] - r[:, 1], r[:, 1] + r[:, 3] - 1
            grid = F.affine_grid(theta, (len(r), x.shape[1], size, size), align_corners=False)
            out.append(F.grid_sample(x[i:i + 1].expand(len(r), -1, -1, -1), grid, mode='bilinear',
                                     padding_mode='border', align_corners=False))
        return torch.cat(out)

    def _pixels(self, k, h, w):
        """Pixel index ranges of region k on an h x w map (pixel centres inside the rectangle) and their region
        coordinates."""
        x1, y1, x2, y2 = self.rect[k].tolist()
        j0, j1 = max(math.ceil(x1 * w - 0.5), 0), min(math.ceil(x2 * w - 0.5), w)
        i0, i1 = max(math.ceil(y1 * h - 0.5), 0), min(math.ceil(y2 * h - 0.5), h)
        dev = self.rect.device
        us = ((torch.arange(j0, j1, device=dev) + 0.5) / w - x1) / max(x2 - x1, 1e-9)
        vs = ((torch.arange(i0, i1, device=dev) + 0.5) / h - y1) / max(y2 - y1, 1e-9)
        return i0, i1, j0, j1, us, vs

    def sampling_grid(self, n, h, w, dtype):
        """grid_sample grid (n, h, w, 2) (align_corners=False) warping every region: canonical pixel -> input."""
        dev = self.rect.device
        ys, xs = (torch.arange(h, device=dev) + 0.5) / h, (torch.arange(w, device=dev) + 0.5) / w
        grid = torch.stack(torch.meshgrid(ys, xs, indexing='ij')[::-1], -1) * 2 - 1
        grid = grid.unsqueeze(0).repeat(n, 1, 1, 1).to(self.G.dtype)
        for k in range(len(self.b)):
            i0, i1, j0, j1, us, vs = self._pixels(k, h, w)
            if i1 <= i0 or j1 <= j0:
                continue
            gx, gy = dense_local(self.G[k:k + 1], us.to(self.G.dtype), vs.to(self.G.dtype))
            x1, y1, x2, y2 = self.rect[k]
            grid[self.b[k], i0:i1, j0:j1] = torch.stack([x1 + gx[0] * (x2 - x1), y1 + gy[0] * (y2 - y1)], -1) * 2 - 1
        return grid.to(dtype)

    def inverse_grid(self, n, h, w):
        """grid_sample grid (n, h, w, 2) un-warping canonical maps back to the image, and the log magnification
        (n, 2, h, w) at each image location (x, y; > 0 where the canonical frame magnifies)."""
        dev = self.rect.device
        idx = torch.full((n, h, w), -1, device=dev, dtype=torch.long)
        for k in range(len(self.b)):
            i0, i1, j0, j1, _, _ = self._pixels(k, h, w)
            if i1 > i0 and j1 > j0:
                idx[self.b[k], i0:i1, j0:j1] = k
        ys, xs = (torch.arange(h, device=dev) + 0.5) / h, (torch.arange(w, device=dev) + 0.5) / w
        py, px = [t.reshape(1, h, w).expand(n, h, w) for t in torch.meshgrid(ys, xs, indexing='ij')]
        grid = torch.stack([px, py], -1).clone().to(self.G.dtype)
        logm = torch.zeros(n, 2, h, w, device=dev, dtype=self.G.dtype)
        m = idx >= 0
        if m.any():
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

    def map_boxes(self, img, xyxy, to_canonical):
        """Boxes (N, 4) normalised xyxy of images img -> canonical frame (to_canonical) or back. Each side goes where the
        warp moves its midpoint, so a box mapped to the canonical frame and back is returned (unlike the bounding box of
        the warped outline, which grows at every mapping). Boxes whose centre is in no region are unchanged."""
        k = self.assign(img, (xyxy[:, 0] + xyxy[:, 2]) / 2, (xyxy[:, 1] + xyxy[:, 3]) / 2)
        m = k >= 0
        if not m.any():
            return xyxy
        k, b = k[m], xyxy[m]
        r = self.rect[k].to(b.dtype)
        rw, rh = r[:, 2] - r[:, 0], r[:, 3] - r[:, 1]
        x1, y1, x2, y2 = ((b[:, 0] - r[:, 0]) / rw, (b[:, 1] - r[:, 1]) / rh,
                          (b[:, 2] - r[:, 0]) / rw, (b[:, 3] - r[:, 1]) / rh)
        xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
        px, py = torch.stack([x1, x2, xm, xm], -1).reshape(-1), torch.stack([ym, ym, y1, y2], -1).reshape(-1)
        kk = k.view(-1, 1).expand(-1, 4).reshape(-1)
        G = self.G.to(b.dtype)
        if to_canonical:
            qx, qy, _, _ = inverse_points(G, kk, px, py)
        else:
            qx, qy = forward_points(G, kk, px, py)
        qx, qy = qx.view(-1, 4), qy.view(-1, 4)
        q = torch.stack([torch.minimum(qx[:, 0], qx[:, 1]), torch.minimum(qy[:, 2], qy[:, 3]),
                         torch.maximum(qx[:, 0], qx[:, 1]), torch.maximum(qy[:, 2], qy[:, 3])], -1)
        q = q * torch.stack([rw, rh, rw, rh], -1) + r[:, [0, 1, 0, 1]]
        out = xyxy.clone()
        out[m] = q
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


def unwarp_predictions(z, regions, h, w, min_conf=1e-4):
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
    shrinking of a set of warps."""
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
        S = s.shape[-1]
        c = (torch.arange(S, device=s.device, dtype=G.dtype) + 0.5) / S
        gx, gy = dense_local(G, c, c)
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


class LSE(nn.Module):
    """Deep equilibrium canonicalizers attached to a Model as model.lse (see Model.attach_lse)."""

    def __init__(self, cfg, stages=None, head_channels=(), layer_from=None):
        super().__init__()
        self.cfg = cfg = load_lse_cfg(cfg)
        self.epoch = 0  # set by train.py (det_grad_epochs)
        self.names = [dec_name(d) for d in cfg['decs']]
        self.spans = {}  # latent DEC name -> [first, last] layer
        for d, name in zip(cfg['decs'], self.names):
            if name == 'input':
                continue
            if isinstance(d, str):
                assert stages and d in stages, f'LSE dec {d}: this model has backbone stages {sorted(stages or {})}'
                self.spans[name] = stages[d]
            else:
                self.spans[name] = d
        validate_spans(list(self.spans.values()), layer_from)
        self.canon = nn.ModuleDict({n: Canonicalizer(cfg['grid'], cfg['channels'], cfg['max_zoom'], cfg['canon_size'],
                                                     cfg['iters'], cfg['grad_iters']) for n in self.names})
        self.has_input = 'input' in self.names
        self.latent = [n for n in self.names if n != 'input']
        self.layer_dec = {i: n for n, (a, b) in self.spans.items() for i in range(a, b + 1)}  # layer -> latent DEC
        if self.latent:  # local scale of every latent DEC -> head features (zero-initialised: no effect at start)
            self.cond = nn.ModuleList(nn.Conv2d(2 * len(self.latent), c, 1) for c in head_channels)
            for m in self.cond:
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

    def _solve(self, name, s):
        return self.canon[name].solve(s)

    def forward_input(self, x, tiles=None, targets=None):
        """Canonicalize a batch x (B, 3, H, W) in [0, 1]. Returns the detector input (randomly rescaled by warp_aug,
        then warped by the input DEC if any), the regions with their warps per DEC, the targets in the frame of the
        predictions (None if no targets), the auxiliary loss and statistics (training; else None)."""
        cfg, n, (h, w) = self.cfg, x.shape[0], x.shape[-2:]
        R, mz = cfg['grid'], cfg['max_zoom']
        regs, loss, stats = {}, None, None
        with torch.autocast(device_type=x.device.type, enabled=False):
            xf = x.float()
            base = Regions.make(n, tiles if self.training and cfg['per_tile'] else None, x.device)
            if self.training and cfg['warp_aug'] > 0:  # rescale some regions at random (labels follow)
                K = len(base.b)
                tx, ty = random_phi(K, R, cfg['warp_aug_strength'], x.device)
                on = (torch.rand(K, 1, 1, device=x.device) < cfg['warp_aug']).float()
                aug = base.with_grid(make_grid(tx * on, ty * on, mz))
                with torch.no_grad():
                    xf = aug.warp(xf)
                    if targets is not None:
                        targets = warp_targets(targets, aug, h, w)
            if self.training:
                loss, stats = xf.new_zeros(()), {'regions/img': len(base.b) / n}
            warm = self.training and self.epoch < cfg['det_grad_epochs']
            for name in (['input'] if self.has_input else []) + self.latent:  # input first: the stages see its output
                s = base.crops(xf, cfg['canon_size'])
                G, phi_x, phi_y = self._solve(name, s)
                reg = base.with_grid(G)
                if self.training:
                    l, st = self._aux(name, s, G, phi_x, phi_y, reg, targets, h, w)
                    loss = loss + l
                    stats.update({f'{name}/{k}': v for k, v in st.items()})
                regs[name] = base.with_grid(G.detach()) if warm else reg  # warm-up: no detection gradient yet
                if name == 'input':
                    if targets is not None:  # with gradients: content and its boxes move together
                        targets = warp_targets(targets, regs[name], h, w)
                    xf = regs[name].warp(xf)
        return xf.to(x.dtype), regs, targets, loss, stats

    def _aux(self, name, s, G, phi_x, phi_y, reg, targets, h, w):
        """Consistency, prior and size losses of one DEC, and its statistics."""
        cfg = self.cfg
        if s.shape[0] > cfg['eq_regions']:  # a random subset of the regions: the second solve is the costly part
            idx = torch.randperm(s.shape[0], device=s.device)[:cfg['eq_regions']]
            s_eq, G_eq = s[idx], G[idx]
        else:
            s_eq, G_eq = s, G
        tx, ty = random_phi(s_eq.shape[0], cfg['grid'], cfg['eq_strength'], s.device)
        c = self.canon[name]
        s_t = c.warp_crops(s_eq, make_grid(tx, ty, cfg['max_zoom']))
        G_t, _, _ = c.solve(s_t)
        eq = F.mse_loss(c.warp_crops(s_t, G_t), c.warp_crops(s_eq, G_eq))
        prior = (torch.tanh(phi_x) ** 2).mean() + (torch.tanh(phi_y) ** 2).mean()
        loss = cfg['eq_weight'] * eq + cfg['prior_weight'] * prior
        stats = {'eq': eq.item(), 'prior': prior.item()}
        if cfg['size_weight'] > 0 and targets is not None and len(targets):
            t = targets.float()
            z, m = reg.zoom_at(t[:, 0].long(), t[:, 2], t[:, 3])
            if m.any():
                size = torch.sqrt(t[m, 4] * w * t[m, 5] * h).clamp_min(1)
                want = torch.log(cfg['target_size'] / size).clamp(-math.log(cfg['max_zoom']), math.log(cfg['max_zoom']))
                size_loss = ((torch.log(z) - want) ** 2).mean()
                loss = loss + cfg['size_weight'] * size_loss
                stats['size'] = size_loss.item()
        stats.update(warp_stats(G.detach()))
        return loss, stats

    def warp_feat(self, x, reg):
        with torch.autocast(device_type=x.device.type, enabled=False):
            return reg.warp(x.float()).to(x.dtype)

    def unwarp_feat(self, x, reg, cache):
        key = tuple(x.shape[-2:])
        with torch.autocast(device_type=x.device.type, enabled=False):
            if key not in cache:
                cache[key] = reg.inverse_grid(x.shape[0], *key)
            return F.grid_sample(x.float(), cache[key][0], mode='bilinear', padding_mode='border',
                                 align_corners=False).to(x.dtype)

    def condition(self, x, regs, caches):
        """Head inputs (list of (B, C_i, h_i, w_i)) + a learned function of every latent DEC's local log magnification."""
        out = []
        for t, conv in zip(x, self.cond):
            key = tuple(t.shape[-2:])
            maps = []
            for name in self.latent:
                cache = caches.setdefault(name, {})
                if key not in cache:
                    with torch.autocast(device_type=t.device.type, enabled=False):
                        cache[key] = regs[name].inverse_grid(t.shape[0], *key)
                maps.append(cache[key][1])
            out.append(t + conv(torch.cat(maps, 1).to(conv.weight.dtype)).to(t.dtype))
        return out + list(x[len(self.cond):])

    @torch.no_grad()
    def preview(self, imgs, targets, tiles=None):
        """What the detector sees for a training batch (input DEC; no random rescaling), with its targets, for plots."""
        training = self.training
        self.train(False)
        try:
            reg = Regions.make(imgs.shape[0], tiles if self.cfg['per_tile'] else None, imgs.device)
            G, _, _ = self._solve('input', reg.crops(imgs.float(), self.cfg['canon_size']))
            reg = reg.with_grid(G)
            return reg.warp(imgs.float()), warp_targets(targets, reg, *imgs.shape[-2:])
        finally:
            self.train(training)


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
def plot_warps(lse, imgs, tiles=None, lines=8):
    """Per DEC, a panel with one row per image: the image the DEC sees with its canonical grid drawn on it (where each
    canonical cell samples from), the local zoom (red: magnified, blue: shrunk, full scale = max_zoom) and the canonical
    image. imgs: (n, 3, H, W) in [0, 1] (RGB); tiles: per image the content rectangle(s) (e.g. without letterbox bars).
    Returns {dec name: (panel, grid overlay of the first image)}, uint8 BGR."""
    training = lse.training
    lse.train(False)
    try:
        cfg = lse.cfg
        x = imgs.float()
        n, _, H, W = x.shape
        base = Regions.make(n, tiles, x.device)
        lim = math.log(cfg['max_zoom'])
        t = torch.linspace(0, 1, 200, device=x.device)
        out = {}
        for name in (['input'] if lse.has_input else []) + lse.latent:
            G, _, _ = lse._solve(name, base.crops(x, cfg['canon_size']))
            reg = base.with_grid(G)
            canon = reg.warp(x)
            _, logm = reg.inverse_grid(n, max(H // 4, 1), max(W // 4, 1))
            zoom = logm.mean(1)  # log geometric magnification at each image location
            rows, first = [], None
            for i in range(n):
                im = np.ascontiguousarray((x[i].permute(1, 2, 0).cpu().numpy()[:, :, ::-1] * 255).clip(0, 255)
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
                for p, label in ((ov, f'{name}: warp grid'), (heat, 'local zoom'), (cv, 'canonical')):
                    cv2.putText(p, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                rows.append(np.hstack([ov, heat, cv]))
                first = ov if first is None else first
            out[name] = (np.vstack(rows), first)
            if name == 'input':
                x = canon  # the stage DECs see the input DEC's canonical image
        return out
    finally:
        lse.train(training)

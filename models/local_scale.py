# Local scale canonicalization for YOLOv7: a deep equilibrium canonicalizer (DEC) that predicts monotone local-scaling
# warps of the input image and / or of backbone feature maps. Port of "Local Scale Equivariance with Latent Deep
# Equilibrium Canonicalizer" (Rahman et al., ICCV 2025, https://github.com/ashiq24/local-scale-equivariance) to
# detection. Enabled with an `lse:` section in the model yaml, e.g. cfg/training/yolov7-tiny-lse-*.yaml; the losses
# and the local-scale augmentation are in utils/local_scale.py.
#
# Warp. Parameters px, py of shape (B, R, R-1) give a monotone R x R grid of sampling positions in [0, 1]
# (softmax + cumsum per row for x, per column for y); px = py = constant is the identity. warp(x, grid) samples
# x at the bilinearly upsampled grid: out(u) = x(grid(u)). The warp maps the image domain onto itself (no crop).
#
# Canonicalizer. One small conv net per warp scores the image seen through the current warp, f(phi) =
# net(warp(image, phi)), and the warp is the fixed point phi* = f(phi*), found with Anderson acceleration
# without gradient, then one differentiable step from phi* (Jacobian-free backpropagation). The reference
# implementation uses torchdeq; this one has no extra dependency and a fixed number of iterations.
#
# Where the warps apply (model yaml):
#   input:  the input image is canonicalized; the network detects in the canonical frame. In eval mode the decoded
#           boxes are mapped back to the input frame here, so test.py / detect.py need no change. In training the
#           targets are mapped into the canonical frame (canonical_targets).
#   stages: [[a, b], ...]: latent canonicalization. The input of layer a is warped and the output of layer b
#           unwarped, so layers a..b see canonical features and everything after b is in the input frame again.
#           Every layer that reads from inside a stage must be in it (checked); only layer b's output leaves it.
#
# Differences from the reference implementation: F.grid_sample instead of a gather-based bilinear sampler (only
# needed there for double backward through the inner optimization, which DEQ mode does not use); exact inverse
# warps by bisection instead of the per-axis approximation; no torchdeq.

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# Warps ----------------------------------------------------------------------------------------------------------------

def _smooth(p, k=5, sigma=0.2):
    # Gaussian blur of the warp parameters (B, R, R-1), as torchvision GaussianBlur((5, 5), 0.2) in the reference
    x = torch.linspace(-(k // 2), k // 2, k, device=p.device, dtype=p.dtype)
    g = torch.exp(-0.5 * (x / sigma) ** 2)
    g = g / g.sum()
    p = F.pad(p.unsqueeze(1), (k // 2,) * 4, mode='reflect')
    p = F.conv2d(F.conv2d(p, g.view(1, 1, 1, k)), g.view(1, 1, k, 1))
    return p.squeeze(1)


def coarse_grid(px, py):
    """Warp parameters px, py (B, R, R-1) -> grid (B, 2, R, R) of sampling positions in [0, 1], channel 0 x
    (increasing along each row), channel 1 y (increasing along each column). Parameters (B, 1, R-1): separable warp,
    the same x warp in every row and y warp in every column (axis-aligned boxes stay axis-aligned)."""
    r = px.shape[-1] + 1
    px, py = px.expand(-1, r, -1), py.expand(-1, r, -1)

    def axis(p):
        w = F.softmax(3 * _smooth(p), dim=-1)
        return torch.cumsum(F.pad(w, (1, 0)), dim=-1)  # 0 ... 1

    return torch.stack((axis(px), axis(py).transpose(-1, -2)), 1)


def identity_grid(n, r, device=None):
    ones = torch.ones(n, r, r - 1, device=device)
    return coarse_grid(ones, ones)


def random_grid(n, r, strength, separable=False, device=None, generator=None):
    """Random local scaling, as random_data_augmentation in the reference: parameters uniform in [0, strength]
    (0.25: neighbouring cells differ in scale by up to ~2x). Non-separable warps start to fold (not invertible)
    above ~0.25."""
    rows = 1 if separable else r
    p = torch.rand(2, n, rows, r - 1, generator=generator) * strength  # CPU: the same warps on every device
    return coarse_grid(p[0].to(device), p[1].to(device))


def sample_grid(grid, pts):
    """Positions the warp samples at points: grid (N, 2, R, R), pts (N, K, 2) as (x, y) in [0, 1] -> (N, K, 2)."""
    out = F.grid_sample(grid, pts.unsqueeze(1) * 2 - 1, mode='bilinear', padding_mode='border', align_corners=True)
    return out[:, :, 0].transpose(1, 2)


def _sample(x, pos):
    # x (B, C, H, W) at positions pos (B, 2, H', W') in [0, 1]
    out = F.grid_sample(x.float(), pos.permute(0, 2, 3, 1) * 2 - 1, mode='bilinear', padding_mode='border',
                        align_corners=True)
    return out.to(x.dtype)


def warp(x, grid):
    """x (B, C, H, W) seen through the warp: out(u) = x(grid(u))."""
    return _sample(x, F.interpolate(grid, size=x.shape[-2:], mode='bilinear', align_corners=True))


def invert_points(grid, pts, rounds=4, steps=16, newton=4):
    """Points u with grid(u) = pts: grid (N, 2, R, R), pts (N, K, 2) -> (N, K, 2). With the other coordinate fixed,
    each coordinate of the warp is monotone, so alternate bisections along x and y (exact in one round for
    separable warps), then Newton steps for the points where strongly non-separable warps leave a residual.
    No gradient."""
    with torch.no_grad():
        grid, pts = grid.float(), pts.float()
        u = pts.clone()
        separable = max(grid[:, 0].diff(dim=1).abs().amax(), grid[:, 1].diff(dim=2).abs().amax()) < 1e-6
        for _ in range(1 if separable else rounds):
            for a in (0, 1):
                lo, hi = torch.zeros_like(u[..., a]), torch.ones_like(u[..., a])
                for _ in range(steps):
                    mid = (lo + hi) / 2
                    v = u.clone()
                    v[..., a] = mid
                    below = sample_grid(grid, v)[..., a] < pts[..., a]
                    lo, hi = torch.where(below, mid, lo), torch.where(below, hi, mid)
                u[..., a] = (lo + hi) / 2
        eps = 1e-3
        for _ in range(newton):
            r = sample_grid(grid, u) - pts
            jx = (sample_grid(grid, u + u.new_tensor([eps, 0])) - sample_grid(grid, u - u.new_tensor([eps, 0]))) / (2 * eps)
            jy = (sample_grid(grid, u + u.new_tensor([0, eps])) - sample_grid(grid, u - u.new_tensor([0, eps]))) / (2 * eps)
            det = jx[..., 0] * jy[..., 1] - jy[..., 0] * jx[..., 1]
            det = torch.where(det.abs() < 1e-8, torch.full_like(det, 1e-8), det)
            step = torch.stack((jy[..., 1] * r[..., 0] - jy[..., 0] * r[..., 1],
                                -jx[..., 1] * r[..., 0] + jx[..., 0] * r[..., 1]), -1) / det[..., None]
            v = (u - step).clamp(0, 1)
            better = (sample_grid(grid, v) - pts).norm(dim=-1) < r.norm(dim=-1)
            u = torch.where(better[..., None], v, u)
    return u


def invert_points_grad(grid, pts):
    # invert_points with a first-order gradient for the grid: u* + (pts - grid(u*)) has the value u* (the residual
    # is ~0) and the gradient -d grid(u*), the implicit gradient with the warp's Jacobian taken as the identity
    u = invert_points(grid.detach(), pts)
    return u + (pts - sample_grid(grid.float(), u))


def inverse_positions(grid, size, res=32):
    """Sampling positions (B, 2, h, w) of the inverse warp at size (h, w): unwarp(warp(x)) = x. Inverted at up to
    res x res points and upsampled (the inverse is as smooth as the warp)."""
    h, w = min(size[0], res), min(size[1], res)
    ys, xs = torch.linspace(0, 1, h, device=grid.device), torch.linspace(0, 1, w, device=grid.device)
    pts = torch.stack(torch.meshgrid(xs, ys, indexing='xy'), -1).view(1, h * w, 2).expand(grid.shape[0], -1, -1)
    pos = invert_points_grad(grid, pts).view(-1, h, w, 2).permute(0, 3, 1, 2)
    return F.interpolate(pos, size=size, mode='bilinear', align_corners=True) if (h, w) != tuple(size) else pos


def unwarp(x, grid):
    """Inverse of warp(): x (B, C, H, W) in the warped frame -> the input frame."""
    return _sample(x, inverse_positions(grid, x.shape[-2:]))


# Boxes ----------------------------------------------------------------------------------------------------------------

def _xywh2xyxy(b):
    return torch.cat((b[..., :2] - b[..., 2:4] / 2, b[..., :2] + b[..., 2:4] / 2), -1)


def _xyxy2xywh(b):
    return torch.cat(((b[..., :2] + b[..., 2:4]) / 2, b[..., 2:4] - b[..., :2]), -1)


def warp_boxes(xyxy, grid, inverse, k=5):
    """Boxes (N, M, 4) normalized x1y1x2y2 under grid (N, 2, R, R), from k points per edge.
    inverse=True: input frame -> warped frame (labels of a warped image): the bounding box of the warped box, which
    is the tight box of a warped object filling its box.
    inverse=False: warped frame -> input frame (detections in the canonical frame): each edge's mean position. Exact
    for separable warps. A non-separable warp shears boxes, and no mapping recovers a box from the bounding box of
    its sheared copy; on random warps (strength 0.1-0.5) the edge means had 20-70% higher IoU with the original box
    than the bounding box did."""
    n, m = xyxy.shape[:2]
    t = torch.linspace(0, 1, k, device=xyxy.device)
    x1, y1, x2, y2 = (xyxy[..., i:i + 1] for i in range(4))  # (N, M, 1)
    xs, ys = x1 + t * (x2 - x1), y1 + t * (y2 - y1)  # (N, M, k)
    pts = torch.cat((torch.stack((xs, y1.expand_as(xs)), -1), torch.stack((xs, y2.expand_as(xs)), -1),
                     torch.stack((x1.expand_as(ys), ys), -1), torch.stack((x2.expand_as(ys), ys), -1)), 2)  # (N, M, 4k, 2)
    pts = pts.view(n, m * 4 * k, 2).float()
    inside = pts.clamp(0, 1)  # outside the image (detections may extend past it) the warp continues as a shift
    if inverse:
        q = (invert_points(grid, inside) + pts - inside).view(n, m, 4 * k, 2)
        return torch.cat((q.amin(2), q.amax(2)), -1)
    q = (sample_grid(grid.float(), inside) + pts - inside).view(n, m, 4, k, 2).mean(3)  # mean of each edge: top, bottom, left, right
    return torch.stack((q[:, :, 2, 0], q[:, :, 0, 1], q[:, :, 3, 0], q[:, :, 1, 1]), -1)


def warp_targets(targets, grid):
    """Labels (n, 6) [image, class, x, y, w, h] (normalized xywh) of images x -> labels of warp(x, grid)
    (grid: (B, 2, R, R), one per image)."""
    if not len(targets):
        return targets
    t = targets.to(grid.device).clone()
    xyxy = _xywh2xyxy(t[:, 2:6].float())
    xyxy = warp_boxes(xyxy[:, None], grid.detach()[t[:, 0].long()], inverse=True)[:, 0]
    t[:, 2:6] = _xyxy2xywh(xyxy).to(t.dtype)
    return t


def unwarp_detections(z, grid, img_hw):
    """Decoded detections z (B, N, 5 + nc), xywh in pixels of the warped image of size img_hw -> pixels of the
    input image."""
    h, w = img_hw
    gain = torch.tensor([w, h, w, h], device=z.device, dtype=torch.float32)
    xyxy = _xywh2xyxy(z[..., :4].float()) / gain
    xyxy = warp_boxes(xyxy, grid, inverse=False) * gain
    z = z.clone()
    z[..., :4] = _xyxy2xywh(xyxy).to(z.dtype)
    return z


# Canonicalizer --------------------------------------------------------------------------------------------------------

def anderson(f, x0, iters, m=5, lam=1e-4, tol=1e-6):
    """Fixed point of f by Anderson acceleration: x0 (B, D) -> (B, D), at most iters evaluations of f. No gradient."""
    n_batch, d = x0.shape
    xs, fs = x0.new_zeros(n_batch, m, d), x0.new_zeros(n_batch, m, d)
    xs[:, 0], fs[:, 0] = x0, f(x0)
    if iters < 2:
        return fs[:, 0]
    xs[:, 1], fs[:, 1] = fs[:, 0], f(fs[:, 0])
    h = x0.new_zeros(n_batch, m + 1, m + 1)
    h[:, 0, 1:] = h[:, 1:, 0] = 1
    y = x0.new_zeros(n_batch, m + 1, 1)
    y[:, 0] = 1
    k = 1
    for k in range(2, iters):
        n = min(k, m)
        g = fs[:, :n] - xs[:, :n]
        h[:, 1:n + 1, 1:n + 1] = g @ g.transpose(1, 2) + lam * torch.eye(n, device=x0.device)[None]
        alpha = torch.linalg.solve(h[:, :n + 1, :n + 1], y[:, :n + 1])[:, 1:n + 1, 0]  # (B, n)
        xs[:, k % m] = (alpha[:, None] @ fs[:, :n])[:, 0]
        fs[:, k % m] = f(xs[:, k % m])
        if (fs[:, k % m] - xs[:, k % m]).norm() / (1e-5 + fs[:, k % m].norm()) < tol:
            break
    return fs[:, k % m]


class DEMBlock(nn.Module):
    # As DEMResidualBlock in the reference: conv + 1x1 projection, instance norm, softplus, max pool
    def __init__(self, c1, c2, k, pool):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, padding=k // 2, padding_mode='reflect')
        self.proj = nn.Conv2d(c1, c2, 1) if c1 != c2 else nn.Identity()
        self.norm = nn.InstanceNorm2d(c2, affine=True)
        self.act = nn.Softplus()
        self.pool = nn.MaxPool2d(pool) if pool > 1 else nn.Identity()

    def forward(self, x):
        return self.pool(self.act(self.norm(self.conv(x) + self.proj(x))))


class Canonicalizer(nn.Module):
    """One warp: phi* = f(phi*), f(phi) = net(warp(image, phi)), as DEMAdapter in the reference."""

    def __init__(self, r, channels, kernels, pools, separable=False):
        super().__init__()
        assert len(channels) == len(kernels) + 1 == len(pools) + 1, 'dem_channels needs one entry more than ' \
                                                                     'dem_kernels and dem_pools'
        self.r, self.separable = r, separable
        self.net = nn.Sequential(*[DEMBlock(c1, c2, k, p) for c1, c2, k, p in
                                   zip(channels[:-1], channels[1:], kernels, pools)],
                                 nn.Conv2d(channels[-1], 2, 1), nn.InstanceNorm2d(2, affine=True),
                                 nn.AdaptiveAvgPool2d(r), nn.Sigmoid())
        nn.init.zeros_(self.net[-3].weight)  # constant output: starts at the identity warp, a pretrained detector
        # is unchanged at the first step (the reference initializes this norm with weight 1)

    def _grid(self, z):
        px, py = z.view(-1, 2, self.r, self.r - 1).unbind(1)
        return coarse_grid(px, py)

    def _f(self, img, z):
        dtype = next(self.net.parameters()).dtype  # FP16 in test.py on GPU
        s = self.net(warp(img, self._grid(z)).to(dtype)).float()  # (B, 2, R, R)
        z = torch.stack((s[:, 0, :, :-1], s[:, 1, 1:, :].transpose(-1, -2)), 1)  # (B, 2, R, R-1)
        if self.separable:  # one x warp for all rows, one y warp for all columns
            z = z.mean(2, keepdim=True).expand_as(z)
        return z.flatten(1)

    def forward(self, img, iters, solver):
        z = torch.ones(img.shape[0], 2 * self.r * (self.r - 1), device=img.device)  # identity
        with torch.no_grad():
            f = lambda z: self._f(img, z)
            if solver == 'anderson':
                z = anderson(f, z, iters)
            else:
                for _ in range(iters):
                    z = f(z)
        return self._grid(self._f(img, z))  # one differentiable step from the fixed point


def _check_stages(stages, layers):
    last = -1
    for a, b in stages:
        assert last < a <= b < len(layers) - 1, f'lse stages {stages}: must be ordered, not overlap and end before ' \
                                                f'the detection layer'
        assert isinstance(layers[a].f, int), f'lse stage {a}-{b}: layer {a} must have one input'
        for m in layers:
            for s in ([m.f] if isinstance(m.f, int) else m.f):
                s = m.i + s if s < 0 else s  # absolute layer index
                if a < m.i <= b:
                    assert a <= s, f'lse stage {a}-{b}: layer {m.i} reads layer {s}, before the stage'
                elif m.i > b:
                    assert not a <= s < b, f'lse stage {a}-{b}: layer {m.i} reads layer {s}, inside the stage ' \
                                           f'(only the last layer {b} may be read after it)'
        last = b


class LocalScale(nn.Module):
    """The model's canonicalizers, from the `lse:` section of the model yaml:
        input: true / false        canonicalize the input image
        stages: [[a, b], ...]      latent canonicalization of layers a..b
        grid: 8 or [8, 6, ...]     coarse warp grid R per warp (input first, then the stages), >= 4
        separable: true            input warp separable (default true: boxes map exactly between the frames)
        separable_stages: false    stage warps separable
        dem_size: 256              the canonicalizers see the image resized to dem_size x dem_size
        dem_channels: [3, 32, 64]  canonicalizer net: channels, kernel sizes and max-pool rates per block
        dem_kernels: [5, 3]
        dem_pools: [4, 2]
        iters: 5                   fixed-point iterations
        solver: anderson           or picard (plain iteration)
    """

    def __init__(self, cfg, layers):
        super().__init__()
        self.input = bool(cfg.get('input', False))
        self.stages = [tuple(int(i) for i in s) for s in cfg.get('stages', [])]
        n = int(self.input) + len(self.stages)
        assert n, 'lse: needs input: true and / or stages'
        _check_stages(self.stages, layers)
        grid = cfg.get('grid', 8)
        grids = list(grid) if isinstance(grid, (list, tuple)) else [grid] * n
        assert len(grids) == n and min(grids) >= 4, f'lse grid {grid}: one size >= 4 per warp ({n} warps)'
        self.size = int(cfg.get('dem_size', 256))
        self.iters = int(cfg.get('iters', 5))
        self.solver = cfg.get('solver', 'anderson')
        assert self.solver in ('anderson', 'picard'), f'lse solver {self.solver}: anderson or picard'
        channels, kernels, pools = cfg.get('dem_channels', [3, 32, 64]), cfg.get('dem_kernels', [5, 3]), \
            cfg.get('dem_pools', [4, 2])
        sep = [bool(cfg.get('separable', True))] * int(self.input) + \
            [bool(cfg.get('separable_stages', False))] * len(self.stages)
        self.canon = nn.ModuleList(Canonicalizer(int(r), channels, kernels, pools, s) for r, s in zip(grids, sep))
        self.starts = {a: k + int(self.input) for k, (a, b) in enumerate(self.stages)}  # layer -> warp index
        self.ends = {b: k + int(self.input) for k, (a, b) in enumerate(self.stages)}
        self.grids, self.img_hw = None, None  # warps of the last batch, (B, 2, R, R) each
        logger.info(f'lse: {"input, " if self.input else ""}stages {self.stages}, grids {grids}, separable {sep}, '
                    f'{self.iters} {self.solver} iterations at {self.size}x{self.size}, '
                    f'{sum(p.numel() for p in self.parameters()) / 1e3:.1f}k params')

    def __getstate__(self):  # deepcopy / pickle (checkpoints, EMA): not the last batch's warps and their graph
        state = self.__dict__.copy()
        state['grids'], state['img_hw'] = None, None
        return state

    def warps(self, x):
        """Warps (B, 2, R, R) for images x (B, 3, H, W) in [0, 1]: input first, then the stages. A stage's
        canonicalizer sees the image in the stage's frame (canonical if the input is canonicalized)."""
        with torch.autocast(device_type=x.device.type, enabled=False):
            img = F.interpolate(x.float(), size=(self.size, self.size), mode='bilinear', align_corners=False,
                                antialias=True)
            grids = []
            for k, canon in enumerate(self.canon):
                grids.append(canon(img, self.iters, self.solver))
                if k == 0 and self.input:
                    img = warp(img, grids[0].detach())
        return grids

    # Warps run in FP32 (outside autocast); feature maps keep their dtype

    def begin(self, x):
        # Model.forward_once, before the first layer: warps for this batch, the canonical input if canonicalized
        self.grids, self.img_hw = self.warps(x), tuple(x.shape[-2:])
        with torch.autocast(device_type=x.device.type, enabled=False):
            return warp(x, self.grids[0]) if self.input else x

    def stage_in(self, i, x):  # before layer i, the first layer of a stage
        with torch.autocast(device_type=x.device.type, enabled=False):
            return warp(x, self.grids[self.starts[i]])

    def stage_out(self, i, x):  # after layer i, the last layer of a stage
        with torch.autocast(device_type=x.device.type, enabled=False):
            return unwarp(x, self.grids[self.ends[i]])

    def end(self, out):
        # Model.forward_once, the detection layer's output: eval-mode boxes back to the input frame
        if self.input and not self.training and isinstance(out, tuple) and torch.is_tensor(out[0]) \
                and out[0].dim() == 3:
            with torch.autocast(device_type=out[0].device.type, enabled=False):
                out = (unwarp_detections(out[0], self.grids[0], self.img_hw),) + tuple(out[1:])
        return out


def get_lse(model):
    """The model's LocalScale, or None (also for models saved before it existed)."""
    model = model.module if hasattr(model, 'module') else model
    return getattr(model, 'lse', None)


def canonical_targets(model, targets):
    """Targets in the frame the model's last forward predicted in (raw outputs): the canonical frame if the input
    is canonicalized, else unchanged."""
    lse = get_lse(model)
    if lse is None or not lse.input:
        return targets
    return warp_targets(targets, lse.grids[0])

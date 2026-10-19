# LSE debugging (README.md section 14): is a canonicalizer stuck in the identity basin, and what warps does it predict?
#
# train.py --lse-debug N writes, in <run>/lse_debug/:
#   status.txt, status.csv  every epoch, per canonicalizer: warp strength (training statistics and on fixed validation
#                           images), solver convergence, how much its parameters moved, and a verdict:
#                             ok         the warps left the identity
#                             identity   warps still ~identity (expected early on, and during det_grad_epochs)
#                             STUCK      identity for `patience` epochs after the warm-up: the identity basin
#                             frozen     its parameters did not change in the epoch (lr_scale 0, frozen layers)
#                             saturated  warps at the max_zoom bound (box mapping less exact, consider prior_weight)
#                           plus "not converged" when the solver's last update stays large.
#   epoch_XXX/<dec>/        every N epochs (and the first and last): per fixed validation image a triplet
#                             img<k>_1_input.jpg   the image the canonicalizer sees
#                             img<k>_2_field.jpg   the predicted warp field: local zoom (red magnified, blue shrunk,
#                                                  full colour at max_zoom), the canonical grid drawn where it samples,
#                                                  arrows from each canonical point to the input point it samples
#                             img<k>_3_warped.jpg  the input warped by the field: what the detector sees
#                             img<k>_triplet.jpg   the three side by side
#                             img<k>_field.npz     the warp knots G (2, R, R), the dense sampling map (canonical pixel
#                                                  -> input pixel) and the log zoom map, for your own checks
# Warnings (STUCK, frozen, saturated, not converged) also go to the training log, with what to try.

import csv
import math

import cv2
import numpy as np
import torch

from utils.lse import forward_points, warp_stats

# verdict thresholds
IDENTITY_DISP = 2e-3   # mean warp displacement (fraction of the region) below which a warp counts as the identity
IDENTITY_ZOOM = 1.05   # ... and strongest local zoom in / out below this
PATIENCE = 3           # identity epochs after the warm-up before STUCK
FROZEN_REL = 1e-7      # relative parameter change in an epoch below which a canonicalizer counts as frozen
SATURATED = 0.9        # zoom_in * zoom_out above this fraction of max_zoom: saturated
STEP_MAX = 0.05        # solver's last update above this: not converged


def debug_images(dataset, img_size, n):
    """n fixed images of a dataset, letterboxed to img_size, and their content rectangles (the regions canonicalized):
    ((n, 3, H, W) in [0, 1] RGB, [rect (1, 4)]), or None."""
    from utils.datasets import letterbox
    ims, rects = [], []
    for f in getattr(dataset, 'img_files', [])[:n]:
        im = cv2.imread(f)
        if im is None:
            continue
        lb, _, (dw, dh) = letterbox(im, img_size, auto=False)
        ims.append(lb[:, :, ::-1])
        H, W = lb.shape[:2]
        rects.append(torch.tensor([[dw / W, dh / H, 1 - dw / W, 1 - dh / H]]))
    if not ims:
        return None
    return torch.from_numpy(np.stack(ims).transpose(0, 3, 1, 2).copy()).float() / 255, rects


def dec_views(model, imgs, tiles=None):
    """One eval pass of a DEC model: per DEC layer (name, layer, regions with warps, the images it sees (n, 3, H, W),
    the canonical images (n, 3, H, W)). Leaves the model in its previous mode."""
    from models.dec import DEC
    ctx, training = model.dec_ctx, model.training
    model.eval()
    ctx.keep = True  # keep the warps and the images the DECs see
    try:
        with torch.no_grad():
            model(imgs.float(), lse_tiles=tiles)
            out = []
            for m in model.model:
                if isinstance(m, DEC):
                    reg, seen = ctx.regs[m.i], ctx.seen[m.i]
                    out.append((m.name, m, reg, seen, reg.warp(seen)))
            return out
    finally:
        ctx.keep = False
        ctx.clear()
        model.train(training)


def to_bgr(t):
    """(3, H, W) RGB in [0, 1] -> uint8 BGR image."""
    return np.ascontiguousarray((t.permute(1, 2, 0).cpu().numpy()[:, :, ::-1] * 255).clip(0, 255).astype(np.uint8))


def field_image(im, reg, i, max_zoom, grid, logm, lines=8, arrows=12):
    """The warp field of image i: local zoom heatmap over the image, the canonical grid where it samples, and arrows
    canonical point -> sampled input point. grid, logm: reg.sampling_grid / inverse_grid of all images. Also returns
    the dense sampling map (H, W, 2, input pixels) and the log zoom map."""
    H, W = im.shape[:2]
    dev = reg.G.device
    lim = math.log(max_zoom)
    zoom = logm[i].mean(0)  # log geometric magnification at each image pixel (> 0: magnified)
    z = ((zoom.cpu().numpy() / lim).clip(-1, 1) * 127 + 128).astype(np.uint8)
    out = cv2.addWeighted(im, 0.35, cv2.applyColorMap(z, cv2.COLORMAP_JET), 0.65, 0)
    inside = np.zeros((H, W), bool)  # outside the canonicalized regions (e.g. letterbox bars): the plain image
    for x1, y1, x2, y2 in reg.rect[reg.b == i].tolist():
        inside[round(y1 * H):round(y2 * H), round(x1 * W):round(x2 * W)] = True
    out[~inside] = im[~inside]
    grid = grid[i]  # canonical pixel -> input, [-1, 1]
    sample = ((grid + 1) / 2 * torch.tensor([W, H], device=dev) - 0.5).cpu().numpy()  # input pixels
    t = torch.linspace(0, 1, 200, device=dev)
    for k in torch.nonzero(reg.b == i)[:, 0].tolist():  # each region of the image
        kk = torch.full_like(t, k, dtype=torch.long)
        x1, y1, x2, y2 = reg.rect[k].tolist()
        for c in torch.linspace(0, 1, lines + 1, device=dev):
            for u, v in ((t, torch.full_like(t, c)), (torch.full_like(t, c), t)):  # canonical rows / columns
                px, py = forward_points(reg.G, kk, u, v)
                pts = torch.stack([(x1 + px * (x2 - x1)) * W, (y1 + py * (y2 - y1)) * H], -1)
                cv2.polylines(out, [pts.cpu().numpy().round().astype(np.int32)], False, (255, 255, 255), 1, cv2.LINE_AA)
    step = max(min(H, W) // arrows, 1)
    for yy in range(step // 2, H, step):
        for xx in range(step // 2, W, step):
            sx, sy = sample[yy, xx]
            if abs(sx - xx) + abs(sy - yy) >= 1:
                cv2.arrowedLine(out, (xx, yy), (int(round(sx)), int(round(sy))), (0, 0, 0), 1, cv2.LINE_AA,
                                tipLength=0.3)
    return out, sample, zoom.cpu().numpy()


class LSEDebugger:
    """Identity-basin checks every epoch and warp triplets every `every` epochs, in save_dir / 'lse_debug'."""

    def __init__(self, save_dir, every, vis, cfg, logger, epochs):
        self.dir = save_dir / 'lse_debug'
        self.dir.mkdir(parents=True, exist_ok=True)
        self.every, self.vis, self.cfg, self.log, self.epochs = every, vis, cfg, logger, epochs
        self.start, self.identity = {}, {}
        self.csv = self.dir / 'status.csv'

    @staticmethod
    def _decs(model):
        from models.dec import DEC
        return [m for m in model.model if isinstance(m, DEC)]

    def epoch_start(self, model):
        """Snapshot of each canonicalizer's parameters (the training model, not the EMA)."""
        self.start = {m.name: torch.cat([p.detach().flatten() for p in m.canon.parameters()]).clone()
                      for m in self._decs(model)}

    def epoch_end(self, epoch, model, ema_model, stats):
        """Verdicts from this epoch's training statistics (floats, models/dec.py names) and parameter change; warps on
        the fixed images from the EMA model; triplets every `every` epochs."""
        views = {v[0]: v for v in dec_views(ema_model, self.vis[0], self.vis[1])} if self.vis is not None else {}
        rows, warn = [], []
        for m in self._decs(model):
            name = m.name
            p = torch.cat([q.detach().flatten() for q in m.canon.parameters()])
            p0 = self.start.get(name, p)
            rel = ((p - p0).norm() / p0.norm().clamp_min(1e-12)).item()
            head = m.canon.head.weight.detach().norm().item()
            s = {k.split('/', 1)[1]: v for k, v in stats.items() if k.startswith(f'{name}/')}
            v = views.get(name)
            vs = {k: float(x) for k, x in warp_stats(v[2].G).items()} if v is not None else {}
            disp, zin, zout = s.get('disp', vs.get('disp', 0.)), s.get('zoom_in', 1.), s.get('zoom_out', 1.)
            ident = disp < IDENTITY_DISP and zin < IDENTITY_ZOOM and zout < IDENTITY_ZOOM
            warm = epoch < self.cfg['det_grad_epochs']
            self.identity[name] = self.identity.get(name, 0) + 1 if ident and not warm else 0
            flags = []
            if self.cfg['lr_scale'] == 0 or rel < FROZEN_REL:
                verdict = 'frozen'
                if self.cfg['lr_scale'] != 0:
                    warn.append(f'{name}: canonicalizer parameters did not change in epoch {epoch} (frozen layers? '
                                f'--freeze counts DEC layers too)')
            elif self.identity[name] >= PATIENCE:
                verdict = 'STUCK'
                warn.append(f'{name}: warps at the identity for {self.identity[name]} epochs (disp {disp:.2g}, zoom '
                            f'in/out {zin:.3f}/{zout:.3f}): identity basin. Try warp_aug 0.5, det_grad_epochs 2-3, '
                            f'lr_scale 10-30, size_weight 0.5 with target_size = obj_px (lse.txt)')
            elif ident:
                verdict = 'identity'
            else:
                verdict = 'ok'
            if zin * zout >= SATURATED * m.canon.max_zoom:
                flags.append('saturated')
                warn.append(f'{name}: warps at the max_zoom bound (zoom in x out {zin * zout:.2f} of '
                            f'{m.canon.max_zoom}): box mapping is less exact; consider prior_weight or a lower max_zoom')
            if s.get('step', 0.) > STEP_MAX:
                flags.append('not converged')
                warn.append(f'{name}: solver not converged (last update {s["step"]:.3g}): raise iters or check warps')
            rows.append({'epoch': epoch, 'dec': name, 'verdict': verdict, 'flags': ' '.join(flags),
                         'disp': disp, 'zoom_in': zin, 'zoom_out': zout, 'step': s.get('step', float('nan')),
                         'eq': s.get('eq', float('nan')), 'val_disp': vs.get('disp', float('nan')),
                         'val_zoom_in': vs.get('zoom_in', float('nan')), 'val_zoom_out': vs.get('zoom_out', float('nan')),
                         'param_change': rel, 'head_norm': head, 'identity_epochs': self.identity[name]})
        self._write(rows)
        for w in warn:
            self.log.warning(f'lse_debug: {w}')
        if views and (epoch % self.every == 0 or epoch == self.epochs - 1):
            self._triplets(epoch, views)
        return rows

    def _write(self, rows):
        new = not self.csv.exists()
        with open(self.csv, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            if new:
                w.writeheader()
            w.writerows(rows)
        with open(self.dir / 'status.txt', 'a') as f:
            for r in rows:
                f.write(f"epoch {r['epoch']:3d} {r['dec']:8s} {r['verdict']:9s} {r['flags']:24s} disp {r['disp']:.3g} "
                        f"zoom in/out {r['zoom_in']:.3f}/{r['zoom_out']:.3f} (val {r['val_zoom_in']:.3f}/"
                        f"{r['val_zoom_out']:.3f}) step {r['step']:.3g} eq {r['eq']:.3g} param change "
                        f"{r['param_change']:.3g} head norm {r['head_norm']:.3g}\n")

    def _triplets(self, epoch, views):
        for name, m, reg, seen, canon in views.values():
            d = self.dir / f'epoch_{epoch:03d}' / name
            d.mkdir(parents=True, exist_ok=True)
            n, _, H, W = seen.shape
            with torch.no_grad():
                grid, logm = reg.sampling_grid(n, H, W, torch.float32), reg.inverse_grid(n, H, W)[1]
            for i in range(n):
                im, wp = to_bgr(seen[i]), to_bgr(canon[i])
                field, sample, logzoom = field_image(im, reg, i, m.canon.max_zoom, grid, logm)
                for p, label in ((im, 'input'), (field, 'warp field'), (wp, 'warped (detector input)')):
                    cv2.putText(p, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imwrite(str(d / f'img{i}_1_input.jpg'), im)
                cv2.imwrite(str(d / f'img{i}_2_field.jpg'), field)
                cv2.imwrite(str(d / f'img{i}_3_warped.jpg'), wp)
                cv2.imwrite(str(d / f'img{i}_triplet.jpg'), np.hstack([im, field, wp]))
                k = torch.nonzero(reg.b == i)[:, 0]
                np.savez_compressed(d / f'img{i}_field.npz', G=reg.G[k].cpu().numpy(), rects=reg.rect[k].cpu().numpy(),
                                    sample_px=sample.astype(np.float32), log_zoom=logzoom.astype(np.float32))

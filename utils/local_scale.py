# Training side of local scale canonicalization (models/local_scale.py, README.md section 14): random local-scale
# augmentation and the canonicalizer losses. Settings are hyp keys (defaults below), so any hyp file works:
#
#   lse_aug       0     chance a training image is warped by a random local scaling (labels follow). Works with any
#                       model, also stock ones: the augmentation baseline of the paper
#   lse_strength  0.25  random warp strength (parameters uniform in [0, strength], as warping_strength in the
#                       reference); non-separable warps fold above ~0.25
#   lse_grid      8     random warp grid (warping_resolution in the reference)
#   lse_separable 1     random warps separable (1) or not (0)
#   lse_canon     1.0   canonicalization consistency (needs an lse model): an image and a randomly warped copy must
#                       give the same canonical image. Cheap: runs the canonicalizers only
#   lse_equi      0.0   equivariance of the objectness map (needs an lse model): the batch is run again warped, and
#                       its objectness must equal the warped objectness of the original. Doubles the training forward
#   lse_prior     0.0   pull the warps towards the identity
#   lse_lr_scale  1.0   learning rate of the canonicalizers relative to lr0 (train.py)
#
# Canonicalization consistency. x' = warp(x, A) for a random warp A; the canonical images warp(x, G(x)) and
# warp(x', G(x')) are equal when G(x) = A o G(x'), i.e. sampling A at the positions of G(x'). The loss is the mean
# squared difference of the two position grids, in coarse grid cells. A stage warp after a canonicalized input sees
# the canonical frame, where the relation is G(x) = G(x') (assuming the input warp canonicalized both).
#
# Objectness equivariance. Detection outputs are not equivariant under local scaling (an object scaled by the warp
# moves to other anchors or levels), so the loss compares a scale-free map: the maximum objectness over anchors and
# levels at the finest level's resolution, in the input frame (unwarped if the input is canonicalized). The map of x'
# must equal the map of x warped by A (Bernoulli KL divergence).

import logging

import torch
import torch.nn.functional as F

from models.local_scale import get_lse, random_grid, identity_grid, sample_grid, warp, unwarp, warp_targets

logger = logging.getLogger(__name__)

DEFAULTS = {'lse_aug': 0.0, 'lse_strength': 0.25, 'lse_grid': 8, 'lse_separable': 1, 'lse_canon': 1.0,
            'lse_equi': 0.0, 'lse_prior': 0.0, 'lse_lr_scale': 1.0}


def objectness_map(pred):
    """Raw outputs [(B, na, h, w, no)] -> (B, 1, h, w) maximum objectness over anchors and levels, at the finest
    level's resolution."""
    size = max((p.shape[2:4] for p in pred), key=lambda s: s[0] * s[1])
    maps = [F.interpolate(p[..., 4].float().sigmoid().amax(1, keepdim=True), size=size, mode='nearest') for p in pred]
    return torch.cat(maps, 1).amax(1, keepdim=True)


def bernoulli_kl(p, q, eps=1e-6):
    p, q = p.clamp(eps, 1 - eps), q.clamp(eps, 1 - eps)
    return (p * (p / q).log() + (1 - p) * ((1 - p) / (1 - q)).log()).mean()


class LocalScaleLoss:
    def __init__(self, hyp, model):
        self.h = {k: type(v)(hyp.get(k, v)) for k, v in DEFAULTS.items()}
        self.lse = get_lse(model)
        if self.lse is None:  # stock model: the augmentation only
            unused = [k for k in ('lse_canon', 'lse_equi', 'lse_prior') if hyp.get(k, 0) > 0]
            if unused:
                logger.warning(f'WARNING: {", ".join(unused)} ignored: the model has no lse: section')
            self.h['lse_canon'] = self.h['lse_equi'] = self.h['lse_prior'] = 0.0
        self.A = None  # random warps of the extra (equivariance) half of the batch
        self.items = torch.zeros(3)  # canon, equi, prior (unweighted)

    @property
    def active(self):  # anything to do in the training loop
        return self.lse is not None or self.h['lse_aug'] > 0

    def _random(self, n, device):
        return random_grid(n, int(self.h['lse_grid']), self.h['lse_strength'], bool(self.h['lse_separable']), device)

    def augment(self, imgs, targets):
        """Random local scaling of a fraction lse_aug of the batch, labels (n, 6) warped with it."""
        if self.h['lse_aug'] <= 0:
            return imgs, targets
        n = imgs.shape[0]
        apply = torch.rand(n) < self.h['lse_aug']
        if not apply.any():
            return imgs, targets
        grids = identity_grid(n, int(self.h['lse_grid']), imgs.device)
        grids[apply] = self._random(int(apply.sum()), imgs.device)
        imgs = imgs.clone()
        imgs[apply] = warp(imgs[apply], grids[apply])
        rows = apply[targets[:, 0].long()]  # labels of warped images
        targets = targets.clone()
        targets[rows] = warp_targets(targets[rows], grids).to(targets.device)
        return imgs, targets

    def inputs(self, imgs):
        """The model's input: the batch, plus a randomly warped copy for the equivariance loss."""
        self.A = None
        if self.h['lse_equi'] <= 0:
            return imgs
        self.A = self._random(imgs.shape[0], imgs.device)
        return torch.cat((imgs, warp(imgs, self.A)), 0)

    def __call__(self, imgs, pred, targets):
        """imgs: the batch (without the warped copy); pred: raw outputs of model(inputs(imgs)); targets (n, 6).
        Returns (pred of imgs, targets in the frame pred is in, loss scaled by batch size like the detection loss)."""
        lse, n = self.lse, imgs.shape[0]
        device = pred[0].device
        if lse is None:
            return pred, targets, torch.zeros(1, device=device)
        grids = lse.grids
        canon = equi = torch.zeros((), device=device)

        if self.A is not None:  # the second half of the batch is warp(imgs, A)
            heat = objectness_map(pred)
            if lse.input:
                heat = unwarp(heat, grids[0])
            equi = bernoulli_kl(warp(heat[:n], self.A), heat[n:])
            pred = [p[:n] for p in pred]
            A, grids_x, grids_xa = self.A, [g[:n] for g in grids], [g[n:] for g in grids]
        else:
            A, grids_x, grids_xa = None, grids, None

        if self.h['lse_canon'] > 0:
            if A is None:  # canonicalizers only, on a warped copy
                A = self._random(n, imgs.device)
                grids_xa = lse.warps(warp(imgs, A))
            u = torch.stack(torch.meshgrid(torch.linspace(0, 1, 16, device=device),
                                           torch.linspace(0, 1, 16, device=device), indexing='xy'), -1)
            u = u.view(1, -1, 2).expand(n, -1, -1)
            for k, (g, ga) in enumerate(zip(grids_x, grids_xa)):
                p, pa = sample_grid(g.float(), u), sample_grid(ga.float(), u)
                if k == 0 or not lse.input:  # warp sees the input frame: G(x) = A o G(x')
                    pa = sample_grid(A, pa)
                canon = canon + ((p - pa) ** 2).sum(-1).mean() * (g.shape[-1] - 1) ** 2  # in coarse cells
            canon = canon / len(grids_x)

        prior = torch.zeros((), device=device)
        for g in grids_x:  # distance from the identity, in coarse cells (logged even when not weighted)
            prior = prior + ((g.float() - identity_grid(1, g.shape[-1], device)) ** 2).sum(1).mean() * \
                (g.shape[-1] - 1) ** 2
        prior = prior / len(grids_x)

        if lse.input:  # the detection loss compares the canonical-frame outputs with canonical-frame targets
            targets = warp_targets(targets, grids_x[0]).to(targets.device)
        self.items = torch.stack((canon, equi, prior)).detach().cpu()
        loss = self.h['lse_canon'] * canon + self.h['lse_equi'] * equi + self.h['lse_prior'] * prior
        return pred, targets, loss.view(1) * n

# Knowledge distillation from a larger YOLOv7 (teacher) into a smaller one (student), e.g. yolov7 / yolov7x / W6 / E6
# -> yolov7-tiny. Enabled with train.py / train_aux.py --teacher <weights.pt>.
#
# Response (output) distillation on the detection heads. For every student output level, the teacher level with the
# same stride is paired cell by cell and anchor by anchor (both see the same augmented batch):
#   - objectness: BCE between student logits and the teacher's objectness probabilities, on every cell
#   - class:      BCE between student logits and the teacher's class probabilities, weighted by teacher objectness
#   (minus the teacher's own entropy, a constant: the terms are KL divergences, 0 when the student matches the teacher)
#   - box:        squared error of the box centre (offset in its cell) and log width / height, in grid units,
#                 weighted by teacher objectness. Not 1 - IoU: IoU has a kink where boxes coincide, so its gradient
#                 does not vanish when the student matches the teacher
# The terms use the detection loss's own gains (hyp box / obj / cls) and level balance, so --distill-weight 1 makes
# distillation about as strong as the ground-truth loss. Total loss = detection loss + distill_weight * distill loss.
#
# Pairing anchor by anchor needs the same anchors in both models: the student takes the teacher's anchors (and
# autoanchor is skipped). Teacher and student must predict the same classes in the same order.

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.experimental import attempt_load
from utils.torch_utils import is_parallel

logger = logging.getLogger(__name__)


def _head(model):
    return (model.module if is_parallel(model) else model).model[-1]  # Detect / IDetect / IAuxDetect


class Distiller:
    MIN_TEACHER_OBJ = 0.01  # box / class terms skip cells where the teacher's objectness is below this (weights ~0)

    def __init__(self, weights, student, hyp, device, weight=1.0):
        self.teacher = attempt_load(weights, map_location=device)  # FP32, fused, eval
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        t, s = _head(self.teacher), _head(student)
        assert t.nc == s.nc, f'teacher predicts {t.nc} classes, student {s.nc}: they must predict the same classes'
        assert t.na == s.na, f'teacher has {t.na} anchors per level, student {s.na}'
        t_strides, s_strides = [int(x) for x in t.stride], [int(x) for x in s.stride]
        missing = [x for x in s_strides if x not in t_strides]
        assert not missing, f'student output strides {missing} have no teacher level (teacher strides {t_strides})'
        self.levels = [t_strides.index(x) for x in s_strides]  # teacher level of each student level
        self.student_head = s
        self.nl, self.nc, self.na = s.nl, s.nc, s.na
        self.stride = max(t_strides)  # input sizes must be multiples of this for the teacher
        self.balance = {3: [4.0, 1.0, 0.4]}.get(self.nl, [4.0, 1.0, 0.25, 0.06, .02])  # as ComputeLoss
        self.hyp, self.weight = hyp, weight  # hyp is read at call time: train.py scales the gains after this
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        t_names, s_names = getattr(self.teacher, 'names', None), getattr(student, 'names', None)
        if t_names and s_names and list(t_names) != list(s_names):
            logger.warning(f'WARNING: teacher class names {list(t_names)} differ from the student\'s {list(s_names)}; '
                           f'distillation pairs classes by index')
        logger.info(f'distillation: teacher {weights} ({sum(p.numel() for p in self.teacher.parameters()) / 1e6:.1f}M '
                    f'params, strides {t_strides}), student levels {s_strides} <- teacher levels '
                    f'{[t_strides[i] for i in self.levels]}, weight {weight}')

    def sync_anchors(self, student):
        # Give the student the teacher's anchors, so predictions pair anchor by anchor (anchors are in grid units,
        # equal between paired levels since their strides are equal)
        t, s = _head(self.teacher), _head(student)
        s.anchors[:] = t.anchors[self.levels].to(s.anchors)
        s.anchor_grid[:] = t.anchor_grid[self.levels].to(s.anchor_grid)

    def _kl(self, s_logits, t_logits):
        # BCE(student, teacher probs) - teacher entropy: Bernoulli KL divergence, same gradient as the BCE
        t = t_logits.sigmoid()
        return self.bce(s_logits, t) - self.bce(t_logits, t)

    def __call__(self, imgs, preds):
        """imgs: the batch the student saw; preds: the student's raw outputs (train mode; aux heads after the first nl
        levels are ignored). Returns (weighted loss scaled by batch size like the detection loss, (box, obj, cls))."""
        with torch.no_grad():
            t_preds = self.teacher(imgs)[1]  # eval mode: (detections, raw outputs of the main levels)
        t_anchors, s_anchors = _head(self.teacher).anchors, self.student_head.anchors
        device = preds[0].device
        lbox, lobj, lcls = (torch.zeros(1, device=device) for _ in range(3))
        for i, ps in enumerate(preds[:self.nl]):
            pt = t_preds[self.levels[i]].float()
            ps = ps.float()
            assert ps.shape == pt.shape, f'student output {tuple(ps.shape)} != teacher output {tuple(pt.shape)}'
            t_obj = pt[..., 4].sigmoid()
            lobj += self.balance[i] * self._kl(ps[..., 4], pt[..., 4]).mean()

            m = t_obj > self.MIN_TEACHER_OBJ
            if m.any():
                w = t_obj[m]
                def box(p, anchors):  # centre offset in its cell and log size, grid units (paired cells share the grid)
                    a = anchors.view(1, self.na, 1, 1, 2).expand(*p.shape[:4], 2)[m]
                    p = p[m]  # wh = (2 sigmoid)^2 * anchor, so log wh = 2 (log 2 + log sigmoid) + log anchor
                    return torch.cat((p[:, :2].sigmoid() * 2. - 0.5, 2 * (0.6931472 + F.logsigmoid(p[:, 2:4])) + a.log()), 1)

                lbox += (w * ((box(ps, s_anchors[i]) - box(pt, t_anchors[self.levels[i]])) ** 2).sum(1)).sum() / w.sum()
                if self.nc > 1:
                    lcls += (w * self._kl(ps[..., 5:][m], pt[..., 5:][m]).mean(1)).sum() / w.sum()

        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = imgs.shape[0]
        return (lbox + lobj + lcls) * bs * self.weight, torch.cat((lbox, lobj, lcls)).detach()

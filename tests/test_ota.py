# ComputeLossOTA's SimOTA assignment (utils/loss.py: simota_match): the same matching as upstream YOLOv7, without the
# memory blow-up, on any device, with correct pixel boxes for non-square images.

import torch
import torch.nn.functional as F
import yaml

from conftest import ROOT
from models.yolo import Model
from utils.loss import ComputeLossOTA, assign_similarity, simota_match


def upstream_match(txyxy, pxyxys, p_cls, p_obj, tcls, nc, topk=10):
    """Upstream YOLOv7's ComputeLossOTA.build_targets matching, verbatim apart from variable names."""
    iou = assign_similarity(txyxy, pxyxys, {})
    iou_loss = -torch.log(iou + 1e-8)
    top_k, _ = torch.topk(iou, min(topk, iou.shape[1]), dim=1)
    dynamic_ks = torch.clamp(top_k.sum(1).int(), min=1)
    num_gt = len(tcls)
    gt_cls = F.one_hot(tcls, nc).float().unsqueeze(1).repeat(1, pxyxys.shape[0], 1)
    y = (p_cls.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_() *
         p_obj.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()).sqrt_()
    cost = F.binary_cross_entropy_with_logits(torch.log(y / (1 - y)), gt_cls, reduction='none').sum(-1) + 3.0 * iou_loss
    mm = torch.zeros_like(cost)
    for g in range(num_gt):
        mm[g][torch.topk(cost[g], k=dynamic_ks[g].item(), largest=False)[1]] = 1.0
    multi = mm.sum(0) > 1
    if multi.sum() > 0:
        _, best = torch.min(cost[:, multi], dim=0)
        mm[:, multi] *= 0.0
        mm[best, multi] = 1.0
    fg = mm.sum(0) > 0.0
    return fg, mm[:, fg].argmax(0)


def dense_scene(seed, t=150, n=1500, nc=3, duplicates=True):
    g = torch.Generator().manual_seed(seed)
    c, wh = torch.rand(t, 2, generator=g) * 600, torch.rand(t, 2, generator=g) * 40 + 3
    txyxy = torch.cat([c - wh / 2, c + wh / 2], 1)
    j = torch.randint(0, t, (n,), generator=g)
    pc, pwh = c[j] + torch.randn(n, 2, generator=g) * 8, wh[j] * (torch.rand(n, 2, generator=g) + 0.5)
    pxyxys = torch.cat([pc - pwh / 2, pc + pwh / 2], 1)
    p_cls, p_obj = torch.randn(n, nc, generator=g) * 2, torch.randn(n, 1, generator=g) * 2
    if duplicates:  # find_3_positive lists a cell / anchor once per target near it: identical candidates
        d = torch.randint(0, n, (n // 5,), generator=g)
        pxyxys, p_cls, p_obj = torch.cat([pxyxys, pxyxys[d]]), torch.cat([p_cls, p_cls[d]]), torch.cat([p_obj, p_obj[d]])
    return txyxy, pxyxys, p_cls, p_obj, torch.randint(0, nc, (t,), generator=g)


def test_matching_equals_upstream():
    for seed in range(20):
        txyxy, pxyxys, p_cls, p_obj, tcls = dense_scene(seed)
        fo, mo = upstream_match(txyxy, pxyxys, p_cls, p_obj, tcls, 3)
        fn, mn = simota_match(txyxy, pxyxys, p_cls, p_obj, tcls, {})
        assert torch.equal(fo, fn) and torch.equal(mo, mn), seed


def test_saturated_scores_do_not_give_nan():
    # under AMP an fp16 sigmoid of 1 made upstream's cost inf / NaN; the new cost stays finite and still matches
    txyxy, pxyxys, p_cls, p_obj, tcls = dense_scene(0, t=20, n=200, duplicates=False)
    p_obj = torch.full_like(p_obj, 30.0).half()  # sigmoid == 1 in fp16
    fg, matched = simota_match(txyxy, pxyxys, p_cls, p_obj, tcls, {})
    assert fg.any() and len(matched) == int(fg.sum())


def build(nc=2):
    with open(ROOT / 'data' / 'hyp.scratch.tiny.yaml') as f:
        hyp = yaml.safe_load(f)
    torch.manual_seed(0)
    m = Model(str(ROOT / 'cfg' / 'training' / 'yolov7-tiny.yaml'), ch=3, nc=nc)
    m.hyp, m.gr, m.nc = hyp, 1.0, nc
    return m.train()


def test_no_targets_runs_on_cpu():
    # upstream created the empty per-level results on 'cuda:0': failed on CPU and on DDP ranks other than 0
    m = build()
    imgs = torch.rand(2, 3, 128, 128)
    loss, items = ComputeLossOTA(m)(m(imgs), torch.zeros(0, 6), imgs)
    assert torch.isfinite(loss).all() and items[0] == 0  # no box loss without targets


def test_non_square_images_use_width_and_height(monkeypatch):
    # target boxes in pixels: x, w scaled by the width and y, h by the height (upstream used the height for all four)
    import utils.loss as L
    seen = []
    real = L.simota_match
    monkeypatch.setattr(L, 'simota_match', lambda txyxy, *a, **k: seen.append(txyxy.clone()) or real(txyxy, *a, **k))
    m = build()
    imgs = torch.rand(1, 3, 128, 256)  # H 128, W 256
    targets = torch.tensor([[0, 0, 0.5, 0.5, 0.25, 0.25]])
    ComputeLossOTA(m).build_targets(m(imgs), targets, imgs)
    assert torch.allclose(seen[0], torch.tensor([[96.0, 48.0, 160.0, 80.0]]))  # centre (128, 64), size 64 x 32


def test_assignment_builds_no_autograd_graph():
    m = build()
    imgs = torch.rand(1, 3, 128, 128)
    out = ComputeLossOTA(m).build_targets(m(imgs), torch.tensor([[0, 0, 0.5, 0.5, 0.2, 0.2]]), imgs)
    assert not any(x.requires_grad for level in out for x in level)

# Model configs: yolov7-tiny and the P2 variant build, run, train a loss step, and take pretrained weights.

import pytest
import torch
import yaml

from conftest import ROOT
from models.yolo import Model
from utils.loss import ComputeLoss, ComputeLossOTA, obj_balance
from utils.torch_utils import intersect_dicts

TINY, P2 = 'cfg/training/yolov7-tiny.yaml', 'cfg/training/yolov7-tiny-p2.yaml'


@pytest.fixture(scope='module')
def hyp():
    with open(ROOT / 'data' / 'hyp.scratch.tiny.yaml') as f:
        return yaml.safe_load(f)


def build(cfg, hyp, nc=2):
    m = Model(str(ROOT / cfg), ch=3, nc=nc)
    m.hyp, m.gr, m.nc = hyp, 1.0, nc
    return m


@pytest.mark.parametrize('cfg, strides, balance', [(TINY, [8, 16, 32], [4.0, 1.0, 0.4]),
                                                   (P2, [4, 8, 16, 32], [4.0, 4.0, 1.0, 0.4])])
def test_build_forward_and_loss(cfg, strides, balance, hyp):
    m = build(cfg, hyp)
    det = m.model[-1]
    assert det.stride.tolist() == strides and obj_balance(det)[:det.nl] == balance
    x = torch.rand(2, 3, 256, 256)
    m.train()
    targets = torch.tensor([[0, 0, 0.5, 0.5, 0.1, 0.1], [1, 1, 0.3, 0.3, 0.05, 0.05]])
    for loss_cls in (ComputeLoss, ComputeLossOTA):  # train.py: OTA unless hyp loss_ota is 0
        out = m(x)
        assert [o.shape[2] for o in out] == [256 // s for s in strides]
        loss = loss_cls(m)(out, targets, x)[0] if loss_cls is ComputeLossOTA else loss_cls(m)(out, targets)[0]
        assert torch.isfinite(loss).all()
        loss.backward()
    m.eval()
    with torch.no_grad():
        y = m(x)[0]
    assert y.shape == (2, 3 * sum((256 // s) ** 2 for s in strides), 5 + 2)


def test_p2_takes_every_pretrained_layer_but_the_new_ones(hyp):
    tiny, p2 = build(TINY, hyp), build(P2, hyp)
    stock = [k for k in tiny.state_dict() if not k.startswith('model.77.')]  # all but the detection layer
    moved = intersect_dicts(tiny.state_dict(), p2.state_dict())
    assert set(stock) <= set(moved)


def test_nwd_values():
    from utils.general import bbox_iou, bbox_nwd
    a = torch.tensor([[10.0, 10, 6, 6]])
    shifted = lambda dx: torch.tensor([[10.0 + dx, 10, 6, 6]])  # noqa: E731
    assert torch.isclose(bbox_nwd(a, a), torch.tensor(1.0), atol=1e-3)
    assert bbox_nwd(a, shifted(2)).item() > 0.85 > bbox_iou(a[0], shifted(2), x1y1x2y2=False).item()  # 0.86 vs 0.50
    assert bbox_iou(a[0], shifted(8), x1y1x2y2=False).item() == 0 and bbox_nwd(a, shifted(8)).item() > 0.5
    assert torch.isclose(bbox_nwd(a, shifted(2), c=6.4), bbox_nwd(a, shifted(2)) ** 2)  # exp(-d / c)


def test_box_loss_nwd_blend(hyp):
    from utils.general import bbox_iou, bbox_nwd
    from utils.loss import box_loss
    p = torch.tensor([[0.5, 0.5, 1.0, 1.2], [0.4, 0.6, 0.5, 0.5]], requires_grad=True)
    t = torch.tensor([[0.6, 0.5, 1.1, 1.0], [0.5, 0.5, 0.5, 0.5]])
    stock, iou = box_loss(p, t, 8, {})
    assert torch.allclose(stock, (1 - bbox_iou(p.T, t, x1y1x2y2=False, CIoU=True)).mean())  # nwd off: unchanged
    blend, iou2 = box_loss(p, t, 8, {'nwd': 0.5, 'nwd_c': 12.8})
    ref = 0.5 * (1 - bbox_iou(p.T, t, x1y1x2y2=False, CIoU=True)) + 0.5 * (1 - bbox_nwd(p * 8, t * 8, 12.8))
    assert torch.allclose(blend, ref.mean()) and torch.equal(iou, iou2)  # objectness target stays CIoU
    blend.backward()
    assert torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0


@pytest.mark.parametrize('cfg', [TINY, P2])
def test_training_step_with_nwd(cfg, hyp):
    m = build(cfg, {**hyp, 'nwd': 0.5})
    m.train()
    x = torch.rand(2, 3, 256, 256)
    targets = torch.tensor([[0, 0, 0.5, 0.5, 0.02, 0.02], [1, 1, 0.3, 0.3, 0.05, 0.05]])
    loss = ComputeLossOTA(m)(m(x), targets, x)[0]
    loss.backward()
    assert torch.isfinite(loss).all()


def test_wise_iou_v3():
    import math
    from utils.loss import WIoUState, wiou_v3
    state = WIoUState({})
    t = torch.tensor([[5.0, 5, 2, 2]])
    with torch.no_grad():
        assert wiou_v3(t, t, state).abs().item() < 1e-6  # perfect box: no loss
        assert state.mean == 1.0  # no autograd (validation): the running mean does not move
    p = torch.tensor([[5.5, 5, 2, 2]])
    l_iou = 1 - (1.5 * 2) / (8 - 1.5 * 2)  # IoU of a 2x2 box shifted by 0.5
    enclosing = 2.5 ** 2 + 2 ** 2
    state.mean = l_iou / 3  # beta = 3 = delta: focusing weight 1
    state.momentum = 0.0
    assert math.isclose(wiou_v3(p, t, state).item(), math.exp(0.25 / enclosing) * l_iou, rel_tol=1e-5)
    # non-monotonic focusing: the weight peaks for ordinary boxes (beta = 1 / ln(alpha), about 1.56) and is low for
    # very good boxes (beta near 0) and outliers (beta large, e.g. loose or wrong labels)
    r = lambda beta: beta / (3 * 1.9 ** (beta - 3))  # noqa: E731
    peak = 1 / math.log(1.9)
    assert r(peak) > max(r(1), r(3)) and r(0.1) < 0.3 and r(10) < 0.05
    state.momentum = 0.5
    wiou_v3(p.requires_grad_(), t, state)  # training: the running mean moves halfway to this batch's mean
    assert math.isclose(state.mean, 0.5 * l_iou / 3 + 0.5 * l_iou, rel_tol=1e-6)


def test_slide_loss_weights():
    import math
    from utils.loss import SlideLoss, slide_mu
    s = SlideLoss(torch.nn.BCEWithLogitsLoss())
    slide_mu(s, torch.tensor([0.5, 0.7, 0.6]))  # mu = 0.6
    pred, true = torch.zeros(4), torch.tensor([0.0, 0.55, 0.6, 0.9])
    bce = torch.nn.functional.binary_cross_entropy_with_logits(pred, true, reduction='none')
    w = torch.tensor([1.0, math.exp(0.4), math.exp(0.4), math.exp(0.1)])  # <= mu - 0.1 | (mu - 0.1, mu) | >= mu
    assert torch.isclose(s(pred, true), (bce * w).mean())
    slide_mu(s, torch.tensor([0.05]))
    assert s.mu == 0.2  # floor


def test_new_losses_off_by_default_and_validated(hyp):
    from utils.loss import SlideLoss
    m = build(TINY, hyp)
    loss = ComputeLossOTA(m)
    assert loss.wiou is None and not isinstance(loss.BCEobj, SlideLoss)  # stock hyp: stock loss
    with pytest.raises(AssertionError):
        ComputeLossOTA(build(TINY, {**hyp, 'iou_loss': 'giou'}))


@pytest.mark.parametrize('cfg', [TINY, P2])
@pytest.mark.parametrize('loss_cls', [ComputeLoss, ComputeLossOTA])
def test_training_step_with_wiou_slide_nwd(cfg, loss_cls, hyp):
    m = build(cfg, {**hyp, 'iou_loss': 'wiou', 'slide': 1, 'nwd': 0.5})
    m.train()
    x = torch.rand(2, 3, 256, 256)
    targets = torch.tensor([[0, 0, 0.5, 0.5, 0.02, 0.02], [1, 1, 0.3, 0.3, 0.05, 0.05]])
    crit = loss_cls(m)
    for _ in range(2):  # the Wise-IoU running mean moves in training
        out = m(x)
        loss = crit(out, targets, x)[0] if loss_cls is ComputeLossOTA else crit(out, targets)[0]
        loss.backward()
        assert torch.isfinite(loss).all()
    assert crit.wiou.mean != 1.0

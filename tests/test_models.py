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

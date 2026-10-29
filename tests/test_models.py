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

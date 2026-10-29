# End to end on CPU: train.py with a multi-source yaml (schedule, simulated key, --close-mosaic, --save-period, NWD
# box loss) and with a plain yaml (P2 model), test.py on the result, and seeded reproducibility. Tiny data and image
# size: this checks that everything runs and is wired together, not accuracy.

import subprocess
import sys

import pytest
import yaml

from conftest import ROOT, make_source

pytestmark = pytest.mark.slow


@pytest.fixture(scope='module')
def data(tmp_path_factory):
    root = tmp_path_factory.mktemp('smoke')
    a = make_source(root, 'camA', (640, 480), (50, 50, 50), 8, seed=1)
    b = make_source(root, 'camB', (320, 320), (90, 90, 90), 8, seed=2)
    s = make_source(root, 'sim', (320, 240), (120, 120, 120), 8, seed=3)
    mixed = {'nc': 1, 'names': ['o'], 'resize': 'native', 'epoch_size': 16,
             'train_sources': [{'name': 'camA', 'path': a, 'val': a, 'target': True, 'weight': 50},
                               {'name': 'camB', 'path': b, 'val': b, 'weight': 30},
                               {'name': 'sim', 'path': s, 'weight': 20, 'simulated': True}],
             'weight_schedule': {'mode': 'linear', 'points': [{'progress': 0.5,
                                                               'weights': {'camA': 60, 'camB': 30, 'sim': 10}}]}}
    plain = {'nc': 1, 'names': ['o'], 'train': [a, b], 'val': a}
    with open(ROOT / 'data' / 'hyp.scratch.tiny.yaml') as f:
        hyp = yaml.safe_load(f)
    hyp['nwd'] = 0.5  # the multi-source run also trains with the NWD box loss
    paths = {}
    for name, d in (('mixed', mixed), ('plain', plain), ('hyp_nwd', hyp)):
        paths[name] = root / f'{name}.yaml'
        paths[name].write_text(yaml.safe_dump(d))
    return root, paths


def train(root, data_yaml, name, *extra, cfg='cfg/training/yolov7-tiny.yaml', epochs=2,
          hyp='data/hyp.scratch.tiny.yaml'):
    cmd = [sys.executable, 'train.py', '--weights', '', '--cfg', cfg, '--data', str(data_yaml),
           '--hyp', str(hyp), '--epochs', str(epochs), '--batch-size', '4', '--img-size', '160',
           '160', '--device', 'cpu', '--workers', '2', '--project', str(root / 'runs'), '--name', name, '--exist-ok',
           '--seed', '0', *extra]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, timeout=1200)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return root / 'runs' / name, r.stdout + r.stderr


def test_train_mixed_then_test(data):
    root, paths = data
    run, log = train(root, paths['mixed'], 'mixed', '--close-mosaic', '1', '--save-period', '1', epochs=3,
                     hyp=paths['hyp_nwd'])
    w = run / 'weights'
    assert (w / 'last.pt').exists() and (w / 'best.pt').exists()
    assert (w / 'epoch_001.pt').exists()  # --save-period 1 (upstream saves epoch 0 and the last 5 anyway)
    assert len((run / 'results.txt').read_text().splitlines()) == 3
    assert (run / 'data_plan.txt').exists() and 'source_results.txt' in {p.name for p in run.iterdir()}
    assert 'Closing mosaic' in log and 'mixed sources from epoch' in log
    r = subprocess.run([sys.executable, 'test.py', '--weights', str(w / 'last.pt'), '--data', str(paths['mixed']),
                        '--img-size', '160', '--batch-size', '4', '--device', 'cpu', '--task', 'val',
                        '--project', str(root / 'runs'), '--name', 'test', '--exist-ok'],
                       capture_output=True, text=True, cwd=ROOT, timeout=600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]


def test_train_plain_p2_is_reproducible(data):
    root, paths = data
    runs = [train(root, paths['plain'], f'plain{i}', cfg='cfg/training/yolov7-tiny-p2.yaml')[0] for i in range(2)]
    a, b = [(r / 'results.txt').read_text() for r in runs]
    assert a == b  # same seed, same CPU: identical losses and metrics

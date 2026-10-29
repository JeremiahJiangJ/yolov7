# tools/propose_weights.py: shares by purpose, simulated sources, --max-repeat caps, the fade schedule, and that the
# yaml it writes is accepted by training.

import subprocess
import sys

import numpy as np
import pytest
import yaml

from conftest import ROOT, SOURCES
from utils.mixed_data import MixedConfig, WeightSchedule, header_frames_per_sample, list_images

FRAMES = {'camA': 5919, 'camB': 40048, 'camC': 13071, 'D': 750, 'F': 287, 'sim': 35596}


def propose(data_path, *args, check=True):
    r = subprocess.run([sys.executable, str(ROOT / 'tools' / 'propose_weights.py'), '--data', str(data_path), *args],
                       capture_output=True, text=True, cwd=ROOT)
    if check:
        assert r.returncode == 0, r.stderr + r.stdout
    return r


def snippet(stdout):
    return yaml.safe_load(stdout.split('Proposed (merge into your data yaml):')[1].split('\nwrote ')[0])


@pytest.fixture
def counts_yaml(tmp_path):
    p = tmp_path / 'data.yaml'
    p.write_text(yaml.safe_dump({'nc': 1, 'names': ['o'], 'train_sources': [
        {'name': n, 'path': '/x', **({'val': '/x'} if n in ('camA', 'F') else {})} for n in FRAMES]}))
    return p


def frames_args():
    return ['--balance-by', 'samples', '--frames'] + [f'{k}={v}' for k, v in FRAMES.items()]


def weights(snip):
    return {s['name']: s['weight'] for s in snip['train_sources']}


def test_generalist_with_simulated_source_and_caps(counts_yaml):
    snip = snippet(propose(counts_yaml, '--purpose', 'generalist', '--simulated', 'sim', '--max-repeat', '300',
                           '--epochs', '100', *frames_args()).stdout)
    assert weights(snip) == {'camA': 15.7, 'camB': 40.9, 'camC': 23.3, 'D': 3.7, 'F': 1.4, 'sim': 15}
    assert snip['epoch_size'] == 60075  # the real frames
    assert [s for s in snip['train_sources'] if s['name'] == 'sim'][0]['simulated'] is True


def test_caps_hold_after_rounding(counts_yaml):
    snip = snippet(propose(counts_yaml, '--purpose', 'generalist', '--simulated', 'sim', '--max-repeat', '300',
                           '--epochs', '100', '--sim-final', '0.08', *frames_args()).stdout)
    names = list(FRAMES)
    w = WeightSchedule(names, [weights(snip)[n] for n in names], snip['weight_schedule'], 100)
    samples = np.array([w(e) for e in range(100)]).sum(0) * snip['epoch_size']
    per_frame = samples / np.array([FRAMES[n] for n in names])
    assert per_frame.max() <= 300, dict(zip(names, per_frame.round()))


def test_sim_final_fades_simulation_and_sums_to_100(counts_yaml):
    snip = snippet(propose(counts_yaml, '--purpose', 'generalist', '--simulated', 'sim', '--max-repeat', '300',
                           '--epochs', '100', '--sim-final', '0.08', *frames_args()).stdout)
    final = snip['weight_schedule']['points'][0]
    assert final['progress'] == 0.6 and final['weights']['sim'] == 8
    assert abs(sum(weights(snip).values()) - 100) < 1e-6 and abs(sum(final['weights'].values()) - 100) < 1e-6
    assert final['weights']['D'] == weights(snip)['D'] and final['weights']['F'] == weights(snip)['F']  # capped: kept


def test_finetune_and_specialist(counts_yaml):
    ft = weights(snippet(propose(counts_yaml, '--purpose', 'finetune', '--target', 'camA', '--simulated', 'sim',
                                 *frames_args()).stdout))
    assert ft['camA'] == 80 and ft['sim'] == 4  # 5919 frames: 80% target; finetune sim share 4%
    sp = snippet(propose(counts_yaml, '--purpose', 'specialist', '--target', 'camB', '--simulated', 'sim',
                         *frames_args()).stdout)
    assert sp['weight_schedule']['points'][0]['weights']['camB'] == 90  # 40048 frames: 90% target at the end
    assert sp['weight_schedule']['points'][0]['weights']['sim'] == 4


@pytest.mark.parametrize('args, message', [
    (['--purpose', 'finetune'], 'needs --target'),
    (['--purpose', 'generalist', '--target', 'sim', '--simulated', 'sim'] + frames_args(), 'must be a real source'),
    (['--purpose', 'generalist', '--frames', 'camA=10'], '--balance-by samples'),
    (['--purpose', 'generalist', '--sim-final', '0.1', '--balance-by', 'samples', '--frames', 'camA=10'],
     'no source is simulated'),
])
def test_errors(counts_yaml, args, message):
    r = propose(counts_yaml, *args, check=False)
    assert r.returncode != 0 and message in r.stderr + r.stdout


def test_auto_target_is_a_real_source_with_a_val(counts_yaml):
    snip = snippet(propose(counts_yaml, '--purpose', 'generalist', '--simulated', 'sim', *frames_args()).stdout)
    assert [s['name'] for s in snip['train_sources'] if s.get('target')] == ['F']  # smallest real source with a val
    r = propose(counts_yaml, '--purpose', 'finetune', '--target', 'camB', *frames_args())
    assert 'WARNING: the target camB has no val' in r.stdout


def test_balance_by_frames_hits_the_requested_shares(sources, tmp_path):
    """With folders: frames balancing gives the requested shares of the frames loaded (simulated: 15% start)."""
    data = {'nc': 1, 'names': ['o'], 'resize': 'native', 'train_sources': [
        {'name': 'A', 'path': sources['A'], 'val': sources['A']}, {'name': 'B', 'path': sources['B']},
        {'name': 'C', 'path': sources['C'], 'simulated': True}]}
    p, out = tmp_path / 'd.yaml', tmp_path / 'out.yaml'
    p.write_text(yaml.safe_dump(data))
    propose(p, '--purpose', 'generalist', '--epochs', '10', '--out', str(out))
    written = yaml.safe_load(out.read_text())
    cfg = MixedConfig(written)  # training accepts the written yaml, simulated key included
    with open(ROOT / 'data' / 'hyp.scratch.tiny.yaml') as f:
        hyp = yaml.safe_load(f)
    fps = np.array(header_frames_per_sample(cfg, [list_images(s.path) for s in cfg.sources], hyp, 640))
    w = np.array([s.weight for s in cfg.sources], float)
    loaded = w * fps / (w * fps).sum()
    assert abs(loaded[2] - 0.15) < 0.01, loaded
    sq = np.sqrt([SOURCES['A'][2], SOURCES['B'][2]])  # real sources: square root of their frame counts
    assert np.allclose(loaded[:2] / loaded[:2].sum(), sq / sq.sum(), atol=0.01)

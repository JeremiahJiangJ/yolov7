# Propose train_sources weights (and epoch_size / weight_schedule) for a multi-source data yaml, by purpose.
#
# Purposes (README.md, section 8, "Proposing weights"):
#   generalist  one model for every source: shares ~ frames ^ power (square root by default), so large sources lead
#               without small ones being drowned out or repeated excessively. epoch_size: total.
#   balanced    equal shares (generalist with power 0): every source trained as much, small ones repeated most.
#   finetune    fine-tune for --target from a model that already knows the domain (e.g. a generalist): a fixed target
#               share sized by the target's frame count, the rest split as generalist to keep the other sources'
#               features. epoch_size: target (--epochs = passes over the target).
#   specialist  train for --target from a generic start (e.g. yolov7-tiny.pt, COCO): the generalist mix first, ramping
#               linearly to the finetune mix (weight_schedule), so the backbone first learns the domain from all data.
#
# Starting points, not optima: check each source's val (source_results.txt) and adjust.
#
# Usage:
#   python tools/propose_weights.py --data data/mixed/template.yaml --purpose generalist --epochs 100
#   python tools/propose_weights.py --data data.yaml --purpose finetune --target camC --epochs 40 --out camC_ft.yaml
#   python tools/propose_weights.py --data data.yaml --purpose generalist --max-repeat 300 --epochs 100
#   python tools/propose_weights.py --data data.yaml --purpose specialist --target camB --frames camA=12285 camB=40048
# The data yaml needs train_sources with name and path; weights and target are not needed (they are proposed).

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
import utils.general  # noqa: F401 (import order: utils.general before utils.metrics)
from utils.mixed_data import WeightSchedule, largest_remainder, list_images

PURPOSES = ('generalist', 'balanced', 'finetune', 'specialist')
TARGET_SHARE_STEPS = ((20000, 0.9), (5000, 0.8), (0, 0.7))  # target frames >= ...: share (finetune, auto)
SCHEDULE_PROGRESS = 0.6  # specialist: the finetune mix is reached at this fraction of --epochs
HEAVY_REPEAT = 500  # warn when a source's frames are sampled more often than this over the run


def power_shares(n, power):
    u = np.asarray(n, dtype=float) ** power
    return u / u.sum()


def capped_shares(n, power, cap):
    """Shares ~ n ** power, no share above cap (per source); the excess goes to the others in the same proportions.
    Returns (shares, capped mask), or None if the caps sum to less than 1 (no shares can respect them)."""
    n, cap = np.asarray(n, dtype=float), np.asarray(cap, dtype=float)
    if cap.sum() < 1:
        return None
    u = n ** power
    fixed = np.zeros(len(n), dtype=bool)
    while True:
        w = np.where(fixed, cap, 0.0)
        free = ~fixed
        w[free] = (1 - cap[fixed].sum()) * u[free] / u[free].sum()
        over = free & (w > cap * (1 + 1e-9))
        if not over.any():
            return w, fixed
        fixed |= over


def auto_target_share(frames):
    return next(s for lo, s in TARGET_SHARE_STEPS if frames >= lo)


def to_weights(shares):
    # integer percentages summing to 100; every source with a positive share keeps at least 1
    w = largest_remainder(shares, 100)
    for k in np.flatnonzero((w == 0) & (np.asarray(shares) > 0)):
        w[k] = 1
        w[int(np.argmax(w))] -= 1
    return [int(x) for x in w]


def main():
    parser = argparse.ArgumentParser(description='Propose train_sources weights for a purpose')
    parser.add_argument('--data', required=True, help='data yaml with train_sources (name, path; weights optional)')
    parser.add_argument('--purpose', required=True, choices=PURPOSES)
    parser.add_argument('--target', default=None, help='the source to train for (required by finetune, specialist; '
                                                        'for generalist / balanced: the source whose val picks best.pt)')
    parser.add_argument('--epochs', type=int, default=100, help='planned --epochs (repetition report, --max-repeat)')
    parser.add_argument('--power', type=float, default=0.5,
                        help='generalist split: shares ~ frames ^ power (1 = proportional, 0 = equal, 0.5 = square root)')
    parser.add_argument('--max-repeat', type=float, default=None,
                        help='generalist: most times any frame may be sampled over the run (caps small sources)')
    parser.add_argument('--target-share', type=float, default=None,
                        help='finetune / specialist: the target share, 0-1 (default: by target frames: '
                             '>= 20k: 0.9, >= 5k: 0.8, else 0.7)')
    parser.add_argument('--frames', nargs='+', default=None, metavar='NAME=N',
                        help='frame counts instead of scanning the source folders, e.g. camA=12285 camB=40048')
    parser.add_argument('--out', default=None, help='write the data yaml with the proposal (comments are not kept)')
    opt = parser.parse_args()
    if opt.purpose in ('finetune', 'specialist') and not opt.target:
        parser.error(f'--purpose {opt.purpose} needs --target <source name>')

    with open(opt.data) as f:
        data = yaml.safe_load(f)
    srcs = data.get('train_sources')
    assert isinstance(srcs, list) and srcs, f'{opt.data}: needs a train_sources list (README.md, section 8)'
    names = [str(s.get('name', f'source{i}')) for i, s in enumerate(srcs)]
    assert len(set(names)) == len(names), f'train_sources names must be unique, got {names}'

    # frame counts
    given = {}
    for item in opt.frames or []:
        assert '=' in item, f'--frames takes NAME=N, got {item}'
        k, v = item.split('=', 1)
        assert k in names, f'--frames: no source named {k} (sources: {names})'
        given[k] = int(v)
    n = []
    for name, s in zip(names, srcs):
        if name in given:
            n.append(given[name])
        else:
            assert 'path' in s, f'source {name} has no path (or give --frames {name}=N)'
            n.append(len(list_images(s['path'])))
        assert n[-1] > 0, f'source {name} has no images'
    n = np.asarray(n, dtype=float)

    # target
    marked = [nm for nm, s in zip(names, srcs) if s.get('target')]
    target = opt.target or (marked[0] if marked else None)
    target_note = ''
    if target is None:  # generalist / balanced without a target: the smallest source, the most at risk
        target = names[int(np.argmin(n))]
        target_note = (f'no target given: {target} (the smallest source) is marked target, so its val picks best.pt '
                       f'and --patience; pass --target to choose another')
    if target not in names:
        sys.exit(f'error: --target {target}: no such source (sources: {names})')
    t = names.index(target)

    notes, schedule = [], None
    if opt.purpose in ('generalist', 'balanced'):
        power = 0.0 if opt.purpose == 'balanced' else opt.power
        epoch_size = 'total'
        shares = power_shares(n, power)
        notes.append(f'shares ~ frames ^ {power:g} (' + ('equal shares' if power == 0 else
                     'proportional to the source sizes' if power == 1 else 'square root' if power == 0.5 else
                     'between proportional and equal') + ')')
        if opt.max_repeat is not None:
            cap = opt.max_repeat * n / (opt.epochs * n.sum())
            res = capped_shares(n, power, cap)
            if res is None:
                notes.append(f'--max-repeat {opt.max_repeat:g} cannot be met: {opt.epochs} epochs x {int(n.sum())} '
                             f'samples sample every frame {opt.epochs:g} times at any shares. Proportional shares '
                             f'(the same repetition for all) used; fewer epochs would respect it')
                shares = n / n.sum()
            else:
                shares, capped = res
                if capped.any():
                    notes.append(f'capped at {opt.max_repeat:g} samples per frame: '
                                 f'{", ".join(nm for nm, c in zip(names, capped) if c)} (the rest of their share went '
                                 f'to the others)')
    else:
        epoch_size = 'target'
        share = opt.target_share if opt.target_share is not None else auto_target_share(n[t])
        assert 0 < share <= 1, f'--target-share must be in (0, 1], got {share}'
        why = 'given' if opt.target_share is not None else f'auto: {int(n[t])} target frames'
        others = [k for k in range(len(n)) if k != t]
        final = np.zeros(len(n))
        final[t] = share if others else 1.0
        if others:
            final[others] = (1 - share) * power_shares(n[others], opt.power)
        notes.append(f'target share {final[t]:.0%} ({why}); the rest split ~ frames ^ {opt.power:g} among the others')
        if opt.purpose == 'finetune':
            shares = final
            notes.append('from a checkpoint that already knows the domain (e.g. a generalist): fixed weights. For a '
                         'small target, also consider freezing the backbone for the first epochs '
                         '(--freeze, --unfreeze-epoch)')
        else:  # specialist: generalist mix first, ramping to the finetune mix
            start = power_shares(n, opt.power)
            shares = start
            schedule = {'mode': 'linear',
                        'points': [{'progress': SCHEDULE_PROGRESS, 'weights': dict(zip(names, to_weights(final)))}]}
            notes.append(f'from a generic start (e.g. yolov7-tiny.pt): the generalist mix first, linear to the target '
                         f'mix at {SCHEDULE_PROGRESS:.0%} of --epochs (weight_schedule), so the backbone first learns '
                         f'the domain from all sources')

    weights = to_weights(shares)

    # what the run would do with these (rounded) weights
    sch = WeightSchedule(names, weights, schedule, opt.epochs)
    per_epoch_w = np.array([sch(e) for e in range(opt.epochs)])  # (epochs, sources), normalised
    if epoch_size == 'total':
        es = n.sum()
    else:  # passes over the target: target frames / its weight averaged over the run
        es = n[t] / per_epoch_w[:, t].mean()
    per_frame = per_epoch_w.sum(0) * es / n  # samples per frame over the run
    mean_share = per_epoch_w.mean(0)

    print(f'purpose {opt.purpose}, {opt.epochs} epochs, epoch_size {epoch_size} ({int(round(es))} samples/epoch, '
          f'{int(round(es * opt.epochs))} training samples)')
    print(f"  {'source':<14}{'frames':>9}{'weight':>8}{'share (mean)':>14}{'samples/frame':>15}")
    for k, nm in enumerate(names):
        flag = '  <- target' if k == t else ''
        print(f'  {nm:<14}{int(n[k]):>9}{weights[k]:>8}{mean_share[k]:>14.1%}{per_frame[k]:>15.0f}{flag}')
    if schedule:
        fw = schedule['points'][0]['weights']
        print(f"  weight_schedule: linear, at {SCHEDULE_PROGRESS:.0%} of --epochs: " +
              ', '.join(f'{nm} {fw[nm]}' for nm in names))
    for x in notes + ([target_note] if target_note else []):
        print(f'  - {x}')
    heavy = [f'{nm} ({per_frame[k]:.0f})' for k, nm in enumerate(names) if per_frame[k] > HEAVY_REPEAT]
    if heavy:
        print(f'  WARNING: frames sampled more than {HEAVY_REPEAT} times over the run: {", ".join(heavy)}. '
              f'Overfitting risk: fewer epochs, a lower share, or --max-repeat (generalist)')
    print('  Shares are of training samples (compute). Frames and objects per sample differ by source: check the '
          'dry run, python -m utils.mixed_data --data <yaml> --epochs N --measure 64')

    # yaml
    out = dict(data)
    out['sampling'] = 'weighted'
    out['epoch_size'] = epoch_size
    out.pop('weight_schedule', None)
    if schedule:
        out['weight_schedule'] = schedule
    new_srcs = []
    for k, s in enumerate(srcs):
        s = dict(s)
        s['weight'] = weights[k]
        s.pop('target', None)
        if k == t:
            s['target'] = True
        new_srcs.append(s)
    out['train_sources'] = new_srcs
    snippet = {'sampling': 'weighted', 'epoch_size': epoch_size,
               'train_sources': [{'name': nm, 'weight': weights[k], **({'target': True} if k == t else {})}
                                 for k, nm in enumerate(names)]}
    if schedule:
        snippet['weight_schedule'] = schedule
    print('\nProposed (merge into your data yaml):')
    print(yaml.safe_dump(snippet, sort_keys=False, default_flow_style=None).rstrip())
    if opt.out:
        with open(opt.out, 'w') as f:
            f.write(f'# {opt.data} with weights proposed by: python tools/propose_weights.py --purpose {opt.purpose}'
                    f'{" --target " + opt.target if opt.target else ""} --epochs {opt.epochs}\n')
            yaml.safe_dump(out, f, sort_keys=False, default_flow_style=None)
        print(f'\nwrote {opt.out} (the original comments are not kept)')


if __name__ == '__main__':
    main()

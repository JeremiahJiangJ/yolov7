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
# Shares of what (--balance-by): the weights set each source's share of the training samples (compute). A sample of a
# source with small frames holds more frames (mosaic), and object density differs, so the shares of the frames loaded
# and of the objects differ from it (the dry run, python -m utils.mixed_data, shows all three). --balance-by frames or
# objects applies the purpose's shares to that quantity instead, and converts them to sample weights using each
# source's frames per sample (from its frame size, --img-size and --hyp) or measured objects per sample.
#
# Starting points, not optima: check each source's val (source_results.txt) and adjust.
#
# Usage:
#   python tools/propose_weights.py --data data/mixed/template.yaml --purpose generalist --epochs 100
#   python tools/propose_weights.py --data data.yaml --purpose generalist --balance-by frames --img-size 640
#   python tools/propose_weights.py --data data.yaml --purpose finetune --target camC --epochs 40 --out camC_ft.yaml
#   python tools/propose_weights.py --data data.yaml --purpose generalist --max-repeat 300 --epochs 100
#   python tools/propose_weights.py --data data.yaml --purpose specialist --target camB --frames camA=12285 camB=40048
# The data yaml needs train_sources with name and path; weights and target are not needed (they are proposed).

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
import utils.general  # noqa: F401 (import order: utils.general before utils.metrics)
from utils.mixed_data import (MixedConfig, WeightSchedule, header_frames_per_sample, largest_remainder, list_images,
                              measure_datasets, measure_sources)

PURPOSES = ('generalist', 'balanced', 'finetune', 'specialist')
BALANCE_BY = ('samples', 'frames', 'objects')
TARGET_SHARE_STEPS = ((20000, 0.9), (5000, 0.8), (0, 0.7))  # target frames >= ...: share (finetune, auto)
SCHEDULE_PROGRESS = 0.6  # specialist: the finetune mix is reached at this fraction of --epochs
MEASURE_DEFAULT = 256  # --balance-by objects: samples built per source to measure objects per sample


def power_shares(n, power):
    u = np.asarray(n, dtype=float) ** power
    return u / u.sum()


def capped(w, cap):
    """Shares w with none above cap (per source): a source above its cap gets exactly the cap, the rest of its share
    goes to the others in proportion to their shares. Returns (shares, capped mask), or None if the caps sum to less
    than 1 (no shares can respect them)."""
    w, cap = np.asarray(w, dtype=float), np.asarray(cap, dtype=float)
    if cap.sum() < 1:
        return None
    fixed = np.zeros(len(w), dtype=bool)
    while True:
        out = np.where(fixed, cap, 0.0)
        free = ~fixed
        out[free] = (1 - cap[fixed].sum()) * w[free] / w[free].sum()
        over = free & (out > cap * (1 + 1e-9))
        if not over.any():
            return out, fixed
        fixed |= over


def to_samples(shares, per_sample):
    """Shares of a quantity (frames loaded, objects) -> shares of the samples that give them."""
    w = np.asarray(shares, dtype=float) / np.asarray(per_sample, dtype=float)
    return w / w.sum()


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
    parser.add_argument('--balance-by', default='samples', choices=BALANCE_BY,
                        help='what the shares are of: samples (compute, the default), frames (frames loaded, i.e. '
                             'distinct content) or objects (supervision; measured, slower)')
    parser.add_argument('--power', type=float, default=0.5,
                        help='generalist split: shares ~ frames ^ power (1 = proportional, 0 = equal, 0.5 = square root)')
    parser.add_argument('--max-repeat', type=float, default=None,
                        help='most times any frame may be the sampled frame over the whole run (all --epochs); caps '
                             'small sources (generalist, balanced)')
    parser.add_argument('--warn-repeat', type=float, default=300,
                        help='warn when a frame is the sampled frame more often than this over the run (a rule of '
                             'thumb for full augmentation; ~100 with light, ~50 with no augmentation)')
    parser.add_argument('--target-share', type=float, default=None,
                        help='finetune / specialist: the target share, 0-1 (default: by target frames: '
                             '>= 20k: 0.9, >= 5k: 0.8, else 0.7)')
    parser.add_argument('--img-size', type=int, default=640, help='training --img-size (frames per sample)')
    parser.add_argument('--hyp', default='data/hyp.scratch.tiny.yaml',
                        help='training hyp yaml (mosaic, mixup, paste-in: frames and objects per sample)')
    parser.add_argument('--measure', type=int, default=None,
                        help=f'samples built per source to measure objects per sample (default: {MEASURE_DEFAULT} '
                             f'with --balance-by objects, else not measured)')
    parser.add_argument('--frames', nargs='+', default=None, metavar='NAME=N',
                        help='frame counts instead of scanning the source folders (--balance-by samples only), '
                             'e.g. camA=12285 camB=40048')
    parser.add_argument('--out', default=None, help='write the data yaml with the proposal (comments are not kept)')
    opt = parser.parse_args()
    if opt.purpose in ('finetune', 'specialist') and not opt.target:
        parser.error(f'--purpose {opt.purpose} needs --target <source name>')
    if opt.frames and opt.balance_by != 'samples':
        parser.error('--frames skips reading the sources, but --balance-by frames / objects needs them')
    measure = opt.measure if opt.measure is not None else (MEASURE_DEFAULT if opt.balance_by == 'objects' else 0)

    with open(opt.data) as f:
        data = yaml.safe_load(f)
    srcs = data.get('train_sources')
    if not (isinstance(srcs, list) and srcs):
        sys.exit(f'error: {opt.data}: needs a train_sources list (README.md, section 8)')
    names = [str(s.get('name', f'source{i}')) for i, s in enumerate(srcs)]
    if len(set(names)) != len(names):
        sys.exit(f'error: train_sources names must be unique, got {names}')

    # target
    marked = [nm for nm, s in zip(names, srcs) if s.get('target')]
    target = opt.target or (marked[0] if marked else None)

    # frame counts (and the files, for frames / objects per sample)
    given = {}
    for item in opt.frames or []:
        if '=' not in item:
            sys.exit(f'error: --frames takes NAME=N, got {item}')
        k, v = item.split('=', 1)
        if k not in names:
            sys.exit(f'error: --frames: no source named {k} (sources: {names})')
        given[k] = int(v)
    files, n = [], []
    for name, s in zip(names, srcs):
        if name in given:
            files.append(None)
            n.append(given[name])
        else:
            if 'path' not in s:
                sys.exit(f'error: source {name} has no path (or give --frames {name}=N)')
            files.append(list_images(s['path']))
            n.append(len(files[-1]))
        if n[-1] <= 0:
            sys.exit(f'error: source {name} has no images')
    n = np.asarray(n, dtype=float)

    target_note = ''
    if target is None:  # generalist / balanced without a target: the smallest source, the most at risk
        target = names[int(np.argmin(n))]
        target_note = (f'no target given: {target} (the smallest source) is marked target, so its val picks best.pt '
                       f'and --patience; pass --target to choose another')
    if target not in names:
        sys.exit(f'error: --target {target}: no such source (sources: {names})')
    t = names.index(target)

    # frames and objects per sample, as training builds the samples
    fps = ops = None
    if all(f is not None for f in files):
        parsed = copy.deepcopy(data)
        parsed['sampling'] = 'weighted'  # settings only: the weights and target here are placeholders
        parsed.pop('weight_schedule', None)
        for k, s in enumerate(parsed['train_sources']):
            s['weight'], s['target'] = 1, k == t
            if k == t and not s.get('val') and not parsed.get('val'):
                s['val'] = s['path']
        cfg = MixedConfig(parsed)
        with open(opt.hyp) as f:
            hyp = yaml.safe_load(f)
        fps = np.asarray(header_frames_per_sample(cfg, files, hyp, opt.img_size), dtype=float)
        if measure:
            print(f'measuring objects per sample on {measure} samples per source ...')
            ops = np.asarray([m[0] for m in measure_sources(measure_datasets(cfg, hyp, opt.img_size), measure)])
            empty = [nm for nm, x in zip(names, ops) if x <= 0]
            if opt.balance_by == 'objects' and empty:
                sys.exit(f'error: --balance-by objects: no objects in the measured samples of {empty} (e.g. background '
                         f'only): use --balance-by samples or frames')
    per_sample = {'samples': np.ones(len(n)), 'frames': fps, 'objects': ops}[opt.balance_by]
    what = {'samples': 'training samples', 'frames': 'frames loaded', 'objects': 'objects'}[opt.balance_by]

    notes, schedule = [], None
    if opt.purpose in ('generalist', 'balanced'):
        power = 0.0 if opt.purpose == 'balanced' else opt.power
        epoch_size = 'total'
        shares = to_samples(power_shares(n, power), per_sample)
        notes.append(f'shares of the {what} ~ frames ^ {power:g} (' + (
            'equal shares' if power == 0 else 'proportional to the source sizes' if power == 1 else
            'square root' if power == 0.5 else 'between proportional and equal') + ')')
        if opt.max_repeat is not None:
            cap = opt.max_repeat * n / (opt.epochs * n.sum())  # sample shares that stay within max_repeat
            res = capped(shares, cap)
            if res is None:
                notes.append(f'--max-repeat {opt.max_repeat:g} cannot be met: {opt.epochs} epochs x {int(n.sum())} '
                             f'samples sample every frame {opt.epochs:g} times at any shares. Proportional shares '
                             f'(the same repetition for all) used; fewer epochs would respect it')
                shares = n / n.sum()
            else:
                shares, hit = res
                if hit.any():
                    notes.append(f'capped at {opt.max_repeat:g} samples per frame over the run: '
                                 f'{", ".join(nm for nm, c in zip(names, hit) if c)} (the rest of their share went '
                                 f'to the others)')
    else:
        epoch_size = 'target'
        share = opt.target_share if opt.target_share is not None else auto_target_share(n[t])
        if not 0 < share <= 1:
            sys.exit(f'error: --target-share must be in (0, 1], got {share}')
        why = 'given' if opt.target_share is not None else f'auto: {int(n[t])} target frames'
        others = [k for k in range(len(n)) if k != t]
        final = np.zeros(len(n))
        final[t] = share if others else 1.0
        if others:
            final[others] = (1 - share) * power_shares(n[others], opt.power)
        final = to_samples(final, per_sample)
        notes.append(f'target share of the {what} {share if others else 1:.0%} ({why}); the rest split '
                     f'~ frames ^ {opt.power:g} among the others')
        if opt.purpose == 'finetune':
            shares = final
            notes.append('from a checkpoint that already knows the domain (e.g. a generalist): fixed weights. For a '
                         'small target, also consider freezing the backbone for the first epochs '
                         '(--freeze, --unfreeze-epoch)')
        else:  # specialist: generalist mix first, ramping to the finetune mix
            shares = to_samples(power_shares(n, opt.power), per_sample)
            schedule = {'mode': 'linear',
                        'points': [{'progress': SCHEDULE_PROGRESS, 'weights': dict(zip(names, to_weights(final)))}]}
            notes.append(f'from a generic start (e.g. yolov7-tiny.pt): the generalist mix first, linear to the target '
                         f'mix at {SCHEDULE_PROGRESS:.0%} of --epochs (weight_schedule), so the backbone first learns '
                         f'the domain from all sources')

    weights = to_weights(shares)

    # what the run would do with these (rounded) weights
    sch = WeightSchedule(names, weights, schedule, opt.epochs)
    per_epoch_w = np.array([sch(e) for e in range(opt.epochs)])  # (epochs, sources), normalised
    es = n.sum() if epoch_size == 'total' else n[t] / per_epoch_w[:, t].mean()  # samples per epoch
    samples = per_epoch_w.sum(0) * es  # per source over the run
    print(f'purpose {opt.purpose}, balanced by {opt.balance_by}, {opt.epochs} epochs, epoch_size {epoch_size} '
          f'({int(round(es))} samples/epoch, {int(round(samples.sum()))} training samples)')
    head = f"  {'source':<14}{'frames':>9}{'weight':>8}{'samples':>9}"
    head += f"{'frames loaded':>15}" if fps is not None else ''
    head += f"{'objects':>9}" if ops is not None else ''
    head += f"{'samples/frame':>15}" + (f"{'loads/frame':>13}" if fps is not None else '')
    print(head)
    loaded = samples * fps if fps is not None else None
    objs = samples * ops if ops is not None else None
    for k, nm in enumerate(names):
        row = f'  {nm:<14}{int(n[k]):>9}{weights[k]:>8}{samples[k] / samples.sum():>9.1%}'
        row += f'{loaded[k] / loaded.sum():>15.1%}' if fps is not None else ''
        row += f'{objs[k] / objs.sum():>9.1%}' if ops is not None else ''
        row += f'{samples[k] / n[k]:>15.0f}' + (f'{loaded[k] / n[k]:>13.0f}' if fps is not None else '')
        print(row + ('  <- target' if k == t else ''))
    print('  (shares of the training samples, the frames loaded and the objects over the run; samples/frame: times '
          'each frame is the sampled frame; loads/frame: also as a mosaic / mixup / paste-in partner)')
    if schedule:
        fw = schedule['points'][0]['weights']
        print(f"  weight_schedule: linear, at {SCHEDULE_PROGRESS:.0%} of --epochs: " +
              ', '.join(f'{nm} {fw[nm]}' for nm in names))
    if fps is None:
        notes.append('frames loaded / objects not shown (--frames given): point the sources at their folders to see '
                     'them, or run the dry run, python -m utils.mixed_data --data <yaml> --epochs N --measure 64')
    elif ops is None:
        notes.append('objects not shown: add --measure 64 (builds samples, reads labels)')
    for x in notes + ([target_note] if target_note else []):
        print(f'  - {x}')
    heavy = [f'{nm} ({samples[k] / n[k]:.0f})' for k, nm in enumerate(names) if samples[k] / n[k] > opt.warn_repeat]
    if heavy:
        print(f'  WARNING: sampled more than {opt.warn_repeat:g} times per frame over the run (--warn-repeat, a rule of '
              f'thumb): {", ".join(heavy)}. Overfitting risk: fewer epochs, a lower share, or --max-repeat; check '
              f'that source\'s val curve (source_results.txt)')

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
                    f'{" --target " + opt.target if opt.target else ""} --balance-by {opt.balance_by} '
                    f'--epochs {opt.epochs}\n')
            yaml.safe_dump(out, f, sort_keys=False, default_flow_style=None)
        print(f'\nwrote {opt.out} (the original comments are not kept)')


if __name__ == '__main__':
    main()

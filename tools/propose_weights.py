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
# Simulated sources (simulated: true on the source, or --simulated NAME ...): a fixed share instead of one by frame
# count (a large simulated set would otherwise lead the mix): --sim-share, by default 15% for generalist / balanced /
# specialist (start) and 4% for finetune, split among the simulated sources ~ frames ^ power; the real sources share the
# rest by the purpose's rule (--max-repeat caps real sources only). --sim-final fades simulation to that share
# (front-loaded, linear weight_schedule; specialist: 4% by default). With simulated sources a generalist's epoch_size is
# the real frame count, and the target must be a real source.
#
# Shares of what (--balance-by): the weights set each source's share of the training samples (compute). A sample of a
# source with small frames holds more frames (mosaic), and object density differs, so the shares of the frames loaded
# and of the objects differ from it (the dry run, python -m utils.mixed_data, shows all three). By default (frames) the
# purpose's shares are of the frames loaded, converted to sample weights using each source's frames per sample (from
# its frame size, --img-size and --hyp); objects uses measured objects per sample, samples the sample shares as they
# are.
#
# Starting points, not optima: check each source's val (source_results.txt) and adjust.
#
# Usage:
#   python tools/propose_weights.py --data data/mixed/template.yaml --purpose generalist --epochs 100
#   python tools/propose_weights.py --data data.yaml --purpose generalist --img-size 640 --hyp my_hyp.yaml
#   python tools/propose_weights.py --data data.yaml --purpose finetune --target camC --epochs 40 --out camC_ft.yaml
#   python tools/propose_weights.py --data data.yaml --purpose generalist --max-repeat 300 --epochs 100
#   python tools/propose_weights.py --data data.yaml --purpose specialist --target camB --balance-by samples \
#       --frames camA=12285 camB=40048
#   python tools/propose_weights.py --data data.yaml --purpose generalist --simulated sim --sim-final 0.08
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
SCHEDULE_PROGRESS = 0.6  # specialist / --sim-final: the final mix is reached at this fraction of --epochs
SIM_SHARE = {'generalist': 0.15, 'balanced': 0.15, 'specialist': 0.15, 'finetune': 0.04}  # --sim-share defaults
SIM_FINAL_SPECIALIST = 0.04  # specialist: --sim-final default (the finetune share)
MEASURE_DEFAULT = 256  # --balance-by objects: samples built per source to measure objects per sample


def power_shares(n, power):
    u = np.asarray(n, dtype=float) ** power
    return u / u.sum()


def split(n, power, mask, total):
    """total split ~ n ^ power among the sources in mask, 0 elsewhere."""
    out = np.zeros(len(n))
    if mask.any():
        out[mask] = total * power_shares(np.asarray(n, dtype=float)[mask], power)
    return out


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


def to_weights(shares, ceiling=None):
    """Percentages to 0.1 summing to 100: every source with a positive share keeps at least 0.1, and none is rounded
    above its ceiling (a share, e.g. the --max-repeat cap)."""
    shares = np.asarray(shares, dtype=float)
    w = largest_remainder(shares, 1000)
    for k in np.flatnonzero((w == 0) & (shares > 0)):
        w[k] = 1
        w[int(np.argmax(w))] -= 1
    if ceiling is not None:
        top = np.floor(np.asarray(ceiling, dtype=float) * 1000 + 1e-6)
        over = w > top
        if over.any() and (~over).any():
            excess = int((w[over] - top[over]).sum())
            w[over] = top[over]
            w[int(np.argmax(np.where(over, -1, w)))] += excess
    return [int(x) // 10 if x % 10 == 0 else int(x) / 10 for x in w]


def main():
    parser = argparse.ArgumentParser(description='Propose train_sources weights for a purpose')
    parser.add_argument('--data', required=True, help='data yaml with train_sources (name, path; weights optional)')
    parser.add_argument('--purpose', required=True, choices=PURPOSES)
    parser.add_argument('--target', default=None, help='the source to train for (required by finetune, specialist; '
                                                        'for generalist / balanced: the source whose val picks best.pt)')
    parser.add_argument('--epochs', type=int, default=100, help='planned --epochs (repetition report, --max-repeat)')
    parser.add_argument('--balance-by', default='frames', choices=BALANCE_BY,
                        help='what the shares are of: frames (frames loaded, i.e. distinct content; the default), '
                             'samples (compute) or objects (supervision; measured, slower)')
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
    parser.add_argument('--simulated', nargs='+', default=None, metavar='NAME',
                        help='simulated sources (as well as those with simulated: true in the data yaml)')
    parser.add_argument('--sim-share', type=float, default=None,
                        help='share of the simulated sources together, 0-1 (default: 0.15, finetune: 0.04)')
    parser.add_argument('--sim-final', type=float, default=None,
                        help='fade the simulated share linearly to this by 60%% of --epochs (front-loaded; default: '
                             'none, specialist: 0.04)')
    parser.add_argument('--out', default=None, help='write the data yaml with the proposal (comments are not kept)')
    opt = parser.parse_args()
    if opt.purpose in ('finetune', 'specialist') and not opt.target:
        parser.error(f'--purpose {opt.purpose} needs --target <source name>')
    if opt.frames and opt.balance_by != 'samples':
        parser.error(f'--frames skips reading the sources, but --balance-by {opt.balance_by} needs them (frames is '
                     f'the default): add --balance-by samples, or drop --frames')
    measure = opt.measure if opt.measure is not None else (MEASURE_DEFAULT if opt.balance_by == 'objects' else 0)

    with open(opt.data) as f:
        data = yaml.safe_load(f)
    srcs = data.get('train_sources')
    if not (isinstance(srcs, list) and srcs):
        sys.exit(f'error: {opt.data}: needs a train_sources list (README.md, section 8)')
    names = [str(s.get('name', f'source{i}')) for i, s in enumerate(srcs)]
    if len(set(names)) != len(names):
        sys.exit(f'error: train_sources names must be unique, got {names}')

    # simulated sources
    for nm in opt.simulated or []:
        if nm not in names:
            sys.exit(f'error: --simulated: no source named {nm} (sources: {names})')
    sim = np.array([bool(s.get('simulated')) or nm in (opt.simulated or []) for nm, s in zip(names, srcs)])
    real = ~sim
    if not real.any():
        sys.exit('error: every source is simulated: at least one real source is needed')
    if sim.any():
        sim_share = opt.sim_share if opt.sim_share is not None else SIM_SHARE[opt.purpose]
        sim_final = opt.sim_final if opt.sim_final is not None else (
            min(SIM_FINAL_SPECIALIST, sim_share) if opt.purpose == 'specialist' else None)
        if not 0 <= sim_share < 1 or (sim_final is not None and not 0 <= sim_final < 1):
            sys.exit('error: --sim-share and --sim-final must be in [0, 1)')
        if sim_final is not None and sim_final > sim_share and opt.purpose != 'specialist':
            sys.exit(f'error: --sim-final {sim_final:g} is above --sim-share {sim_share:g} (it is the share simulation '
                     f'fades to)')
    else:
        if opt.sim_share is not None or opt.sim_final is not None:
            sys.exit('error: --sim-share / --sim-final given, but no source is simulated (simulated: true or '
                     '--simulated)')
        sim_share, sim_final = 0.0, None

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
    if target is None:  # generalist / balanced without a target: the smallest real source, the most at risk
        target = names[int(np.argmin(np.where(real, n, np.inf)))]
        target_note = (f'no target given: {target} (the smallest {"real " if sim.any() else ""}source) is marked '
                       f'target, so its val picks best.pt and --patience; pass --target to choose another')
    if target not in names:
        sys.exit(f'error: --target {target}: no such source (sources: {names})')
    t = names.index(target)
    if sim[t]:
        sys.exit(f'error: the target {target} is simulated: the target (its val picks best.pt) must be a real source')

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

    notes, schedule, hit, ceiling = [], None, np.zeros(len(n), dtype=bool), None
    if opt.purpose in ('generalist', 'balanced'):
        power = 0.0 if opt.purpose == 'balanced' else opt.power
        # with simulated sources an epoch is the real frame count (a pass's worth of real data)
        epoch_size = int(n[real].sum()) if sim.any() else 'total'
        shares = to_samples(split(n, power, real, 1 - sim_share) + split(n, power, sim, sim_share), per_sample)
        notes.append(f'shares of the {what} ~ frames ^ {power:g} (' + (
            'equal shares' if power == 0 else 'proportional to the source sizes' if power == 1 else
            'square root' if power == 0.5 else 'between proportional and equal') + ')' +
            (' among the real sources' if sim.any() else ''))
        if opt.max_repeat is not None:
            es_plan = n.sum() if epoch_size == 'total' else epoch_size
            cap = opt.max_repeat * n / (opt.epochs * es_plan)  # sample shares that stay within max_repeat
            tot = shares[real].sum()  # the real sources' samples; simulated shares stay fixed
            res = capped(shares[real] / tot, cap[real] / tot)
            if res is None:
                notes.append(f'--max-repeat {opt.max_repeat:g} cannot be met: {opt.epochs} epochs x '
                             f'{int(round(tot * es_plan))} real samples sample every real frame '
                             f'{opt.epochs * tot * es_plan / n[real].sum():.0f} times at any split. Proportional '
                             f'shares (the same repetition for all) used; fewer epochs would respect it')
                shares[real] = tot * n[real] / n[real].sum()
            else:
                shares[real] = tot * res[0]
                hit[real] = res[1]
                ceiling = np.where(real, cap, np.inf)
                if hit.any():
                    notes.append(f'capped at {opt.max_repeat:g} samples per frame over the run: '
                                 f'{", ".join(nm for nm, c in zip(names, hit) if c)} (the rest of their share went '
                                 f'to the other {"real " if sim.any() else ""}sources)')
    else:
        epoch_size = 'target'
        share = opt.target_share if opt.target_share is not None else auto_target_share(n[t])
        if not 0 < share <= 1:
            sys.exit(f'error: --target-share must be in (0, 1], got {share}')
        why = 'given' if opt.target_share is not None else f'auto: {int(n[t])} target frames'
        final_sim = sim_final if opt.purpose == 'specialist' and sim_final is not None else sim_share
        others = real.copy()
        others[t] = False
        if share + final_sim > 1:
            sys.exit(f'error: target share {share:g} + simulated share {final_sim:g} > 1')
        final = split(n, opt.power, sim, final_sim)
        final[t] = share if others.any() else 1 - final_sim
        final += split(n, opt.power, others, 1 - share - final_sim)
        final = to_samples(final, per_sample)
        notes.append(f'target share of the {what} {final[t] if not others.any() else share:.0%} ({why}); the rest '
                     f'split ~ frames ^ {opt.power:g} among the other {"real " if sim.any() else ""}sources')
        if opt.purpose == 'finetune':
            shares = final
            notes.append('from a checkpoint that already knows the domain (e.g. a generalist): fixed weights. For a '
                         'small target, also consider freezing the backbone for the first epochs '
                         '(--freeze, --unfreeze-epoch)')
        else:  # specialist: generalist mix first, ramping to the finetune mix
            shares = to_samples(split(n, opt.power, real, 1 - sim_share) + split(n, opt.power, sim, sim_share),
                                per_sample)
            schedule = {'mode': 'linear',
                        'points': [{'progress': SCHEDULE_PROGRESS, 'weights': dict(zip(names, to_weights(final)))}]}
            notes.append(f'from a generic start (e.g. yolov7-tiny.pt): the generalist mix first, linear to the target '
                         f'mix at {SCHEDULE_PROGRESS:.0%} of --epochs (weight_schedule), so the backbone first learns '
                         f'the domain from all sources')

    if sim.any():
        sim_names = ', '.join(nm for nm, x in zip(names, sim) if x)
        notes.append(f'simulated ({sim_names}): a fixed {sim_share:.0%} of the {what} together, not sized by their '
                     f'frame count (they would lead the mix); the real sources share the rest' +
                     (f'. epoch_size {epoch_size}: the real frames, so an epoch is a pass\'s worth of real data'
                      if isinstance(epoch_size, int) else ''))
        notes.append('simulated: no val set needed (best.pt is picked on real data); check their objects have the '
                     'real cameras\' pixel size (resize, section 9) and watch the real val curves')
        if sim_final is not None and opt.purpose != 'specialist':  # front-loaded: fade simulation out
            # sample shares: capped sources fixed, simulated scaled by a, uncapped real by b, so that simulated is
            # sim_final of the --balance-by quantity: a*Sq*(1-f) - b*f*Uq = f*Cq and a*Ss + b*Us = 1 - Cs
            recv = real & ~hit  # capped sources keep their share
            recv = recv if recv.any() else real
            fixed = ~sim & ~recv
            p = shares * per_sample
            f = sim_final
            a, b = np.linalg.solve([[p[sim].sum() * (1 - f), -f * p[recv].sum()],
                                    [shares[sim].sum(), shares[recv].sum()]],
                                   [f * p[fixed].sum(), 1 - shares[fixed].sum()])
            final = shares.copy()
            final[sim] *= a
            final[recv] *= b
            schedule = {'mode': 'linear',
                        'points': [{'progress': SCHEDULE_PROGRESS,
                                    'weights': dict(zip(names, to_weights(final, ceiling)))}]}
            notes.append(f'simulated faded linearly to {sim_final:.0%} of the {what} at {SCHEDULE_PROGRESS:.0%} of '
                         f'--epochs '
                         f'(weight_schedule, front-loaded); the freed share goes to the '
                         f'{"uncapped " if hit.any() else ""}real sources')
        elif opt.purpose == 'specialist':
            notes.append(f'simulated: {sim_share:.0%} in the generalist mix, {sim_final:.0%} in the target mix')

    weights = to_weights(shares, ceiling)

    # what the run would do with these (rounded) weights
    sch = WeightSchedule(names, weights, schedule, opt.epochs)
    per_epoch_w = np.array([sch(e) for e in range(opt.epochs)])  # (epochs, sources), normalised
    es = n.sum() if epoch_size == 'total' else epoch_size if isinstance(epoch_size, int) else \
        n[t] / per_epoch_w[:, t].mean()  # samples per epoch
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
        print(row + ('  <- target' if k == t else '  (simulated)' if sim[k] else ''))
    print('  (shares of the training samples, the frames loaded and the objects over the run; samples/frame: times '
          'each frame is the sampled frame; loads/frame: also as a mosaic / mixup / paste-in partner)')
    if schedule:
        fw = schedule['points'][0]['weights']
        print(f"  weight_schedule: linear to, at {SCHEDULE_PROGRESS:.0%} of --epochs: " +
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
        if sim[k]:
            s['simulated'] = True
        new_srcs.append(s)
    out['train_sources'] = new_srcs
    snippet = {'sampling': 'weighted', 'epoch_size': epoch_size,
               'train_sources': [{'name': nm, 'weight': weights[k], **({'target': True} if k == t else {}),
                                  **({'simulated': True} if sim[k] else {})} for k, nm in enumerate(names)]}
    if schedule:
        snippet['weight_schedule'] = schedule
    print('\nProposed (merge into your data yaml):')
    print(yaml.safe_dump(snippet, sort_keys=False, default_flow_style=None).rstrip())
    if opt.out:
        with open(opt.out, 'w') as f:
            f.write(f'# {opt.data} with weights proposed by: python tools/propose_weights.py '
                    f'{" ".join(sys.argv[1:])}\n')
            yaml.safe_dump(out, f, sort_keys=False, default_flow_style=None)
        print(f'\nwrote {opt.out} (the original comments are not kept)')


if __name__ == '__main__':
    main()

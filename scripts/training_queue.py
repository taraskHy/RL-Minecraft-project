"""Two-slot Windows training queue. Control: runs/training_control.json.

Only launches training; the main Codex chat publishes results. Existing logical
training process trees count towards the global two-job cap. A gaming change
requests a clean checkpoint pause and resumes the same run with the new limits.
"""
import argparse
import json
import msvcrt
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'runs'
STATE = RUNS / 'training_queue_state.json'
CONTROL = RUNS / 'training_control.json'
PLAN = ROOT / 'configs' / 'training_queue.json'
FLAGS = subprocess.CREATE_NO_WINDOW


def save(path, obj):
    tmp = path.with_suffix('.tmp')
    try:
        tmp.write_text(json.dumps(obj, indent=2))
        for attempt in range(20):
            try:
                os.replace(tmp, path)
                return True
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.25)
    except OSError as exc:
        # OneDrive, antivirus, or a reader can briefly deny replacement. Keep
        # the previous valid state file and the live in-memory job inventory;
        # retry on the next loop instead of abandoning the running trainers.
        print(f'State save deferred; queue continues tracking jobs: {exc}', flush=True)
        return False


def processes():
    code = "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress"
    raw = subprocess.check_output(['powershell', '-NoProfile', '-Command', code], text=True, creationflags=FLAGS)
    rows = json.loads(raw or '[]')
    return rows if isinstance(rows, list) else [rows]


def trainings(rows):
    jobs = {r['ProcessId']: r for r in rows if re.search(r'-m\s+parkour\.train\b', r.get('CommandLine') or '')}
    roots = [r for r in jobs.values() if r['ParentProcessId'] not in jobs]
    for r in roots:
        match = re.search(r'--run\s+([^\s"]+)', r['CommandLine'])
        r['run'] = match[1] if match else None
    return roots


def summary(name):
    p = RUNS / name / 'summary.json'
    try:
        return json.loads(p.read_text()) if p.exists() else {}
    except (OSError, ValueError):
        return {}


def next_seed(state, plan):
    # After the requested three arms, replicate the best recipe on a COMMON
    # validation metric (mean success on levels 0-5, standard + obstacles).
    # No test result enters this decision.
    candidates = []
    for spec in plan['jobs']:
        best = summary(spec['name']).get('best', {})
        levels = best.get('validation_per_level', {})
        if all(str(i) in levels for i in range(6)) and summary(spec['name']).get('completed'):
            score = sum(levels[str(i)]['success'] for i in range(6)) / 6
            candidates.append((score, spec))
    if not candidates:
        return None
    score, base = max(candidates, key=lambda x: x[0])
    seed = 1
    while f"{base['name']}-s{seed}" in state['jobs'] or (RUNS / f"{base['name']}-s{seed}").exists():
        seed += 1
    return {'name': f"{base['name']}-s{seed}", 'flags': base['flags'] + ['--seed', str(seed)],
            'selection': {'metric': 'common validation levels 0-5', 'score': score, 'base': base['name']}}


def main():
    RUNS.mkdir(exist_ok=True)
    lock = open(RUNS/'training_queue.lock', 'a+b')
    lock.seek(0)
    if not lock.read(1):
        lock.write(b'0'); lock.flush()
    lock.seek(0)
    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        raise SystemExit('A training queue already owns the lock')
    state = json.loads(STATE.read_text()) if STATE.exists() else {'jobs': {}}
    handles = {}
    print('Queue started; at most two logical training jobs globally', flush=True)
    while True:
        try:
            plan = json.loads(PLAN.read_text())
            control = json.loads(CONTROL.read_text()) if CONTROL.exists() else {'gaming': False, 'hold': False}
        except (OSError, ValueError) as exc:
            print(f'Configuration unavailable; no launch: {exc}', flush=True)
            time.sleep(5)
            continue
        try:
            rows = processes()  # fails closed: never launch without a valid process inventory
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            print(f'Process inventory unavailable; no launch: {exc}', flush=True)
            time.sleep(5)
            continue
        active = trainings(rows)
        names = {r['run'] for r in active}
        for spec in plan['jobs']:
            job = state['jobs'].setdefault(spec['name'], {**spec, 'status': 'pending'})
            if job['status'] == 'pending':
                job['flags'] = spec['flags']
        for name, job in list(state['jobs'].items()):
            if job['status'] == 'running' and name not in names:
                if name in handles:
                    rc = handles.pop(name).wait()
                else:
                    rc = None
                if summary(name).get('completed'):
                    job['status'] = 'completed'
                    print(f'COMPLETED {name}', flush=True)
                elif (RUNS/name/'pause_ready.json').exists():
                    job['status'] = 'pending'
                    job['resume'] = True
                    print(f'CHECKPOINT PAUSED {name}', flush=True)
                else:
                    job['status'] = 'failed'
                    job['exit_code'] = rc
                    print(f'FAILED {name}: {rc}; inspect its log', flush=True)
            if job['status'] == 'running' and job.get('gaming', False) != control.get('gaming', False):
                (RUNS/name/'pause_requested').touch()
        if not control.get('hold', False):
            if not any(j['status'] == 'pending' for j in state['jobs'].values()) and len(active) < 2:
                spec = next_seed(state, plan)
                if spec:
                    state['jobs'][spec['name']] = {**spec, 'status': 'pending'}
            for name, job in state['jobs'].items():
                if len(active) >= 2:
                    break
                if job['status'] != 'pending' or name in names:
                    continue
                folder = RUNS/name
                latest = folder/'checkpoints/latest.pt'
                if folder.exists() and (folder/'train_log.csv').exists() and not job.get('resume'):
                    job['status'] = 'blocked_existing'
                    print(f'Existing evidence blocks a duplicate: {name}', flush=True)
                    continue
                init = str(latest) if job.get('resume') else 'results/final_policy.pt'
                for marker in ['pause_requested', 'pause_ready.json']:
                    (folder/marker).unlink(missing_ok=True)
                args = [sys.executable, '-u', '-m', 'parkour.train', '--run', name, '--init-from', init,
                        '--obstacle-prob', '0.3', '--gamma', '0.995', '--iterations', '1000', '--test-once', '--compact-ice', '--eval-batch-size', '256'] + job['flags']
                fraction = float(control.get('memory_fraction_per_job', 0))
                if fraction:
                    args += ['--gpu-mem-fraction', str(fraction)]
                if control.get('gaming', False):
                    args += ['--gpu-duty', '0.9', '--threads', '8']
                # Smaller simultaneous tensor working set; the rollout still
                # contains 524,288 game ticks, as in the original recipe.
                if control.get('small_memory', False):
                    args += ['--num-envs', '8192', '--rollout', '64', '--minibatches', '32']
                logs = RUNS/'queue_logs'; logs.mkdir(exist_ok=True)
                with open(logs/f'{name}.log', 'a') as log:
                    proc = subprocess.Popen(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=FLAGS)
                handles[name] = proc
                job.update(status='running', pid=proc.pid, gaming=control.get('gaming', False), command=args)
                active.append({'run': name}); names.add(name)
                print(f'START {name} pid={proc.pid} gaming={job["gaming"]}', flush=True)
        state['updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
        save(STATE, state)
        time.sleep(5)


if __name__ == '__main__':
    main()

"""Resume the interrupted batch, then build the diagonal-PMMH comparison.

Uses the saved states and frozen parent implementation; no warmup is repeated.
The worker waits for sampler exit, without polling or restarting failed jobs.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import psutil

ROOT = Path(__file__).resolve().parent
CODE = ROOT.parents[1]
REPO = CODE.parents[1]
RUNTIME = CODE / '.runtime'
PARENT = ROOT / 'nuts'
OUTPUT = ROOT / 'nuts-resumed'


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def environment():
    cfg = json.loads((PARENT/'chain0/configuration.json').read_text())
    env = os.environ.copy()
    isolation = RUNTIME/'mps-isolated'
    isolation.mkdir(parents=True, exist_ok=True)
    env.update(CUDA_VISIBLE_DEVICES='0', JAX_PLATFORMS='cuda',
               JAX_ENABLE_X64='true', XLA_FLAGS=cfg['xla_flags'],
               XLA_PYTHON_CLIENT_PREALLOCATE='false',
               CUDA_MPS_PIPE_DIRECTORY=str(isolation),
               PYTHONPATH=str(RUNTIME/'deps'),
               MPLCONFIGDIR=str(RUNTIME/'matplotlib'))
    return env


def worker():
    command = [sys.executable, str(PARENT/'source/batch_extend.py'),
               '--parent', str(PARENT), '--output', str(OUTPUT),
               '--recover-interrupted', '--checkpoint', '25']
    process = subprocess.Popen(command, cwd=REPO, env=environment())
    record = dict(utc=datetime.now(timezone.utc).isoformat(), pid=process.pid,
                  create_time=psutil.Process(process.pid).create_time(), command=command,
                  parent='nuts', output='nuts-resumed', warmup_repeated=False,
                  comparison_iterations=[501, 1500],
                  gpu=0, mps=False, log=str(RUNTIME/'identity-resume.log'))
    write(ROOT/'resume-launch.json', record)
    result = process.wait()
    if result:
        write(RUNTIME/'identity-pipeline-status.json', dict(phase='sampler failed', returncode=result))
        raise SystemExit(result)
    comparison = [sys.executable, str(CODE/'identity_comparison.py'),
                  '--nuts', str(OUTPUT),
                  '--pmmh', str(CODE/'results/historical-proposal-control-v1/pmrw'),
                  '--parent', str(PARENT), '--output', str(ROOT/'comparison')]
    subprocess.run(comparison, cwd=REPO, env=environment(), check=True)
    write(RUNTIME/'identity-pipeline-status.json', dict(phase='comparison ready',
          utc=datetime.now(timezone.utc).isoformat(), output=str(ROOT/'comparison')))


def main():
    if '--worker' in sys.argv:
        return worker()
    import numpy as np
    if OUTPUT.exists() or (ROOT/'resume-launch.json').exists():
        raise ValueError('Recovery already launched; inspect before continuing')
    old = json.loads((ROOT/'launch.json').read_text())
    if psutil.pid_exists(old['pid']):
        process = psutil.Process(old['pid'])
        if process.create_time() == old['create_time'] and process.status() != 'zombie':
            raise ValueError('Original process still alive')
    checkpoints = {}
    for c in range(4):
        path = PARENT/f'chain{c}/checkpoint.npz'
        with np.load(path) as cp:
            assert len(cp['trace']) == 500
            assert np.isfinite(cp['next_step_size']) and cp['next_step_size'] > 0
        checkpoints[str(path.relative_to(REPO))] = hashlib.sha256(path.read_bytes()).hexdigest()
    write(PARENT/'interruption.json', dict(verified_terminal=True, pid=old['pid'],
          create_time=old['create_time'], checked_utc=datetime.now(timezone.utc).isoformat(),
          evidence='Original process absent or PID reused; workspace HANDOFF records GPU MPS fault and reboot.',
          checkpoint_sha256=checkpoints, authorization='User requested checkpoint continuation without repeating warmup.'))
    RUNTIME.mkdir(exist_ok=True)
    with (RUNTIME/'identity-resume.log').open('x') as log:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker'],
                                   cwd=REPO, env=environment(), stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps(dict(worker_pid=process.pid, log=str(RUNTIME/'identity-resume.log'))))


if __name__ == '__main__':
    main()

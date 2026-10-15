"""Freeze shared starts and a proposal metric from the previous PMMH run.

This is sampler preparation, not a prior fit. Both methods receive identical
inputs. The old chains remain development runs and are not pooled with these.
"""
from pathlib import Path
import hashlib
import io
import json
import numpy as np

root = Path(__file__).resolve().parent
out = root/'inputs'
out.mkdir(exist_ok=False)
prefixes, manifests = [], []
for c in range(4):
    parent = root.parent/'sampling-recovery-v1'/f'pmrw-chain{c}'
    payload = (parent/'checkpoint.npz').read_bytes()
    cp = np.load(io.BytesIO(payload))
    assert len(cp['trace']) >= 1950
    prefix = cp['trace'][:1950].copy()
    prefixes.append(prefix)
    np.save(out/f'initial{c}.npy', prefix[-1, 2:])
    manifests.append(dict(parent=str(parent.relative_to(root.parent)),
                          checkpoint_sha256=hashlib.sha256(payload).hexdigest(),
                          source_iteration_count=len(cp['trace']), selected_start_iteration=1950,
                          preparation_iteration_seconds=float(cp['iteration_seconds'][:1950].sum()),
                          parent_configuration=json.loads((parent/'configuration.json').read_text())))
prefixes = np.stack(prefixes)
np.savez_compressed(out/'preparation_traces.npz', trace=prefixes)
# Use only the second half of the old warmup to estimate a common proposal
# covariance. Retain between-chain spread; regularize before factorization.
cloud = prefixes[:, 500:1000, 2:].reshape(-1, 18)
cov = np.cov(cloud.T)
cov = .99*cov + .01*np.diag(np.diag(cov)) + 1e-6*np.eye(18)
old = np.load(root.parent/'candidate-pmrw-4chain-v1/chain0/initial_chol.npy')
cov *= np.sum(old**2)/np.trace(cov)
chol = np.linalg.cholesky(cov)
np.save(out/'chol.npy', chol)
manifest = dict(purpose='Shared initialization and proposal scale only; common proper prior unchanged',
                covariance_rows='Iterations 501 through 1000 of each old PMMH chain',
                covariance_regularization='0.99 covariance + 0.01 diagonal + 1e-6 I; trace normalized to original score metric',
                covariance_eigenvalues=np.linalg.eigvalsh(cov).tolist(),
                preparation_cost='Include original shared optimization/score metric and four PMMH prefixes through iteration 1950. These are preparation costs for both methods, not free starts.',
                parents=manifests,
                files_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in out.iterdir()})
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

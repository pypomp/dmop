"""Freeze the longest chain0 plateau and comparison states through iteration1400."""
from pathlib import Path
import hashlib
import io
import json
import numpy as np

root=Path(__file__).resolve().parent
assert not (root/'inputs.npz').exists()
parent=root.parent/'gpu-comparison-v1/pmrw'
traces={};provenance={}
for c in [0,1,3]:
    payload=(parent/f'chain{c}/checkpoint.npz').read_bytes()
    with np.load(io.BytesIO(payload)) as cp:
        assert len(cp['trace'])>=1400
        traces[c]=cp['trace'][:1400].copy()
        provenance[c]=dict(parent=str(parent/f'chain{c}'),available_iterations=len(cp['trace']),
                           checkpoint_sha256=hashlib.sha256(payload).hexdigest())
z=traces[0][500:,2:]
starts=np.flatnonzero(np.r_[True,np.any(np.diff(z,axis=0)!=0,axis=1)])
lengths=np.diff(np.r_[starts,len(z)])
which=int(np.argmax(lengths));plateau=500+int(starts[which])
selection=[(0,plateau),(0,1399),(1,1399),(3,1399)]
np.savez_compressed(root/'inputs.npz',z=np.stack([traces[c][i,2:] for c,i in selection]),
                    cached_loglik=np.array([traces[c][i,0] for c,i in selection]),recorded_particles=5000)
manifest=dict(selection='Longest chain0 plateau after warmup through1400 (earliest tie), then states at1400 from chains0/1/3',
              points=[dict(point=p,chain=c,iteration=i+1) for p,(c,i) in enumerate(selection)],
              plateau_iterations=int(lengths[which]),parents=provenance,
              inputs_sha256=hashlib.sha256((root/'inputs.npz').read_bytes()).hexdigest(),
              protocol=dict(particles=[5000,10000],reps=32,batch_size=16,precision='float64',
                            gpu=1,purpose='Diagnose filter noise and retained likelihood overestimation; no posterior draws',
                            refresh_check='Mean min(1,exp(fresh_loglik-cached_loglik)) at fixed theta, only for J5000',
                            uncertainty='Thirty-two replicates provide a local diagnostic, not a proof of particle adequacy throughout the posterior'))
(root/'protocol.json').write_text(json.dumps(manifest,indent=2)+'\n')

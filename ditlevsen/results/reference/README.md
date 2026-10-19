# Shared IFAD and IF2 reference inputs

Both comparison methods use these inputs from the main Dhaka experiment.

| Tracked file | Contents |
|---|---|
| [ifad097_comparable_post_if2.npz](ifad097_comparable_post_if2.npz) | The 100 parameter vectors after the IF2 stage used to initialize IFAD-0.97 |
| [manuscript_likelihood.csv](manuscript_likelihood.csv) | Independent reference likelihoods, indexed by method and effort |
| [manuscript_parameters.csv](manuscript_parameters.csv) | Reference parameter estimates |
| [manuscript_trace_summary.csv](manuscript_trace_summary.csv) | Reference optimization progress |

The current comparable-effort IFAD-0.97 export has median -3744.40 and maximum
-3743.68. The SI uses the intermediate 175-iteration IF2 checkpoint as its
warm-start baseline, not the full-budget IF2 result. The independent baseline
evaluations are in
[corenflos/results/ifad097_post_if2_reference/](../../../corenflos/results/ifad097_post_if2_reference/).

[ditlevsen.warm_starts](../../ditlevsen/warm_starts.py) extracts and checks the
IF2 vectors; [ditlevsen.reference_results](../../ditlevsen/reference_results.py)
exports the reference summaries. Both read the main experiment's original
objects under `code/global_search/` (and historical Git objects for some
references). Re-extraction requires those archived inputs. A fresh clone can
use these tracked exports directly.

The companion provenance JSON and original result objects are covered by the
[archive](../../../artifacts/README.md). Older development notes contain
superseded IFAD summaries; the tracked CSVs here and
[imgs/precise_table.tex](../../../imgs/precise_table.tex) are the current
manuscript reference.

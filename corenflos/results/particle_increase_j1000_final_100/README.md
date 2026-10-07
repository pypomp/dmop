# Four 1,000-particle comparisons

This experiment is produced by
[corenflos.particle_experiment](../../corenflos/particle_experiment.py).
It is stored under Corenflos because that package owns the shared driver.

| Directory | Method | Initialization |
|---|---|---|
| [corenflos_vanilla/](corenflos_vanilla/) | Corenflos | Global bounding box |
| [corenflos_warm/](corenflos_warm/) | Corenflos | Shared IF2 estimates |
| [ditlevsen_vanilla/](ditlevsen_vanilla/) | Ditlevsen | Global bounding box |
| [ditlevsen_warm/](ditlevsen_warm/) | Ditlevsen | Shared IF2 estimates |

`manifest.json` records all four configurations, including original absolute
paths. `status.json` and audit outputs record completion. The driver copied
the corresponding 100-particle settings, changed the fitting particle count,
and ran serial partitions. All 100 starts contribute to each final summary.
See the [parent result index](../README.md) for CSV meanings and archive needs.

All four variants appear in the SI tables. Only the Ditlevsen variants from
this directory appear in the current figures, paired with the 100-particle
Corenflos runs. Those figures and their settings exports are in
[imgs/competitors/corenflos100_ditlevsen1000/](../../../imgs/competitors/corenflos100_ditlevsen1000/README.md).
The `figures/` folder here is the earlier comparison using 1,000 particles
for both methods.

For a new run, use a new output directory following the
[reproduction guide](../../../code/competitor_reproduction.md). An existing
manifest is checked exactly, including paths; a relocated archive is not
an instruction to overwrite that provenance and resume with new settings.

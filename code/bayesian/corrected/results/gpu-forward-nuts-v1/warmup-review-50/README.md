# First 50 NUTS warmup updates

All four chains move in every coordinate, with no recorded divergences.
Mean trajectory acceptance ranges from 0.638 to 0.687. Every trajectory
uses 31 integration steps and reaches depth 5 without a U-turn. The median
postcompilation batch time is 46.1 seconds; this is shared-machine timing.

The weakest movement is in the fifth immigration coefficient. Its RMS jump
is 0.20--0.24 times the standard deviation in the frozen J5000 PMMH review.
The sixth coefficient's visited range is only 1.06--1.92 times that reference
standard deviation. The movement plot was inspected across all 18 parameters:
chains remain separated in several coordinates, and the immigration lower
tails are poorly explored. These are warmup observations, not posterior
samples or a convergence diagnosis.

Grey bands show the 95% intervals from the **unconverged** 5,063-draw PMMH
snapshot. They provide provisional scales, not known posterior intervals.
The initial title/legend overlap was corrected by moving the band description
into the legend; the figure was rerendered from the same frozen positions and
visually inspected again.

This supports a small depth-5 versus depth-6 comparison of movement per unit
time, recorded in `../../nuts-depth-check-v1`. The current production run
continues unchanged while that comparison is made.

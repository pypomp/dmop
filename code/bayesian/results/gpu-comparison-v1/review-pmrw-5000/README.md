# Interim J5000 PMMH review

This snapshot contains 5,063 retained draws from each of four chains, after
500 warmup updates. The draws come from `../pmrw-continued`; its parent run
is part of the same chains and must not be counted again. `audit.json`
records the target, checkpoint hashes and retained-draw hash.

The chains have not converged. Across all 18 coordinates, rank split R-hat
ranges from 1.012 to 1.133, minimum bulk ESS is 25.3, and minimum tail ESS is
30.8. Chain acceptance rates are 0.145, 0.201, 0.289 and 0.288. Agreement has
improved since the 2,338-draw review, whose maximum R-hat was 1.235, but
the seasonal immigration coefficients remain slow to explore.

All three figures were visually inspected. The trend marginal curves overlap
reasonably, although autocorrelation remains substantial at lag 100.
The fifth seasonal immigration coefficient has a long lower tail visited
unevenly by the chains; the second chain also makes a late lower-tail
excursion in the sixth coefficient. These features are visible in both the
18-coordinate traces and marginal plots. Concentration relative to the
plotted prior does not establish insensitivity to the prior.

The four-panel trend figure retains the original histogram, density,
autocorrelation and trace layout. The warmup legend now has an opaque
background so the vertical marker does not cross its text. All axes remain
linear in the displayed estimation coordinates; no numerical report
calculation changed.

Continue the existing planned run, with the next review at its 10,000-draw
completion. The J10000 primary comparison, particle-count agreement and
wider-prior comparison remain outstanding. These are development figures,
not manuscript results.

# JRSSB review manuscript

`../ms-jrssb.tex` is the separate submission source. Scientific content comes
from `../ms.tex`; formatting and the cover page are applied by
`../code/build_jrssb.py`. The original manuscript and supplement are unchanged.

From the repository root:

```sh
python3 code/build_jrssb.py
make -C .. dmop/ms-jrssb.pdf
```

The second command uses the shared workspace's LaTeX setup. Elsewhere, run
`latexmk -pdf ms-jrssb.tex` from the repository root with a full TeX Live
installation. On Overleaf, select `ms-jrssb.tex` as the main document. The
generated PDF and auxiliary files are ignored by Git.

Edit manuscript content in `ms.tex`, layout in `preamble.tex`, and cover-page
details in the generator, then regenerate. The submission copy uses A4,
12pt body text with a 24pt baseline and room for 28 text lines, a separate
title/contact page, and tables at the end. Captions, references, tables and
algorithms retain 12pt text. Figures retain the existing PDF assets. The
corresponding author is Edward L. Ionides, using his departmental address.

The November 3 build is 38 pages including the cover, abstract, references and
tables. This exceeds the journal's suggested length; the conversion preserves
the text rather than shortening it. It implements the requested review format,
not the journal's final production preparation.

## Template sources

The [JRSSB instructions](https://academic.oup.com/jrsssb/pages/general-instructions)
link to the [RSS Overleaf template](https://www.overleaf.com/latex/templates/template-for-journal-of-the-royal-statistical-society-series-c-applied-statistics/xhkktkrcfswc).
`template.tex` preserves that template's source as retrieved on November 3,
2026. It is a reference example, not a manuscript build target; its sample
figures and bibliography are not bundled here.

`oup-authoring-template.cls` is the unmodified OUP class, version 1.5, from
[CTAN](https://ctan.org/pkg/oup-authoring-template). Its license is LPPL 1.3 or
later; the upstream `manifest.txt` is included. `provenance.json` records URLs,
retrieval date and SHA-256 checksums. Review-layout overrides live separately
in `preamble.tex`, including the A4 crop stock and caption/float spacing that
the production class otherwise overrides.

The OUP class needs packages beyond the original manuscript's dependencies.
For a minimal TinyTeX installation, the additional packages used here are
`crop sttools changepage totcount subfloat fontsize anyfontsize footmisc
mdwtools wrapfig caption listings jknapltx rsfs multirow setspace pgf`.

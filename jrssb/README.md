# JRSSB review manuscript

`../ms-jrssb.tex` is the separate submission source. Scientific content comes
from `../ms.tex`; formatting and the RSS title metadata are applied by
`../code/build_jrssb.py`. The original manuscript and supplement are unchanged.

From the repository root:

```sh
python3 code/build_jrssb.py
make -C .. dmop/ms-jrssb.pdf
```

The second command uses the shared workspace's LaTeX setup. Elsewhere, run
`latexmk -pdf ms-jrssb.tex` from the repository root with a full TeX Live
installation. On Overleaf, select `ms-jrssb.tex` as the main document. The
submission PDFs `ms-jrssb.pdf` and `si.pdf` are tracked in Git. Auxiliary
files and other generated documents remain ignored.

Edit manuscript content in `ms.tex`, layout in `preamble.tex`, and title-page
details in the generator, then regenerate. The submission copy uses A4,
12pt body text with a 24pt baseline and room for 28 text lines, the native RSS
title/author/contact/abstract layout, and tables at the end. Captions, references, tables and
algorithms retain 12pt text; the title block retains the template's own font
sizes, including its 9pt abstract. Figures retain the existing PDF assets. The
corresponding author is Edward L. Ionides, using his departmental address.

The November 3 build is 37 pages including the title/abstract, references and
tables. This exceeds the journal's suggested length; the conversion preserves
the text rather than shortening it. It implements the requested review format,
not the journal's final production preparation.

## Template sources

The [JRSSB instructions](https://academic.oup.com/jrsssb/pages/general-instructions)
link to the [RSS Overleaf template](https://www.overleaf.com/latex/templates/template-for-journal-of-the-royal-statistical-society-series-c-applied-statistics/xhkktkrcfswc).
`template.tex` preserves that template's source as retrieved on November 3,
2026. It is a reference example, not a manuscript build target; its sample
figures and bibliography are not bundled here.

`oup-authoring-template.cls` is the RSS fork dated July 8, 2024, recovered
unchanged from the [original diffPomp manuscript](https://github.com/hetankevin/diffPomp/blob/9e5aaa2b4dcdc453a94806967b5c9989c788b4bc/paper/oup-authoring-template.cls).
Its executable code matches the publisher's OUP 1.1 bundle; only three RSS
copyright/version comments differ. This replaces the incompatible generic
CTAN 1.5 class initially used here. The license is LPPL 1.2 or later; the
upstream 1.1 `manifest.txt` is included. `provenance.json` records retrieval
sources and checksums.

The generator uses the template's native title, author, address, correspondence,
abstract, keywords and `\maketitle` commands. The native running heads are
retained. At the author's request, the original journal banner, DOI/year/date
placeholders, gray logo box and publisher copyright footer are retained
exactly as in the template. Review-layout overrides in `preamble.tex` set
the A4 crop stock and body/caption/float spacing. They do not replace the
native title layout or publisher header.

The OUP class needs packages beyond the original manuscript's dependencies.
For a minimal TinyTeX installation, the additional packages used here are
`crop sttools changepage totcount subfloat fontsize anyfontsize footmisc
mdwtools wrapfig caption listings jknapltx rsfs multirow setspace pgf cm-super`.

# Figure assets used by the manuscript

| Asset | Use | Source and reproduction |
|---|---|---|
| [mop.png](mop.png) | Fig. 1A: MOP likelihood illustration | [code/fig1a/](../../code/fig1a/README.md), with input data and saved curves |
| [biasvar.png](biasvar.png) | Fig. 1B: bias/variance illustration | Separate from the Fig. 1A generator |
| [tikzcholera.png](tikzcholera.png) | Dhaka compartment diagram | Included directly by `ms.tex` |

For Fig. 1A, edit `code/fig1a/plot.py` and run it from the repository root.
That command changes labels and redraws the saved numerical curves without
rerunning the particle filters. The parent code guide links the other
[manuscript experiments](../../code/README.md).

# Implementation correspondence

The resampling code follows FilterFlow's `RegularisedTransform` and
`regularized_transport` modules:

- pairwise cost is half the squared Euclidean distance;
- particles are centered and divided by the largest coordinate standard
  deviation times the square root of the state dimension;
- Sinkhorn starts at the squared particle-cloud diameter, reduces epsilon by
  `scaling^2`, and averages consecutive potential updates;
- converged potentials are detached and followed by one differentiable
  fixed-point update;
- the transport adjoint is clipped elementwise to `[-1, 1]`;
- columns of the transport matrix sum exactly to `J w`, rows sum approximately
  to one, and transformed particles are `T @ x`;
- resampling is triggered when relative ESS falls below 0.5.

The Dacca transition is the same 20-step Gaussian Euler recurrence used by
Pypomp. Process normals are reparameterized, making the filter likelihood
differentiable with respect to the 23 estimated parameters. As in the
Corenflos experiments, particle randomness changes between optimizer updates.

The transport uses all six physical Dacca state coordinates, including `Mn`,
the within-month death accumulator. This matches FilterFlow's treatment of the
particle state even though Pypomp resets `Mn` before the next interval.
`count` only records whether a discretized trajectory crossed the state
boundary. It receives the Pypomp likelihood floor but is not included in the
transport. After transport it is reset, since an entropic matrix is dense and
transporting this indicator would otherwise mark every output particle invalid
whenever any negligible input particle failed.

The differentiable-filter likelihood is used only for fitting. Stored
parameters and every optimizer update are scored afterward with Pypomp's
ordinary Euler-20 particle filter. Consequently the figures report the same
scientific target used for IFAD, IF2, and Ditlevsen rather than the biased
training likelihood.

Signal Extraction and Baseline Refinement
=========================================

:class:`jacscanomaly.PlanetSignalExtractor` addresses a feedback problem in
planetary-anomaly searches: a strong local anomaly can bias a single-lens fit,
and the biased baseline can in turn obscure the anomaly. The extractor proposes
signal intervals from the standard local-template grid, reduces their influence
on the baseline fit, and evaluates the resulting baseline on the full light
curve.

Common setup
------------

The extractor begins with a fitted baseline, or with fixed nonlinear
parameters when ``refit=False``. It repeatedly uses the finder's grid scanner
to locate the strongest residual candidate. Candidates below
``seed_min_dchi2`` stop the procedure.

For each accepted refinement, the fit is always evaluated again on the full
data. The result consequently has two residual series:

``initial_residual``
   Residual against the initial baseline.

``refined_residual``
   Residual against the final baseline, including signal points in the
   evaluation but not necessarily in its fit.

``signal_mask`` identifies hard-excluded points. ``point_weight`` records the
continuous weights used by robust mode. ``iterations`` records every accepted
or terminating mask/refit decision.

Hard-mask mode
--------------

``baseline_mode="mask"`` is a greedy procedure. For the best grid seed
``(t0, teff)``, it first opens a window with half-width

.. math::

   \max(\mathtt{mask\_teff\_coeff}\,t_\mathrm{eff},
        \mathtt{mask\_min\_half\_width}).

Inside the window, points join the mask only if both their absolute z-score and
their local template improvement exceed relative-or-absolute thresholds. The
core can be padded by ``mask_core_pad_teff``. The baseline is refit on the
remaining points and the full light curve is re-evaluated.

The addition is rejected when it masks more than ``max_mask_fraction`` of the
data or makes the retained-data reduced chi-square worse by more than
``max_unmasked_chi2_dof_increase``. The process repeats up to ``max_iter``.

Robust-weight mode
------------------

``baseline_mode="robust"`` does not choose discrete intervals first. It maps
the absolute residual z-score to a target weight, smooths those weights in
time, and relaxes the current weight toward the target by ``robust_eta``. The
new baseline is then fitted with weighted residuals. Iteration stops when the
largest weight change is below ``robust_min_weight_change`` or the
``robust_max_iter`` limit is reached.

The final hard ``signal_mask`` is derived from low weights and residual support
for reporting. Use this mode for broad or poorly bounded structure; use a hard
mask when the excluded intervals themselves are scientifically meaningful.

Beam-interval mode
------------------

``baseline_mode="beam_interval"`` is the default. Rather than committing to a
single mask immediately, it keeps up to ``beam_width`` partial solutions. For
each current branch it proposes several interval widths around the best seed,
plus a residual-grown interval based on ``beam_grow_min_abs_z``. Duplicate
intervals are removed and only the best
``beam_candidates_per_iter`` proposals are considered.

Each candidate branch is ranked by

.. math::

   S = \chi^2_\mathrm{kept}
       + \lambda_p N_\mathrm{masked}
       + \lambda_I N_\mathrm{intervals}
       + \lambda_w W_\mathrm{masked},

where the three penalties are ``beam_point_penalty``,
``beam_interval_penalty``, and ``beam_width_penalty``. Lower is better. The
same mask-fraction, retained-data, and catastrophic-fit safeguards used by hard
mask mode apply to each branch. After ``beam_max_iter`` rounds, the lowest-score
branch becomes the refined result.

Safeguards and prior windows
----------------------------

If a candidate refinement has a reduced chi-square catastrophically larger
than its reference fit according to ``max_refined_chi2_dof_ratio``, it is
discarded. A final flat-baseline diagnostic can replace an unsupported masked
PSPL peak with a flat model. This protects events in which the apparent peak
was entirely masked away.

``prior_signal_windows=((center, half_width), ...)`` forces known intervals
into the final signal mask and performs one guarded refit. It is appropriate
for manual follow-up windows, not for encoding a final physical-model result.

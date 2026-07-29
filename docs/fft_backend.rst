FFT anomaly-grid backend
========================

``jacscanomaly`` can evaluate the residual anomaly grid with FFT correlations:

.. code-block:: python

   from jacscanomaly import Finder, FinderConfig

   config = FinderConfig(
       grid_backend="fft",
       fft_oversample=4,
   )
   result = Finder(config).run(time, flux, ferr, x0=x0)

The public workflow and result objects are unchanged. The option affects the
``(t0, teff)`` residual scan and the same scan used for automatic single-lens
initialization when ``x0`` is omitted.

The two existing templates
--------------------------

The backend evaluates both anomaly templates already used by the JAX and C++
implementations. With

.. math::

   \tau = \frac{t-t_0}{t_{\rm eff}},
   \qquad Q = 1 + \tau^2,

they are

.. math::

   A_0(\tau) = \frac{1}{\sqrt{1+\tau^2}}

and

.. math::

   A_1(\tau) = \frac{Q+2}{\sqrt{Q(Q+4)}}.

``A0`` is the high-magnification approximation. ``A1`` is the lower-
magnification template used by the existing anomaly scan. For every grid
point, both weighted line fits are evaluated and the one with the smaller
chi-square is retained. The template amplitude remains unconstrained, so the
same scan detects positive and negative residual structures.

The local constant fit is also a correlation
--------------------------------------------

For each ``(t0, teff)``, the scan only uses observations inside

.. math::

   |t-t_0| < c\,t_{\rm eff},

where ``c`` is ``teff_coeff``. Let :math:`b(t-t_0)` denote this translated box
window. Irregular observations are accumulated on a regular calculation grid
as

.. math::

   N_j = \sum_{i\in j} 1,
   \qquad W_j = \sum_{i\in j} w_i,

.. math::

   (WY)_j = \sum_{i\in j} w_i y_i,
   \qquad (WY^2)_j = \sum_{i\in j} w_i y_i^2.

A box correlation supplies the local sufficient statistics for every center
simultaneously:

.. math::

   N = N \star b,
   \quad W = W \star b,
   \quad Y = WY \star b,
   \quad Y_2 = WY^2 \star b.

The analytically minimized weighted constant-model chi-square is then

.. math::

   \chi^2_{\rm flat} = Y_2 - \frac{Y^2}{W}.

Thus the null model does not require an independent fit at every trial center.

Profiled template fits
----------------------

For either translated template :math:`x`, the FFT pass also computes

.. math::

   X = W \star (b x),
   \qquad XX = W \star (b x^2),
   \qquad XY = WY \star (b x).

After centering within the same local window,

.. math::

   S_{xx} = XX - \frac{X^2}{W},
   \qquad
   S_{xy} = XY - \frac{X Y}{W}.

Here ``XY`` on the right-hand side denotes the weighted data-template
correlation, while :math:`X Y` is the product of the local weighted template
sum and local weighted flux sum.

The improvement over the local constant model is

.. math::

   \Delta\chi^2 = \frac{S_{xy}^2}{S_{xx}}.

The calculation is performed for both ``A0`` and ``A1``. Numerically singular
templates are rejected, and the larger valid improvement is stored.

Irregular observations
----------------------

The observations themselves do not need to be evenly spaced. For a trial
``teff``, the existing center grid has spacing

.. math::

   \Delta t_0 = \mathtt{dt0\_coeff}\,t_{\rm eff}.

The FFT calculation grid uses

.. math::

   \Delta t_{\rm FFT}
   = \frac{\Delta t_0}{\mathtt{fft\_oversample}}.

Each observation contributes its count, weight, weighted flux, and weighted
squared flux to its nearest calculation bin. Empty bins have zero weight.
Keeping the original weighted squared-flux sums means that only the window and
template values are approximated as constant within a bin.

The default ``fft_oversample=4`` gives about 24 calculation cells per
``teff`` with the default ``dt0_coeff=0.17``. Increase it when unusually sharp
features or very uneven weights make the binning approximation important.

Exact candidate refinement
--------------------------

The full FFT surface is used to extract the same non-overlapping cluster
population as the other backends. Each extracted cluster representative is
then evaluated directly on the original timestamps.

This exact pass:

* recomputes the local constant fit,
* fits both ``A0`` and ``A1`` and applies the existing tie rule,
* replaces the representative ``dchi2``, and
* computes ``n_window``, ``n_contrib``, ``n_eff``, ``peak_frac``, ``rho1``,
  and ``longest_run`` with the same definitions as the JAX and C++ backends.

Consequently, candidate criteria and automatic-initialization quality cuts use
exact representative metrics. In ``grid_metrics``, the dense FFT grid contains
FFT ``dchi2`` and window counts; the detailed quality columns are materialized
for extracted representatives, which are the rows used downstream.

Configuration
-------------

``grid_backend`` accepts ``"jax"``, ``"cpp"``, or ``"fft"``. FFT-specific
controls are:

``fft_oversample``
   Calculation-grid cells per ``t0`` interval. Larger values improve the
   approximation and increase memory and runtime.

``fft_max_grid_points``
   Maximum calculation-grid length for one timescale. This prevents an
   unexpectedly fine shortest-timescale grid from allocating a very large FFT.

``fft_singular_rtol``
   Relative threshold for rejecting a template whose centered weighted norm is
   numerically zero.

The JAX chunking options do not affect the FFT backend.

Complexity and when it helps
----------------------------

Let ``M`` be the observation count, ``K`` the number of trial centers for one
timescale, and ``G`` the oversampled regular calculation-grid length. A direct
scan evaluates roughly ``M`` observations at each center, giving
:math:`O(MK)`. The FFT pass costs :math:`O(G\log G)` for that timescale, plus a
direct evaluation of the much smaller set of extracted representatives.

FFT is most useful for long, densely searched seasons. For short seasons or a
small grid, the C++ backend can remain faster because it avoids FFT setup and
binning.

Relation to ``PSPLFFTScanner``
------------------------------

The two FFT facilities solve different stages:

* :class:`jacscanomaly.PSPLFFTScanner` searches the full light curve with exact
  PSPL shapes over ``(u0, teff)`` to produce single-lens initial values.
* ``FinderConfig(grid_backend="fft")`` accelerates the localized residual
  anomaly grid with the two established ``A0`` and ``A1`` templates and the
  local constant null model.

They can be used independently or together.

from __future__ import annotations

import numpy as np


def polynomial_design(time: np.ndarray, center: float, order: int) -> np.ndarray:
    order = max(0, int(order))
    tau = np.asarray(time, dtype=float) - float(center)
    scale = max(float(np.max(np.abs(tau))) if tau.size else 1.0, 1.0)
    x = tau / scale
    return np.column_stack([x ** k for k in range(order + 1)])


def weighted_linear_fit(
    design: np.ndarray,
    y: np.ndarray,
    ferr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    X = np.asarray(design, dtype=float)
    y = np.asarray(y, dtype=float)
    ferr = np.maximum(np.asarray(ferr, dtype=float), 1e-12)
    if X.ndim != 2 or X.shape[0] != y.size:
        raise ValueError("Design matrix must have shape (n_data, n_columns).")
    finite = np.isfinite(y) & np.isfinite(ferr) & np.all(np.isfinite(X), axis=1)
    if int(np.sum(finite)) < X.shape[1]:
        coeff = np.full(X.shape[1], np.nan, dtype=float)
        model = np.zeros_like(y, dtype=float)
        return coeff, model, float("inf"), False

    Xf = X[finite]
    yf = y[finite]
    wf = 1.0 / ferr[finite]
    Xw = Xf * wf[:, None]
    yw = yf * wf
    try:
        coeff, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    except np.linalg.LinAlgError:
        coeff = np.full(X.shape[1], np.nan, dtype=float)
        model = np.zeros_like(y, dtype=float)
        return coeff, model, float("inf"), False

    model = X @ coeff
    z = (y - model) / ferr
    chi2 = float(np.sum(z[finite] * z[finite]))
    return coeff.astype(float), model.astype(float), chi2, bool(np.isfinite(chi2))

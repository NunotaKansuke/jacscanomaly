"""JSON output of the pipeline parts.

The parts give their own records (:meth:`AnomalyResult.to_dict
<jacscanomaly.AnomalyResult.to_dict>`, :meth:`RobustFit.to_dict
<jacscanomaly.RobustFit.to_dict>`); the pipeline arranges them::

    write_json("results.json", {"evt1": {"scan": result, "host": host}, ...})
"""

from __future__ import annotations

import json

import numpy as np


def _json_safe(x):
    if hasattr(x, "to_dict"):
        return _json_safe(x.to_dict())
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_json_safe(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    return x


def _dump(x, ind: int = 0) -> str:
    """Indented JSON; number lists and short flat dicts on one line."""
    pad, sub = "  " * ind, "  " * (ind + 1)
    flat = lambda v: not isinstance(v, (dict, list)) or (
        isinstance(v, list) and all(not isinstance(y, (dict, list)) for y in v))
    if isinstance(x, list) and x and not all(flat(y) for y in x):
        return "[\n" + ",\n".join(sub + _dump(y, ind + 1) for y in x) + "\n" + pad + "]"
    if isinstance(x, dict) and (not all(flat(v) for v in x.values()) or len(json.dumps(x)) > 200):
        sep = ",\n\n" if ind == 0 else ",\n"  # a blank line between top-level entries
        return "{\n" + sep.join(f"{sub}{json.dumps(k)}: {_dump(v, ind + 1)}"
                                for k, v in x.items()) + "\n" + pad + "}"
    return json.dumps(x, separators=(", ", ": "))


def write_json(path, obj) -> dict:
    """Write ``obj`` to ``path``; returns what was written.

    ``obj`` is any nesting of dicts, lists, numbers and objects with
    ``to_dict()`` (an :class:`~jacscanomaly.AnomalyResult`, a
    :class:`~jacscanomaly.RobustFit`). ``nan`` is written as ``null``.
    """
    doc = _json_safe(obj)
    with open(path, "w") as fh:
        fh.write(_dump(doc) + "\n")
    return doc


__all__ = ["write_json"]

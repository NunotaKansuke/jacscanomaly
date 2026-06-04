from __future__ import annotations

import importlib.util
import sys
import types


if importlib.util.find_spec("microjax") is None:
    microjax = types.ModuleType("microjax")
    fastlens = types.ModuleType("microjax.fastlens")

    class _DummyFSPLDisk:
        def A(self, *_args, **_kwargs):
            raise RuntimeError("microjax is required for FSPL magnification tests")

    def fspl_disk():
        return _DummyFSPLDisk()

    fastlens.fspl_disk = fspl_disk
    microjax.fastlens = fastlens
    sys.modules["microjax"] = microjax
    sys.modules["microjax.fastlens"] = fastlens

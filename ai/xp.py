"""
Backend GPU/CPU pour TruePneumoniaAI.

Toutes les couches importent ce module via :
    from xp import xp as np

Si CuPy est disponible et qu'un GPU CUDA est détecté, xp = cupy.
Sinon, xp = numpy (fallback transparent).
"""

try:
    import cupy as _cp
    _cp.cuda.runtime.getDeviceCount()          # lève une exception si pas de GPU
    xp = _cp
    GPU = True
    _name = _cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    print(f"[xp] Backend GPU : CuPy  ({_name})")
except Exception:
    import numpy as _np
    xp = _np
    GPU = False
    print("[xp] Backend CPU : NumPy (CuPy non disponible ou aucun GPU détecté)")

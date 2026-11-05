"""
Fast APSPN Kernel - Python wrappers for high-performance APSPN-based algorithms

This package provides Python interfaces to C++ implementations of:
- Brandes-APSPN-kernel
- CC-APSPN-kernel
- Parallel CC-APSPN-kernel (OpenMP)
- CC-ApproxBC and CC-ApproxGC
- CM-GC
- CC-SE-BCC-Opts-APSPN-kernel

Supported problems:
- APSPN: All-Pairs Shortest Path Number
- GC: Network Geometrical Congruence (requires reference distance)
- BC: Betweenness Centrality
- APSLP: All-Pairs Shortest-Lightest Path (requires reference distance)
- PHD: Path Hesitation Degree
- Approximate GC and approximate BC
"""

from .brandes_apspn_kernel import brandes_apspn_kernel
from .cc_apspn_kernel_approx import cc_apspn_kernel_approx
from .cc_apspn_kernel_par import cc_apspn_kernel_par
from .cm_gc import cm_gc
from .cc_sebccopts_apspn_kernel import cc_sebccopts_apspn_kernel

__version__ = "1.0.0"
__all__ = [
    "brandes_apspn_kernel",
    "cc_apspn_kernel_approx",
    "cc_apspn_kernel_par",
    "cm_gc",
    "cc_sebccopts_apspn_kernel"
]


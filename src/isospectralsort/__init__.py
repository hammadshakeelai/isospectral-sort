"""
IsospectralSort: Continuous Dynamical, Lie-Algebraic, and Soliton Sorting Systems.

Provides pure mathematical sorting algorithms from differential geometry,
Hamiltonian mechanics, nonlinear wave physics, and optimal transport.
"""

from .brockett import brockett_sort, BrockettDiagnostics
from .toda import toda_sort, TodaDiagnostics
from .optimal_transport import optimal_transport_sort, OptimalTransportDiagnostics
from .box_ball import box_ball_sort, BBSDiagnostics

__version__ = "0.1.0"
__all__ = [
    "brockett_sort",
    "BrockettDiagnostics",
    "toda_sort",
    "TodaDiagnostics",
    "optimal_transport_sort",
    "OptimalTransportDiagnostics",
    "box_ball_sort",
    "BBSDiagnostics",
]

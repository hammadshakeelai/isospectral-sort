"""
Performance Benchmarking Suite: Continuous Mathematical Sorts vs Classical Timsort.

Evaluates execution time, iteration count, and scalability across dimensions n.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from isospectralsort.brockett import brockett_sort
from isospectralsort.toda import toda_sort
from isospectralsort.optimal_transport import optimal_transport_sort
from isospectralsort.box_ball import box_ball_sort


def run_benchmark():
    np.random.seed(42)
    sizes = [4, 8, 12, 16]
    
    print("=" * 80)
    print(f"{'n':<6} | {'Timsort (CPU)':<14} | {'Brockett Flow':<16} | {'Toda Lattice':<16} | {'Optimal Transport':<18} | {'Box-Ball (BBS)':<14}")
    print("-" * 80)
    
    for n in sizes:
        arr = list(np.random.randn(n) * 20.0)
        
        # 1. Timsort
        t0 = time.perf_counter()
        _ = sorted(arr)
        t_timsort = (time.perf_counter() - t0) * 1000.0  # ms
        
        # 2. Brockett
        t0 = time.perf_counter()
        _ = brockett_sort(arr)
        t_brockett = (time.perf_counter() - t0) * 1000.0
        
        # 3. Toda
        t0 = time.perf_counter()
        _ = toda_sort(arr)
        t_toda = (time.perf_counter() - t0) * 1000.0
        
        # 4. Optimal Transport
        t0 = time.perf_counter()
        _ = optimal_transport_sort(arr)
        t_ot = (time.perf_counter() - t0) * 1000.0
        
        # 5. Box-Ball
        t0 = time.perf_counter()
        _ = box_ball_sort(arr)
        t_bbs = (time.perf_counter() - t0) * 1000.0
        
        print(f"{n:<6} | {t_timsort:8.4f} ms    | {t_brockett:10.2f} ms   | {t_toda:10.2f} ms   | {t_ot:12.2f} ms     | {t_bbs:8.2f} ms")
        
    print("=" * 80)
    print("\nTheoretical Note:")
    print("While discrete Timsort runs in O(n log n) digital cycles, continuous")
    print("dynamical systems (Brockett & Toda) are intended for analog/optical")
    print("hardware architectures where commutators and flows evolve in O(1) continuous physical time.\n")


if __name__ == "__main__":
    run_benchmark()

"""
Example usage of fast_apspn_kernel package

This script demonstrates how to use the Brandes-APSPN-kernel and CC-APSPN-kernel
algorithms for various APSPN-based problems.
"""

import numpy as np
import sys
import os

# Add the package to path if not installed
sys.path.insert(0, os.path.dirname(__file__))

from fast_apspn_kernel import (
    brandes_apspn_kernel, 
    cc_apspn_kernel_approx,
    cc_sebccopts_apspn_kernel,
    cm_gc
)


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def example_small_network():
    """Example 1: Small network (5 nodes, path graph)"""
    print_section("Example 1: Small Network (5 nodes, path graph)")
    
    # Create a path graph: 0-1-2-3-4 (not fully connected)
    adj_matrix = np.array([
        [0, 1, 0, 0, 0],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [0, 0, 0, 1, 0]
    ], dtype=np.float64)
    
    print("Adjacency matrix (path graph):")
    print(adj_matrix)
    print("Network structure: 0-1-2-3-4")
    
    # Run APSPN problem
    print("\n--- APSPN (All-Pairs Shortest Path Number) ---")
    result = brandes_apspn_kernel(adj_matrix, problem_type='apspn')
    print(f"APSPN matrix:\n{result['result']}")
    print(f"Elapsed time: {result['elapsed_time']:.6f} seconds")
    
    # Run Betweenness Centrality
    print("\n--- Betweenness Centrality ---")
    result = brandes_apspn_kernel(adj_matrix, problem_type='bc')
    print(f"BC values: {result['result']}")
    print(f"Elapsed time: {result['elapsed_time']:.6f} seconds")


def example_with_reference_distance():
    """Example 2: Network with reference distance (for GC)"""
    print_section("Example 2: Network with Reference Distance (5 nodes)")
    
    # Create star graph: node 2 at center connected to 0, 1, 3, 4 (not fully connected)
    adj_matrix = np.array([
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 0, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0]
    ], dtype=np.float64)
    
    # Create reference distance matrix (e.g., geographical distances)
    rd_matrix = np.array([
        [0.0, 3.5, 2.0, 4.1, 3.8],
        [3.5, 0.0, 1.5, 3.9, 4.2],
        [2.0, 1.5, 0.0, 2.5, 2.8],
        [4.1, 3.9, 2.5, 0.0, 3.6],
        [3.8, 4.2, 2.8, 3.6, 0.0]
    ], dtype=np.float64)
    
    print("Adjacency matrix (star graph, center at node 2):")
    print(adj_matrix)
    print("\nReference distance matrix:")
    print(rd_matrix)
    
    # Run GC (Network Geometrical Congruence)
    print("\n--- Network Geometrical Congruence (GC) ---")
    result = cc_apspn_kernel_approx(adj_matrix, problem_type='gc', rd_matrix=rd_matrix)
    print(f"GC value: {result['result']}")
    print(f"Elapsed time: {result['elapsed_time']:.6f} seconds")
    
    # Run APSLP (All-Pairs Shortest-Lightest Path)
    print("\n--- All-Pairs Shortest-Lightest Path (APSLP) ---")
    result = cc_apspn_kernel_approx(adj_matrix, problem_type='apslp', rd_matrix=rd_matrix)
    print(f"APSLP matrix:\n{result['result']}")
    print(f"Elapsed time: {result['elapsed_time']:.6f} seconds")


def example_algorithm_comparison():
    """Example 3: Compare Brandes vs CC-APSPN"""
    print_section("Example 3: Algorithm Comparison (5 nodes)")
    
    # Create a network with 5 nodes (not fully connected)
    # Network structure: partial mesh with some missing connections
    adj_matrix = np.array([
        [0, 1, 1, 0, 0],
        [1, 0, 1, 1, 0],
        [1, 1, 0, 1, 1],
        [0, 1, 1, 0, 1],
        [0, 0, 1, 1, 0]
    ], dtype=np.float64)
    
    print("Network: Partial mesh with 5 nodes")
    print("Adjacency matrix:")
    print(adj_matrix)
    
    # Run Brandes-APSPN-kernel
    print("\n--- Brandes-APSPN-kernel ---")
    result_brandes = brandes_apspn_kernel(adj_matrix, problem_type='apspn')
    print(f"Elapsed time: {result_brandes['elapsed_time']:.6f} seconds")
    
    # Run CC-APSPN-kernel
    print("\n--- CC-APSPN-kernel (full optimizations) ---")
    result_cc = cc_apspn_kernel_approx(adj_matrix, problem_type='apspn', ablation_type='None')
    print(f"Elapsed time: {result_cc['elapsed_time']:.6f} seconds")
    
    # Verify results match
    if np.allclose(result_brandes['result'], result_cc['result']):
        print("\nResults match between algorithms.")
    else:
        print("\nWarning: Results differ.")


def example_ablation_study():
    """Example 4: Ablation study with CC-APSPN"""
    print_section("Example 4: Ablation Study (CC-APSPN-kernel)")
    
    # Create a small network
    adj_matrix = np.array([
        [0, 1, 1, 0, 0],
        [1, 0, 1, 1, 0],
        [1, 1, 0, 1, 1],
        [0, 1, 1, 0, 1],
        [0, 0, 1, 1, 0]
    ], dtype=np.float64)
    
    print("Network: 5 nodes")
    
    ablation_types = ['None', 'Mem', 'Symm', 'DAP']
    
    print("\n--- Running ablation study ---")
    for ablation in ablation_types:
        result = cc_apspn_kernel_approx(adj_matrix, problem_type='apspn', ablation_type=ablation)
        print(f"{ablation:20s}: {result['elapsed_time']:.6f} seconds")




def example_sebccopts_optimization():
    """Example 5: SE and BCC optimization toggles"""
    print_section("Example 5: SE and BCC Optimization Options")
    
    # Create a network with 5 nodes
    adj_matrix = np.array([
        [0, 1, 1, 0, 0],
        [1, 0, 1, 1, 0],
        [1, 1, 0, 1, 1],
        [0, 1, 1, 0, 1],
        [0, 0, 1, 1, 0]
    ], dtype=np.float64)
    
    print("Network: 5 nodes, partial mesh")
    print("\n--- Testing different SE/BCC combinations ---")
    
    # Test different combinations
    configs = [
        (True, True, "SE + BCC (both enabled)"),
        (True, False, "SE only (BCC disabled)"),
        (False, True, "BCC only (SE disabled)"),
        (False, False, "None (both disabled)")
    ]
    
    for use_se, use_bcc, desc in configs:
        result = cc_sebccopts_apspn_kernel(
            adj_matrix, 
            problem_type='apspn', 
            use_se=use_se, 
            use_bcc=use_bcc
        )
        print(f"{desc:30s}: {result['elapsed_time']:.6f} seconds")


def example_cm_gc():
    """Example 6: Specialized CM-GC algorithm"""
    print_section("Example 6: CM-GC (Specialized for GC)")
    
    # Create a small network (5 nodes star graph)
    adj_matrix = np.array([
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 0, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0]
    ], dtype=np.float64)
    
    # Reference distance matrix
    rd_matrix = np.array([
        [0.0, 3.5, 2.0, 4.1, 3.8],
        [3.5, 0.0, 1.5, 3.9, 4.2],
        [2.0, 1.5, 0.0, 2.5, 2.8],
        [4.1, 3.9, 2.5, 0.0, 3.6],
        [3.8, 4.2, 2.8, 3.6, 0.0]
    ], dtype=np.float64)
    
    print("Network: Star graph (5 nodes, center at node 2)")
    
    # Compare CM-GC with CC-APSPN-kernel GC
    print("\n--- Comparing GC implementations ---")
    
    result_cm = cm_gc(adj_matrix, rd_matrix)
    print(f"CM-GC (specialized):         GC = {result_cm['result']:.6f}, Time = {result_cm['elapsed_time']:.6f}s")
    
    result_cc = cc_apspn_kernel_approx(adj_matrix, problem_type='gc', rd_matrix=rd_matrix)
    print(f"CC-APSPN-kernel:             GC = {result_cc['result']:.6f}, Time = {result_cc['elapsed_time']:.6f}s")
    
    # Verify results match
    if np.abs(result_cm['result'] - result_cc['result']) < 1e-6:
        print("\nResults match between implementations.")
    else:
        print("\nWarning: Results differ.")


def example_cc_approxgc():
    """Example 7: CC-ApproxGC"""
    print_section("Example 7: CC-ApproxGC")

    adj_matrix = np.array([
        [0, 1, 1, 0, 0],
        [1, 0, 1, 1, 0],
        [1, 1, 0, 1, 1],
        [0, 1, 1, 0, 1],
        [0, 0, 1, 1, 0]
    ], dtype=np.float64)

    rd_matrix = np.array([
        [0.0, 1.5, 2.0, 3.2, 3.7],
        [1.5, 0.0, 1.1, 2.0, 2.8],
        [2.0, 1.1, 0.0, 1.4, 1.9],
        [3.2, 2.0, 1.4, 0.0, 1.2],
        [3.7, 2.8, 1.9, 1.2, 0.0]
    ], dtype=np.float64)

    print("\n--- Approximate GC with CC-ApproxGC ---")
    result_gc = cc_apspn_kernel_approx(
        adj_matrix,
        problem_type='gc_approx',
        rd_matrix=rd_matrix,
        sampling_factor=2,
        verbose=False
    )
    print(f"Approximate GC value from CC-ApproxGC: {result_gc['result']}")
    print(f"Elapsed time: {result_gc['elapsed_time']:.6f} seconds")


def example_cc_approxbc():
    """Example 8: CC-ApproxBC"""
    print_section("Example 8: CC-ApproxBC")

    adj_matrix = np.array([
        [0, 1, 0, 0, 0],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [0, 0, 0, 1, 0]
    ], dtype=np.float64)

    print("\n--- Approximate BC with CC-ApproxBC ---")
    result_bc = cc_apspn_kernel_approx(
        adj_matrix,
        problem_type='bc_approx'
    )
    print(f"Approximate BC values from CC-ApproxBC: {result_bc['result']}")
    print(f"Elapsed time: {result_bc['elapsed_time']:.6f} seconds")


def main():
    """Run all examples"""
    print("\n")
    print("=" * 70)
    print("Fast APSPN Kernel Examples")
    print("=" * 70)
    
    try:
        example_small_network()
        example_with_reference_distance()
        example_algorithm_comparison()
        example_ablation_study()
        example_sebccopts_optimization()
        example_cm_gc()
        example_cc_approxgc()
        example_cc_approxbc()
        
        print("\n" + "="*70)
        print("  All examples completed successfully!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure you have:")
        print("  1. Compiled the C++ code using compile_windows_msvc.bat")
        print("  2. Installed numpy: pip install numpy")


if __name__ == "__main__":
    main()


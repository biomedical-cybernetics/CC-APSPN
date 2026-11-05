#!/usr/bin/env python3
"""
Benchmark APSPN Kernels on Example Networks

Problems tested:
- APSPN: All-Pairs Shortest Path Number
- BC: Betweenness Centrality
- GC: Network Geometrical Congruence
- APSLP: All-Pairs Shortest-Lightest Path
- PHD: Path Hesitation Degree
- BC_APPROX: approximate Betweenness Centrality
- GC_APPROX: approximate Network Geometrical Congruence
"""

import os
import sys
import glob
import numpy as np
from pathlib import Path
import pandas as pd

# Add the package to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent / 'source' / 'python'))

from fast_apspn_kernel import (
    brandes_apspn_kernel, cc_apspn_kernel_approx, cc_apspn_kernel_par,
    cc_sebccopts_apspn_kernel, cm_gc,
)

# Import utilities
utils_dir = script_dir / 'utils' / 'python_utils'
sys.path.insert(0, str(utils_dir))
from network_loader import load_mat_variable
from io_utils import suppress_c_stdout_stderr, format_number


def main():
    print("\n" + "="*80)
    print("  APSPN KERNEL BENCHMARK ON EXAMPLE NETWORKS")
    print("="*80 + "\n")
    
    # Set paths
    data_folder = script_dir.parent / "data" / "example_networks"
    result_folder = script_dir.parent / "results" / "benchmark"
    
    # Create result folder if it doesn't exist
    result_folder.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading networks from: {data_folder}\n")
    
    if not data_folder.exists():
        print(f"Error: Networks folder not found: {data_folder}")
        print(f"   Please create the folder and add .mat network files")
        return
    
    # Get all .mat files
    mat_files = sorted(glob.glob(str(data_folder / "*.mat")))
    
    if not mat_files:
        print(f"Error: No .mat files found in {data_folder}")
        print(f"   Please add network files in MATLAB .mat format")
        return
    
    print(f"   Found {len(mat_files)} network(s)\n")
    
    # Problem types to test
    problems = ['apspn', 'bc', 'gc', 'apslp', 'phd', 'bc_approx', 'gc_approx']

    # Requested parallel threads; keep within the machine's physical core count.
    num_threads = 4
    
    # Algorithm configurations
    algorithms = [
        {
            'name': 'Brandes-APSPN-kernel',
            'func': brandes_apspn_kernel,
            'supported_problems': {'apspn', 'bc', 'gc', 'apslp', 'phd'},
        },
        {
            'name': 'CC-APSPN-kernel',
            'func': cc_apspn_kernel_approx,
            'supported_problems': {'apspn', 'bc', 'gc', 'apslp', 'phd'},
        },
        {
            'name': 'Parallel-CC-APSPN-kernel',
            'func': cc_apspn_kernel_par,
            'supported_problems': {'apspn', 'bc', 'gc', 'apslp', 'phd'},
        },
        {
            'name': 'CC-APSPN-kernel (+SE+BCC)',
            'func': cc_sebccopts_apspn_kernel,
            'supported_problems': {'apspn', 'bc'},
        },
        {
            'name': 'CM-GC',
            'func': cm_gc,
            'supported_problems': {'gc'},
        },
        {
            'name': 'CC-ApproxBC',
            'func': cc_apspn_kernel_approx,
            'supported_problems': {'bc_approx'},
        },
        {
            'name': 'CC-ApproxGC',
            'func': cc_apspn_kernel_approx,
            'supported_problems': {'gc_approx'},
        },
    ]
    
    # Storage for results
    results_data = []
    
    for file_idx, filepath in enumerate(mat_files, 1):
        network_name = Path(filepath).stem
        
        print("="*80)
        print(f"Network {file_idx}/{len(mat_files)}: {network_name}")
        print("="*80)
        
        try:
            # Load adjacency matrix
            x, _ = load_mat_variable(filepath, ['x', 'adjMatrix', 'x_matrices'])
            
            n = x.shape[0]
            m = np.count_nonzero(x) // 2  # Assuming undirected
            density = (2 * m) / (n * (n - 1)) if n > 1 else 0
            
            # Network info
            print(f"   Nodes:       {format_number(n)}")
            print(f"   Edges:       {format_number(m)}")
            print(f"   Density:     {density*100:.4f}%\n")
            
            # Load reference distance matrix for GC and APSLP
            try:
                rd_matrix, rd_var_name = load_mat_variable(filepath, ['GSP', 'geo_matrices'], expected_shape=(n, n))
                has_rd_from_file = True
                print(f"   Loaded reference distance matrix from {rd_var_name}\n")
            except ValueError:
                rd_matrix = None
                has_rd_from_file = False
                print(f"   Reference distance matrix (GSP or geo_matrices) not found - skipping GC and APSLP\n")
            
            # Test each problem type
            for problem in problems:
                # Skip GC and APSLP if reference distance matrix is not available
                if problem in ['gc', 'gc_approx', 'apslp'] and not has_rd_from_file:
                    print(f"--- Problem: {problem.upper()} ---")
                    print(f"   Skipped (GSP or geo_matrices not available)\n")
                    continue
                
                print(f"--- Problem: {problem.upper()} ---")
                
                network_result = {
                    'Network': network_name,
                    'Nodes': n,
                    'Edges': m,
                    'Density': density,
                    'Problem': problem.upper()
                }
                
                # Run each algorithm
                for algo in algorithms:
                    algo_name = algo['name']
                    algo_func = algo['func']

                    if problem not in algo['supported_problems']:
                        continue
                    
                    print(f"   {algo_name:25s}", end=" ")
                    
                    try:
                        # Prepare arguments based on problem type
                        kwargs = {} if algo_func is cm_gc else {'problem_type': problem}
                        if problem in ['gc', 'gc_approx', 'apslp']:
                            kwargs['rd_matrix'] = rd_matrix
                        if algo_func is cc_apspn_kernel_par:
                            kwargs['num_threads'] = num_threads
                            network_result['Parallel_requested_threads'] = num_threads
                        if algo_func is cc_sebccopts_apspn_kernel:
                            kwargs.update(use_se=True, use_bcc=True)
                        if problem == 'gc_approx':
                            kwargs['sampling_factor'] = min(10, n)
                            kwargs['verbose'] = False
                        
                        # Call the algorithm with suppressed C output
                        with suppress_c_stdout_stderr():
                            result = algo_func(x, **kwargs)
                        
                        elapsed_time = result['elapsed_time']
                        
                        if algo_func is cc_apspn_kernel_par:
                            effective_threads = result['effective_threads']
                            network_result['Parallel_effective_threads'] = effective_threads
                            print(f"OK {elapsed_time:.6f} s ({effective_threads} threads used; {num_threads} requested)")
                        else:
                            print(f"OK {elapsed_time:.6f} s")
                        
                        network_result[algo_name] = elapsed_time
                        
                    except Exception as e:
                        print(f"Error: {str(e)}")
                        network_result[algo_name] = None
                
                print()  # Empty line between problems
                results_data.append(network_result)
            
            print()  # Empty line between networks
            
        except Exception as e:
            print(f"   Error loading network: {str(e)}\n")
            continue
    
    # Print summary
    print("\n" + "="*80)
    print("  SUMMARY")
    print("="*80 + "\n")
    
    if results_data:
        # Create DataFrame
        df = pd.DataFrame(results_data)
        
        # Reorganize by problem type
        for problem in problems:
            problem_df = df[df['Problem'] == problem.upper()].copy()
            if not problem_df.empty:
                print(f"\n{problem.upper()} Results (Time in seconds):")
                print("-" * 80)
                supported_algo_names = [
                    algo['name']
                    for algo in algorithms
                    if problem in algo['supported_problems']
                ]
                display_columns = [
                    col for col in ['Network', 'Nodes', 'Edges', 'Density', *supported_algo_names]
                    if col in problem_df.columns
                ]
                display_df = problem_df[display_columns]
                print(display_df.to_string(index=False))
                print()
        
        # Save to CSV
        csv_file = result_folder / "benchmark_results.csv"
        df.to_csv(csv_file, index=False)
        print(f"Results saved to: {csv_file}\n")
        
        # Calculate and display speedups
        print("\nSpeedup Summary (CC-APSPN-kernel vs Brandes-APSPN-kernel):")
        print("-" * 80)
        for problem in problems:
            problem_df = df[df['Problem'] == problem.upper()].copy()
            if not problem_df.empty and 'Brandes-APSPN-kernel' in problem_df.columns and 'CC-APSPN-kernel' in problem_df.columns:
                # Calculate speedup
                brandes_times = problem_df['Brandes-APSPN-kernel'].dropna()
                cc_times = problem_df['CC-APSPN-kernel'].dropna()
                if len(brandes_times) > 0 and len(cc_times) > 0:
                    speedups = brandes_times.values / cc_times.values
                    avg_speedup = np.mean(speedups)
                    print(f"   {problem.upper():10s}: {avg_speedup:.2f}x faster (average)")
        print()
    
    print("="*80)
    print("  BENCHMARK COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()


clear all;
clc;

script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(fileparts(script_dir));
output_dir = fullfile(project_root, 'source', 'matlab', 'mex_functions');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

cpp_dir = fullfile(project_root, 'source', 'cpp');
graph_utils = fullfile(project_root, 'source', 'common_include', 'graph_utils.cpp');

compile_mex_helper(fullfile(cpp_dir, 'brandes_apspn_kernel', 'brandes_apspn_kernel_cpp.cpp'), output_dir, {graph_utils});
compile_mex_helper(fullfile(cpp_dir, 'cc_sebccopts_apspn_kernel', 'cc_sebccopts_apspn_kernel_cpp.cpp'), output_dir, {graph_utils});
compile_mex_helper(fullfile(cpp_dir, 'cm_gc', 'cm_gc_cpp.cpp'), output_dir, {graph_utils});

approx_dir = fullfile(cpp_dir, 'cc_apspn_kernel_approx');
approx_bc_dir = fullfile(approx_dir, 'cc_approxbc_lib');
approx_gc_dir = fullfile(approx_dir, 'cc_approxgc_lib');
approx_extra_sources = {
    graph_utils, ...
    fullfile(approx_gc_dir, 'cc_approxgc.cpp'), ...
    fullfile(approx_bc_dir, 'cc_approxbc_wrapper.cpp'), ...
    fullfile(approx_bc_dir, 'Probabilistic.cpp'), ...
    fullfile(approx_bc_dir, 'Graph.cpp'), ...
    fullfile(approx_bc_dir, 'Rand_gen.cpp'), ...
    fullfile(approx_bc_dir, 'utilities.cpp'), ...
    fullfile(approx_bc_dir, 'Sp_sampler.cpp'), ...
    fullfile(approx_bc_dir, 'Ranking_list.cpp')
};
compile_mex_helper(fullfile(approx_dir, 'cc_apspn_kernel_cpp_ablation_approx.cpp'), output_dir, approx_extra_sources);

% Parallel exact kernel, with sequential BC_APPROX/GC_APPROX compatibility.
parallel_dir = fullfile(cpp_dir, 'cc_apspn_kernel_par');
parallel_bc = fullfile(parallel_dir, 'cc_approxbc_lib');
parallel_extra = {graph_utils, ...
    fullfile(project_root, 'source', 'common_include', 'cc_bfs_p_apsp_l.cpp'), ...
    fullfile(parallel_dir, 'cc_approxgc_lib', 'cc_approxgc.cpp')};
bc_sources = {'cc_approxbc_wrapper.cpp', 'Probabilistic.cpp', 'Graph.cpp', ...
    'Rand_gen.cpp', 'utilities.cpp', 'Sp_sampler.cpp', 'Ranking_list.cpp'};
for k = 1:numel(bc_sources)
    parallel_extra{end + 1} = fullfile(parallel_bc, bc_sources{k}); %#ok<SAGROW>
end
compile_mex_helper(fullfile(parallel_dir, 'cc_apspn_kernel_par.cpp'), ...
    output_dir, parallel_extra);

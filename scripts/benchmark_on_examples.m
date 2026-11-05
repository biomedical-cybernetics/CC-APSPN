%
% This script benchmarks APSPN kernels on networks:
%   - Brandes-APSPN kernel
%   - CC-APSPN kernel
%   - Parallel CC-APSPN kernel (exact modes)
%   - CC-APSPN kernel with SE and BCC (APSPN and BC)
%   - CM-GC
%   - CC-ApproxBC and CC-ApproxGC
%
% Problems tested:
%   - APSPN:  All-Pairs Shortest Path Number
%   - GC:     Network Geometrical Congruence (requires reference distance matrix rd)
%   - BC:     Betweenness Centrality
%   - APSLP:  All-Pairs Shortest-Lightest Path (requires reference distance matrix rd)
%   - PHD:    Path Hesitation Degree
%   - BC_APPROX: approximate Betweenness Centrality
%   - GC_APPROX: approximate Network Geometrical Congruence
%
% Usage:
%   Run this script directly in MATLAB
%
% The script will:
%   1. Load all .mat files from the configured data folder (default: ../data/example_networks/)
%   2. Run each algorithm on each applicable problem for each network
%   3. Display timing results
%   4. Save results as CSV to ../results/benchmark/
%
% Requirements:
%   - MEX files must be compiled (run ../source/matlab/compile_all_mex.m first)
%   - Networks must be in the configured data folder as .mat files
%   - Networks should contain adjacency matrix as variable 'x', 'adjMatrix', or 'x_matrices'
%   - For GC and APSLP: networks should contain 'geo_matrices' for reference distance matrix

%% ======================================================================
%% Configuration
%% ======================================================================
clear all;
clc;
fprintf('\n');
fprintf('================================================================\n');
fprintf('          APSPN-KERNEL ALGORITHM BENCHMARK\n');
fprintf('================================================================\n');
fprintf('\n');

% Get script directory and set paths
script_dir = fileparts(mfilename('fullpath'));
mex_folder = fullfile(script_dir, '..', 'source', 'matlab', 'mex_functions');
data_folder = fullfile(script_dir, '..', 'data', 'example_networks');
result_folder = fullfile(script_dir, '..', 'results', 'benchmark');

% Add MEX functions to path
addpath(mex_folder);

% Create result folder if it doesn't exist
if ~exist(result_folder, 'dir')
    mkdir(result_folder);
end

% Number of repetitions for timing
num_repetitions = 1;

% Requested threads for parallel CC-APSPN; results record the actual count.
num_threads = 4;

%% ======================================================================
%% Algorithm Configuration
%% ======================================================================

% Define algorithms
algorithms = struct(...
    'name', {}, ...
    'func', {}, ...
    'param_template', {}, ...
    'save_counters', {}, ...
    'supported_problems', {});

% Brandes-APSPN-kernel
algorithms(end+1).name = 'Brandes-APSPN-kernel';
algorithms(end).func = 'brandes_apspn_kernel_cpp';
algorithms(end).param_template = '(''PROBLEM_TYPE'', x, EXTRA_PARAMS)';
algorithms(end).save_counters = true;
algorithms(end).supported_problems = {'APSPN', 'GC', 'BC', 'APSLP', 'PHD'};

% CC-APSPN-kernel (Full version - no ablation)
algorithms(end+1).name = 'CC-APSPN-kernel';
algorithms(end).func = 'cc_apspn_kernel_cpp_ablation_approx';
algorithms(end).param_template = '(''PROBLEM_TYPE'', ''None'', x, EXTRA_PARAMS)';
algorithms(end).save_counters = true;
algorithms(end).supported_problems = {'APSPN', 'GC', 'BC', 'APSLP', 'PHD'};

% Parallel CC-APSPN-kernel (approximate modes are benchmarked separately below)
algorithms(end+1).name = 'Parallel-CC-APSPN-kernel';
algorithms(end).func = 'cc_apspn_kernel_par';
algorithms(end).param_template = '(''PROBLEM_TYPE'', x, num_threads, EXTRA_PARAMS, false)';
algorithms(end).save_counters = true;
algorithms(end).supported_problems = {'APSPN', 'GC', 'BC', 'APSLP', 'PHD'};

% CC-APSPN-kernel with structural equivalence and biconnected components
algorithms(end+1).name = 'CC-APSPN-kernel (+SE+BCC)';
algorithms(end).func = 'cc_sebccopts_apspn_kernel_cpp';
algorithms(end).param_template = '(''PROBLEM_TYPE'', true, true, x, false)';
algorithms(end).save_counters = true;
algorithms(end).supported_problems = {'APSPN', 'BC'};

% CM-GC
algorithms(end+1).name = 'CM-GC';
algorithms(end).func = 'cm_gc_cpp';
algorithms(end).param_template = '(x, rd)';
algorithms(end).save_counters = false;
algorithms(end).supported_problems = {'GC'};

% CC-ApproxBC
algorithms(end+1).name = 'CC-ApproxBC';
algorithms(end).func = 'cc_apspn_kernel_cpp_ablation_approx';
algorithms(end).param_template = '(''PROBLEM_TYPE'', ''None'', x, EXTRA_PARAMS)';
algorithms(end).save_counters = true;
algorithms(end).supported_problems = {'BC_APPROX'};

% CC-ApproxGC
algorithms(end+1).name = 'CC-ApproxGC';
algorithms(end).func = 'cc_apspn_kernel_cpp_ablation_approx';
algorithms(end).param_template = '(''PROBLEM_TYPE'', ''None'', x, EXTRA_PARAMS)';
algorithms(end).save_counters = true;
algorithms(end).supported_problems = {'GC_APPROX'};

% Define problems to benchmark
problems_to_run = {'APSPN', 'GC', 'BC', 'APSLP', 'PHD', 'BC_APPROX', 'GC_APPROX'};

%% ======================================================================
%% Load Networks
%% ======================================================================

fprintf('Loading networks from: %s\n', data_folder);
fprintf('\n');

if ~exist(data_folder, 'dir')
    error('Networks folder not found: %s\n', data_folder);
end

% Get all .mat files
files = dir(fullfile(data_folder, '*.mat'));

if isempty(files)
    error('No .mat files found in: %s\n', data_folder);
end

fprintf('   Found %d network(s)\n', length(files));
fprintf('\n');

%% ======================================================================
%% Run Benchmarks
%% ======================================================================

% Storage for results
% One row per (network, problem, algorithm) combination
results_data = {};
result_row_idx = 0;

for file_idx = 1:length(files)
    filepath = fullfile(data_folder, files(file_idx).name);
    [~, network_name, ~] = fileparts(filepath);
    
    % Print network header
    fprintf('===============================================================\n');
    fprintf('Network %d/%d: %s\n', file_idx, length(files), network_name);
    fprintf('===============================================================\n');
    
    % Load adjacency matrix
    [x, has_rd, rd] = load_network(filepath);
    n = size(x, 1);
    m = nnz(x) / 2;  % Assuming undirected
    density = 2 * m / (n * (n - 1));
    
    % Network info
    fprintf('   Nodes:       %s\n', format_number(n));
    fprintf('   Edges:       %s\n', format_number(m));
    fprintf('   Density:     %.4f%%\n', density * 100);
    fprintf('   Has RD:      %s\n', bool_to_str(has_rd));
    fprintf('   Repetitions: %d\n', num_repetitions);
    fprintf('\n');
    
    % Run each problem type
    for prob_idx = 1:length(problems_to_run)
        problem_type = problems_to_run{prob_idx};
        
        % Check if problem requires rd
        needs_rd = ismember(problem_type, {'GC', 'GC_APPROX', 'APSLP'});
        if needs_rd && ~has_rd
            fprintf('Skipping %s (requires referential distance)\n\n', problem_type);
            continue;
        end
        
        fprintf('---------------------------------------------------------------\n');
        fprintf('Problem: %s\n', problem_type);
        fprintf('---------------------------------------------------------------\n');
        
        % Run each algorithm
        for algo_idx = 1:length(algorithms)
            algo = algorithms(algo_idx);
            
            % Check if algorithm supports this problem
            if ~ismember(problem_type, algo.supported_problems)
                continue;
            end
            
            fprintf('   Algorithm: %s (%s)\n', algo.name, algo.func);
            
            % Build parameter string
            param_str = build_param_string(algo.param_template, problem_type, has_rd);
            
            % Run multiple times for timing
            times = zeros(num_repetitions, 1);
            counters_array = cell(num_repetitions, 1);
            requested_threads = 1;
            effective_threads = 1;
            
            for rep = 1:num_repetitions
                fprintf('      Run %d/%d... ', rep, num_repetitions);
                
                try
                    % Execute algorithm (use evalc to suppress function output)
                    if algo.save_counters
                        evalc(['[result, elapsed, counters] = ' algo.func param_str ';']);
                        counters_array{rep} = counters;
                        if strcmp(algo.func, 'cc_apspn_kernel_par')
                            requested_threads = counters.requested_threads;
                            effective_threads = counters.effective_threads;
                        end
                    else
                        evalc(['[result, elapsed] = ' algo.func param_str ';']);
                    end
                    times(rep) = elapsed;
                    fprintf('%.4f s\n', elapsed);
                catch ME
                    fprintf('ERROR: %s\n', ME.message);
                    times(rep) = NaN;
                end
            end
            
            % Compute statistics
            valid_times = times(~isnan(times));
            if isempty(valid_times)
                fprintf('      All runs failed.\n\n');
                continue;
            end
            
            time_mean = mean(valid_times);
            time_std = std(valid_times);
            time_min = min(valid_times);
            time_max = max(valid_times);
            
            % Print statistics
            fprintf('      +----------------------+--------------------+\n');
            fprintf('      | Mean Time            | %10.6f s      |\n', time_mean);
            fprintf('      | Std Dev              | %10.6f s      |\n', time_std);
            fprintf('      | Min Time             | %10.6f s      |\n', time_min);
            fprintf('      | Max Time             | %10.6f s      |\n', time_max);
            fprintf('      +----------------------+--------------------+\n');
            if strcmp(algo.func, 'cc_apspn_kernel_par')
                fprintf('      Threads: %d requested, %d effective\n', ...
                    requested_threads, effective_threads);
            end
            fprintf('\n');
            
            % Store result
            result_row_idx = result_row_idx + 1;
            results_data{result_row_idx, 1} = network_name;
            results_data{result_row_idx, 2} = n;
            results_data{result_row_idx, 3} = m;
            results_data{result_row_idx, 4} = density;
            results_data{result_row_idx, 5} = problem_type;
            results_data{result_row_idx, 6} = algo.name;
            results_data{result_row_idx, 7} = time_mean;
            results_data{result_row_idx, 8} = time_std;
            results_data{result_row_idx, 9} = time_min;
            results_data{result_row_idx, 10} = time_max;
            
            % Store counter statistics if available
            if algo.save_counters
                counters_mean = average_counters(counters_array);
                results_data{result_row_idx, 11} = counters_mean;
            else
                results_data{result_row_idx, 11} = [];
            end
            results_data{result_row_idx, 12} = requested_threads;
            results_data{result_row_idx, 13} = effective_threads;
        end
        
        fprintf('\n');
    end
    
    fprintf('\n');
end

%% ======================================================================
%% Summary
%% ======================================================================

fprintf('================================================================\n');
fprintf('                         SUMMARY\n');
fprintf('================================================================\n');
fprintf('\n');

if result_row_idx == 0
    fprintf('No results to display.\n\n');
    return;
end

% Group results by problem type
for prob_idx = 1:length(problems_to_run)
    problem_type = problems_to_run{prob_idx};
    
    % Filter rows for this problem
    prob_mask = false(result_row_idx, 1);
    for i = 1:result_row_idx
        if strcmp(results_data{i, 5}, problem_type)
            prob_mask(i) = true;
        end
    end
    
    if ~any(prob_mask)
        continue;
    end
    
    fprintf('Problem: %s\n', problem_type);
    fprintf('%s\n', repmat('-', 1, 80));
    
    % Print header
    fprintf('%-35s %10s', 'Network', 'Nodes');
    for algo_idx = 1:length(algorithms)
        if ismember(problem_type, algorithms(algo_idx).supported_problems)
            fprintf(' %18s', algorithms(algo_idx).name);
        end
    end
    fprintf('\n');
    fprintf('%s\n', repmat('-', 1, 80));
    
    % Get unique networks for this problem
    networks_in_problem = {};
    for i = 1:result_row_idx
        if prob_mask(i)
            net_name = results_data{i, 1};
            if ~ismember(net_name, networks_in_problem)
                networks_in_problem{end+1} = net_name;
            end
        end
    end
    
    % Print data rows
    for net_idx = 1:length(networks_in_problem)
        net_name = networks_in_problem{net_idx};
        
        % Find the first row with this network and problem
        first_row = [];
        for i = 1:result_row_idx
            if strcmp(results_data{i, 1}, net_name) && strcmp(results_data{i, 5}, problem_type)
                first_row = i;
                break;
            end
        end

        if isempty(first_row)
            continue;
        end

        % Print network name and nodes once
        fprintf('%-35s %10d', shorten_name(net_name, 35), results_data{first_row, 2});
        
        % Print times for each algorithm
        for algo_idx = 1:length(algorithms)
            if ~ismember(problem_type, algorithms(algo_idx).supported_problems)
                continue;
            end
            
            algo_name = algorithms(algo_idx).name;
            
            % Find matching result
            found = false;
            for i = 1:result_row_idx
                if strcmp(results_data{i, 1}, net_name) && ...
                   strcmp(results_data{i, 5}, problem_type) && ...
                   strcmp(results_data{i, 6}, algo_name)
                    fprintf(' %18.6f', results_data{i, 7});
                    found = true;
                    break;
                end
            end
            
            if ~found
                fprintf(' %18s', 'N/A');
            end
        end
        fprintf('\n');
    end
    
    fprintf('\n');
end

%% ======================================================================
%% Save Results
%% ======================================================================

% Convert to table for CSV export
header = {'Network', 'Nodes', 'Edges', 'Density', 'Problem', 'Algorithm', ...
          'Mean_Time_s', 'Std_Time_s', 'Min_Time_s', 'Max_Time_s', ...
          'Requested_Threads', 'Effective_Threads'};

% Prepare data for CSV (exclude counters struct)
csv_data = results_data(:, [1:10, 12:13]);

result_table = cell2table(csv_data, 'VariableNames', header);

% Save as CSV
csv_file = fullfile(result_folder, 'benchmark_results.csv');
writetable(result_table, csv_file);
fprintf('Results saved to: %s\n', csv_file);

% Save full results (including counters) as .mat
mat_file = fullfile(result_folder, 'benchmark_results.mat');
save(mat_file, 'results_data', 'algorithms', 'problems_to_run', 'num_threads');
fprintf('Full results (with counters) saved to: %s\n', mat_file);
fprintf('\n');

fprintf('================================================================\n');
fprintf('                   BENCHMARK COMPLETE\n');
fprintf('================================================================\n');
fprintf('\n');

%% ======================================================================
%% Helper Functions
%% ======================================================================

function [x, has_rd, rd] = load_network(filepath)
%LOAD_NETWORK Load adjacency matrix and optional referential distance from .mat file
    fileInfo = whos('-file', filepath);
    varNames = {fileInfo.name};
    
    % Check if GSP (reference distance) exists - if so, adj matrix is stored directly
    if ismember('GSP', varNames)
        % Load adjacency matrix directly (not in cell array)
        if ismember('x', varNames)
            data = load(filepath, 'x');
            x = data.x;
        elseif ismember('adjMatrix', varNames)
            data = load(filepath, 'adjMatrix');
            x = data.adjMatrix;
        else
            error('No adjacency matrix found in file: %s\nExpected variable name: ''x'' or ''adjMatrix''', filepath);
        end
        
        % Load reference distance from GSP
        data = load(filepath, 'GSP');
        rd = data.GSP;
        has_rd = true;
    else
        % Standard loading: check for cell arrays or direct matrices
        if ismember('x_matrices', varNames)
            data = load(filepath, 'x_matrices');
            x = data.x_matrices{1};
        elseif ismember('x', varNames)
            data = load(filepath, 'x');
            x = data.x;
        elseif ismember('adjMatrix', varNames)
            data = load(filepath, 'adjMatrix');
            x = data.adjMatrix;
        else
            error('No adjacency matrix found in file: %s\nExpected variable name: ''x'', ''adjMatrix'', or ''x_matrices''', filepath);
        end
        
        % Load referential distance if available from geo_matrices
        has_rd = false;
        rd = [];
        if ismember('geo_matrices', varNames)
            try
                data = load(filepath, 'geo_matrices');
                rd = data.geo_matrices{1};
                has_rd = true;
            catch
                warning('Failed to load geo_matrices from %s', filepath);
            end
        end
    end
    
end

function param_str = build_param_string(template, problem_type, has_rd)
%BUILD_PARAM_STRING Build parameter string for algorithm based on problem type
    % Replace PROBLEM_TYPE
    param_str = strrep(template, 'PROBLEM_TYPE', lower(problem_type));
    
    % Replace EXTRA_PARAMS based on problem
    switch problem_type
        case {'GC', 'APSLP'}
            if has_rd
                param_str = strrep(param_str, 'EXTRA_PARAMS', 'rd');
            else
                error('Problem %s requires referential distance (rd)', problem_type);
            end
        case 'GC_APPROX'
            if has_rd
                param_str = strrep(param_str, 'EXTRA_PARAMS', 'rd, min(10, size(x, 1)), false');
            else
                error('Problem %s requires referential distance (rd)', problem_type);
            end
        otherwise
            param_str = strrep(param_str, ', EXTRA_PARAMS', '');
    end
end

function str = format_number(num)
%FORMAT_NUMBER Format number with thousands separators
    str = num2str(num);
    if num >= 1000
        % Add commas as thousands separators
        str = regexprep(str, '(\d{1,3})(?=(\d{3})+(?!\d))', '$1,');
    end
end

function str = bool_to_str(bool_val)
%BOOL_TO_STR Convert boolean to Yes/No string
    if bool_val
        str = 'Yes';
    else
        str = 'No';
    end
end

function short_name = shorten_name(name, max_len)
%SHORTEN_NAME Shorten string to maximum length
    if length(name) <= max_len
        short_name = name;
    else
        short_name = [name(1:max_len-3) '...'];
    end
end

function counters_mean = average_counters(counters_array)
%AVERAGE_COUNTERS Average numeric counters and retain descriptive metadata.
    counters_array = counters_array(~cellfun(@isempty, counters_array));
    if isempty(counters_array)
        counters_mean = [];
        return;
    end
    
    % Get field names from first counter
    fields = fieldnames(counters_array{1});
    counters_mean = struct();
    
    % Average each field
    for f = 1:length(fields)
        field_name = fields{f};
        first_value = counters_array{1}.(field_name);
        if ~(isnumeric(first_value) && isscalar(first_value))
            counters_mean.(field_name) = first_value;
            continue;
        end
        values = zeros(length(counters_array), 1);
        
        for i = 1:length(counters_array)
            if ~isempty(counters_array{i}) && isfield(counters_array{i}, field_name)
                values(i) = counters_array{i}.(field_name);
            else
                values(i) = NaN;
            end
        end
        
        counters_mean.(field_name) = mean(values(~isnan(values)));
    end
end


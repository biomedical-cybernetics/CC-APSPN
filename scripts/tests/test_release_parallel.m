function report = test_release_parallel(action, target, output_path)
%TEST_RELEASE_PARALLEL Small release correctness/compatibility checks.
%   test_release_parallel runs all exact modes and input-contract checks.
%   test_release_parallel('approx', 'sequential', FILE) and
%   test_release_parallel('approx', 'parallel', FILE) must run in separate
%   fresh MATLAB processes. Compare their JSON values, not their runtimes.
%   test_release_parallel('compare', BASELINE_FILE, PARALLEL_FILE) checks them.
if nargin < 1, action = 'exact'; end
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(root, 'source', 'matlab', 'mex_functions'));
if strcmp(action, 'compare')
    baseline = jsondecode(fileread(target));
    candidate = jsondecode(fileread(output_path));
    assert(numel(baseline) == numel(candidate), 'Probe count differs.');
    for k = 1:numel(baseline)
        assert(strcmp(baseline(k).key, candidate(k).key), 'Probe order differs.');
        assert_close(candidate(k).value, baseline(k).value, baseline(k).key);
    end
    report = struct('approximate_comparisons', numel(baseline));
    fprintf('RELEASE APPROXIMATE PASS: %d comparisons.\n', numel(baseline));
    return
end
fixtures = {'grid_4x5', 'cycle_n11'};
if strcmp(action, 'approx')
    report = approximate_probes(target);
    fid = fopen(output_path, 'w');
    assert(fid >= 0, 'Cannot write report: %s', output_path);
    cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fprintf(fid, '%s\n', jsonencode(report));
    fprintf('RELEASE APPROXIMATE PROBES PASS: %s, %d calls.\n', target, numel(report));
    return
end
assert(strcmp(action, 'exact'), 'Unknown test action.');
modes = {'APSPN', 'BC', 'GC', 'PHD', 'APSLP'};
calls = 0;
for f = 1:numel(fixtures)
    x = load_fixture(fixtures{f});
    n = size(x, 1);
    rd = reference_distance(n);
    inputs = {full(x), x, logical(full(x)), logical(x)};
    for m = 1:numel(modes)
        mode = modes{m};
        extra = {};
        if ismember(mode, {'GC', 'APSLP'}), extra = {rd}; end
        reference = brandes_apspn_kernel_cpp(mode, x, extra{:});
        sequential = cc_apspn_kernel_cpp_ablation_approx(mode, 'None', x, extra{:}, false);
        assert_close(sequential, reference, [mode ' sequential vs Brandes']);
        for input_index = 1:numel(inputs)
            before = inputs{input_index};
            for threads = [1 4 1]
                [actual, elapsed, counters] = cc_apspn_kernel_par( ...
                    mode, before, threads, extra{:}, false);
                assert_close(actual, reference, [fixtures{f} ' ' mode]);
                assert(isequal(before, inputs{input_index}), 'Adjacency was modified.');
                assert_shape(actual, mode, n);
                assert(isfinite(elapsed) && elapsed >= 0, 'Invalid elapsed time.');
                assert(counters.requested_threads == threads, 'Wrong requested-thread count.');
                assert(counters.effective_threads >= 1 && ...
                    counters.effective_threads <= threads, 'Invalid effective-thread count.');
                calls = calls + 1;
            end
        end
    end
    % Referential distances are symmetric; both supported storage formats
    % must preserve the same numerical result and leave the input unchanged.
    expected = cc_apspn_kernel_cpp_ablation_approx('APSLP', 'None', x, rd, false);
    for rd_input = {rd, sparse(rd)}
        before = rd_input{1};
        actual = cc_apspn_kernel_par('APSLP', x, 4, rd_input{1}, false);
        assert_close(actual, expected, 'APSLP reference-distance storage');
        assert(isequal(rd_input{1}, before), 'RD was modified.');
        calls = calls + 1;
    end
end
x = load_fixture('grid_4x5');
bad_weight = full(x); bad_weight(1, 2) = 0.5;
asymmetric = full(x); [i, j] = find(triu(x), 1); asymmetric(i, j) = 0;
bad_nan = full(x); bad_nan(1, 2) = NaN;
invalid = {zeros(2, 3), bad_weight, sparse(bad_weight), asymmetric, ...
    sparse(asymmetric), bad_nan, complex(full(x), full(x)), single(full(x))};
for k = 1:numel(invalid)
    assert_rejected(@() cc_apspn_kernel_par('BC', invalid{k}, 1, false));
end
for threads = {0, -1, 1.5, NaN, Inf, [1 2], '4'}
    assert_rejected(@() cc_apspn_kernel_par('BC', x, threads{1}, false));
end
assert_rejected(@() cc_apspn_kernel_par('GC', x, 1));
assert_rejected(@() cc_apspn_kernel_par('GC', x, 1, zeros(2), false));
assert_rejected(@() cc_apspn_kernel_par('UNKNOWN', x, 1, false));
rd = reference_distance(size(x, 1));
rd(1, 2) = rd(1, 2) + 0.25;
assert_rejected(@() cc_apspn_kernel_par('APSLP', x, 1, rd, false));
report = struct('exact_parallel_calls', calls, 'invalid_input_checks', numel(invalid) + 11);
fprintf('RELEASE EXACT PASS: %d parallel calls and %d invalid-input checks.\n', ...
    report.exact_parallel_calls, report.invalid_input_checks);
end

function report = approximate_probes(target)
assert(ismember(target, {'sequential', 'parallel'}), 'Unknown target.');
report = struct('key', {}, 'value', {}, 'elapsed_time', {});
x = load_fixture('grid_4x5');
n = size(x, 1);
rd = reference_distance(n);
inputs = {full(x), x, logical(full(x)), logical(x)};
for mode = {'BC_APPROX', 'GC_APPROX'}
    for k = 1:numel(inputs)
        for threads = [1 4 1]
            extra = {rd, 2, false};
            if strcmp(mode{1}, 'BC_APPROX')
                extra = {0.1, 0.1, 2, 'log_n_w', 42, false};
            end
            value = invoke_approx(target, mode{1}, inputs{k}, threads, extra);
            assert_shape(value.result, mode{1}, n);
            assert(all(isfinite(value.result(:))) && all(value.result(:) >= 0), ...
                'Invalid approximate output.');
            report(end + 1) = struct('key', sprintf('%s format=%d threads=%d', ...
                mode{1}, k, threads), 'value', value.result, ...
                'elapsed_time', value.elapsed_time); %#ok<AGROW>
        end
    end
end
% One direction of the existing cycle fixture checks row/column adjacency
% interpretation. This is a transformation, not a separately generated graph.
cycle = load_fixture('cycle_n11');
directed = triu(cycle); directed(1, end) = 0; directed(end, 1) = 1;
for graph_input = {full(directed), directed}
    value = invoke_approx(target, 'BC_APPROX', graph_input{1}, 4, ...
        {0.1, 0.1, 2, 'log_n_w', 42, false});
    assert_shape(value.result, 'BC_APPROX', size(directed, 1));
    assert(all(isfinite(value.result(:))) && all(value.result(:) >= 0));
    report(end + 1) = struct('key', sprintf('directed BC_APPROX sparse=%d', ...
        issparse(graph_input{1})), 'value', value.result, ...
        'elapsed_time', value.elapsed_time); %#ok<AGROW>
end
end

function value = invoke_approx(target, mode, x, threads, extra)
before = x;
if strcmp(target, 'parallel')
    [actual, elapsed, counters] = cc_apspn_kernel_par(mode, x, threads, extra{:});
    assert(counters.requested_threads == threads && counters.effective_threads == 1);
else
    [actual, elapsed] = cc_apspn_kernel_cpp_ablation_approx(mode, 'None', x, extra{:});
end
assert(isequal(x, before), 'Approximate call modified input.');
assert(isfinite(elapsed) && elapsed >= 0, 'Invalid approximate runtime.');
value = struct('result', actual, 'elapsed_time', elapsed);
end

function x = load_fixture(name)
data = load(fullfile(fileparts(mfilename('fullpath')), 'fixtures', [name '.mat']), 'x');
x = data.x;
end

function rd = reference_distance(n)
index = (1:n).';
coordinates = [0.19 * index + 0.003 * index .^ 2, ...
    sin(0.61 * index) + 0.07 * cos(1.13 * index) + 0.017 * index .^ 2];
rd = sqrt((coordinates(:, 1) - coordinates(:, 1).').^2 + ...
    (coordinates(:, 2) - coordinates(:, 2).').^2);
end

function assert_shape(value, mode, n)
if ismember(mode, {'BC', 'PHD', 'BC_APPROX'})
    assert(isequal(size(value), [n 1]), 'Expected a column vector.');
elseif ismember(mode, {'GC', 'GC_APPROX'})
    assert(isscalar(value), 'Expected a scalar.');
else
    assert(isequal(size(value), [n n]), 'Expected an n-by-n matrix.');
end
end

function assert_close(actual, expected, label)
assert(isequal(size(actual), size(expected)), '%s: output shape differs.', label);
assert(all(isfinite(actual(:))) && all(isfinite(expected(:))), '%s: nonfinite output.', label);
assert(all(abs(actual(:) - expected(:)) <= 1e-10 + 1e-10 * abs(expected(:))), ...
    '%s: numerical disagreement.', label);
end

function assert_rejected(callback)
rejected = false;
try
    value = callback(); %#ok<NASGU>
catch
    rejected = true;
end
assert(rejected, 'Invalid input was accepted.');
end

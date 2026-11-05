function start_parallel_pool(num_threads)
    % START_PARALLEL_POOL Manages the parallel pool with the given number of threads
    %
    % Input:
    %   num_threads - Desired number of parallel threads (workers)
    %                 0 to shutdown the parallel pool
    %                 Positive integer to start/modify pool
    %                 Limited to the number of physical cores
    %
    % Example:
    %   start_parallel_pool(4);  % Starts a pool with 4 workers
    %   start_parallel_pool(0);  % Shuts down the parallel pool

    % Get the current parallel pool without creating one
    pool = gcp('nocreate');

    % Handle shutdown request
    if num_threads == 0
        if ~isempty(pool)
            fprintf('Shutting down parallel pool...\n');
            delete(pool);
        else
            fprintf('No parallel pool is running.\n');
        end
        return;
    end

    % Get number of physical cores
    max_cores = feature('numcores');

    % Validate the input argument
    if ~isscalar(num_threads) || num_threads < 0 || floor(num_threads) ~= num_threads
        error('Input must be zero or a positive integer.');
    end

    % Limit number of threads to physical cores
    num_threads = min(num_threads, max_cores);

    % Manage parallel pool
    if isempty(pool)
        % No pool exists, create a new one
        fprintf('Starting a parallel pool with %d threads (max cores: %d)...\n', ...
            num_threads, max_cores);
        parpool(num_threads);
    elseif pool.NumWorkers ~= num_threads
        % Pool exists but with different number of workers
        fprintf(['Current parallel pool has %d threads. ', ...
            'Reinitializing with %d threads (max cores: %d)...\n'], ...
            pool.NumWorkers, num_threads, max_cores);
        delete(pool);
        parpool(num_threads);
    else
        % Pool already has desired number of workers
        fprintf('Parallel pool with %d threads is already running (max cores: %d).\n', ...
            num_threads, max_cores);
    end
end

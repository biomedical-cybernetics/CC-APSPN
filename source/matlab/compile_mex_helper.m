function compile_mex_helper(cpp_file_path, output_dir, extra_sources_or_eigen_dir, eigen_include_dir)
    % COMPILE_MEX_HELPER Compile a MEX target with shared project defaults.
    %
    % Supported forms:
    %   compile_mex_helper('path/to/file.cpp')
    %   compile_mex_helper('path/to/file.cpp', 'output/dir/')
    %   compile_mex_helper('path/to/file.cpp', 'output/dir/', {'extra1.cpp','extra2.cpp'})
    %   compile_mex_helper('path/to/file.cpp', 'output/dir/', [], 'path/to/eigen')
    %   compile_mex_helper('path/to/file.cpp', 'output/dir/', {'extra.cpp'}, 'path/to/eigen')

    if nargin < 2 || isempty(output_dir)
        output_dir = 'source/matlab/mex_functions/';
    end

    extra_sources = {};
    if nargin < 3
        extra_sources_or_eigen_dir = [];
    end
    if nargin < 4 || isempty(eigen_include_dir)
        eigen_include_dir = getenv('EIGEN_INCLUDE_DIR');
        if isempty(eigen_include_dir)
            if ispc
                eigen_include_dir = 'C:/ALGLIB/eigen-3.4.0';
            else
                eigen_include_dir = '/usr/include/eigen3';
            end
        end
    end

    if iscell(extra_sources_or_eigen_dir)
        extra_sources = extra_sources_or_eigen_dir;
    elseif isstring(extra_sources_or_eigen_dir) || ischar(extra_sources_or_eigen_dir)
        candidate = char(extra_sources_or_eigen_dir);
        if nargin < 4 && is_probably_eigen_dir(candidate)
            eigen_include_dir = candidate;
        elseif ~isempty(candidate)
            extra_sources = {candidate};
        end
    elseif ~isempty(extra_sources_or_eigen_dir)
        error('compile_mex_helper:InvalidThirdArgument', ...
            'Third argument must be extra source(s) or an Eigen include directory.');
    end

    extra_sources = normalize_extra_sources(extra_sources);
    validate_eigen_include_dir(eigen_include_dir);

    if ispc
        build_flags = {'COMPFLAGS="$COMPFLAGS /O2 /std:c++20 /openmp /DMATLAB_BUILD"', ...
                       'LDFLAGS="$LDFLAGS /openmp"'};
    elseif isunix && ~ismac
        build_flags = {'CXXFLAGS="$CXXFLAGS -O3 -std=c++20 -fopenmp -DMATLAB_BUILD"', ...
                       'LDFLAGS="$LDFLAGS -fopenmp"'};
    else
        error('compile_mex_helper:UnsupportedPlatform', ...
              'These build commands support Windows/MSVC and Linux/GCC.');
    end
    project_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    mex_args = [{'-largeArrayDims'}, build_flags, { ...
        ['-I' eigen_include_dir], ...
        ['-I' fullfile(project_root, 'source', 'common_include')], ...
        ['-I' fullfile(project_root, 'third_party', 'cc_bfs_p', 'cpp', 'include')], ...
        '-outdir', output_dir, ...
        cpp_file_path ...
    }];

    if ispc
        % Avoid silently selecting an older installed MSVC without C++20.
        compiler_options = getenv('MEX_CXX_OPTIONS');
        if isempty(compiler_options)
            compiler_options = fullfile(matlabroot, 'bin', 'win64', ...
                'mexopts', 'msvcpp2022.xml');
        end
        mex_args = [{'-f', compiler_options}, mex_args];
    end

    for i = 1:numel(extra_sources)
        mex_args{end + 1} = extra_sources{i}; %#ok<AGROW>
    end

    mex(mex_args{:});
    fprintf('Successfully compiled: %s\n', cpp_file_path);
end


function tf = is_probably_eigen_dir(path_str)
    eigen_dense_header = fullfile(char(path_str), 'Eigen', 'Dense');
    tf = isfolder(path_str) && exist(eigen_dense_header, 'file') == 2;
end


function validate_eigen_include_dir(eigen_include_dir)
    eigen_dense_header = fullfile(eigen_include_dir, 'Eigen', 'Dense');
    if ~isfolder(eigen_include_dir) || exist(eigen_dense_header, 'file') ~= 2
        error('compile_mex_helper:InvalidEigenIncludeDir', ...
            ['Invalid Eigen include directory: %s\nExpected to find: %s'], ...
            eigen_include_dir, eigen_dense_header);
    end
end


function extra_sources = normalize_extra_sources(extra_sources)
    if isempty(extra_sources)
        extra_sources = {};
        return;
    end

    if ischar(extra_sources) || isstring(extra_sources)
        extra_sources = cellstr(extra_sources);
    end

    if ~iscell(extra_sources)
        error('compile_mex_helper:InvalidExtraSources', ...
            'Extra sources must be a char vector, string array, or cell array of paths.');
    end

    extra_sources = cellfun(@char, extra_sources, 'UniformOutput', false);
end

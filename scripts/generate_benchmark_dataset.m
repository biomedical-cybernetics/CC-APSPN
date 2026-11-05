script_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(script_dir, 'utils', 'npso_model_generation'));

base_folder = fullfile(fileparts(script_dir), 'data', 'benchmark_dataset');
low_high_den_sn = [1e-3, 1e-1];
low_high_den_ln = [1e-4, 1e-2];
result_folder = base_folder;
par_realizations =  5;

%% set 1: fixed size, varying densities
Network_Sizes = 10000; 
Densities = [0.001, 0.005, 0.01, 0.05, 0.1]; 
Ts = [0.1];                 % Temperatures
Gammas = [2.5];                       % Gamma values
Comms = [4];                             % Community numbers
realizations = 5;

generate_npso_networks(Network_Sizes, Densities, Ts, Gammas, Comms, realizations, result_folder, par_realizations, false, true);

%% set 2: low-high density, varying sizes
Network_Sizes = [10000, 30000]; 
Ts = [0.1];                 % Temperatures
Gammas = [2.5];                       % Gamma values
Comms = [4];                             % Community numbers
realizations = 5;

generate_npso_networks(Network_Sizes, low_high_den_sn, Ts, Gammas, Comms, realizations, result_folder, par_realizations, false, true);

Network_Sizes = [100000]; 
Ts = [0.1];                 % Temperatures
Gammas = [2.5];                       % Gamma values
Comms = [4];                             % Community numbers
realizations = 1;

generate_npso_networks(Network_Sizes, low_high_den_ln, Ts, Gammas, Comms, realizations, result_folder, false, false, true);


%% set 3: low-high density, varying gamma
Network_Sizes = 100000; 
Ts = [0.1];                 % Temperatures
Gammas = [2 2.5 3];                       % Gamma values
Comms = [4];                             % Community numbers
realizations = 1;

generate_npso_networks(Network_Sizes, low_high_den_ln, Ts, Gammas, Comms, realizations, result_folder, false, false, true);

%% set 4: low-high density, varying T
Network_Sizes = 100000; 
Ts = [0.1 0.5 1];                 % Temperatures
Gammas = [2.5];                       % Gamma values
Comms = [4];                             % Community numbers
realizations = 1;

generate_npso_networks(Network_Sizes, low_high_den_ln, Ts, Gammas, Comms, realizations, result_folder, false, false, true);

%% set 5: low-high density, varying C
Network_Sizes = 100000; 
Ts = [0.1];                 % Temperatures
Gammas = [2.5];                       % Gamma values
Comms = [4 10 100];                             % Community numbers
realizations = 1;

generate_npso_networks(Network_Sizes, low_high_den_ln, Ts, Gammas, Comms, realizations, result_folder, false, false, true);


%% set 6: nPSO networks for comprehensive APSPN-based problems evaluation
Network_Sizes = 10000; % Different network sizes
Densities = [0.001 0.0025 0.005 0.01 0.025 0.05 0.1 0.25];                   % Densities

Ts = [0.1];                 % Temperatures
Gammas = [2 2.5 3];                       % Gamma values
Comms = [4];                             % Community numbers
realizations = 5;
generate_npso_networks(Network_Sizes, Densities, Ts, Gammas, Comms, realizations, result_folder, par_realizations, false, true);

%% set 7: ablation-test nPSO networks
Network_Sizes = 30000;
Densities = [0.00025, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.25];
Ts = [0.1];                 % Temperatures
Gammas = [2, 3];            % Gamma values
Comms = [4];                % Community numbers
realizations = 5;
generate_npso_networks(Network_Sizes, Densities, Ts, Gammas, Comms, realizations, result_folder, par_realizations, false, true);



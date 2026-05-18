function parameter_sweep_schrodinger(calibration_mat_path, output_path)
% PARAMETER_SWEEP_SCHRODINGER  Joint sweep over anneal_time x s_max x schedule_reduction.
%
% Phase 5A.2 of the Investigation 5 plan. Uses the same split-operator solver
% as calibrate_schrodinger.m but parallelises board evaluation with parfor
% and explores a wider parameter space than just anneal_time.
%
% USAGE
%   parameter_sweep_schrodinger('../../data/calibration_boards.mat', ...
%                                '../../data/phase5a2_sweep.mat')
%
% Sweep dimensions (designed jointly):
%   anneal_time:   focused on SHORT times per Phase 5A.3 finding (>1.85 ns
%                  was empirically ruled out by the eigsh adiabatic-limit test)
%   s_max:         test stopping the anneal early (website may sample before s=1)
%   schedule_red:  the schedule_energy_reduction_factor in the Python adjudicator
%                  is hardcoded at 0.5 — try other values to see if the website
%                  uses a different schedule scaling

if nargin < 2 || isempty(output_path)
    [d, ~, ~] = fileparts(calibration_mat_path);
    output_path = fullfile(d, 'phase5a2_sweep.mat');
end

% -------------------------------------------------------------------------
% Load raw D-Wave Advantage2 schedule (apply reduction in the sweep loop)
% -------------------------------------------------------------------------
sched_path = fullfile('C:', 'Users', 'murr2', 'projects', ...
    'snowdrop-adjudicators', 'snowdrop_adjudicators', ...
    'schrodinger', 'advantage2.1.3.txt');
if ~isfile(sched_path)
    error('Schedule file not found:\n  %s', sched_path);
end
fprintf('Loading D-Wave schedule from: %s\n', sched_path);
raw = readmatrix(sched_path);    % 1001 x 3
delta_raw = raw(:, 2) / 2;       % corresponds to load_schedule_data() Python output BEFORE reduction
A_raw     = raw(:, 3);           % corresponds to load_schedule_data() Python output BEFORE reduction

% -------------------------------------------------------------------------
% Petersen fixed parameters
% -------------------------------------------------------------------------
num_qubits = 10;
dim        = 2^num_qubits;   % 1024
num_edges  = 15;
edges = [0,2; 0,3; 0,6; 1,3; 1,4; 1,7; 2,4; 2,8; 3,9; 4,5; ...
         5,6; 5,9; 6,7; 7,8; 8,9];
p1_node  = 5;
p2_node  = 7;
s_min    = 0.001;
max_step = 0.0005;

% -------------------------------------------------------------------------
% Precompute spin table
% -------------------------------------------------------------------------
fprintf('Precomputing spin table (%d x %d)...\n', dim, num_qubits);
spins = zeros(dim, num_qubits);
for qi = 0:num_qubits-1
    mask = 2^qi;
    for b = 0:dim-1
        spins(b+1, qi+1) = 2 * double(bitand(b, mask) > 0) - 1;
    end
end

% -------------------------------------------------------------------------
% Load calibration boards
% -------------------------------------------------------------------------
fprintf('Loading calibration boards from: %s\n', calibration_mat_path);
calib = load(calibration_mat_path);
board_indices  = double(calib.board_indices(:));
website_scores = double(calib.website_scores(:));
N = length(board_indices);
fprintf('  %d calibration boards loaded.\n', N);
fprintf('  Website scores: [%.3f, %.3f]\n', min(website_scores), max(website_scores));

% Precompute J per board (cheap, do once)
J_all = zeros(num_edges, N);
for i = 1:N
    J_all(:, i) = soe_idx_to_couplings(board_indices(i) + 1, num_edges);
end

% -------------------------------------------------------------------------
% Sweep grid (joint design)
%   anneal_times: dense in the short regime per Phase 5A.3 (adiabatic-limit
%                 test ruled out >>1 ns); also include longer values as
%                 sanity check.
%   s_max_values: test stopping the anneal early — website might sample
%                 before s=1 (where delta=0 and ground state degenerates).
%   sched_reds:   the Python schedule_energy_reduction_factor; currently
%                 hardcoded at 0.5. Try other scalings.
% -------------------------------------------------------------------------
anneal_times = [0.1, 0.3, 0.7, 1.0, 1.85, 3.0, 5.0, 10.0, 30.0, 100.0];
s_max_values = [0.5, 0.7, 0.9, 0.99, 0.999];
sched_reds   = [0.25, 0.5, 1.0, 2.0];

n_at = length(anneal_times);
n_sm = length(s_max_values);
n_sr = length(sched_reds);
total = n_at * n_sm * n_sr;
fprintf('\nSweep grid: %d anneal_time x %d s_max x %d sched_red = %d combos\n', ...
        n_at, n_sm, n_sr, total);
fprintf('Per combo: %d boards (parfor parallelised)\n', N);

% Set up parpool
pool = gcp('nocreate');
if isempty(pool)
    fprintf('\nStarting parallel pool...\n');
    pool = parpool();
end
fprintf('Pool: %d workers\n', pool.NumWorkers);
n_workers = pool.NumWorkers;

% Per-combo eval ETA
fprintf('Estimated per-combo time: %.0f sec (%.1f min)\n', ...
        N * 0.7 / n_workers, N * 0.7 / n_workers / 60);
fprintf('Estimated total time: %.1f hr\n', total * N * 0.7 / n_workers / 3600);

% -------------------------------------------------------------------------
% Result arrays
% -------------------------------------------------------------------------
r2_grid      = zeros(n_at, n_sm, n_sr);
rmse_grid    = zeros(n_at, n_sm, n_sr);
mae_grid     = zeros(n_at, n_sm, n_sr);
bias_grid    = zeros(n_at, n_sm, n_sr);
scores_grid  = zeros(n_at, n_sm, n_sr, N);  % full predicted scores per combo
elapsed_grid = zeros(n_at, n_sm, n_sr);

fprintf('\n%-3s  %-9s  %-8s  %-8s  %-8s  %-8s  %-8s  %-8s\n', ...
        '#', 'sched', 's_max', 'tf(ns)', 'R^2', 'RMSE', 'MAE', 'time(s)');
fprintf('%s\n', repmat('-', 1, 72));

combo_idx = 0;
t_total = tic;
for k_sr = 1:n_sr
    sr = sched_reds(k_sr);
    delta_sched = delta_raw * sr;
    A_sched     = A_raw * sr;

    for k_sm = 1:n_sm
        sm = s_max_values(k_sm);
        for k_at = 1:n_at
            tf = anneal_times(k_at);
            combo_idx = combo_idx + 1;
            t0 = tic;

            local_scores = zeros(N, 1);
            parfor i = 1:N
                local_scores(i) = soe_solve(J_all(:, i), edges, num_qubits, dim, ...
                                            delta_sched, A_sched, tf, ...
                                            s_min, sm, max_step, spins, ...
                                            p1_node, p2_node); %#ok<PFBNS>
            end

            elapsed = toc(t0);
            res = website_scores - local_scores;
            ss_res = sum(res.^2);
            ss_tot = sum((website_scores - mean(website_scores)).^2);
            r2  = 1 - ss_res / ss_tot;
            rmse = sqrt(mean(res.^2));
            mae  = mean(abs(res));
            bias = mean(res);

            r2_grid(k_at, k_sm, k_sr)      = r2;
            rmse_grid(k_at, k_sm, k_sr)    = rmse;
            mae_grid(k_at, k_sm, k_sr)     = mae;
            bias_grid(k_at, k_sm, k_sr)    = bias;
            scores_grid(k_at, k_sm, k_sr, :) = local_scores;
            elapsed_grid(k_at, k_sm, k_sr) = elapsed;

            fprintf('%3d  sr=%4.2f   sm=%5.3f   tf=%6.2f  R^2=%+6.4f  RMSE=%6.4f  MAE=%6.4f  %6.1f\n', ...
                    combo_idx, sr, sm, tf, r2, rmse, mae, elapsed);

            % Save intermediate progress every 10 combos
            if mod(combo_idx, 10) == 0
                save(output_path, 'r2_grid', 'rmse_grid', 'mae_grid', 'bias_grid', ...
                     'elapsed_grid', 'anneal_times', 's_max_values', 'sched_reds', ...
                     'website_scores', 'board_indices', 'combo_idx', '-v7.3');
            end
        end
    end
end
t_total_elapsed = toc(t_total);
fprintf('\nFull sweep: %d combos in %.1f hr (%.1f sec/combo avg)\n', ...
        total, t_total_elapsed / 3600, t_total_elapsed / total);

% -------------------------------------------------------------------------
% Find best
% -------------------------------------------------------------------------
[best_r2, lin_idx] = max(r2_grid(:));
[best_at_idx, best_sm_idx, best_sr_idx] = ind2sub(size(r2_grid), lin_idx);
fprintf('\nBest combo: R^2 = %.4f at sched_red=%.3f, s_max=%.4f, anneal_time=%.3f ns\n', ...
        best_r2, sched_reds(best_sr_idx), s_max_values(best_sm_idx), ...
        anneal_times(best_at_idx));
fprintf('Baseline (calib at sr=0.5, sm=0.999, tf=1.85): R^2 = %.4f\n', ...
        r2_grid(find(anneal_times==1.85), find(s_max_values==0.999), find(sched_reds==0.5)));

% -------------------------------------------------------------------------
% Save final results
% -------------------------------------------------------------------------
best_anneal_time      = anneal_times(best_at_idx);
best_s_max            = s_max_values(best_sm_idx);
best_sched_red        = sched_reds(best_sr_idx);
best_scores           = squeeze(scores_grid(best_at_idx, best_sm_idx, best_sr_idx, :));

save(output_path, 'r2_grid', 'rmse_grid', 'mae_grid', 'bias_grid', ...
     'scores_grid', 'elapsed_grid', ...
     'anneal_times', 's_max_values', 'sched_reds', ...
     'website_scores', 'board_indices', ...
     'best_anneal_time', 'best_s_max', 'best_sched_red', 'best_scores', 'best_r2', ...
     '-v7.3');
fprintf('\nResults saved to: %s\n', output_path);
fprintf('Total wall time: %.1f hr\n', t_total_elapsed / 3600);

end  % function parameter_sweep_schrodinger


% =========================================================================
% Helpers (copied from calibrate_schrodinger.m for self-containment)
% =========================================================================
function J = soe_idx_to_couplings(idx, n_edges)
J    = zeros(n_edges, 1);
bits = idx - 1;
for e = 1:n_edges
    if bitand(bits, 2^(e-1)) > 0
        J(e) = -1;
    else
        J(e) =  1;
    end
end
end


function score = soe_solve(J, edges, n_qubits, dim, ...
    delta_sched, A_sched, tf, s_min, s_max, max_step, spins, p1_node, p2_node)
phase_factor = 2 * pi * tf;
n_edges = size(edges, 1);
E_prob = zeros(dim, 1);
for e = 1:n_edges
    if J(e) ~= 0
        E_prob = E_prob + J(e) * spins(:, edges(e,1)+1) .* spins(:, edges(e,2)+1);
    end
end
psi = ones(dim, 1) / sqrt(dim);
s  = s_min;
ds = max_step;
while s < s_max - 1e-12
    ds = min(ds, s_max - s);
    ns   = s * 1000;
    nlo  = min(floor(ns), 999);
    frac = ns - nlo;
    nlo  = nlo + 1;
    nhi  = nlo + 1;
    delta_s = delta_sched(nlo) + frac * (delta_sched(nhi) - delta_sched(nlo));
    A_s     = A_sched(nlo)     + frac * (A_sched(nhi)     - A_sched(nlo));
    ph  = exp(-1i * A_s * E_prob * (phase_factor * ds * 0.5));
    psi = ph .* psi;
    theta = delta_s * phase_factor * ds;
    psi = soe_driver(psi, n_qubits, theta);
    psi = ph .* psi;
    s = s + ds;
end
probs = real(psi .* conj(psi));
mag   = probs' * spins;
corr = zeros(n_qubits);
for i = 1:n_qubits
    for j = i+1:n_qubits
        corr(i,j) = probs' * (spins(:,i) .* spins(:,j)) - mag(i)*mag(j);
    end
end
corr = corr + corr';
influence = sum(corr, 2);
score = influence(p1_node + 1) - influence(p2_node + 1);
end


function psi = soe_driver(psi, n_qubits, theta)
c = cos(theta);
s = sin(theta);
for q = 0:n_qubits-1
    stride = 2^q;
    psi = reshape(psi, stride, 2, []);
    a0 = psi(:,1,:);
    a1 = psi(:,2,:);
    psi(:,1,:) =  c * a0 + 1i*s * a1;
    psi(:,2,:) = 1i*s * a0 +  c * a1;
    psi = psi(:);
end
end

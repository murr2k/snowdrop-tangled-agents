function calibrate_schrodinger(calibration_mat_path, output_path)
% CALIBRATE_SCHRODINGER  Grid-search anneal_time to match website Schrodinger scores.
%
% Loads calibration boards + website scores exported by
%   python scripts/calibrate_adjudicator.py --export-mat
%
% Evaluates each board at a log-spaced grid of anneal_time values using the
% fast MATLAB split-operator Schrodinger solver (~0.7 s/board), computes
% R^2 against website scores, then refines with fminbnd.
%
% USAGE
%   calibrate_schrodinger('../../data/calibration_boards.mat')
%   calibrate_schrodinger('../../data/calibration_boards.mat', '../../data/matlab_calib_results.mat')
%
% OUTPUT
%   Saves best_anneal_time, best_local_scores, r2_grid, anneal_time_grid
%   to output_path (default: same directory as input, named matlab_calib_results.mat).

if nargin < 2 || isempty(output_path)
    [d, ~, ~] = fileparts(calibration_mat_path);
    output_path = fullfile(d, 'matlab_calib_results.mat');
end

% -------------------------------------------------------------------------
% Load D-Wave Advantage2 schedule
% -------------------------------------------------------------------------
sched_path = fullfile('C:', 'Users', 'murr2', 'projects', ...
    'snowdrop-adjudicators', 'snowdrop_adjudicators', ...
    'schrodinger', 'advantage2.1.3.txt');

if ~isfile(sched_path)
    error('Schedule file not found:\n  %s\nAdjust sched_path inside calibrate_schrodinger.m', sched_path);
end

fprintf('Loading D-Wave schedule from: %s\n', sched_path);
raw = readmatrix(sched_path);    % 1001 x 3
delta_sched = raw(:, 2) / 4;    % GHz (matches Python convention)
A_sched     = raw(:, 3) * 0.5;  % GHz

% -------------------------------------------------------------------------
% Petersen graph fixed parameters
% -------------------------------------------------------------------------
num_qubits = 10;
dim        = 2^num_qubits;   % 1024
num_edges  = 15;

% Edge list: lexical order, 0-indexed (matches Python GraphProperties.graph_database[5])
edges = [0,2; 0,3; 0,6; 1,3; 1,4; 1,7; 2,4; 2,8; 3,9; 4,5; ...
         5,6; 5,9; 6,7; 7,8; 8,9];

p1_node = 5;   % 0-indexed; soe_solve uses p1_node+1 internally
p2_node = 7;

% Fixed Schrodinger sweep parameters
s_min    = 0.001;
s_max    = 0.999;
max_step = 0.0005;

% -------------------------------------------------------------------------
% Precompute spin table (dim x num_qubits)
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

board_indices  = calib.board_indices(:);    % (N,1) 0-indexed Python integers
website_scores = calib.website_scores(:);   % (N,1) website adjudicator scores
N = length(board_indices);
fprintf('  %d calibration boards loaded.\n', N);
fprintf('  Website scores: [%.3f, %.3f]\n', min(website_scores), max(website_scores));

% -------------------------------------------------------------------------
% Grid search over anneal_time
% -------------------------------------------------------------------------
anneal_time_grid = unique(round(logspace(log10(5), log10(20000), 15)));
n_grid = length(anneal_time_grid);

fprintf('\nGrid search over %d anneal_time values on %d boards...\n', n_grid, N);
fprintf('  %-14s  %8s  %8s  %8s\n', 'anneal_time', 'R^2', 'MAE', 'time(s)');
fprintf('  %s\n', repmat('-', 1, 50));

r2_grid  = zeros(1, n_grid);
mae_grid = zeros(1, n_grid);

best_r2     = -Inf;
best_at     = anneal_time_grid(1);
best_scores = zeros(N, 1);

for gi = 1:n_grid
    at = anneal_time_grid(gi);
    t0 = tic;

    local_scores = eval_all_boards(board_indices, edges, num_edges, ...
                                   num_qubits, dim, delta_sched, A_sched, ...
                                   s_min, s_max, max_step, spins, ...
                                   p1_node, p2_node, at);

    elapsed = toc(t0);
    r2  = compute_r2(website_scores, local_scores);
    mae = mean(abs(website_scores - local_scores));

    r2_grid(gi)  = r2;
    mae_grid(gi) = mae;

    fprintf('  %-14.1f  %8.4f  %8.4f  %8.1f\n', at, r2, mae, elapsed);

    if r2 > best_r2
        best_r2     = r2;
        best_at     = at;
        best_scores = local_scores;
    end
end

fprintf('\nBest from grid: anneal_time=%.1f ns  R^2=%.4f\n', best_at, best_r2);

% -------------------------------------------------------------------------
% Refinement: fminbnd in log space around best grid point
% -------------------------------------------------------------------------
lo = best_at * 0.3;
hi = best_at * 3.0;
fprintf('\nRefining in [%.1f, %.1f] ns (log space)...\n', lo, hi);

call_count = 0;

    function neg_r2 = objective(log_at)
        at_val = exp(log_at);
        sc = eval_all_boards(board_indices, edges, num_edges, ...
                             num_qubits, dim, delta_sched, A_sched, ...
                             s_min, s_max, max_step, spins, ...
                             p1_node, p2_node, at_val);
        r2_val = compute_r2(website_scores, sc);
        neg_r2 = -r2_val;
        call_count = call_count + 1;
        fprintf('  [%2d] at=%8.2f  R^2=%.4f\n', call_count, at_val, r2_val);
    end

opt = optimset('TolX', 0.01, 'MaxFunEvals', 20, 'Display', 'off');
log_best = fminbnd(@objective, log(lo), log(hi), opt);
best_at_refined = exp(log_best);

% Evaluate at refined best
best_scores_refined = eval_all_boards(board_indices, edges, num_edges, ...
                                      num_qubits, dim, delta_sched, A_sched, ...
                                      s_min, s_max, max_step, spins, ...
                                      p1_node, p2_node, best_at_refined);
best_r2_refined  = compute_r2(website_scores, best_scores_refined);
best_mae_refined = mean(abs(website_scores - best_scores_refined));

fprintf('\nRefined: anneal_time=%.2f ns  R^2=%.4f  MAE=%.4f\n', ...
        best_at_refined, best_r2_refined, best_mae_refined);

% -------------------------------------------------------------------------
% Save results
% -------------------------------------------------------------------------
best_anneal_time  = best_at_refined;
best_local_scores = best_scores_refined;
best_r2_final     = best_r2_refined;
best_mae_final    = best_mae_refined;

save(output_path, ...
     'best_anneal_time', 'best_local_scores', ...
     'best_r2_final', 'best_mae_final', ...
     'anneal_time_grid', 'r2_grid', 'mae_grid', ...
     'website_scores', 'board_indices', '-v7.3');

fprintf('\nResults saved to: %s\n', output_path);
fprintf('\nNext step:\n');
fprintf('  python scripts/calibrate_adjudicator.py --load-matlab-results %s\n', output_path);

end  % function calibrate_schrodinger


% =========================================================================
function local_scores = eval_all_boards(board_indices, edges, num_edges, ...
                                        num_qubits, dim, delta_sched, A_sched, ...
                                        s_min, s_max, max_step, spins, ...
                                        p1_node, p2_node, anneal_time)
% Evaluate all boards at the given anneal_time.
% board_indices: 0-indexed Python integers (soe_idx_to_couplings expects 1-indexed,
%                so we pass board_indices(i)+1).

N = length(board_indices);
local_scores = zeros(N, 1);

for i = 1:N
    J = soe_idx_to_couplings(board_indices(i) + 1, num_edges);
    local_scores(i) = soe_solve(J, edges, num_qubits, dim, ...
                                delta_sched, A_sched, anneal_time, ...
                                s_min, s_max, max_step, spins, p1_node, p2_node);
end

end  % eval_all_boards


% =========================================================================
function r2 = compute_r2(y_true, y_pred)
ss_res = sum((y_true - y_pred).^2);
ss_tot = sum((y_true - mean(y_true)).^2);
if ss_tot > 0
    r2 = 1 - ss_res / ss_tot;
else
    r2 = 0;
end
end


% =========================================================================
% The three soe_* functions below are copied from
% generate_petersen_lut_schrodinger.m so this file is self-contained.
% =========================================================================

function J = soe_idx_to_couplings(idx, n_edges)
%   bit e = 1  ->  Green / FM    ->  J = -1
%   bit e = 0  ->  Purple / AFM  ->  J = +1
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
E_prob  = zeros(dim, 1);
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
    psi   = soe_driver(psi, n_qubits, theta);

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
corr      = corr + corr';
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

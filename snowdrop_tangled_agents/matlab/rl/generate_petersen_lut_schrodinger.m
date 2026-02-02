function generate_petersen_lut_schrodinger(varargin)
%GENERATE_PETERSEN_LUT_SCHRODINGER  Ground-truth terminal LUT via Schrödinger.
%
%   Scores all 2^15 = 32 768 terminal states of the Petersen graph (graph 5)
%   using the time-dependent Schrödinger equation with the D-Wave Advantage2
%   annealing schedule.  Output is drop-in compatible with the .mat file
%   produced by generate_terminal_lut.py.
%
%   Algorithm: Strang splitting exploits the Hamiltonian structure.
%     H(s) = −Δ(s)·Σ_i σ_x^i              ← driver:  tensor product of 2×2 rotations
%           + A(s)·Σ_{i<j} J_ij σ_z^i σ_z^j  ← problem: diagonal in computational basis
%
%   Per-step cost: O(n·2^n)  vs  O(2^{3n}) for the Python eigh-based solver.
%   Benchmark result: ~0.7 s/state → full LUT in ~47 min with 8 workers.
%
%   Usage:
%       generate_petersen_lut_schrodinger()
%       generate_petersen_lut_schrodinger('NumWorkers', 4)

% ─── Parse arguments ──────────────────────────────────────────────
ip = inputParser;
addParameter(ip, 'NumWorkers', 0);     % 0 = use all available cores
parse(ip, varargin{:});

fprintf('╔════════════════════════════════════════════════════════════╗\n');
fprintf('║   Petersen Terminal LUT — Schrödinger Ground Truth        ║\n');
fprintf('║   Split-Operator Method (Strang Splitting)                ║\n');
fprintf('╚════════════════════════════════════════════════════════════╝\n\n');

% ─── Petersen graph (graph_number = 5 in game engine) ─────────────
n_qubits = 10;
dim      = 2^n_qubits;           % 1024
n_edges  = 15;
n_states = 2^n_edges;            % 32 768
graph_id = 5;

% Edge list from GraphProperties.graph_database[5], lexical order.
edges = [0,2; 0,3; 0,6; 1,3; 1,4; 1,7; 2,4; 2,8; 3,9; 4,5; ...
         5,6; 5,9; 6,7; 7,8; 8,9];

p1_node = 5;   % 0-indexed
p2_node = 7;

% ─── Schrödinger parameters (match Python adjudicator) ───────────
tf       = 40.0;
s_min    = 0.001;
s_max    = 0.999;
max_step = 0.0005;
epsilon  = 0.0005;

% ─── Load D-Wave Advantage2 schedule ──────────────────────────────
fprintf('[1/4] Loading D-Wave schedule...\n');
sched_path = fullfile('C:', 'Users', 'murr2', 'projects', ...
    'snowdrop-adjudicators', 'snowdrop_adjudicators', ...
    'schrodinger', 'advantage2.1.3.txt');

if ~isfile(sched_path)
    error('Schedule file not found:\n  %s', sched_path);
end

raw = readmatrix(sched_path);          % 1001 × 3
% Python convention: delta = col[1]/2 * 0.5 = col[1]/4
%                    A     = col[2]   * 0.5
delta_sched = raw(:, 2) / 4;
A_sched     = raw(:, 3) * 0.5;
fprintf('    Δ ∈ [%.4f, %.4f] GHz,  A ∈ [%.4f, %.4f] GHz\n', ...
    min(delta_sched), max(delta_sched), min(A_sched), max(A_sched));

% ─── Precompute spin table ────────────────────────────────────────
fprintf('\n[2/4] Precomputing spin table...\n');
spins = zeros(dim, n_qubits);
for i = 0:n_qubits-1
    mask = 2^i;
    for b = 0:dim-1
        spins(b+1, i+1) = 2 * double(bitand(b, mask) > 0) - 1;
    end
end
fprintf('    %d × %d spin table ready\n', dim, n_qubits);

% ─── Parallel pool ────────────────────────────────────────────────
fprintf('\n[3/4] Initializing parallel pool...\n');
if ip.Results.NumWorkers > 0
    pool = parpool('local', ip.Results.NumWorkers);
else
    pool = parpool('local');
end
n_workers = pool.NumWorkers;
fprintf('    %d workers\n', n_workers);

% ─── Main loop ────────────────────────────────────────────────────
fprintf('\n[4/4] Scoring %d terminal states...\n', n_states);
tic;

terminal_scores = zeros(1, n_states, 'single');

parfor idx = 1:n_states
    J = soe_idx_to_couplings(idx, n_edges);
    terminal_scores(idx) = soe_solve(J, edges, n_qubits, dim, ...
        delta_sched, A_sched, tf, s_min, s_max, max_step, ...
        spins, p1_node, p2_node);
end

generation_time_sec = toc;
delete(pool);

% ─── Save .mat (drop-in replacement for Python generate_terminal_lut.py) ─
output_file = fullfile(fileparts(mfilename('fullpath')), 'data', 'terminal_scores.mat');
fprintf('\nSaving to %s...\n', output_file);

num_states      = n_states;
num_edges       = n_edges;
scorer          = 'schrodinger';
num_reads       = 0;
generated_at    = datestr(now, 'yyyy-mm-dd HH:MM:SS UTC');
description     = sprintf(['Terminal state scores for graph %d ' ...
    '(schrodinger, split-operator MATLAB). Index i: bit j=1 means ' ...
    'edge j is G (green/FM), 0 means P (purple/AFM). ' ...
    'Scores are from Player 1 perspective.'], graph_id);

save(output_file, ...
    'terminal_scores', 'num_states', 'num_edges', 'graph_id', ...
    'scorer', 'num_reads', 'generation_time_sec', ...
    'generated_at', 'description', '-v7.3');

% ─── Summary ──────────────────────────────────────────────────────
fprintf('\n╔════════════════════════════════════════════════════════════╗\n');
fprintf('║              LUT GENERATION COMPLETE                       ║\n');
fprintf('╠════════════════════════════════════════════════════════════╣\n');
fprintf('║  Graph:           Petersen  (graph_id = 5)                 ║\n');
fprintf('║  Scorer:          Schrödinger (split-operator)             ║\n');
fprintf('║  States scored:   %8d                                     ║\n', n_states);
fprintf('║  Score range:     [%+7.3f, %+7.3f]                        ║\n', ...
    min(terminal_scores), max(terminal_scores));
fprintf('║  Generation time: %8.1f s  (%.2f hrs)                     ║\n', ...
    generation_time_sec, generation_time_sec/3600);
fprintf('║  Avg time/state:  %8.3f s                                  ║\n', ...
    generation_time_sec / n_states);
fprintf('║  Workers:         %8d                                     ║\n', n_workers);
fprintf('║                                                            ║\n');
fprintf('║  Next step: run generate_expanded_lut_parallel() to        ║\n');
fprintf('║  extend to 1/2/3-grey states (~2–5 min with parfor).       ║\n');
fprintf('╚════════════════════════════════════════════════════════════╝\n');

end


% ═══════════════════════════════════════════════════════════════════
%  Core solver — one terminal state → score
%  (prefixed soe_ to avoid name collisions in parfor scope)
% ═══════════════════════════════════════════════════════════════════
function score = soe_solve(J, edges, n_qubits, dim, ...
    delta_sched, A_sched, tf, s_min, s_max, max_step, spins, p1_node, p2_node)

phase_factor = 2 * pi * tf;

% Problem-Hamiltonian diagonal: E_prob(b) = Σ_e J_e · spin_i(b) · spin_j(b)
n_edges = size(edges, 1);
E_prob  = zeros(dim, 1);
for e = 1:n_edges
    if J(e) ~= 0
        E_prob = E_prob + J(e) * spins(:, edges(e,1)+1) .* spins(:, edges(e,2)+1);
    end
end

% Initialise: ground state of driver = |+⟩^⊗n
psi = ones(dim, 1) / sqrt(dim);

% Strang splitting loop
s  = s_min;
ds = max_step;

while s < s_max - 1e-12
    ds = min(ds, s_max - s);

    % Interpolate schedule
    ns   = s * 1000;
    nlo  = min(floor(ns), 999);
    frac = ns - nlo;
    nlo  = nlo + 1;                % 1-based
    nhi  = nlo + 1;

    delta_s = delta_sched(nlo) + frac * (delta_sched(nhi) - delta_sched(nlo));
    A_s     = A_sched(nlo)     + frac * (A_sched(nhi)     - A_sched(nlo));

    % Half-step: problem (diagonal)
    ph  = exp(-1i * A_s * E_prob * (phase_factor * ds * 0.5));
    psi = ph .* psi;

    % Full step: driver (tensor product)
    theta = delta_s * phase_factor * ds;
    psi   = soe_driver(psi, n_qubits, theta);

    % Half-step: problem
    psi = ph .* psi;

    s = s + ds;
end

% Correlation matrix → influence → score
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


% ═══════════════════════════════════════════════════════════════════
%  Driver rotation: exp(i·θ·Σ_i σ_x^i) via per-qubit butterfly
% ═══════════════════════════════════════════════════════════════════
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


% ═══════════════════════════════════════════════════════════════════
%  Index → Ising J couplings
% ═══════════════════════════════════════════════════════════════════
function J = soe_idx_to_couplings(idx, n_edges)
%   bit e = 1  →  Green / FM    →  J = −1
%   bit e = 0  →  Purple / AFM  →  J = +1
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

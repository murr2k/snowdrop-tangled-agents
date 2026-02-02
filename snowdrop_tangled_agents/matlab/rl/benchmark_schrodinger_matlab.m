function benchmark_schrodinger_matlab()
%BENCHMARK_SCHRODINGER_MATLAB  Time the split-operator Schrödinger solver
%   on Petersen-graph terminal states and project full-LUT generation cost.
%
%   The Python SchrodingerEquationAdjudicator diagonalises H(s) at every
%   time step (la.eigh on a 1024×1024 dense matrix — the developer's own
%   comment on line 140 reads "this is taking up most of the time!!!").
%   With ~2000 steps that gives O(2^{3n}) total work at n = 10 qubits.
%
%   This benchmark uses Strang splitting, which exploits the structure of
%   the D-Wave annealing Hamiltonian:
%
%     H(s) = −Δ(s)·Σ_i σ_x^i              ← driver
%           + A(s)·Σ_{i<j} J_ij σ_z^i σ_z^j  ← problem
%
%   • Driver:  −Δ·Σσ_x  is a sum of COMMUTING single-qubit terms, so
%     exp(+i·Δ·Σσ_x·dt) = Π_i R_x(θ).  Each qubit rotation is applied
%     via a reshape butterfly in O(2^n).  Total: O(n·2^n) per step.
%
%   • Problem: A·Σ J σ_z σ_z  is DIAGONAL in the computational basis.
%     exp(-i·A·E_prob·dt) is an elementwise phase.  Cost: O(2^n).
%
%   Strang splitting per step:  U ≈ U_prob(dt/2)·U_drv(dt)·U_prob(dt/2)
%   Total per-step cost: O(n·2^n) ≈ 10 K flops  vs  O(2^{3n}) ≈ 10^9 flops.
%
%   Usage:  benchmark_schrodinger_matlab()

fprintf('╔════════════════════════════════════════════════════════════╗\n');
fprintf('║   Schrödinger Split-Operator Benchmark — Petersen (n=10)  ║\n');
fprintf('╚════════════════════════════════════════════════════════════╝\n\n');

% ─── Petersen graph ────────────────────────────────────────────────
n_qubits = 10;
dim      = 2^n_qubits;           % 1024
n_edges  = 15;
n_states = 2^n_edges;            % 32 768

% Edge list — from GraphProperties.graph_database[5], lexical order.
edges = [0,2; 0,3; 0,6; 1,3; 1,4; 1,7; 2,4; 2,8; 3,9; 4,5; ...
         5,6; 5,9; 6,7; 7,8; 8,9];

% Player nodes (graph_number = 5 in game engine)
p1_node = 5;  % 0-indexed
p2_node = 7;

% ─── Schrödinger parameters (match Python adjudicator exactly) ────
tf       = 40.0;       % anneal time in ns
s_min    = 0.001;
s_max    = 0.999;
max_step = 0.0005;     % ~2 000 steps nominal
epsilon  = 0.0005;     % draw threshold

% ─── Load D-Wave Advantage2 schedule ──────────────────────────────
fprintf('[1/3] Loading D-Wave Advantage2 schedule...\n');

sched_path = fullfile('C:', 'Users', 'murr2', 'projects', 'snowdrop-adjudicators', 'snowdrop_adjudicators', 'schrodinger', 'advantage2.1.3.txt');
if ~isfile(sched_path)
    error('Schedule file not found: %s', sched_path);
end

raw = readmatrix(sched_path);          % 1001 × 3

% Python convention (0-indexed columns):
%   delta_qubit = col[1] / 2    then *= 0.5   →  col[1] / 4
%   big_e_qubit = col[2]        then *= 0.5   →  col[2] * 0.5
% MATLAB 1-based: col[1]→2, col[2]→3.
delta_sched = raw(:, 2) / 4;
A_sched     = raw(:, 3) * 0.5;

fprintf('    Schedule: %d points\n', size(raw, 1));
fprintf('    Δ  ∈ [%.4f, %.4f] GHz\n', min(delta_sched), max(delta_sched));
fprintf('    A  ∈ [%.4f, %.4f] GHz\n', min(A_sched), max(A_sched));

% ─── Precompute spin table ────────────────────────────────────────
fprintf('\n[2/3] Precomputing spin table...\n');

% spins(b+1, i+1) = ±1 for basis state b ∈ [0..1023], qubit i ∈ [0..9]
spins = zeros(dim, n_qubits);
for i = 0:n_qubits-1
    mask = 2^i;
    for b = 0:dim-1
        spins(b+1, i+1) = 2 * double(bitand(b, mask) > 0) - 1;
    end
end
fprintf('    %d × %d spin table ready\n', dim, n_qubits);

% ─── Benchmark ─────────────────────────────────────────────────────
fprintf('\n[3/3] Solving test states...\n\n');

test_idx = [1, 100, 1000, 5000, 10000, 16384, 20000, 30000, 32768];
n_tests  = length(test_idx);

fprintf('  %8s  %12s  %6s  %6s  %8s\n', 'Index', 'Score', 'Winner', 'Norm', 'Time_s');
fprintf('  %s\n', repmat('─', 1, 50));

times  = zeros(n_tests, 1);
scores = zeros(n_tests, 1);
norms  = zeros(n_tests, 1);

for t = 1:n_tests
    J = idx_to_couplings(test_idx(t), n_edges);

    tic;
    [scores(t), norms(t)] = solve_one(J, edges, n_qubits, dim, ...
        delta_sched, A_sched, tf, s_min, s_max, max_step, ...
        spins, p1_node, p2_node);
    times(t) = toc;

    if    scores(t) >  epsilon
        w = 'P1';
    elseif scores(t) < -epsilon
        w = 'P2';
    else
        w = 'Draw';
    end

    fprintf('  %8d  %+12.6f  %6s  %6.4f  %8.3f\n', ...
        test_idx(t), scores(t), w, norms(t), times(t));
end

% ─── Projection ────────────────────────────────────────────────────
med_t  = median(times);
total  = med_t * n_states;

fprintf('\n╔════════════════════════════════════════════════════════════╗\n');
fprintf('║                   TIMING PROJECTION                        ║\n');
fprintf('╠════════════════════════════════════════════════════════════╣\n');
fprintf('║  Median time / state  :  %8.3f s                           ║\n', med_t);
fprintf('║  Steps per state      :  ~%d (max_step = %.4f)             ║\n', ...
    round((s_max - s_min) / max_step), max_step);
fprintf('║                                                            ║\n');
fprintf('║  Full LUT  (32 768 terminal states)                        ║\n');
fprintf('║    1 worker  :  %7.1f min  (%6.2f hrs)                     ║\n', ...
    total/60, total/3600);
fprintf('║    4 workers :  %7.1f min                                  ║\n', total/60/4);
fprintf('║    8 workers :  %7.1f min                                  ║\n', total/60/8);
fprintf('║    + MATLAB expanded_lut extension (parfor):  2–5 min      ║\n');
fprintf('╚════════════════════════════════════════════════════════════╝\n');

fprintf('\nValidation: compare scores against Python output for the same');
fprintf('\nstate indices using generate_terminal_lut.py --graph 5.\n');

end


% ═══════════════════════════════════════════════════════════════════
%  Core solver — one terminal state → score + final norm
% ═══════════════════════════════════════════════════════════════════
function [score, final_norm] = solve_one(J, edges, n_qubits, dim, ...
    delta_sched, A_sched, tf, s_min, s_max, max_step, spins, p1_node, p2_node)

phase_factor = 2 * pi * tf;     % GHz × ns → radians

% ─ Problem-Hamiltonian diagonal energies ─────────────────────────
% For the tangled game h = 0 (all local fields zero), so:
%   E_prob(b) = Σ_e J_e · spin_{i_e}(b) · spin_{j_e}(b)
n_edges = size(edges, 1);
E_prob  = zeros(dim, 1);
for e = 1:n_edges
    if J(e) ~= 0
        E_prob = E_prob + J(e) * spins(:, edges(e,1)+1) .* spins(:, edges(e,2)+1);
    end
end

% ─ Initialise: ground state of transverse field = |+⟩^⊗n ─────────
psi = ones(dim, 1) / sqrt(dim);

% ─ Split-operator (Strang splitting) evolution ──────────────────
s  = s_min;
ds = max_step;

while s < s_max - 1e-12
    ds = min(ds, s_max - s);

    % Schedule interpolation  (s ∈ [0,1] → schedule index ∈ [0,1000])
    ns   = s * 1000;
    nlo  = min(floor(ns), 999);    % 0-based
    frac = ns - nlo;
    nlo  = nlo + 1;                % 1-based MATLAB index
    nhi  = nlo + 1;

    delta_s = delta_sched(nlo) + frac * (delta_sched(nhi) - delta_sched(nlo));
    A_s     = A_sched(nlo)     + frac * (A_sched(nhi)     - A_sched(nlo));

    % Half-step: problem (diagonal phase)
    %   exp(-i · A(s) · E_prob · 2π·tf · ds/2)
    ph  = exp(-1i * A_s * E_prob * (phase_factor * ds * 0.5));
    psi = ph .* psi;

    % Full step: driver
    %   H_drv = −Δ·Σσ_x  →  U_drv = exp(+i·Δ·2π·tf·ds · Σσ_x)
    %                               = Π_i R_x(θ),  θ = Δ·2π·tf·ds
    theta = delta_s * phase_factor * ds;
    psi   = apply_driver(psi, n_qubits, theta);

    % Half-step: problem (same phases, already computed)
    psi = ph .* psi;

    s = s + ds;
end

final_norm = norm(psi);

% ─ Correlation matrix → influence → score ───────────────────────
probs = real(psi .* conj(psi));            % |ψ_b|²  (real by construction)
mag   = probs' * spins;                    % 1 × n_qubits magnetisations

corr = zeros(n_qubits);
for i = 1:n_qubits
    for j = i+1:n_qubits
        corr(i,j) = probs' * (spins(:,i) .* spins(:,j)) - mag(i)*mag(j);
    end
end
corr      = corr + corr';                  % symmetrise
influence = sum(corr, 2);                  % row sums

score = influence(p1_node + 1) - influence(p2_node + 1);

end


% ═══════════════════════════════════════════════════════════════════
%  Driver rotation — tensor product of single-qubit σ_x rotations
% ═══════════════════════════════════════════════════════════════════
function psi = apply_driver(psi, n_qubits, theta)
%APPLY_DRIVER  Apply exp(i·θ·Σ_i σ_x^i) via per-qubit butterfly.
%
%   Decomposes into n independent single-qubit rotations:
%     R_x(θ) = cos θ · I + i sin θ · σ_x
%            = [ cos θ     i sin θ  ]
%              [ i sin θ   cos θ    ]
%
%   Reshape trick isolates qubit q as the 2nd dimension of a 3-D array.
%   Cost per qubit: O(2^n).  Total: O(n · 2^n).

c = cos(theta);
s = sin(theta);

for q = 0:n_qubits-1
    stride = 2^q;
    psi = reshape(psi, stride, 2, []);     % (lower_bits, qubit_q, upper_bits)

    a0 = psi(:,1,:);                       % amplitudes where qubit q = 0
    a1 = psi(:,2,:);                       % amplitudes where qubit q = 1

    psi(:,1,:) =  c * a0 + 1i*s * a1;
    psi(:,2,:) = 1i*s * a0 +  c * a1;

    psi = psi(:);                          % flatten
end

end


% ═══════════════════════════════════════════════════════════════════
%  Index → Ising J couplings
% ═══════════════════════════════════════════════════════════════════
function J = idx_to_couplings(idx, n_edges)
%IDX_TO_COUPLINGS  1-based LUT index → Ising coupling vector.
%   Bit e = 1  →  Green / FM    →  J = −1
%   Bit e = 0  →  Purple / AFM  →  J = +1

J    = zeros(n_edges, 1);
bits = idx - 1;                        % 0-based bit pattern
for e = 1:n_edges
    if bitand(bits, 2^(e-1)) > 0
        J(e) = -1;
    else
        J(e) =  1;
    end
end

end

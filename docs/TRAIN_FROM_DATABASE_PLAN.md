# Plan: Train Value Network from Database

## Goal

Train the RL agent's neural networks using the ~44+ games already collected in the SQLite database, rather than starting from scratch with self-play.

---

## Research Findings

### MATLAB Toolbox Options

| Approach | Toolbox | Function | Supported Agents |
|----------|---------|----------|------------------|
| **Offline RL** | Reinforcement Learning | [`trainFromData`](https://www.mathworks.com/help/reinforcement-learning/ref/trainfromdata.html) | DQN, DDPG, TD3, SAC (off-policy only) |
| **Supervised Learning** | Deep Learning | [`trainnet`](https://www.mathworks.com/help/deeplearning/ref/trainnet.html) | Any network architecture |
| **Behavior Cloning** | Reinforcement Learning | [`rlBehaviorCloningRegularizerOptions`](https://www.mathworks.com/help/reinforcement-learning/ug/train-agent-offline-to-control-furuta-pendulum.html) | Off-policy agents |

### Key Constraint

**Our current PPO agent is on-policy and NOT supported by `trainFromData`.**

Options:
1. **Switch to SAC** (off-policy) - Use MATLAB's native offline RL
2. **Supervised pre-training** - Train networks with Deep Learning Toolbox, then initialize PPO
3. **Hybrid approach** - Supervised warm-start, then self-play fine-tuning

---

## Recommended Approach: Supervised Pre-Training + PPO Fine-Tuning

This approach uses the Deep Learning Toolbox for initial training, then transfers weights to PPO for self-play refinement.

### Why This Approach?

1. **Keeps PPO architecture** - No need to redesign agent
2. **Uses collected data** - Leverages the 44+ games in database
3. **Best of both worlds** - Supervised learning for warm-start, RL for optimization
4. **MATLAB best practice** - Aligns with [Train Network with Numeric Features](https://www.mathworks.com/help/deeplearning/ug/train-network-on-data-set-of-numeric-features.html) workflow

---

## Implementation Plan

### Phase 1: Data Extraction Pipeline

**File:** `snowdrop_tangled_agents/matlab/rl/extract_training_data.m`

```
Database (game_stats.db)
    │
    ├─ games table: id, opponent, result, final_score, our_score, opp_score
    │
    └─ moves table: game_id, move_num, player, edge, color, state_after, score_after
            │
            ▼
    Extract (state, action, outcome) tuples
            │
            ▼
    Feature Engineering
            │
            ├─ state_features (50): board encoding + metadata
            ├─ action_label (30): one-hot action taken
            └─ outcome_value (1): normalized game result [-1, +1]
```

**Tasks:**
1. Connect to SQLite using Database Toolbox
2. Query moves joined with games (WHERE result IS NOT NULL)
3. Convert state strings to 50-element feature vectors
4. Convert actions to indices (edge × 2 + color)
5. Normalize outcomes: `tanh(final_score_delta / 3)`
6. Save as `.mat` file for training

### Phase 2: Value Network Training

**File:** `snowdrop_tangled_agents/matlab/rl/train_value_network.m`

Train a regression network to predict game outcome from position.

**Architecture:**
```
featureInputLayer(50, Normalization="zscore")
    ↓
fullyConnectedLayer(128)
batchNormalizationLayer
reluLayer
dropoutLayer(0.3)
    ↓
fullyConnectedLayer(64)
batchNormalizationLayer
reluLayer
dropoutLayer(0.2)
    ↓
fullyConnectedLayer(32)
reluLayer
    ↓
fullyConnectedLayer(1)
tanhLayer                    % Output: value ∈ [-1, +1]
```

**Training:**
```matlab
options = trainingOptions("adam", ...
    MaxEpochs=100, ...
    MiniBatchSize=64, ...
    ValidationFrequency=50, ...
    ValidationData={X_val, y_val}, ...
    Shuffle="every-epoch", ...
    Plots="training-progress");

valueNet = trainnet(X_train, y_train, layers, "mse", options);
```

### Phase 3: Policy Network Training (Imitation Learning)

**File:** `snowdrop_tangled_agents/matlab/rl/train_policy_network.m`

Train a classification network to imitate expert moves.

**Architecture:**
```
featureInputLayer(50, Normalization="zscore")
    ↓
fullyConnectedLayer(128)
batchNormalizationLayer
reluLayer
dropoutLayer(0.3)
    ↓
fullyConnectedLayer(64)
reluLayer
    ↓
fullyConnectedLayer(30)
softmaxLayer                 % Output: P(action) for 30 actions
```

**Training:**
```matlab
options = trainingOptions("adam", ...
    MaxEpochs=100, ...
    MiniBatchSize=64, ...
    ValidationFrequency=50, ...
    Shuffle="every-epoch", ...
    Plots="training-progress");

policyNet = trainnet(X_train, y_actions, layers, "crossentropy", options);
```

**Note:** Only train on winning games or weight samples by outcome to learn good moves.

### Phase 4: Transfer Weights to PPO Agent

**File:** `snowdrop_tangled_agents/matlab/rl/initialize_ppo_from_pretrained.m`

Transfer the pre-trained networks into a PPO agent.

```matlab
function agent = initialize_ppo_from_pretrained(env, valueNet, policyNet)
    % Create base PPO agent
    agent = createPPOAgent(env);

    % Get actor/critic from agent
    actor = getActor(agent);
    critic = getCritic(agent);

    % Extract dlnetworks
    actorNet = getModel(actor);
    criticNet = getModel(critic);

    % Transfer weights from pre-trained networks
    % (Layer-by-layer weight copying)
    actorNet = transferWeights(actorNet, policyNet);
    criticNet = transferWeights(criticNet, valueNet);

    % Update agent with new networks
    actor = setModel(actor, actorNet);
    critic = setModel(critic, criticNet);
    agent = setActor(agent, actor);
    agent = setCritic(agent, critic);
end
```

### Phase 5: Fine-Tune with Self-Play

Use existing `trainParallel.m` to refine the pre-trained agent:

```matlab
% Load pre-trained agent
agent = initialize_ppo_from_pretrained(env, valueNet, policyNet);

% Fine-tune with self-play (fewer episodes needed now)
[trainedAgent, stats] = trainParallel(agent, ...
    'MaxEpisodes', 500, ...   % Less than starting from scratch
    'NumWorkers', 4);
```

---

## File Structure

```
snowdrop_tangled_agents/matlab/rl/
├── extract_training_data.m      # Phase 1: Database → training data
├── train_value_network.m        # Phase 2: Value network training
├── train_policy_network.m       # Phase 3: Policy network training
├── initialize_ppo_from_pretrained.m  # Phase 4: Weight transfer
├── train_from_database.m        # Orchestrator script
└── test_pretrained_agent.m      # Validation tests
```

---

## Data Requirements

| Metric | Minimum | Recommended |
|--------|---------|-------------|
| Total games | 30 | 100+ |
| Winning games | 10 | 30+ |
| Moves per game | ~15 | ~15 |
| Total samples | 450 | 1500+ |

Current database: ~44 games → ~660 move samples (borderline sufficient)

---

## Validation Plan

1. **Value Network Accuracy**
   - Test on held-out games
   - MSE < 0.2 on outcome prediction
   - Correlation > 0.5 with actual outcomes

2. **Policy Network Accuracy**
   - Top-5 accuracy > 50% on held-out moves
   - Action distribution matches expert play

3. **Agent Performance**
   - Pre-trained agent wins > 30% vs random (before fine-tuning)
   - After fine-tuning: wins > 50% vs heuristic opponent

---

## Alternative: Switch to SAC Agent

If supervised pre-training proves insufficient, we could switch to SAC (Soft Actor-Critic) which supports `trainFromData`:

```matlab
% Create SAC agent
agent = rlSACAgent(obsInfo, actInfo);

% Configure behavior cloning regularization
agent.AgentOptions.BatchDataRegularizerOptions = rlBehaviorCloningRegularizerOptions;

% Create file datastore from exported data
fds = fileDatastore("training_data/*.mat", ReadFcn=@readExperience);

% Train offline
options = rlTrainingFromDataOptions(MaxEpochs=100, NumStepsPerEpoch=500);
trainFromData(agent, fds, options);
```

This is more complex but uses MATLAB's native offline RL infrastructure.

---

## Implementation Order

1. **extract_training_data.m** - Get data out of SQLite
2. **train_value_network.m** - Train value prediction
3. **train_policy_network.m** - Train action imitation
4. **initialize_ppo_from_pretrained.m** - Transfer weights
5. **train_from_database.m** - Orchestrator
6. **test_pretrained_agent.m** - Validation

---

## Success Criteria

- [ ] Extract 500+ training samples from database
- [ ] Value network achieves MSE < 0.2 on validation set
- [ ] Policy network achieves top-5 accuracy > 50%
- [ ] Pre-trained agent beats random opponent > 30%
- [ ] Fine-tuned agent beats heuristic opponent > 50%

---

## Sources

- [trainnet - MATLAB Deep Learning Toolbox](https://www.mathworks.com/help/deeplearning/ref/trainnet.html)
- [trainFromData - MATLAB RL Toolbox](https://www.mathworks.com/help/reinforcement-learning/ref/trainfromdata.html)
- [Train Network with Numeric Features](https://www.mathworks.com/help/deeplearning/ug/train-network-on-data-set-of-numeric-features.html)
- [Train Agent Offline (Quanser QUBE)](https://www.mathworks.com/help/reinforcement-learning/ug/train-agent-offline-to-control-furuta-pendulum.html)
- [Behavioral Cloning - GeeksforGeeks](https://www.geeksforgeeks.org/deep-learning/behavioral-cloning/)

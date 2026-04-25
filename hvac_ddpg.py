"""
HVAC Energy Optimization using CNN-LSTM + DDPG
================================================
References:
- DDPG Algorithm: Lillicrap et al. (2015) "Continuous control with deep reinforcement learning"
- DDPG Structure: XinJingHao/DDPG-Pytorch (clean actor/critic/replay/OUNoise pattern)
- HVAC Control concept: Li et al. (2019) "Cooling Control Algorithm using DDPG"
- CNN-LSTM Prediction: Gebreyesus et al. (2024) "Hybrid CNN-LSTM for DC energy"

Architecture:
  CNN-LSTM → predicts T_Return (room temperature, 15 mins ahead)
  DDPG     → adjusts SP_Return (cooling setpoint) to minimize PUE
             while maintaining T_Return near comfort target
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import collections
import random
import argparse
from tqdm import tqdm
from keras.models import load_model, Model
from keras.layers import Dense, Input, Concatenate
from keras.optimizers import Adam

# ─── Load CNN-LSTM Model & Scalers ───────────────────────────────────────────

print("Loading CNN-LSTM model...")
cnn_lstm_model = load_model('cnn_lstm_hvac.keras')

with open('feature_scaler.pkl', 'rb') as f:
    feature_scaler = pickle.load(f)
with open('target_scaler.pkl', 'rb') as f:
    target_scaler = pickle.load(f)
with open('model_config.pkl', 'rb') as f:
    config = pickle.load(f)

FEATURE_COLS = config['FEATURE_COLS']
n_input      = config['n_input']

print(f"CNN-LSTM loaded | Features: {FEATURE_COLS} | Lookback: {n_input} steps")

# ─── Load and Prepare Dataset ────────────────────────────────────────────────

FILE_PATH = r'C:\Users\xin37\github\CNN-LSTM-model-for-energy-usage-forecasting-1\data\HVAC_NE_EC_19-21.csv'

print("Loading dataset...")
df = pd.read_csv(FILE_PATH, parse_dates=['Timestamp'], index_col='Timestamp')
df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)

# Build df_model with delta features
df_model = df.copy()
df_model['T_Supply_diff']  = df_model['T_Supply'].diff()
df_model['T_Return_diff']  = df_model['T_Return'].diff()
df_model['SP_Return_diff'] = df_model['SP_Return'].diff()
df_model['Power_diff']     = df_model['Power'].diff()
df_model.dropna(inplace=True)

print(f"Dataset loaded: {df_model.shape}")
print(f"Date range: {df_model.index[0]} to {df_model.index[-1]}")
print(f"T_Return mean: {df_model['T_Return'].mean():.2f}°C (comfort target)")
print(f"SP_Return distribution:\n{df_model['SP_Return'].value_counts().head(5)}")

# ─── Pre-compute CNN-LSTM Predictions (Offline Batch) ────────────────────────

print("\n" + "="*60)
print("Training Mode: OFFLINE BATCH RL")
print("CNN-LSTM predictions: pre-computed once on full dataset")
print("Note: SP_Return changes by DDPG do NOT affect CNN-LSTM predictions")
print("      Consistent with offline RL approach (Yi et al., 2020)")
print("="*60)

print("\nPre-computing CNN-LSTM predictions...")
df_reset     = df_model.reset_index(drop=True)
X_all_scaled = feature_scaler.transform(df_reset[FEATURE_COLS].values)
n_samples    = len(X_all_scaled) - n_input

windows_scaled = np.array([
    X_all_scaled[i: i + n_input]
    for i in range(n_samples)
], dtype=np.float32)

print(f"Windows shape: {windows_scaled.shape}")
predictions_scaled  = cnn_lstm_model.predict(windows_scaled, batch_size=256, verbose=1)
predictions_celsius = target_scaler.inverse_transform(predictions_scaled).flatten()

# Pad beginning so index aligns with df_reset
all_predictions = np.concatenate([
    np.full(n_input, predictions_celsius[0]),
    predictions_celsius
])
print(f"Pre-computed {len(all_predictions)} predictions")
print(f"Prediction range: {all_predictions.min():.2f} - {all_predictions.max():.2f}°C")

# ─── HVAC Environment ────────────────────────────────────────────────────────

class HVACEnv:
    """
    Data-driven HVAC simulation environment for offline RL training.

    State (8 features — aligned with Chapter 3 state space definition):
        [predicted_T_return, T_outdoor, SP_Return, Power,
         RH_outdoor, RH_return, T_saturation, comfort_error]

    Action:
        delta SP_Return — continuous adjustment to cooling setpoint (°C)
        bounded within [-action_range, +action_range] per step
        setpoint constrained to [sp_min, sp_max] operational range

    Reward (Chapter 3, Section 3.3.4):
        r(s,a) = -|T_Return_actual - T_target|
                 - pue_weight * (PUE - 1.0)
                 - thermal_penalty (if T_Return > safety threshold)

    PUE (Chapter 3, Section 3.5.4):
        PUE = (P_IT + P_cooling) / P_IT
        P_IT assumed = 10 kW (typical small data centre)
        PUE = 1.0 → ideal (no cooling overhead)
    """

    # Safety threshold from Chapter 3 (Section 3.3.3 Constraints)
    SAFETY_THRESHOLD = 25.0   # max safe T_Return (°C)
    ASSUMED_IT_POWER = 10.0   # assumed IT load (kW)

    def __init__(self, df, feature_cols,
                 t_target=20.0,     # T_Return comfort target (°C)
                 sp_min=19.0,       # SP_Return min (matches dataset range 18.5-23.5)
                 sp_max=23.5,       # SP_Return max
                 action_range=0.5,  # max delta per step (°C)
                 pue_weight=0.3,    # PUE penalty weight in reward
                 thermal_penalty_weight=10.0,  # safety violation penalty
                 episode_len=200):  # steps per episode

        self.df                     = df.reset_index(drop=True)
        self.feature_cols           = feature_cols
        self.t_target               = t_target
        self.sp_min                 = sp_min
        self.sp_max                 = sp_max
        self.action_range           = action_range
        self.pue_weight             = pue_weight
        self.thermal_penalty_weight = thermal_penalty_weight
        self.episode_len            = episode_len

        # State dim = 8 (Chapter 3 Section 3.3.3)
        self.state_dim  = 8
        self.action_dim = 1

        self.current_idx  = None
        self.current_sp   = None
        self.episode_pues = []
        self.step_count   = 0

    def _predict_temperature(self, idx):
        """
        Return pre-computed CNN-LSTM prediction for T_Return at idx.
        Offline batch prediction — predictions do not respond to SP changes.
        """
        return float(all_predictions[idx])

    def _calculate_pue(self, idx):
        """
        PUE = (P_IT + P_cooling) / P_IT
        P_IT = 10 kW (assumed, typical small DC)
        P_cooling = fan power from dataset (kW)
        PUE = 1.0 → ideal efficiency (no overhead)
        PUE > 1.0 → overhead from cooling system
        (Chapter 3, Section 3.5.4)
        """
        cooling_power = float(self.df['Power'].iloc[idx])
        pue = (self.ASSUMED_IT_POWER + cooling_power) / self.ASSUMED_IT_POWER
        return float(np.clip(pue, 1.0, 3.0))

    def _build_state(self, predicted_t, idx):
        """
        Build 8-feature state vector (Chapter 3, Section 3.3.3 State Space).

        Features:
            predicted_t   — CNN-LSTM predicted T_Return (°C)
            T_outdoor     — outdoor air temperature (°C)
            SP_Return     — current cooling setpoint controlled by DDPG (°C)
            Power         — fan power consumption (kW)
            RH_outdoor    — outdoor relative humidity (%)
            RH_return     — return air relative humidity (%)
            T_saturation  — saturation temperature in humidifier (°C)
            comfort_error — deviation from target: predicted_t - t_target (°C)
        """
        t_outdoor    = float(self.df['T_Outdoor'].iloc[idx])
        power        = float(self.df['Power'].iloc[idx])
        rh_outdoor   = float(self.df['RH_Outdoor'].iloc[idx])
        rh_return    = float(self.df['RH_Return'].iloc[idx])
        t_saturation = float(self.df['T_Saturation'].iloc[idx])
        comfort_error= float(predicted_t - self.t_target)

        return np.array([
            predicted_t,    # CNN-LSTM state observation
            t_outdoor,      # environmental condition
            self.current_sp,# DDPG-controlled setpoint
            power,          # energy consumption
            rh_outdoor,     # humidity (environmental)
            rh_return,      # humidity (return air)
            t_saturation,   # cooling system status
            comfort_error   # thermal deviation from target
        ], dtype=np.float32)

    def reset(self):
        """Start a new episode at a random position in the historical dataset."""
        max_start        = len(self.df) - self.episode_len - 1
        self.current_idx = random.randint(n_input, max_start)
        self.current_sp  = float(self.df['SP_Return'].iloc[self.current_idx])
        self.episode_pues= []
        self.step_count  = 0

        predicted_t = self._predict_temperature(self.current_idx)
        return self._build_state(predicted_t, self.current_idx)

    def step(self, action):
        """
        Apply DDPG action, advance one timestep, return (next_state, reward, done, info).

        Action: delta SP_Return (°C), clipped to [-action_range, +action_range]
        Reward: -comfort_error - pue_weight*(PUE-1) - thermal_penalty
                (Chapter 3, Section 3.3.4)
        """
        # 1. Apply action — adjust SP_Return
        delta_sp        = float(np.clip(np.squeeze(action), -self.action_range, self.action_range))
        self.current_sp = float(np.clip(self.current_sp + delta_sp, self.sp_min, self.sp_max))

        # 2. Advance timestep
        self.current_idx += 1
        self.step_count  += 1
        done = (self.step_count >= self.episode_len) or \
               (self.current_idx >= len(self.df) - 1)

        # 3. CNN-LSTM predicted T_Return (state observation)
        predicted_t = self._predict_temperature(self.current_idx)

        # 4. Actual T_Return from dataset (ground truth for reward)
        actual_t = float(self.df['T_Return'].iloc[self.current_idx])

        # 5. PUE calculation
        pue = self._calculate_pue(self.current_idx)
        self.episode_pues.append(pue)

        # 6. Reward function (Chapter 3, Section 3.3.4)
        #    Term 1: thermal comfort — penalize deviation from target
        #    Term 2: energy efficiency — penalize PUE overhead
        #    Term 3: safety constraint — strongly penalize temp violations
        comfort_error = abs(actual_t - self.t_target)

        # Safety penalty (log-barrier style) — Chapter 3 Section 3.3.3 Constraints
        if actual_t > self.SAFETY_THRESHOLD:
            thermal_penalty = self.thermal_penalty_weight * (actual_t - self.SAFETY_THRESHOLD)
        else:
            thermal_penalty = 0.0

        # Log-barrier safety penalty
        # λ * ln(1 + exp(T_predicted - Φ))
        temp_diff = np.clip(predicted_t - self.SAFETY_THRESHOLD, -50, 50)
        thermal_penalty = self.thermal_penalty_weight * math.log1p(math.exp(temp_diff))

        reward = (-comfort_error
                  - self.pue_weight * (pue - 1.0)
                  - thermal_penalty)

        # 7. Next state
        next_state = self._build_state(predicted_t, self.current_idx)

        info = {
            'predicted_T_return': predicted_t,
            'actual_T_return':    actual_t,
            'SP_Return':          self.current_sp,
            'PUE':                pue,
            'comfort_error':      comfort_error,
            'thermal_penalty':    thermal_penalty,
            'reward':             reward
        }

        return next_state, reward, done, info

    def average_pue(self):
        if len(self.episode_pues) == 0:
            return 1.0
        return float(np.mean(self.episode_pues))

# ─── Replay Buffer ───────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    Experience replay buffer for off-policy DDPG training.
    Stores (state, action, reward, next_state, done) tuples.
    Random sampling breaks temporal correlation for stable learning.
    """
    def __init__(self, limit=100000):
        self.buffer = collections.deque(maxlen=limit)

    def add(self, s, a, r, ns, done):
        self.buffer.append((s, a, r, ns, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (np.array(s,  dtype=np.float32),
                np.array(a,  dtype=np.float32),
                np.array(r,  dtype=np.float32),
                np.array(ns, dtype=np.float32),
                np.array(d,  dtype=np.float32))

    def __len__(self):
        return len(self.buffer)

# ─── Ornstein-Uhlenbeck Noise ────────────────────────────────────────────────

class OUNoise:
    """
    Ornstein-Uhlenbeck process for temporally correlated exploration noise.
    Produces smooth, mean-reverting noise suitable for physical control systems.
    Reference: Uhlenbeck & Ornstein (1930); used in DDPG (Lillicrap et al., 2015).
    """
    def __init__(self, size, mu=0., theta=0.15, sigma=0.2):
        self.size  = size
        self.mu    = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = self.mu.copy()

    def sample(self):
        dx = self.theta * (self.mu - self.state) + \
             self.sigma * np.random.randn(self.size)
        self.state += dx
        return self.state

# ─── Actor Network ───────────────────────────────────────────────────────────

def build_actor(state_dim, action_dim):
    """
    Actor network: maps state → deterministic action (delta SP_Return).
    tanh activation bounds output to [-1, 1], scaled by action_range.
    Architecture: Dense(128) → Dense(64) → Dense(32) → Dense(action_dim, tanh)
    (Chapter 3, Section 3.3.2 — Actor Network)
    """
    inp = Input(shape=(state_dim,), name='state_input')
    x   = Dense(128, activation='relu')(inp)
    x   = Dense(64,  activation='relu')(x)
    x   = Dense(32,  activation='relu')(x)
    out = Dense(action_dim, activation='tanh', name='action')(x)
    return Model(inputs=inp, outputs=out, name='actor')

# ─── Critic Network ──────────────────────────────────────────────────────────

def build_critic(state_dim, action_dim):
    """
    Critic network: maps (state, action) → Q-value.
    Estimates expected cumulative reward (Bellman equation).
    Architecture: Concat(state, action) → Dense(128) → Dense(64) → Dense(32) → Dense(1)
    (Chapter 3, Section 3.3.2 — Critic Network)
    """
    state_inp  = Input(shape=(state_dim,),  name='state_input')
    action_inp = Input(shape=(action_dim,), name='action_input')
    x = Concatenate()([state_inp, action_inp])
    x = Dense(128, activation='relu')(x)
    x = Dense(64,  activation='relu')(x)
    x = Dense(32,  activation='relu')(x)
    out = Dense(1, activation='linear', name='q_value')(x)
    return Model(inputs=[state_inp, action_inp], outputs=out, name='critic')

# ─── DDPG Agent ──────────────────────────────────────────────────────────────

class DDPGAgent:
    """
    Deep Deterministic Policy Gradient (DDPG) agent.

    Key components:
    - Actor  : deterministic policy μ(s) → a
    - Critic : Q-function Q(s,a) → expected return
    - Target networks: soft-updated copies for stable learning
    - Replay buffer: off-policy experience storage
    - OUNoise: smooth exploration during training

    Reference: Lillicrap et al. (2015) arXiv:1509.02971
    (Chapter 3, Section 3.3.2)
    """
    def __init__(self, state_dim, action_dim, action_range,
                 gamma=0.99,      # discount factor γ
                 tau=0.005,       # soft update rate τ
                 lr_actor=1e-4,   # actor learning rate
                 lr_critic=1e-3,  # critic learning rate
                 buffer_size=100000,
                 batch_size=64,
                 warmup=500):

        self.action_range = action_range
        self.gamma        = gamma
        self.tau          = tau
        self.batch_size   = batch_size
        self.warmup       = warmup

        # Online networks
        self.actor  = build_actor(state_dim, action_dim)
        self.critic = build_critic(state_dim, action_dim)

        # Target networks (initialized as copies, soft-updated during training)
        self.target_actor  = build_actor(state_dim, action_dim)
        self.target_critic = build_critic(state_dim, action_dim)
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

        self.actor_opt  = Adam(learning_rate=lr_actor)
        self.critic_opt = Adam(learning_rate=lr_critic)

        self.memory = ReplayBuffer(buffer_size)
        self.noise  = OUNoise(action_dim)

    def act(self, state, add_noise=True):
        """Select action: actor output scaled to action_range, with optional OU noise."""
        state_t = tf.convert_to_tensor([state], dtype=tf.float32)
        action  = self.actor(state_t, training=False).numpy()[0]
        if add_noise:
            action += self.noise.sample()
        action = action * self.action_range
        return np.clip(action, -self.action_range, self.action_range)

    def remember(self, s, a, r, ns, done):
        self.memory.add(s, a, r, ns, done)

    def learn(self):
        """
        Sample minibatch and update actor and critic via gradient descent.
        Critic: minimize Bellman error (MSE between Q and target Q)
        Actor:  maximize Q(s, μ(s)) via policy gradient
        Targets: soft-updated each step (θ' = τθ + (1-τ)θ')
        """
        if len(self.memory) < self.batch_size:
            return

        s, a, r, ns, d = self.memory.sample(self.batch_size)
        s_t  = tf.convert_to_tensor(s,  dtype=tf.float32)
        a_t  = tf.convert_to_tensor(a,  dtype=tf.float32)
        ns_t = tf.convert_to_tensor(ns, dtype=tf.float32)

        # ── Update Critic (Bellman equation) ──────────────────────────
        with tf.GradientTape() as tape:
            next_a      = self.target_actor(ns_t,  training=False)
            target_q    = self.target_critic([ns_t, next_a], training=False)
            y           = r[:, None] + self.gamma * target_q * (1 - d[:, None])
            q           = self.critic([s_t, a_t], training=True)
            critic_loss = tf.reduce_mean(tf.square(y - q))
        grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_opt.apply_gradients(zip(grads, self.critic.trainable_variables))

        # ── Update Actor (policy gradient) ────────────────────────────
        with tf.GradientTape() as tape:
            actions    = self.actor(s_t, training=True)
            actor_loss = -tf.reduce_mean(self.critic([s_t, actions], training=False))
        grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_opt.apply_gradients(zip(grads, self.actor.trainable_variables))

        # ── Soft Update Target Networks ────────────────────────────────
        # θ' = τ*θ + (1-τ)*θ'
        self._soft_update(self.target_actor,  self.actor)
        self._soft_update(self.target_critic, self.critic)

    def _soft_update(self, target, source):
        for tw, sw in zip(target.trainable_variables, source.trainable_variables):
            tw.assign(self.tau * sw + (1 - self.tau) * tw)

    def save(self, path):
        self.actor.save_weights(path + '_actor.weights.h5')
        self.critic.save_weights(path + '_critic.weights.h5')
        print(f"Weights saved → {path}")

    def load(self, path):
        self.actor.load_weights(path + '_actor.weights.h5')
        self.critic.load_weights(path + '_critic.weights.h5')
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())
        print(f"Weights loaded ← {path}")

# ─── Training Loop ───────────────────────────────────────────────────────────

def train(df_model, total_steps=50000):

    env   = HVACEnv(df=df_model, feature_cols=FEATURE_COLS)
    agent = DDPGAgent(
        state_dim    = env.state_dim,
        action_dim   = env.action_dim,
        action_range = env.action_range
    )

    print(f"\n{'='*60}")
    print(f"DDPG Training — {total_steps} steps")
    print(f"State dim:      {env.state_dim} features")
    print(f"Action dim:     {env.action_dim} (delta SP_Return)")
    print(f"T_Return target:{env.t_target}°C")
    print(f"Safety limit:   {env.SAFETY_THRESHOLD}°C")
    print(f"SP_Return range:{env.sp_min} - {env.sp_max}°C")
    print(f"Action range:   ±{env.action_range}°C per step")
    print(f"PUE weight:     {env.pue_weight}")
    print(f"Thermal penalty:{env.thermal_penalty_weight}x violation")
    print(f"{'='*60}\n")

    reward_history  = []
    pue_history     = []
    sp_history      = []
    temp_history    = []
    episode_rewards = []
    episode_pues    = []
    violation_count = 0

    state          = env.reset()
    episode        = 0
    episode_reward = 0

    pbar = tqdm(total=total_steps, desc='Training', unit='step')

    for step in range(total_steps):
        if step < agent.warmup:
            action = np.array([np.random.uniform(-env.action_range, env.action_range)])
        else:
            action = agent.act(state)

        next_state, reward, done, info = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        agent.learn()

        reward_history.append(reward)
        pue_history.append(info['PUE'])
        sp_history.append(info['SP_Return'])
        temp_history.append(info['actual_T_return'])

        if info['actual_T_return'] > env.SAFETY_THRESHOLD:
            violation_count += 1

        state          = next_state
        episode_reward += reward

        pbar.set_postfix({
            'ep':     episode,
            'reward': f'{episode_reward:.1f}',
            'PUE':    f'{info["PUE"]:.3f}',
            'SP':     f'{info["SP_Return"]:.2f}',
            'T_ret':  f'{info["actual_T_return"]:.2f}'
        })
        pbar.update(1)

        if done:
            episode += 1
            episode_rewards.append(episode_reward)
            episode_pues.append(env.average_pue())

            if episode % 10 == 0:
                print(f"\n  Ep {episode:4d} | Step {step:6d} | "
                      f"Reward {episode_reward:8.2f} | "
                      f"Avg PUE {env.average_pue():.4f} | "
                      f"Avg SP {np.mean(sp_history[-200:]):.2f}°C")

            state          = env.reset()
            episode_reward = 0
            agent.noise.reset()

    pbar.close()
    agent.save('ddpg_hvac')

    _plot_training(episode_rewards, episode_pues,
                   pue_history, sp_history, temp_history, env)

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Final Avg PUE   (last 500 steps): {np.mean(pue_history[-500:]):.4f}")
    print(f"Final Avg SP    (last 500 steps): {np.mean(sp_history[-500:]):.4f}°C")
    print(f"Final Avg Temp  (last 500 steps): {np.mean(temp_history[-500:]):.4f}°C")
    print(f"Target T_Return: {env.t_target}°C")
    print(f"Safety violations (T>{env.SAFETY_THRESHOLD}°C): {violation_count} steps")
    print(f"{'='*60}")

    return agent, pue_history

# ─── Plot Training ───────────────────────────────────────────────────────────

def _plot_training(episode_rewards, episode_pues,
                   pue_history, sp_history, temp_history, env):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle('DDPG Training Results — CNN-LSTM + DDPG HVAC Control',
                 fontsize=13, fontweight='bold')

    def smooth(x, w=20):
        if len(x) < w:
            return np.array(x), np.arange(len(x))
        smoothed = np.convolve(x, np.ones(w)/w, mode='valid')
        return smoothed, np.arange(len(smoothed))

    episodes = np.arange(len(episode_rewards))
    steps    = np.arange(len(pue_history))

    # 1. Episode Reward
    axes[0,0].plot(episodes, episode_rewards, alpha=0.3, color='blue',
                   label='Raw episode reward')
    if len(episode_rewards) > 5:
        sm, idx = smooth(episode_rewards, 5)
        axes[0,0].plot(idx, sm, color='blue', linewidth=2,
                       label='Smoothed reward (window=5)')
    axes[0,0].axhline(y=max(episode_rewards), color='green', linestyle=':',
                      alpha=0.7, label=f'Best: {max(episode_rewards):.1f}')
    axes[0,0].set_title('Episode Reward (higher = better)')
    axes[0,0].set_xlabel('Episode')
    axes[0,0].set_ylabel('Total Reward')
    axes[0,0].legend(fontsize=8)
    axes[0,0].grid(True, alpha=0.3)

    # 2. Episode PUE
    axes[0,1].plot(episodes, episode_pues, alpha=0.3, color='green',
                   label='Raw episode PUE')
    if len(episode_pues) > 5:
        sm, idx = smooth(episode_pues, 5)
        axes[0,1].plot(idx, sm, color='green', linewidth=2,
                       label='Smoothed PUE (window=5)')
    axes[0,1].axhline(y=1.0, color='red', linestyle='--', linewidth=1.5,
                      label='Ideal PUE = 1.0')
    axes[0,1].axhline(y=np.mean(episode_pues), color='orange', linestyle='-.',
                      linewidth=1.5,
                      label=f'Mean PUE = {np.mean(episode_pues):.3f}')
    axes[0,1].set_title('Episode Average PUE (lower = better)')
    axes[0,1].set_xlabel('Episode')
    axes[0,1].set_ylabel('PUE')
    axes[0,1].legend(fontsize=8)
    axes[0,1].grid(True, alpha=0.3)

    # 3. SP_Return
    sm_sp, _ = smooth(sp_history, 50)
    axes[1,0].plot(steps, sp_history, alpha=0.2, color='orange',
                   linewidth=0.5, label='Raw SP_Return (each step)')
    axes[1,0].plot(np.arange(len(sm_sp)), sm_sp, color='darkorange',
                   linewidth=2, label='Smoothed SP_Return (window=50)')
    axes[1,0].axhline(y=env.t_target, color='red', linestyle='--',
                      linewidth=1.5, label=f'T_Return target={env.t_target}°C')
    axes[1,0].axhline(y=env.sp_min, color='gray', linestyle=':',
                      linewidth=1.2, label=f'SP min={env.sp_min}°C')
    axes[1,0].axhline(y=env.sp_max, color='black', linestyle=':',
                      linewidth=1.2, label=f'SP max={env.sp_max}°C')
    axes[1,0].axhline(y=np.mean(sp_history), color='blue', linestyle='-.',
                      linewidth=1.5,
                      label=f'Mean SP={np.mean(sp_history):.2f}°C')
    axes[1,0].set_title('SP_Return (Cooling Setpoint) Over Training')
    axes[1,0].set_xlabel('Step')
    axes[1,0].set_ylabel('Setpoint (°C)')
    axes[1,0].legend(fontsize=8)
    axes[1,0].grid(True, alpha=0.3)

    # 4. T_Return
    sm_t, _ = smooth(temp_history, 50)
    axes[1,1].plot(steps, temp_history, alpha=0.2, color='purple',
                   linewidth=0.5, label='Actual T_Return (each step)')
    axes[1,1].plot(np.arange(len(sm_t)), sm_t, color='purple',
                   linewidth=2, label='Smoothed T_Return (window=50)')
    axes[1,1].axhline(y=env.t_target, color='red', linestyle='--',
                      linewidth=1.5, label=f'Target={env.t_target}°C')
    axes[1,1].axhline(y=env.SAFETY_THRESHOLD, color='darkred', linestyle='-.',
                      linewidth=1.5,
                      label=f'Safety limit={env.SAFETY_THRESHOLD}°C')
    axes[1,1].axhline(y=np.mean(temp_history), color='blue', linestyle='-.',
                      linewidth=1.5,
                      label=f'Mean={np.mean(temp_history):.2f}°C')
    axes[1,1].fill_between(np.arange(len(sm_t)), sm_t, env.t_target,
                            where=(np.array(sm_t) > env.t_target),
                            alpha=0.15, color='red', label='Above target (too warm)')
    axes[1,1].fill_between(np.arange(len(sm_t)), sm_t, env.t_target,
                            where=(np.array(sm_t) <= env.t_target),
                            alpha=0.15, color='blue', label='Below target (too cool)')
    axes[1,1].set_title('Actual T_Return Over Training')
    axes[1,1].set_xlabel('Step')
    axes[1,1].set_ylabel('Temperature (°C)')
    axes[1,1].legend(fontsize=8)
    axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Plot saved → training_results.png")

# ─── Test Loop ───────────────────────────────────────────────────────────────

def test(df_model, num_episodes=5):
    import matplotlib.pyplot as plt

    env   = HVACEnv(df=df_model, feature_cols=FEATURE_COLS)
    agent = DDPGAgent(
        state_dim    = env.state_dim,
        action_dim   = env.action_dim,
        action_range = env.action_range
    )
    agent.load('ddpg_hvac')

    print(f"\nTesting {num_episodes} episodes (no exploration noise)...")

    all_pues, all_errors, all_rewards, all_violations = [], [], [], []
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle('DDPG Test Results — CNN-LSTM + DDPG HVAC Control',
                 fontsize=13, fontweight='bold')
    colors = ['blue', 'green', 'red', 'orange', 'purple']

    for ep in range(num_episodes):
        state        = env.reset()
        total_reward = 0
        pues, temps, sps, errors = [], [], [], []
        violations   = 0

        for _ in range(env.episode_len):
            action = agent.act(state, add_noise=False)
            state, reward, done, info = env.step(action)
            total_reward += reward
            pues.append(info['PUE'])
            temps.append(info['actual_T_return'])
            sps.append(info['SP_Return'])
            errors.append(info['comfort_error'])
            if info['actual_T_return'] > env.SAFETY_THRESHOLD:
                violations += 1
            if done:
                break
        
        # DUE calculation for this episode
        ep_predictions = [all_predictions[env.current_idx - len(temps) + i]
                          for i in range(len(temps))]
        due = calculate_due_validation(temps, ep_predictions)

        avg_pue   = np.mean(pues)
        avg_temp  = np.mean(temps)
        avg_error = np.mean(errors)
        all_pues.append(avg_pue)
        all_errors.append(avg_error)
        all_rewards.append(total_reward)
        all_violations.append(violations)

        c     = colors[ep % len(colors)]
        label = f'Ep {ep+1} (PUE={avg_pue:.3f})'
        steps = np.arange(len(temps))

        axes[0,0].plot(steps, temps,  color=c, alpha=0.8, linewidth=1.2, label=label)
        axes[0,1].plot(steps, pues,   color=c, alpha=0.8, linewidth=1.2, label=label)
        axes[1,0].plot(steps, sps,    color=c, alpha=0.8, linewidth=1.2, label=label)
        axes[1,1].plot(steps, errors, color=c, alpha=0.8, linewidth=1.2, label=label)

        print(f"  Ep {ep+1:2d} | Reward: {total_reward:8.2f} | "
            f"Avg PUE: {avg_pue:.4f} | "
            f"Avg T_Return: {avg_temp:.2f}°C | "
            f"Comfort Error: {avg_error:.4f}°C | "
            f"Safety violations: {violations} | "
            f"DUE: {due:.4f}")

    # Finalize plots
    axes[0,0].axhline(y=env.t_target, color='black', linestyle='--',
                      linewidth=2, label=f'Target={env.t_target}°C')
    axes[0,0].axhline(y=env.SAFETY_THRESHOLD, color='darkred', linestyle='-.',
                      linewidth=1.5, label=f'Safety limit={env.SAFETY_THRESHOLD}°C')
    axes[0,0].set_title('Actual T_Return per Episode')
    axes[0,0].set_xlabel('Step')
    axes[0,0].set_ylabel('Temperature (°C)')
    axes[0,0].legend(fontsize=8)
    axes[0,0].grid(True, alpha=0.3)

    axes[0,1].axhline(y=1.0, color='black', linestyle='--',
                      linewidth=2, label='Ideal PUE=1.0')
    axes[0,1].axhline(y=np.mean(all_pues), color='red', linestyle='-.',
                      linewidth=1.5, label=f'Mean PUE={np.mean(all_pues):.3f}')
    axes[0,1].set_title('PUE per Episode (lower = better)')
    axes[0,1].set_xlabel('Step')
    axes[0,1].set_ylabel('PUE')
    axes[0,1].legend(fontsize=8)
    axes[0,1].grid(True, alpha=0.3)

    axes[1,0].axhline(y=env.sp_min, color='gray', linestyle=':',
                      linewidth=1.5, label=f'SP min={env.sp_min}°C')
    axes[1,0].axhline(y=env.sp_max, color='black', linestyle=':',
                      linewidth=1.5, label=f'SP max={env.sp_max}°C')
    axes[1,0].axhline(y=env.t_target, color='red', linestyle='--',
                      linewidth=1.5, label=f'T_Return target={env.t_target}°C')
    axes[1,0].set_title('SP_Return (Setpoint) Decided by DDPG')
    axes[1,0].set_xlabel('Step')
    axes[1,0].set_ylabel('Setpoint (°C)')
    axes[1,0].legend(fontsize=8)
    axes[1,0].grid(True, alpha=0.3)

    axes[1,1].axhline(y=0, color='black', linestyle='--',
                      linewidth=2, label='Perfect comfort (error=0)')
    axes[1,1].axhline(y=np.mean(all_errors), color='red', linestyle='-.',
                      linewidth=1.5,
                      label=f'Mean error={np.mean(all_errors):.3f}°C')
    axes[1,1].set_title('Comfort Error = |T_Return - Target|')
    axes[1,1].set_xlabel('Step')
    axes[1,1].set_ylabel('Error (°C)')
    axes[1,1].legend(fontsize=8)
    axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('test_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Plot saved → test_results.png")

    print(f"\n{'='*55}")
    print(f"Test Summary ({num_episodes} episodes)")
    print(f"{'='*55}")
    print(f"  Mean PUE:              {np.mean(all_pues):.4f}")
    print(f"  Mean Comfort Error:    {np.mean(all_errors):.4f}°C")
    print(f"  Mean Reward:           {np.mean(all_rewards):.2f}")
    print(f"  Total Safety Violations (T>{env.SAFETY_THRESHOLD}°C): {sum(all_violations)}")
    print(f"  Target T_Return:       {env.t_target}°C")
    print(f"{'='*55}")

# ─── DUE Metric Calculation ────────────────────────────────────────────────
def calculate_due_validation(y_true, y_pred):
    """
    De-Underestimation (DUE) metric from Li et al. (2019).
    Penalizes cases where model underestimates temperature.
    Underestimation is dangerous — agent thinks it's safe when it's not.
    """
    y_true      = np.array(y_true)
    y_pred      = np.array(y_pred)
    differences = y_true - y_pred        # actual - predicted
    penalties   = np.maximum(differences, 0)  # only count underestimates
    return float(np.sum(penalties))
# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HVAC CNN-LSTM + DDPG Training/Testing')
    parser.add_argument('--train',    action='store_true', help='Train DDPG agent')
    parser.add_argument('--test',     action='store_true', help='Test DDPG agent')
    parser.add_argument('--steps',    type=int, default=50000, help='Training steps')
    parser.add_argument('--episodes', type=int, default=5,     help='Test episodes')
    args = parser.parse_args()

    if args.train:
        agent, pues = train(df_model, total_steps=args.steps)
    if args.test:
        test(df_model, num_episodes=args.episodes)
    if not args.train and not args.test:
        agent, pues = train(df_model, total_steps=args.steps)
        test(df_model, num_episodes=args.episodes)
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

df_model = df.copy()
df_model['T_Supply_diff']       = df_model['T_Supply'].diff()
df_model['T_Return_diff']       = df_model['T_Return'].diff()
df_model['SP_Return_diff']      = df_model['SP_Return'].diff()
df_model['Power_diff']          = df_model['Power'].diff()
df_model.dropna(inplace=True)

print(f"Dataset loaded: {df_model.shape}")
print(f"Date range: {df_model.index[0]} to {df_model.index[-1]}")
print(f"T_Return mean: {df_model['T_Return'].mean():.2f}°C (used as comfort baseline)")
print(f"SP_Return distribution:\n{df_model['SP_Return'].value_counts().head(5)}")

# ─── Pre-compute CNN-LSTM Predictions ────────────────────────────────────────

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
    Data-driven HVAC environment.

    State:  [predicted_T_return, T_outdoor, SP_Return, Power, comfort_error]
    Action: delta SP_Return (how much to adjust cooling setpoint)
    Reward: -comfort_error - pue_weight * (PUE - 1.0)

    CNN-LSTM predicts T_Return 15 mins ahead given current sensor window.
    DDPG adjusts SP_Return to maintain T_Return near target while minimizing PUE.
    PUE = (IT_power + cooling_power) / IT_power  [assumed IT power = 10 kW]
    """

    def __init__(self, df, feature_cols,
                 t_target=20.0,    # T_Return mean from dataset
                 sp_min=19.0,      # SP_Return realistic min (data shows 18.5-23.5)
                 sp_max=23.5,      # SP_Return realistic max
                 action_range=0.5, # max setpoint change per step (°C)
                 pue_weight=0.3,   # weight of PUE in reward
                 episode_len=200): # steps per episode

        self.df           = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.t_target     = t_target
        self.sp_min       = sp_min
        self.sp_max       = sp_max
        self.action_range = action_range
        self.pue_weight   = pue_weight
        self.episode_len  = episode_len

        # State: [predicted_T_return, T_outdoor, SP_Return, Power, comfort_error]
        self.state_dim  = 5
        self.action_dim = 1

        self.current_idx  = None
        self.current_sp   = None
        self.episode_pues = []
        self.step_count   = 0

    def _predict_temperature(self, idx):
        """Use pre-computed CNN-LSTM prediction for T_Return."""
        return float(all_predictions[idx])

    def _calculate_pue(self, idx):
        """
        PUE = Total Facility Power / IT Equipment Power
        Since dataset only has fan power, assume IT load = 10 kW (typical small DC).
        PUE = (10 + fan_power) / 10
        Lower fan_power → lower PUE → better energy efficiency.
        """
        ASSUMED_IT_POWER = 10.0  # kW
        cooling_power    = float(self.df['Power'].iloc[idx])
        pue              = (ASSUMED_IT_POWER + cooling_power) / ASSUMED_IT_POWER
        return float(np.clip(pue, 1.0, 3.0))

    def _build_state(self, predicted_t, idx):
        """Build normalized state vector for DDPG."""
        t_outdoor     = float(self.df['T_Outdoor'].iloc[idx])
        power         = float(self.df['Power'].iloc[idx])
        comfort_error = float(predicted_t - self.t_target)
        return np.array([
            predicted_t,
            t_outdoor,
            self.current_sp,
            power,
            comfort_error
        ], dtype=np.float32)

    def reset(self):
        """Start a new episode at a random dataset position."""
        max_start = len(self.df) - self.episode_len - 1
        self.current_idx  = random.randint(n_input, max_start)
        self.current_sp   = float(self.df['SP_Return'].iloc[self.current_idx])
        self.episode_pues = []
        self.step_count   = 0

        predicted_t = self._predict_temperature(self.current_idx)
        return self._build_state(predicted_t, self.current_idx)

    def step(self, action):
        """
        Apply action (delta SP_Return), advance one timestep, return transition.
        """
        # 1. Adjust setpoint
        delta_sp        = float(np.clip(np.squeeze(action), -self.action_range, self.action_range))
        self.current_sp = float(np.clip(self.current_sp + delta_sp, self.sp_min, self.sp_max))

        # 2. Advance timestep
        self.current_idx += 1
        self.step_count  += 1
        done = (self.step_count >= self.episode_len) or \
               (self.current_idx >= len(self.df) - 1)

        # 3. CNN-LSTM predicted T_Return (for state)
        predicted_t = self._predict_temperature(self.current_idx)

        # 4. Actual T_Return from dataset (for reward — ground truth comfort)
        actual_t = float(self.df['T_Return'].iloc[self.current_idx])

        # 5. PUE calculation
        pue = self._calculate_pue(self.current_idx)
        self.episode_pues.append(pue)

        # 6. Reward function:
        #    - comfort_error: how far actual room temp is from target
        #    - pue penalty: energy inefficiency cost
        #    Goal: DDPG learns to keep T_Return near target with minimum fan power
        comfort_error = abs(actual_t - self.t_target)
        reward        = -comfort_error - self.pue_weight * (pue - 1.0)

        # 7. Next state uses predicted temperature
        next_state = self._build_state(predicted_t, self.current_idx)

        info = {
            'predicted_T_return': predicted_t,
            'actual_T_return':    actual_t,
            'SP_Return':          self.current_sp,
            'PUE':                pue,
            'comfort_error':      comfort_error,
            'reward':             reward
        }

        return next_state, reward, done, info

    def average_pue(self):
        if len(self.episode_pues) == 0:
            return 1.0
        return float(np.mean(self.episode_pues))

# ─── Replay Buffer ───────────────────────────────────────────────────────────

class ReplayBuffer:
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
    """Temporally correlated noise for continuous action exploration.
    Better than Gaussian noise for physical control systems (HVAC).
    Reference: Uhlenbeck & Ornstein (1930), used in DDPG paper."""
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
    Actor: state → action (delta SP_Return)
    tanh output ensures action stays in [-1, 1], scaled by action_range.
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
    Critic: (state, action) → Q-value
    Estimates expected cumulative reward for taking action in state.
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
    Deep Deterministic Policy Gradient agent.
    Uses actor-critic with experience replay and soft target updates.
    Reference: Lillicrap et al. (2015) arXiv:1509.02971
    """
    def __init__(self, state_dim, action_dim, action_range,
                 gamma=0.99,   # discount factor
                 tau=0.005,    # soft update rate
                 lr_actor=1e-4,
                 lr_critic=1e-3,
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

        # Target networks (soft-updated copies)
        self.target_actor  = build_actor(state_dim, action_dim)
        self.target_critic = build_critic(state_dim, action_dim)
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

        self.actor_opt  = Adam(learning_rate=lr_actor)
        self.critic_opt = Adam(learning_rate=lr_critic)

        self.memory = ReplayBuffer(buffer_size)
        self.noise  = OUNoise(action_dim)

    def act(self, state, add_noise=True):
        """Select action with optional exploration noise."""
        state_t = tf.convert_to_tensor([state], dtype=tf.float32)
        action  = self.actor(state_t, training=False).numpy()[0]
        if add_noise:
            action += self.noise.sample()
        # Scale from tanh range [-1,1] to action range
        action = action * self.action_range
        return np.clip(action, -self.action_range, self.action_range)

    def remember(self, s, a, r, ns, done):
        self.memory.add(s, a, r, ns, done)

    def learn(self):
        """Sample batch and update actor and critic networks."""
        if len(self.memory) < self.batch_size:
            return

        s, a, r, ns, d = self.memory.sample(self.batch_size)

        s_t  = tf.convert_to_tensor(s,  dtype=tf.float32)
        a_t  = tf.convert_to_tensor(a,  dtype=tf.float32)
        ns_t = tf.convert_to_tensor(ns, dtype=tf.float32)

        # ── Update Critic ──────────────────────────────────────────────
        with tf.GradientTape() as tape:
            next_a   = self.target_actor(ns_t,  training=False)
            target_q = self.target_critic([ns_t, next_a], training=False)
            y        = r[:, None] + self.gamma * target_q * (1 - d[:, None])
            a_t = tf.convert_to_tensor(a, dtype=tf.float32)
            q   = self.critic([s_t, a_t], training=True)
            critic_loss = tf.reduce_mean(tf.square(y - q))
        grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_opt.apply_gradients(zip(grads, self.critic.trainable_variables))

        # ── Update Actor ───────────────────────────────────────────────
        with tf.GradientTape() as tape:
            actions    = self.actor(s_t, training=True)
            actor_loss = -tf.reduce_mean(self.critic([s_t, actions], training=False))
        grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_opt.apply_gradients(zip(grads, self.actor.trainable_variables))

        # ── Soft Update Target Networks ────────────────────────────────
        self._soft_update(self.target_actor,  self.actor)
        self._soft_update(self.target_critic, self.critic)

    def _soft_update(self, target, source):
        """θ_target = τ*θ_source + (1-τ)*θ_target"""
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

    env = HVACEnv(df=df_model, feature_cols=FEATURE_COLS)
    agent = DDPGAgent(
        state_dim    = env.state_dim,
        action_dim   = env.action_dim,
        action_range = env.action_range
    )

    print(f"\n{'='*60}")
    print(f"DDPG Training — {total_steps} steps")
    print(f"State dim: {env.state_dim} | Action dim: {env.action_dim}")
    print(f"T_Return target: {env.t_target}°C")
    print(f"SP_Return range: {env.sp_min} - {env.sp_max}°C")
    print(f"Action range: ±{env.action_range}°C per step")
    print(f"{'='*60}\n")

    # Tracking
    reward_history  = []
    pue_history     = []
    sp_history      = []
    temp_history    = []
    episode_rewards = []
    episode_pues    = []

    state          = env.reset()
    episode        = 0
    episode_reward = 0

    pbar = tqdm(total=total_steps, desc='Training', unit='step')

    for step in range(total_steps):
        # Warmup: random actions to fill replay buffer
        if step < agent.warmup:
            action = np.array([np.random.uniform(-env.action_range, env.action_range)])
        else:
            action = agent.act(state)

        next_state, reward, done, info = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        agent.learn()

        # Track metrics
        reward_history.append(reward)
        pue_history.append(info['PUE'])
        sp_history.append(info['SP_Return'])
        temp_history.append(info['actual_T_return'])

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

    # ── Plot Results ──────────────────────────────────────────────────
    _plot_training(episode_rewards, episode_pues,
                   pue_history, sp_history, temp_history, env)

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Final Avg PUE  (last 500 steps): {np.mean(pue_history[-500:]):.4f}")
    print(f"Final Avg SP   (last 500 steps): {np.mean(sp_history[-500:]):.4f}°C")
    print(f"Final Avg Temp (last 500 steps): {np.mean(temp_history[-500:]):.4f}°C")
    print(f"Target T_Return: {env.t_target}°C")
    print(f"{'='*60}")

    return agent, pue_history

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
        idx      = np.arange(len(smoothed))
        return smoothed, idx

    episodes = np.arange(len(episode_rewards))
    steps    = np.arange(len(pue_history))

    # ── 1. Episode Reward ─────────────────────────────────────────────
    axes[0,0].plot(episodes, episode_rewards,
                   alpha=0.3, color='blue', label='Raw episode reward')
    if len(episode_rewards) > 5:
        sm, idx = smooth(episode_rewards, 5)
        axes[0,0].plot(idx, sm, color='blue', linewidth=2,
                       label='Smoothed reward (window=5)')
    axes[0,0].axhline(y=max(episode_rewards), color='green', linestyle=':',
                      alpha=0.7, label=f'Best reward: {max(episode_rewards):.1f}')
    axes[0,0].set_title('Episode Reward (higher = better)')
    axes[0,0].set_xlabel('Episode')
    axes[0,0].set_ylabel('Total Reward')
    axes[0,0].legend(fontsize=8)
    axes[0,0].grid(True, alpha=0.3)

    # ── 2. Episode PUE ────────────────────────────────────────────────
    axes[0,1].plot(episodes, episode_pues,
                   alpha=0.3, color='green', label='Raw episode PUE')
    if len(episode_pues) > 5:
        sm, idx = smooth(episode_pues, 5)
        axes[0,1].plot(idx, sm, color='green', linewidth=2,
                       label='Smoothed PUE (window=5)')
    axes[0,1].axhline(y=1.0,  color='red',    linestyle='--', linewidth=1.5,
                      label='Ideal PUE = 1.0 (no cooling overhead)')
    axes[0,1].axhline(y=np.mean(episode_pues), color='orange', linestyle='-.',
                      linewidth=1.5,
                      label=f'Mean PUE = {np.mean(episode_pues):.3f}')
    axes[0,1].set_title('Episode Average PUE (lower = better)')
    axes[0,1].set_xlabel('Episode')
    axes[0,1].set_ylabel('PUE')
    axes[0,1].legend(fontsize=8)
    axes[0,1].grid(True, alpha=0.3)

    # ── 3. SP_Return over steps ───────────────────────────────────────
    sm_sp, _ = smooth(sp_history, 50)
    axes[1,0].plot(steps, sp_history,
                   alpha=0.2, color='orange', linewidth=0.5,
                   label='Raw SP_Return (each step)')
    axes[1,0].plot(np.arange(len(sm_sp)), sm_sp,
                   color='darkorange', linewidth=2,
                   label='Smoothed SP_Return (window=50)')
    axes[1,0].axhline(y=env.t_target, color='red',    linestyle='--', linewidth=1.5,
                      label=f'T_Return target = {env.t_target}°C')
    axes[1,0].axhline(y=env.sp_min,   color='gray',   linestyle=':',  linewidth=1.2,
                      label=f'SP min = {env.sp_min}°C')
    axes[1,0].axhline(y=env.sp_max,   color='black',  linestyle=':',  linewidth=1.2,
                      label=f'SP max = {env.sp_max}°C')
    axes[1,0].axhline(y=np.mean(sp_history), color='blue', linestyle='-.',
                      linewidth=1.5,
                      label=f'Mean SP = {np.mean(sp_history):.2f}°C')
    axes[1,0].set_title('SP_Return (Cooling Setpoint) Over Training')
    axes[1,0].set_xlabel('Step')
    axes[1,0].set_ylabel('Setpoint (°C)')
    axes[1,0].legend(fontsize=8)
    axes[1,0].grid(True, alpha=0.3)

    # ── 4. Actual T_Return over steps ─────────────────────────────────
    sm_t, _ = smooth(temp_history, 50)
    axes[1,1].plot(steps, temp_history,
                   alpha=0.2, color='purple', linewidth=0.5,
                   label='Actual T_Return (each step)')
    axes[1,1].plot(np.arange(len(sm_t)), sm_t,
                   color='purple', linewidth=2,
                   label='Smoothed T_Return (window=50)')
    axes[1,1].axhline(y=env.t_target, color='red', linestyle='--', linewidth=1.5,
                      label=f'Target T_Return = {env.t_target}°C')
    axes[1,1].axhline(y=np.mean(temp_history), color='blue', linestyle='-.',
                      linewidth=1.5,
                      label=f'Mean T_Return = {np.mean(temp_history):.2f}°C')
    axes[1,1].fill_between(
        np.arange(len(sm_t)), sm_t,
        env.t_target,
        where=(np.array(sm_t) > env.t_target),
        alpha=0.15, color='red',   label='Above target (too warm)'
    )
    axes[1,1].fill_between(
        np.arange(len(sm_t)), sm_t,
        env.t_target,
        where=(np.array(sm_t) <= env.t_target),
        alpha=0.15, color='blue',  label='Below target (too cool)'
    )
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

    env = HVACEnv(df=df_model, feature_cols=FEATURE_COLS)
    agent = DDPGAgent(
        state_dim    = env.state_dim,
        action_dim   = env.action_dim,
        action_range = env.action_range
    )
    agent.load('ddpg_hvac')

    print(f"\nTesting {num_episodes} episodes (no exploration noise)...")

    all_pues, all_errors, all_rewards = [], [], []

    # For plotting
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle('DDPG Test Results — CNN-LSTM + DDPG HVAC Control',
                 fontsize=13, fontweight='bold')
    colors = ['blue', 'green', 'red', 'orange', 'purple']

    for ep in range(num_episodes):
        state        = env.reset()
        total_reward = 0
        pues, temps, sps, errors = [], [], [], []

        for _ in range(env.episode_len):
            action = agent.act(state, add_noise=False)
            state, reward, done, info = env.step(action)
            total_reward   += reward
            pues.append(info['PUE'])
            temps.append(info['actual_T_return'])
            sps.append(info['SP_Return'])
            errors.append(info['comfort_error'])
            if done:
                break

        avg_pue   = np.mean(pues)
        avg_temp  = np.mean(temps)
        avg_error = np.mean(errors)
        all_pues.append(avg_pue)
        all_errors.append(avg_error)
        all_rewards.append(total_reward)

        steps = np.arange(len(temps))
        c     = colors[ep % len(colors)]
        label = f'Ep {ep+1} (PUE={avg_pue:.3f})'

        # Plot each episode
        axes[0,0].plot(steps, temps, color=c, alpha=0.8,
                       linewidth=1.2, label=label)
        axes[0,1].plot(steps, pues,  color=c, alpha=0.8,
                       linewidth=1.2, label=label)
        axes[1,0].plot(steps, sps,   color=c, alpha=0.8,
                       linewidth=1.2, label=label)
        axes[1,1].plot(steps, errors, color=c, alpha=0.8,
                       linewidth=1.2, label=label)

        print(f"  Ep {ep+1:2d} | Reward: {total_reward:8.2f} | "
              f"Avg PUE: {avg_pue:.4f} | "
              f"Avg T_Return: {avg_temp:.2f}°C | "
              f"Comfort Error: {avg_error:.4f}°C")

    # ── Finalize plots ────────────────────────────────────────────────

    # T_Return
    axes[0,0].axhline(y=env.t_target, color='black', linestyle='--',
                      linewidth=2, label=f'Target T_Return = {env.t_target}°C')
    axes[0,0].set_title('Actual T_Return per Episode')
    axes[0,0].set_xlabel('Step')
    axes[0,0].set_ylabel('Temperature (°C)')
    axes[0,0].legend(fontsize=8)
    axes[0,0].grid(True, alpha=0.3)

    # PUE
    axes[0,1].axhline(y=1.0, color='black', linestyle='--',
                      linewidth=2, label='Ideal PUE = 1.0')
    axes[0,1].axhline(y=np.mean(all_pues), color='red', linestyle='-.',
                      linewidth=1.5,
                      label=f'Mean PUE = {np.mean(all_pues):.3f}')
    axes[0,1].set_title('PUE per Episode (lower = better)')
    axes[0,1].set_xlabel('Step')
    axes[0,1].set_ylabel('PUE')
    axes[0,1].legend(fontsize=8)
    axes[0,1].grid(True, alpha=0.3)

    # SP_Return
    axes[1,0].axhline(y=env.sp_min, color='gray', linestyle=':',
                      linewidth=1.5, label=f'SP min = {env.sp_min}°C')
    axes[1,0].axhline(y=env.sp_max, color='black', linestyle=':',
                      linewidth=1.5, label=f'SP max = {env.sp_max}°C')
    axes[1,0].axhline(y=env.t_target, color='red', linestyle='--',
                      linewidth=1.5, label=f'T_Return target = {env.t_target}°C')
    axes[1,0].set_title('SP_Return (Setpoint) Decided by DDPG')
    axes[1,0].set_xlabel('Step')
    axes[1,0].set_ylabel('Setpoint (°C)')
    axes[1,0].legend(fontsize=8)
    axes[1,0].grid(True, alpha=0.3)

    # Comfort Error
    axes[1,1].axhline(y=0, color='black', linestyle='--',
                      linewidth=2, label='Perfect comfort (error=0)')
    axes[1,1].axhline(y=np.mean(all_errors), color='red', linestyle='-.',
                      linewidth=1.5,
                      label=f'Mean error = {np.mean(all_errors):.3f}°C')
    axes[1,1].set_title('Comfort Error = |T_Return - Target|')
    axes[1,1].set_xlabel('Step')
    axes[1,1].set_ylabel('Error (°C)')
    axes[1,1].legend(fontsize=8)
    axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('test_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Plot saved → test_results.png")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"Test Summary ({num_episodes} episodes)")
    print(f"{'='*50}")
    print(f"  Mean PUE:           {np.mean(all_pues):.4f}")
    print(f"  Mean Comfort Error: {np.mean(all_errors):.4f}°C")
    print(f"  Mean Reward:        {np.mean(all_rewards):.2f}")
    print(f"  Target T_Return:    {env.t_target}°C")
    print(f"{'='*50}")
# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='HVAC DDPG Training/Testing'
    )
    parser.add_argument('--train',  action='store_true', help='Train DDPG agent')
    parser.add_argument('--test',   action='store_true', help='Test DDPG agent')
    parser.add_argument('--steps',  type=int, default=50000, help='Training steps')
    args = parser.parse_args()

    if args.train:
        agent, pues = train(df_model, total_steps=args.steps)
    if args.test:
        test(df_model)
    if not args.train and not args.test:
        agent, pues = train(df_model, total_steps=args.steps)
        test(df_model)
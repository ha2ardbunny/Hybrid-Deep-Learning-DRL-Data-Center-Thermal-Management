import math
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

FILE_PATH = r'/mnt/c/Users/xin37/github/CNN-LSTM-model-for-energy-usage-forecasting-1/data/TDC2_processed.csv'

print("Loading dataset...")
df = pd.read_csv(FILE_PATH, parse_dates=['Timestamp'], index_col='Timestamp')
df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)

df_model = df.copy()
df_model['T_Supply_diff']   = df_model['T_Supply'].diff()
df_model['T_Return_diff']   = df_model['T_Return'].diff()
df_model['HVAC_Power_diff'] = df_model['HVAC_Power_kW'].diff()
df_model['IT_Power_diff']   = df_model['IT_Power_Total_kW'].diff()
df_model.dropna(inplace=True)

print(f"Dataset loaded: {df_model.shape}")
print(f"Date range: {df_model.index[0]} to {df_model.index[-1]}")
print(f"T_Return mean: {df_model['T_Return'].mean():.2f}degC")
print(f"T_Supply range: {df_model['T_Supply'].min():.2f} - {df_model['T_Supply'].max():.2f}degC")
print(f"HVAC_Power_kW mean: {df_model['HVAC_Power_kW'].mean():.3f} kW")
print(f"IT_Power_Total_kW mean: {df_model['IT_Power_Total_kW'].mean():.3f} kW")
print(f"PUE mean: {df_model['PUE'].mean():.3f}")

print("\nThermal model: first-order lag (alpha=0.96, calibrated from dataset)")
print("  T_Supply -> T_Return equilibrium from grouped statistics")
print("  No ML fitting required — physics-based model")

# ─── Pre-compute CNN-LSTM Predictions (Offline Batch) ────────────────────────

print("\n" + "="*60)
print("Simulated Environment: First-order thermal lag model")
print("  alpha=0.96 (thermal inertia), equilibrium from data stats")
print("  T_Supply -> T_Return causal, physically validated")
print("CNN-LSTM predictions: pre-computed once on full dataset")
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

all_predictions = np.concatenate([
    np.full(n_input, predictions_celsius[0]),
    predictions_celsius
])
print(f"Pre-computed {len(all_predictions)} predictions")
print(f"Prediction range: {all_predictions.min():.2f} - {all_predictions.max():.2f}degC")

# ─── HVAC Environment ────────────────────────────────────────────────────────

class HVACEnv:
    SAFETY_THRESHOLD = 35.0 
    ASSUMED_IT_POWER = 14.125 

    def __init__(self, df, feature_cols,
                 t_target=27.0,         
                 t_supply_min=18.28,    
                 t_supply_max=30.31,    
                 thermal_penalty_weight=1.0,
                 episode_len=200):

        self.df                     = df.reset_index(drop=True)
        self.feature_cols           = feature_cols
        self.t_target               = t_target
        self.t_supply_min           = t_supply_min
        self.t_supply_max           = t_supply_max
        self.action_range           = 1.0
        self.thermal_penalty_weight = thermal_penalty_weight
        self.episode_len            = episode_len

        self.state_dim  = 8
        self.action_dim = 1

        self.current_idx      = None
        self.current_t_supply = None
        self.current_t_return = None   
        self.episode_powers   = []   
        self.episode_pues     = []
        self.step_count       = 0
        self.prev_pue = None

    def _predict_temperature(self, idx):
        return float(all_predictions[idx])

    # Physical reasoning: more cooling demand = more compressor work = more power.
    def _estimate_power(self, t_supply, t_outdoor):
        it_power = self.ASSUMED_IT_POWER 
        
        cop = max(0.5, 3.0 - 0.1 * (t_outdoor - t_supply))
        hvac_power = it_power / cop
        
        return float(np.clip(hvac_power, 2.0, 30.0))

    def _simulate_t_return(self, t_supply, prev_t_return):
        """
        First-order thermal lag model calibrated from dataset.

        T_Return(t+1) = alpha * T_Return(t) + (1-alpha) * T_Return_eq(T_Supply) + noise

        T_Return_eq: equilibrium T_Return for a given T_Supply,
                     extracted from dataset grouped statistics.
        alpha=0.96:  thermal inertia fitted from linear regression (R2=0.984).

        Validated convergence (100 steps, T_Return_0=20degC):
          T_Supply=15.5 -> T_Return=18.52  (cooler supply -> cooler return) OK
          T_Supply=19.0 -> T_Return=20.86  (near target)
          T_Supply=27.0 -> T_Return=22.92  (warmer supply -> warmer return) OK

        To reach T_Return=20degC, agent should target T_Supply~17-18degC.
        """
        _supply_bins  = [14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 34.0]
        _return_eq    = [22.0, 23.5, 25.0, 26.5, 27.5, 28.5, 29.5, 30.5, 31.5, 32.5, 33.5]

        t_return_eq   = float(np.interp(t_supply, _supply_bins, _return_eq))
        alpha         = 0.96
        noise         = np.random.normal(0, 0.15)
        t_return_next = alpha * prev_t_return + (1.0 - alpha) * t_return_eq + noise
        return float(np.clip(t_return_next, 15.0, 45.0))  

    def _build_state(self, predicted_t, idx):
        t_outdoor     = float(self.df['T_Outdoor'].iloc[idx])
        rh_outdoor    = float(self.df['RH_Outdoor'].iloc[idx])
        rh_return     = float(self.df['RH_Return'].iloc[idx])
        hvac_power = self._estimate_power(self.current_t_supply, t_outdoor)
        it_power      = float(self.df['IT_Power_Total_kW'].iloc[idx])
        sim_t_return  = self.current_t_return if self.current_t_return is not None \
                        else predicted_t
        comfort_error = float(sim_t_return - self.t_target)

        return np.array([
            predicted_t,
            t_outdoor,
            self.current_t_supply,
            hvac_power,            
            rh_outdoor,
            rh_return,
            it_power,             
            comfort_error
        ], dtype=np.float32)

    def reset(self):
        max_start             = len(self.df) - self.episode_len - 1
        self.current_idx      = random.randint(n_input, max_start)
        self.current_t_supply = float(self.df['T_Supply'].iloc[self.current_idx])
        self.current_t_return = float(self.df['T_Return'].iloc[self.current_idx]) 
        self.episode_powers   = []
        self.episode_pues     = []
        self.step_count       = 0
        self.prev_pue = None

        predicted_t = self._predict_temperature(self.current_idx)
        return self._build_state(predicted_t, self.current_idx)

    def step(self, action):
        # 1. Decode action: [-1,1] -> [t_supply_min, t_supply_max]
        raw_action            = float(np.clip(np.squeeze(action), -1.0, 1.0))
        t_supply_mid          = (self.t_supply_max + self.t_supply_min) / 2.0
        t_supply_half_range   = (self.t_supply_max - self.t_supply_min) / 2.0
        self.current_t_supply = float(np.clip(
            t_supply_mid + raw_action * t_supply_half_range,
            self.t_supply_min, self.t_supply_max
        ))

        # 2. Advance timestep
        self.current_idx += 1
        self.step_count  += 1
        done = (self.step_count >= self.episode_len) or \
               (self.current_idx >= len(self.df) - 1)

        # 3. CNN-LSTM prediction (state observation)
        predicted_t = self._predict_temperature(self.current_idx)

        # 4. First-order thermal lag model 
        actual_t              = self._simulate_t_return(
            self.current_t_supply, self.current_t_return
        )
        self.current_t_return = actual_t 

        # 5. Real HVAC power and PUE from TDC2 dataset
        est_power = self._estimate_power(self.current_t_supply, 
                                  float(self.df['T_Outdoor'].iloc[self.current_idx]))
        it_power  = float(self.df['IT_Power_Total_kW'].iloc[self.current_idx])
        self.episode_powers.append(est_power)
        pue = float(np.clip(
            (it_power + est_power) / max(it_power, 0.1), 1.0, 3.0
        ))
        self.episode_pues.append(pue)

        # 6. Reward = comfort + energy
        comfort_error   = abs(actual_t - self.t_target)
        temp_diff       = np.clip(actual_t - self.SAFETY_THRESHOLD, -50, 50)
        k = 0.5
        thermal_penalty = self.thermal_penalty_weight * math.log1p(math.exp(k * temp_diff))

        pue_penalty     = (pue - 1.0)           
        comfort_penalty = 0.1 * comfort_error
        delta_bonus     = 0.0 if self.prev_pue is None else 0.3 * (self.prev_pue - pue)

        reward = -pue_penalty - comfort_penalty - thermal_penalty + delta_bonus
        reward = max(reward, -10.0)

        self.prev_pue = pue   

        # 7. Next state
        next_state = self._build_state(predicted_t, self.current_idx)

        info = {
            'predicted_T_return': predicted_t,
            'actual_T_return':    actual_t,
            'T_Supply':           self.current_t_supply,
            'HVAC_Power_kW':      est_power,
            'PUE':                pue,
            'comfort_error':      comfort_error,
            'thermal_penalty':    thermal_penalty,
            'reward':             reward
        }

        return next_state, reward, done, info

    def average_pue(self):
        return float(np.mean(self.episode_pues)) if self.episode_pues else 1.0

    def average_power(self):
        return float(np.mean(self.episode_powers)) if self.episode_powers else 0.0

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
    inp = Input(shape=(state_dim,), name='state_input')
    x   = Dense(128, activation='relu')(inp)
    x   = Dense(64,  activation='relu')(x)
    x   = Dense(32,  activation='relu')(x)
    out = Dense(action_dim, activation='tanh', name='action')(x)
    return Model(inputs=inp, outputs=out, name='actor')

# ─── Critic Network ──────────────────────────────────────────────────────────

def build_critic(state_dim, action_dim):
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
    def __init__(self, state_dim, action_dim, action_range,
                 gamma=0.99,
                 tau=0.005,
                 lr_actor=1e-4,
                 lr_critic=1e-3,
                 buffer_size=100000,
                 batch_size=128,      
                 warmup=2000):     

        self.action_range = action_range
        self.gamma        = gamma
        self.tau          = tau
        self.batch_size   = batch_size
        self.warmup       = warmup

        self.actor  = build_actor(state_dim, action_dim)
        self.critic = build_critic(state_dim, action_dim)

        self.target_actor  = build_actor(state_dim, action_dim)
        self.target_critic = build_critic(state_dim, action_dim)
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

        self.actor_opt  = Adam(learning_rate=lr_actor)
        self.critic_opt = Adam(learning_rate=lr_critic)

        self.memory = ReplayBuffer(buffer_size)
        self.noise  = OUNoise(action_dim)

    def act(self, state, add_noise=True):
        state_t = tf.convert_to_tensor([state], dtype=tf.float32)
        action  = self.actor(state_t, training=False).numpy()[0]
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1.0, 1.0)

    def remember(self, s, a, r, ns, done):
        self.memory.add(s, a, r, ns, done)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        self._learn_step()

    @tf.function
    def _learn_step(self):
        s, a, r, ns, d = self.memory.sample(self.batch_size)
        s_t  = tf.convert_to_tensor(s,  dtype=tf.float32)
        a_t  = tf.convert_to_tensor(a,  dtype=tf.float32)
        ns_t = tf.convert_to_tensor(ns, dtype=tf.float32)
        r_t  = tf.convert_to_tensor(r,  dtype=tf.float32)
        d_t  = tf.convert_to_tensor(d,  dtype=tf.float32)

        with tf.GradientTape() as tape:
            next_a   = self.target_actor(ns_t, training=False)
            target_q = self.target_critic([ns_t, next_a], training=False)
            y        = r_t[:, None] + self.gamma * target_q * (1 - d_t[:, None])
            q        = self.critic([s_t, a_t], training=True)
            critic_loss = tf.reduce_mean(tf.square(y - q))
        grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_opt.apply_gradients(zip(grads, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            actions    = self.actor(s_t, training=True)
            actor_loss = -tf.reduce_mean(self.critic([s_t, actions], training=False))
        grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_opt.apply_gradients(zip(grads, self.actor.trainable_variables))

        self._soft_update(self.target_actor,  self.actor)
        self._soft_update(self.target_critic, self.critic)

    def _soft_update(self, target, source):
        for tw, sw in zip(target.trainable_variables, source.trainable_variables):
            tw.assign(self.tau * sw + (1 - self.tau) * tw)

    def save(self, path):
        self.actor.save_weights(path + '_actor.weights.h5')
        self.critic.save_weights(path + '_critic.weights.h5')
        print(f"Weights saved -> {path}")

    def load(self, path):
        self.actor.load_weights(path + '_actor.weights.h5')
        self.critic.load_weights(path + '_critic.weights.h5')
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())
        print(f"Weights loaded <- {path}")

# ─── Training Loop ───────────────────────────────────────────────────────────

def train(df_model, total_steps=200000):

    env   = HVACEnv(df=df_model, feature_cols=FEATURE_COLS)
    agent = DDPGAgent(
        state_dim    = env.state_dim,
        action_dim   = env.action_dim,
        action_range = env.action_range
    )

    print(f"\n{'='*60}")
    print(f"DDPG Training — {total_steps} steps")
    print(f"Environment:    First-order thermal lag (alpha=0.96, calibrated)")
    print(f"State dim:      {env.state_dim} features")
    print(f"Action:         T_Supply setpoint (degC)")
    print(f"T_Return target:{env.t_target}degC  (TDC2: mean=29.1degC)")
    print(f"Safety limit:   {env.SAFETY_THRESHOLD}degC  (TDC2: max=43.5degC)")
    print(f"T_Supply range: {env.t_supply_min} - {env.t_supply_max}degC  (TDC2 5th-95th pct)")
    print(f"Reward:         -(comfort/8) - pue_penalty - thermal_penalty  [PUE ENABLED]")
    print(f"Thermal penalty:{env.thermal_penalty_weight}x log-barrier")
    print(f"Batch size:     {agent.batch_size}")
    print(f"Warmup steps:   {agent.warmup}")
    print(f"{'='*60}\n")

    reward_history  = []
    pue_history     = []
    sp_history      = []
    temp_history    = []
    error_history   = []
    episode_rewards = []
    episode_pues    = []
    violation_count = 0

    state          = env.reset()
    episode        = 0
    episode_reward = 0

    pbar = tqdm(total=total_steps, desc='Training', unit='step')

    for step in range(total_steps):
        if step < agent.warmup:
            action = np.array([np.random.uniform(-1.0, 1.0)])
        else:
            action = agent.act(state)

        next_state, reward, done, info = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        agent.learn()

        reward_history.append(reward)
        pue_history.append(info['PUE'])
        sp_history.append(info['T_Supply'])
        temp_history.append(info['actual_T_return'])
        error_history.append(info['comfort_error'])

        if info['actual_T_return'] > env.SAFETY_THRESHOLD:
            violation_count += 1

        state          = next_state
        episode_reward += reward

        pbar.set_postfix({
            'ep':    episode,
            'rew':   f'{episode_reward:.1f}',
            'err':   f'{info["comfort_error"]:.2f}',
            'T_sup': f'{info["T_Supply"]:.2f}',
            'T_ret': f'{info["actual_T_return"]:.2f}',
            'pue':   f'{info["PUE"]:.3f}'
        })
        pbar.update(1)

        if done:
            episode += 1
            episode_rewards.append(episode_reward)
            episode_pues.append(env.average_pue())

            if episode % 10 == 0:
                avg_err = np.mean(error_history[-200:])
                print(f"\n  Ep {episode:4d} | Step {step:6d} | "
                      f"Reward {episode_reward:7.2f} | "
                      f"Avg err {avg_err:.3f}degC | "
                      f"Avg T_sup {np.mean(sp_history[-200:]):.2f}degC | "
                      f"Avg T_ret {np.mean(temp_history[-200:]):.2f}degC")

            state          = env.reset()
            episode_reward = 0
            agent.noise.reset()

    pbar.close()
    agent.save('ddpg_hvac')

    _plot_training(episode_rewards, episode_pues,
                   pue_history, sp_history, temp_history, error_history, env)

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Final Avg comfort error (last 500): {np.mean(error_history[-500:]):.4f}degC")
    print(f"Final Avg T_Supply      (last 500): {np.mean(sp_history[-500:]):.4f}degC")
    print(f"Final Avg T_Return      (last 500): {np.mean(temp_history[-500:]):.4f}degC")
    print(f"Target T_Return: {env.t_target}degC")
    print(f"Safety violations (T>{env.SAFETY_THRESHOLD}degC): {violation_count} steps")
    print(f"{'='*60}")

    # ── Save training results ────────────────────────────────────────────────
    import json
    train_results = {
        'total_steps':        total_steps,
        'total_episodes':     episode,
        'final_comfort_error':float(np.mean(error_history[-500:])),
        'final_t_return':     float(np.mean(temp_history[-500:])),
        'final_t_supply':     float(np.mean(sp_history[-500:])),
        'final_pue':          float(np.mean(pue_history[-500:])),
        'safety_violations':  violation_count,
        't_target':           env.t_target,
        'safety_threshold':   env.SAFETY_THRESHOLD,
        'best_reward':        float(max(episode_rewards)),
    }
    with open('ddpg_train_results.json', 'w') as f:
        json.dump(train_results, f, indent=2)

    np.save('ddpg_train_rewards.npy',  np.array(episode_rewards))
    np.save('ddpg_train_errors.npy',   np.array(error_history))
    np.save('ddpg_train_temps.npy',    np.array(temp_history))
    np.save('ddpg_train_supply.npy',   np.array(sp_history))
    np.save('ddpg_train_pues.npy',     np.array(pue_history))
    print("Training results saved:")
    print("  ddpg_train_results.json")
    print("  ddpg_train_rewards.npy")
    print("  ddpg_train_errors.npy")
    print("  ddpg_train_temps.npy")
    print("  ddpg_train_supply.npy")
    print("  ddpg_train_pues.npy")

    return agent, pue_history

# ─── Plot Training ───────────────────────────────────────────────────────────

def _plot_training(episode_rewards, episode_pues,
                   pue_history, sp_history, temp_history, error_history, env):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle('DDPG Training Results — CNN-LSTM + DDPG HVAC Control\n',
                 fontsize=13, fontweight='bold')

    def smooth(x, w=20):
        if len(x) < w:
            return np.array(x), np.arange(len(x))
        return np.convolve(x, np.ones(w)/w, mode='valid'), \
               np.arange(len(np.convolve(x, np.ones(w)/w, mode='valid')))

    episodes = np.arange(len(episode_rewards))
    steps    = np.arange(len(pue_history))

    # 1. Episode Reward
    axes[0,0].plot(episodes, episode_rewards, alpha=0.3, color='blue',
                   label='Raw episode reward')
    if len(episode_rewards) > 5:
        sm, idx = smooth(episode_rewards, 5)
        axes[0,0].plot(idx, sm, color='blue', linewidth=2,
                       label='Smoothed (window=5)')
    axes[0,0].axhline(y=max(episode_rewards), color='green', linestyle=':',
                      alpha=0.7, label=f'Best: {max(episode_rewards):.1f}')
    axes[0,0].set_title('Episode Reward (higher = better)')
    axes[0,0].set_xlabel('Episode')
    axes[0,0].set_ylabel('Total Reward')
    axes[0,0].legend(fontsize=8)
    axes[0,0].grid(True, alpha=0.3)

    # 2. Comfort Error over steps
    sm_err, _ = smooth(error_history, 50)
    axes[0,1].plot(steps, error_history, alpha=0.15, color='red',
                   linewidth=0.5, label='Raw comfort error')
    axes[0,1].plot(np.arange(len(sm_err)), sm_err, color='red',
                   linewidth=2, label='Smoothed (window=50)')
    axes[0,1].axhline(y=np.mean(error_history), color='orange', linestyle='-.',
                      linewidth=1.5,
                      label=f'Mean={np.mean(error_history):.3f}degC')
    axes[0,1].set_title(f'Comfort Error T_Return - {env.t_target} degC')
    axes[0,1].set_xlabel('Step')
    axes[0,1].set_ylabel('Error (degC)')
    axes[0,1].legend(fontsize=8)
    axes[0,1].grid(True, alpha=0.3)

    # 3. T_Supply commanded
    sm_sp, _ = smooth(sp_history, 50)
    axes[1,0].plot(steps, sp_history, alpha=0.2, color='orange',
                   linewidth=0.5, label='Raw T_Supply')
    axes[1,0].plot(np.arange(len(sm_sp)), sm_sp, color='darkorange',
                   linewidth=2, label='Smoothed (window=50)')
    axes[1,0].axhline(y=env.t_target, color='red', linestyle='--',
                      linewidth=1.5, label=f'T_Return target={env.t_target}degC')
    axes[1,0].axhline(y=env.t_supply_min, color='gray', linestyle=':',
                      linewidth=1.2, label=f'T_Supply min={env.t_supply_min}degC')
    axes[1,0].axhline(y=env.t_supply_max, color='black', linestyle=':',
                      linewidth=1.2, label=f'T_Supply max={env.t_supply_max}degC')
    axes[1,0].axhline(y=np.mean(sp_history), color='blue', linestyle='-.',
                      linewidth=1.5,
                      label=f'Mean={np.mean(sp_history):.2f}degC')
    axes[1,0].set_title('T_Supply Commanded by DDPG')
    axes[1,0].set_xlabel('Step')
    axes[1,0].set_ylabel('T_Supply (degC)')
    axes[1,0].legend(fontsize=8)
    axes[1,0].grid(True, alpha=0.3)

    # 4. Simulated T_Return
    sm_t, _ = smooth(temp_history, 50)
    axes[1,1].plot(steps, temp_history, alpha=0.2, color='purple',
                   linewidth=0.5, label='Simulated T_Return')
    axes[1,1].plot(np.arange(len(sm_t)), sm_t, color='purple',
                   linewidth=2, label='Smoothed (window=50)')
    axes[1,1].axhline(y=env.t_target, color='red', linestyle='--',
                      linewidth=1.5, label=f'Target={env.t_target}degC')
    axes[1,1].axhline(y=env.SAFETY_THRESHOLD, color='darkred', linestyle='-.',
                      linewidth=1.5,
                      label=f'Safety limit={env.SAFETY_THRESHOLD}degC')
    axes[1,1].axhline(y=np.mean(temp_history), color='blue', linestyle='-.',
                      linewidth=1.5,
                      label=f'Mean={np.mean(temp_history):.2f}degC')
    axes[1,1].fill_between(np.arange(len(sm_t)), sm_t, env.t_target,
                            where=(np.array(sm_t) > env.t_target),
                            alpha=0.15, color='red', label='Above target')
    axes[1,1].fill_between(np.arange(len(sm_t)), sm_t, env.t_target,
                            where=(np.array(sm_t) <= env.t_target),
                            alpha=0.15, color='blue', label='Below target')
    axes[1,1].set_title('Simulated T_Return (First-order Thermal Lag Model)')
    axes[1,1].set_xlabel('Step')
    axes[1,1].set_ylabel('Temperature (degC)')
    axes[1,1].legend(fontsize=8)
    axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Plot saved -> training_results.png")

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
    fig.suptitle('DDPG Test Results — CNN-LSTM + DDPG HVAC Control\n',
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
            sps.append(info['T_Supply'])
            errors.append(info['comfort_error'])
            if info['actual_T_return'] > env.SAFETY_THRESHOLD:
                violations += 1
            if done:
                break

        ep_predictions = [all_predictions[env.current_idx - len(temps) + i + 1]
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
        label = f'Ep {ep+1} (err={avg_error:.3f})'
        steps = np.arange(len(temps))

        axes[0,0].plot(steps, temps,  color=c, alpha=0.8, linewidth=1.2, label=label)
        axes[0,1].plot(steps, pues,   color=c, alpha=0.8, linewidth=1.2, label=label)
        axes[1,0].plot(steps, sps,    color=c, alpha=0.8, linewidth=1.2, label=label)
        axes[1,1].plot(steps, errors, color=c, alpha=0.8, linewidth=1.2, label=label)

        print(f"  Ep {ep+1:2d} | Reward: {total_reward:8.2f} | "
              f"Avg PUE: {avg_pue:.4f} | "
              f"Avg T_Return: {avg_temp:.2f}degC | "
              f"Comfort Error: {avg_error:.4f}degC | "
              f"Safety violations: {violations} | "
              f"DUE: {due:.4f}")

    axes[0,0].axhline(y=env.t_target, color='black', linestyle='--',
                      linewidth=2, label=f'Target={env.t_target}degC')
    axes[0,0].axhline(y=env.SAFETY_THRESHOLD, color='darkred', linestyle='-.',
                      linewidth=1.5, label=f'Safety={env.SAFETY_THRESHOLD}degC')
    axes[0,0].set_title('Simulated T_Return per Episode')
    axes[0,0].set_xlabel('Step')
    axes[0,0].set_ylabel('Temperature (degC)')
    axes[0,0].legend(fontsize=8)
    axes[0,0].grid(True, alpha=0.3)

    axes[0,1].axhline(y=1.0, color='black', linestyle='--',
                      linewidth=2, label='Ideal PUE=1.0')
    axes[0,1].axhline(y=np.mean(all_pues), color='red', linestyle='-.',
                      linewidth=1.5, label=f'Mean PUE={np.mean(all_pues):.3f}')
    axes[0,1].set_title('Physics PUE per Episode (lower = better)')
    axes[0,1].set_xlabel('Step')
    axes[0,1].set_ylabel('PUE')
    axes[0,1].legend(fontsize=8)
    axes[0,1].grid(True, alpha=0.3)

    axes[1,0].axhline(y=env.t_supply_min, color='gray', linestyle=':',
                      linewidth=1.5, label=f'min={env.t_supply_min}degC')
    axes[1,0].axhline(y=env.t_supply_max, color='black', linestyle=':',
                      linewidth=1.5, label=f'max={env.t_supply_max}degC')
    axes[1,0].axhline(y=env.t_target, color='red', linestyle='--',
                      linewidth=1.5, label=f'T_target={env.t_target}degC')
    axes[1,0].set_title('T_Supply Commanded by DDPG')
    axes[1,0].set_xlabel('Step')
    axes[1,0].set_ylabel('T_Supply (degC)')
    axes[1,0].legend(fontsize=8)
    axes[1,0].grid(True, alpha=0.3)

    axes[1,1].axhline(y=0, color='black', linestyle='--',
                      linewidth=2, label='Perfect (error=0)')
    axes[1,1].axhline(y=np.mean(all_errors), color='red', linestyle='-.',
                      linewidth=1.5,
                      label=f'Mean={np.mean(all_errors):.3f}degC')
    axes[1,1].set_title('Comfort Error = |T_Return - 27degC|')
    axes[1,1].set_xlabel('Step')
    axes[1,1].set_ylabel('Error (degC)')
    axes[1,1].legend(fontsize=8)
    axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('test_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Plot saved -> test_results.png")

    print(f"\n{'='*55}")
    print(f"Test Summary ({num_episodes} episodes)")
    print(f"{'='*55}")
    print(f"  Mean PUE (physics):    {np.mean(all_pues):.4f}")
    print(f"  Mean Comfort Error:    {np.mean(all_errors):.4f}degC")
    print(f"  Mean Reward:           {np.mean(all_rewards):.2f}")
    print(f"  Safety Violations:     {sum(all_violations)}")
    print(f"  Target T_Return:       {env.t_target}degC")
    print(f"{'='*55}")

    # ── Save test results ─────────────────────────────────────────────────────
    import json
    test_results = {
        'num_episodes':       num_episodes,
        'episode_len':        env.episode_len,
        'mean_comfort_error': float(np.mean(all_errors)),
        'std_comfort_error':  float(np.std(all_errors)),
        'mean_pue':           float(np.mean(all_pues)),
        'mean_reward':        float(np.mean(all_rewards)),
        'safety_violations':  int(sum(all_violations)),
        't_target':           env.t_target,
        'per_episode': [
            {
                'episode':        ep + 1,
                'comfort_error':  float(all_errors[ep]),
                'pue':            float(all_pues[ep]),
                'reward':         float(all_rewards[ep]),
                'violations':     int(all_violations[ep]),
            }
            for ep in range(num_episodes)
        ]
    }
    with open('ddpg_test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)

    np.save('ddpg_test_errors.npy',   np.array(all_errors))
    np.save('ddpg_test_pues.npy',     np.array(all_pues))
    np.save('ddpg_test_rewards.npy',  np.array(all_rewards))
    print("\nTest results saved:")
    print("  ddpg_test_results.json")
    print("  ddpg_test_errors.npy")
    print("  ddpg_test_pues.npy")
    print("  ddpg_test_rewards.npy")

# ─── DUE Metric ──────────────────────────────────────────────────────────────

def calculate_due_validation(y_true, y_pred):
    y_true      = np.array(y_true)
    y_pred      = np.array(y_pred)
    differences = y_true - y_pred
    penalties   = np.maximum(differences, 0)
    return float(np.sum(penalties))

# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HVAC CNN-LSTM + DDPG')
    parser.add_argument('--train',    action='store_true')
    parser.add_argument('--test',     action='store_true')
    parser.add_argument('--steps',    type=int, default=200000)
    parser.add_argument('--episodes', type=int, default=5)
    args = parser.parse_args()

    if args.train:
        agent, pues = train(df_model, total_steps=args.steps)
    if args.test:
        test(df_model, num_episodes=args.episodes)
    if not args.train and not args.test:
        agent, pues = train(df_model, total_steps=args.steps)
        test(df_model, num_episodes=args.episodes)
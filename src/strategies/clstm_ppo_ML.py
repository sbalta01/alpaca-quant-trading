import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sb3_contrib import RecurrentPPO
from sb3_contrib.ppo_recurrent.policies import MlpLstmPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from src.env.trading_env import TradingEnv

# ──────────────────────────────────────────────────────────────────────────────

class CLSTMFeatureExtractor(BaseFeaturesExtractor):
    """
    First LSTM extracts hidden features from a sequence of T raw states.
    """
    def __init__(self, observation_space, features_dim, seq_len=30):
        obs_dim = observation_space.shape[0] // seq_len
        super().__init__(observation_space, features_dim=features_dim)
        self.seq_len    = seq_len
        self.obs_dim    = obs_dim
        self.hidden_dim = features_dim
        
        # LSTM + three tanh layers
        self.lstm = nn.LSTM(obs_dim, self.hidden_dim, batch_first=True)
        self.fc1  = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc2  = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.act  = nn.Tanh()

    def forward(self, observations):
        # observations: [batch, seq_len * obs_dim]
        batch_size = observations.size(0)
        # reshape into [batch, seq_len, obs_dim]
        x = observations.view(batch_size, self.seq_len, self.obs_dim)
        lstm_out, _ = self.lstm(x)
        h = lstm_out[:, -1, :]        # last time step
        h = self.act(self.fc1(h))
        h = self.act(self.fc2(h))
        return h

# ──────────────────────────────────────────────────────────────────────────────
def make_vec_env(env_kwargs, n_envs=1, seq_len=30):
    venv = DummyVecEnv([lambda: TradingEnv(**env_kwargs) for _ in range(n_envs)])
    return VecFrameStack(venv, n_stack=seq_len)

def train_clstm_ppo(
    price_df, tech_cols, macro_cols,
    seq_len=30, lstm_hidden=512,
    total_timesteps=1_000_000, n_envs=1
):
    # 1) fetch and slice data
    df = price_df
    env_kwargs = dict(
        price_df=df, tech_cols=tech_cols, macro_cols=macro_cols,
        initial_cash=1e3, transaction_cost=0.001,
        turbulence_col="turbulence", turbulence_percentile=0.9
    )
    features_dim = len(df.index.get_level_values(level='symbol').unique())*len(tech_cols) + len(macro_cols)
    env = make_vec_env(env_kwargs, n_envs, seq_len)

    # 2) Recurrent PPO hyperparams (two LSTMs: your extractor + internal LSTM)
    policy_kwargs = dict(
        features_extractor_class = CLSTMFeatureExtractor,
        features_extractor_kwargs = dict(seq_len=seq_len, features_dim=features_dim),
        lstm_hidden_size  = lstm_hidden, #hidden size for PPO's LSTM
        net_arch=dict(
            pi=[lstm_hidden],         # One hidden layers of size lstm_hidden for the actor
            vf=[lstm_hidden]          # and for the critic
        )
    )

    if torch.cuda.is_available():
        device = 'cuda'
        print("Current device name: ", torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        device = 'cpu'
        print("Current device name: ", device)

    model = RecurrentPPO(
        policy = MlpLstmPolicy,
        env    = env,
        learning_rate   = 3e-4,
        n_steps         = 128,
        batch_size      = 128,
        n_epochs=10,
        gamma=0.99,
        clip_range      = 0.2,
        ent_coef        = 0.01,
        vf_coef         = 0.5,
        max_grad_norm   = 0.5,
        device   = device,
        policy_kwargs   = policy_kwargs,
        tensorboard_log="./tensorboard/"
     )
            
    # 3) train
    model.learn(total_timesteps=total_timesteps, progress_bar= True)
    return model
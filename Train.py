import os
import torch
import optuna
import retro
import numpy as np
import cv2

from gym import Env
from gym.spaces import MultiBinary, Box

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback

# Paths
LOG_DIR = './logs/'
OPT_DIR = './opt/'
CHECKPOINT_DIR = './train/'

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OPT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Custom Environment
class StreetFighter(Env): 
    def __init__(self):
        super().__init__()
        self.observation_space = Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.action_space = MultiBinary(12)
        self.game = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis', use_restricted_actions=retro.Actions.FILTERED)
    
    def reset(self):
        obs = self.game.reset()
        obs = self.preprocess(obs)
        self.previous_frame = obs
        self.score = 0
        return obs
    
    def preprocess(self, observation): 
        gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (84,84), interpolation=cv2.INTER_CUBIC)
        return np.reshape(resize, (84,84,1))
    
    def step(self, action): 
        obs, reward, done, info = self.game.step(action)
        obs = self.preprocess(obs)
        frame_delta = obs - self.previous_frame
        self.previous_frame = obs
        reward = info['score'] - self.score 
        self.score = info['score'] 
        return frame_delta, reward, done, info
    
    def render(self, *args, **kwargs):
        self.game.render()
        
    def close(self):
        self.game.close()

# Callback for saving models
class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, f'best_model_{self.n_calls}')
            self.model.save(model_path)
        return True

# Check CUDA availability
print("Checking device availability...")
if torch.cuda.is_available():
    print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("❌ Using CPU")

# Re-create environment
env = StreetFighter()
env = Monitor(env, LOG_DIR)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 4, channels_order='last')

# Best parameters (insert manually or load from Optuna result)
best_params = {
    'n_steps': 7989,
    'gamma': 0.9126461334090934,
    'learning_rate': 1.2679898046516959e-05,
    'clip_range': 0.263489569972742,
    'gae_lambda': 0.9095722601337441
}

# Create PPO model with best params
model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, device='cuda' if torch.cuda.is_available() else 'cpu', **best_params)

# Load weights from the best model checkpoint
model_path = os.path.join(OPT_DIR, 'trial_7_best_model.zip')
model = PPO.load(model_path, env=env)

# Training callback
callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)

# Continue training
model.learn(total_timesteps=3000000, callback=callback)

# Final save
model.save(os.path.join(CHECKPOINT_DIR, 'final_trained_model'))

print("✅ Training complete and model saved.")

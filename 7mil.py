import os
import torch
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
CHECKPOINT_DIR = './train/'

os.makedirs(LOG_DIR, exist_ok=True)
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
            model_path = os.path.join(self.save_path, f'continued_model_{self.n_calls}')
            self.model.save(model_path)
        return True

# Check CUDA availability
print("Checking device availability...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("❌ Using CPU")

# Re-create environment
env = StreetFighter()
env = Monitor(env, LOG_DIR)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 4, channels_order='last')

# Load previously trained model
model_path = os.path.join(CHECKPOINT_DIR, 'final_trained_model.zip')
model = PPO.load(model_path, env=env, device=device)

# Training callback
callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)

# Continue training
model.learn(total_timesteps=7_000_000, callback=callback)

# Final save
model.save(os.path.join(CHECKPOINT_DIR, 'final_trained_model_continued'))

print("✅ Continued training complete and model saved.")

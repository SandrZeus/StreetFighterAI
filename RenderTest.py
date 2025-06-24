import retro
import os
import time
import numpy as np
import cv2
from gym import Env
from gym.spaces import MultiBinary, Box
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy

LOG_DIR = './logs/'
MODEL_PATH = './train/final_trained_model.zip'

# Custom Street Fighter environment
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
        resize = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_CUBIC)
        channels = np.reshape(resize, (84, 84, 1))
        return channels 
    
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
        time.sleep(0.01)
        
    def close(self):
        self.game.close()

if __name__ == "__main__":
    print("🎮 Loading trained model...")

    env = StreetFighter()
    env = Monitor(env, LOG_DIR)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, 4, channels_order='last')

    model = PPO.load(MODEL_PATH)

    print("✅ Model loaded! Starting evaluation with rendering...")
    mean_reward, _ = evaluate_policy(model, env, render=True, n_eval_episodes=5)
    print(f"🎯 Mean reward over 5 episodes: {mean_reward}")

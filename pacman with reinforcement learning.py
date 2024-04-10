import gym
from stable_baselines3.ppo import PPO
import os
import random

log_dir = 'logs'
if not os.path.exists(log_dir):
    os.mkdir(log_dir)


#env = gym.make("ALE/MsPacman-v5", obs_type = 'ram', render_mode = 'human')
env = gym.make('LunarLander-v2')
agent = PPO('MlpPolicy', env, tensorboard_log = log_dir, verbose = 1)
agent.learn(total_timesteps=10000, tb_log_name='ppo_pacman_agent')
for i in range(3):
    done = False
    obs = env.reset()
    score = 0
    while not done:
        #action, _states = agent.predict(obs)
        action = random.randint(0,8)
        obs, reward, done, info, truncated = env.step(action)
        score += reward
        env.render()
    print(score)
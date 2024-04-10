import gym
from time import perf_counter
from stable_baselines3.ppo import PPO
import os

start = perf_counter()
logs = 'logs'
if not os.path.exists(logs):
    os.makedirs(logs)

env = gym.make('CarRacing-v2')
agent = PPO('CnnPolicy', env, verbose = 1)
agent.learn(total_timesteps = 1000)
#agent.save('ppo_car_racing_1000000')
#agent = PPO.load('ppo_car_racing_1000000')
end = perf_counter()

for i in range(3):
    score = 0
    obs = env.reset()
    done = False
    while not done:
        action, _states = agent.predict(obs)
        obs, reward, done, info = env.step(action)
        score += reward
        env.render()
    print(score)

print(f'elapsed time: {round(end - start, 3)} s')

env.close()
import gym
from stable_baselines3.a2c import A2C # a2c has better performance than dqn on this task
from stable_baselines3.common.evaluation import evaluate_policy
from time import perf_counter

start_time = perf_counter()
env = gym.make('LunarLander-v2')
#model = A2C.load('lunar_lander_a2c')
model = A2C('MlpPolicy', env, verbose = 1)
model.learn(total_timesteps = 1000)
#model.save('lunar_lander_a2c')

# to visualize

for i in range (10):
    obs = env.reset()
    score = 0
    terminated = False
    truncated = False
    while not terminated and not truncated:
        action, _states = model.predict(obs)
        print('fatto 0')
        obs, reward, terminated, truncated, info,a,b,c = env.step(action)
        print('fatto 1')
        env.render()
        score += reward
    print(f'episode {i+1} -- score: {round(score,3)}')

env.close()
end_time = perf_counter()
time_elapsed = end_time - start_time
print(f'time spent: {round(time_elapsed,3)} s')

import gymnasium as gym
import ale_py

gym.register_envs(ale_py)
env = gym.make("ALE/Assault-v5", render_mode="human")
obs, info = env.reset()


done = False
while not done:
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    print("action", action, "reward", reward, "terminated", terminated, "truncated", truncated, "info", info)
    done = terminated or truncated

env.close()


print("Observation shape:", obs.shape)
print("Action space:", env.action_space)

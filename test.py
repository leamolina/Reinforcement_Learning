import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing
from frame_stack import FrameStack  # Corrected import
import ale_py

gym.register_envs(ale_py)

def Q_learning(D, N):
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

# Hyperparameters
gamma = 0.99  # Discount factor
epsilon = 1.0  # Initial exploration rate
epsilon_min = 0.1  # Minimum exploration rate
epsilon_decay = 0.995  # Epsilon decay factor
learning_rate = 0.00025  # Learning rate
batch_size = 32  # Mini-batch size
memory_size = 100000  # Replay memory size
target_update_frequency = 1000  # Steps before target network update
num_episodes = 500  # Total number of episodes
max_steps_per_episode = 10000  # Maximum steps per episode

# Neural network for Q-value approximation
class DQNetwork(nn.Module):
    def __init__(self, action_space):
        super(DQNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),  # Input: (4, 84, 84)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, action_space)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Replay memory
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def store(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Initialize environment and preprocessing
env = gym.make("ALE/Assault-v5", render_mode=None)
env = AtariPreprocessing(env, grayscale_obs=True, frame_skip=4)
env = FrameStack(env, num_stack=4)  # Stacks the last 4 frames

action_space = env.action_space.n
policy_net = DQNetwork(action_space).to("cuda")
target_net = DQNetwork(action_space).to("cuda")
target_net.load_state_dict(policy_net.state_dict())  # Synchronize weights
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
memory = ReplayMemory(memory_size)

# Training loop
for episode in range(num_episodes):
    state = env.reset()[0]
    state = np.transpose(state, (2, 0, 1))  # Reshape to (C, H, W)
    total_reward = 0
    done = False

    for t in range(max_steps_per_episode):
        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to("cuda")
                q_values = policy_net(state_tensor)
                action = torch.argmax(q_values).item()

        # Take action and observe transition
        next_state, reward, done, _, _ = env.step(action)
        next_state = np.transpose(next_state, (2, 0, 1))  # Reshape to (C, H, W)

        memory.store((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        # Sample and train if memory has enough transitions
        if len(memory) > batch_size:
            batch = memory.sample(batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.tensor(np.array(states), dtype=torch.float32).to("cuda")
            actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to("cuda")
            rewards = torch.tensor(rewards, dtype=torch.float32).to("cuda")
            next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to("cuda")
            dones = torch.tensor(dones, dtype=torch.bool).to("cuda")

            # Q-values
            current_q = policy_net(states).gather(1, actions).squeeze()
            next_q = target_net(next_states).max(1)[0]
            target_q = rewards + gamma * next_q * (~dones)

            # Loss and optimization
            loss = nn.MSELoss()(current_q, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Update target network periodically
        if t % target_update_frequency == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if done:
            break

    # Epsilon decay
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

env.close()

import gymnasium as gym
import numpy as np
import random
import ale_py
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Load the environment
gym.register_envs(ale_py)
env = gym.make("ALE/Assault-v5", render_mode="human")
"""
BBF (Bigger, Better, Faster), 
a value-based reinforcement learning agent that achieves superhuman performance on the Atari 100K benchmark while maintaining human-level efficiency. 
It focuses on improving sample efficiency and scalability in deep reinforcement learning (RL).
State-of-the-Art Performance with BBF:
BBF outperforms previous RL agents like DQN, Rainbow, EfficientZero, and SR-SPR.
Achieves superhuman scores with significantly fewer environment interactions and computational cost compared to prior agents.

Scaling Neural Networks:
BBF employs larger neural networks for value estimation (4x wider Impala-CNN).
Effective scaling techniques help avoid overfitting and maintain sample efficiency.
Components :
Replay Ratio: Increased to 8, balancing computation and sample efficiency.
Periodic Network Resets: Prevents overfitting by resetting parts of the network during training.
Update Horizon Annealing: Gradually reduces the update horizon (n-step) from 10 to 3 steps during early training.
Increased Discount Factor (γ): Gradually increases from 0.97 to 0.997 to balance reward weighting.
Weight Decay: Added to control overfitting, using AdamW optimizer with a decay of 0.1.
No Noisy Networks: Removed as they didn't contribute to improved performance.

Benchmarking and Comparisons:
BBF shows a 5x improvement over SR-SPR and 16x improvement over DQN and Rainbow.
Computational efficiency: BBF requires just 6 hours of GPU training time compared to 8.5 hours for EfficientZero.
"""


# Hyperparameters
# Number of samples per gradient step.
BATCH_SIZE = 32
# Discount factor for future rewards.
GAMMA = 0.99
# Epsilon-Greedy Parameters: For exploration:
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 1000000
# Target Network Update Frequency
TARGET_UPDATE = 10000
# Replay Buffer Size ( Capacity for storing past experiences)
MEMORY_SIZE = 1000000
# Optimizer's step size
LEARNING_RATE = 1e-4
# Number of gradient steps per environment interaction
REPLAY_RATIO = 8

# Preprocessing
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) # Reduces dimensionality by converting the frame to a single channel.
    resized = cv2.resize(gray, (84, 84)) # The original Atari frame is resized to 84x84 pixels to reduce computation.
    return resized

# Experience Replay Buffer (Stores experiences : state, action, reward, next_state, done for training.)
"""
push(): Adds a new experience to the buffer.
sample(): Randomly samples a batch for training.
deque: Automatically removes the oldest experience if the buffer is full.
"""
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), actions, rewards, np.array(next_states), dones

    def __len__(self):
        return len(self.buffer)

#Neural Network
# Define the Q-Network (ResNet inspired Impala-CNN)
"""
CNN Architecture (Impala-style):
    3 Convolutional layers extract features from the input frames.
    ReLU activations introduce non-linearity.
    Flatten and fully connected layers map extracted features to Q-values.
Input: 4 stacked grayscale frames.
Output: Q-values for each action.
"""
class QNetwork(nn.Module):
    def __init__(self, action_size):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, action_size)
        )

    def forward(self, x):
        return self.net(x)

# Initialize networks and replay buffer
action_size = env.action_space.n
policy_net = QNetwork(action_size).cuda()
target_net = QNetwork(action_size).cuda()
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.AdamW(policy_net.parameters(), lr=LEARNING_RATE, weight_decay=0.1)
memory = ReplayBuffer(MEMORY_SIZE)

# Epsilon-greedy action selection
"""
Uses a decaying epsilon-greedy strategy.
Chooses a random action with probability ε.
Selects the action with the highest Q-value otherwise.
"""
def select_action(state, steps_done):
    eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
    if random.random() > eps_threshold:
        with torch.no_grad():
            return policy_net(state.cuda()).argmax().item()
    else:
        return random.randrange(action_size)

# Training loop:
"""
Process:
Environment Reset: The game environment is reset at the beginning of each episode.
State Stacking: Four consecutive frames are stacked for richer state representation.
Action Selection: The agent selects an action using the ε-greedy policy.
Environment Interaction: The agent interacts with the environment and stores the experience.
State Transition: The next frame is preprocessed and added to the state stack.
"""

steps_done = 0
for episode in range(1000):
    state = preprocess_frame(env.reset()[0])
    state_stack = np.stack([state] * 4, axis=0)
    total_reward = 0

    while True:
        state_tensor = torch.FloatTensor(state_stack).unsqueeze(0).cuda()
        action = select_action(state_tensor, steps_done)
        next_state, reward, done, _, _ = env.step(action)

        next_state = preprocess_frame(next_state)
        next_state_stack = np.append(state_stack[1:], next_state[np.newaxis, ...], axis=0)

        memory.push(state_stack, action, reward, next_state_stack, done)
        state_stack = next_state_stack
        total_reward += reward
        steps_done += 1

        # Sample batch and train
        """
        Minibatch Training: A batch of experiences is sampled from the buffer.
        Q-Value Calculation:
        Predicted Q-value: From the policy network.
        Target Q-value: Reward + discounted future reward from the target network.
        Loss Calculation: Mean Squared Error (MSE) between predicted and target Q-values.
        Optimizer Step: The network is updated using AdamW."""
        if len(memory) > BATCH_SIZE:
            states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)
            states = torch.FloatTensor(states).cuda()
            next_states = torch.FloatTensor(next_states).cuda()
            actions = torch.LongTensor(actions).cuda()
            rewards = torch.FloatTensor(rewards).cuda()
            dones = torch.FloatTensor(dones).cuda()

            q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            next_q_values = target_net(next_states).max(1)[0].detach()
            expected_q_values = rewards + GAMMA * next_q_values * (1 - dones)

            loss = nn.MSELoss()(q_values, expected_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Target Network Syncing: Every 10,000 steps, the target network is updated to match the policy network.
        if steps_done % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if done:
            print(f"Episode {episode} Reward: {total_reward}")
            break

env.close()

import gymnasium as gym
import numpy as np
import random
import ale_py
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque




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



# Epsilon-greedy action selection
"""
Uses a decaying epsilon-greedy strategy.
Chooses a random action with probability ε.
Selects the action with the highest Q-value otherwise.
"""
def select_action(state, steps_done, policy_net, device, action_size, epsilon, epsilon_min, epsilon_decay):
    eps_threshold = epsilon_min + (epsilon - epsilon_min) * np.exp(-1. * steps_done / epsilon_decay)
    if random.random() > eps_threshold:
        with torch.no_grad():
            return policy_net(state.to(device)).argmax().item()
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
def train_bbf(batch_size, gamma, epsilon, epsilon_min, epsilon_decay, target_update, memory_size, learning_rate, replay_ratio, episodes, model_path):
    # Load the environment
    gym.register_envs(ale_py)
    env = gym.make("ALE/Assault-v5", render_mode="rgb_array")

    # Initialize networks and replay buffer
    action_size = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = QNetwork(action_size).to(device)
    target_net = QNetwork(action_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.AdamW(policy_net.parameters(), lr=learning_rate, weight_decay=0.1)
    memory = ReplayBuffer(memory_size)

    steps_done = 0
    for episode in range(episodes):
        state = preprocess_frame(env.reset()[0])
        state_stack = np.stack([state] * 4, axis=0)
        total_reward = 0

        while True:
            state_tensor = torch.FloatTensor(state_stack).unsqueeze(0).to(device)
            action = select_action(state_tensor, steps_done, policy_net, device, action_size, epsilon, epsilon_min, epsilon_decay)
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
            if len(memory) > batch_size:
                states, actions, rewards, next_states, dones = memory.sample(batch_size)
                states = torch.FloatTensor(states).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                actions = torch.LongTensor(actions).to(device)
                rewards = torch.FloatTensor(rewards).to(device)
                dones = torch.FloatTensor(dones).to(device)

                q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                next_q_values = target_net(next_states).max(1)[0].detach()
                expected_q_values = rewards + gamma * next_q_values * (1 - dones)

                loss = nn.MSELoss()(q_values, expected_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # Target Network Syncing: Every 10,000 steps, the target network is updated to match the policy network.
            if steps_done % target_update < batch_size:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                print("Episode", episode+1 , "/", episodes, "Reward:", total_reward, ", Epsilon :", epsilon)
                # Décroissance d'epsilon :
                epsilon = max(epsilon_min, epsilon * epsilon_decay)
                break

    # Sauvegarde du modèle entraîné
    torch.save(policy_net.state_dict(), model_path)
    print("Modèle sauvegardé avec succès")

    env.close()

def run_bbf(model_path):
    # Load environment
    env = gym.make("ALE/Assault-v5", render_mode="human")
    action_size = env.action_space.n

    # Load trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = QNetwork(action_size).to(device)
    policy_net.load_state_dict(torch.load(model_path, map_location=device))
    policy_net.eval()  # Set model to evaluation mode

    state = preprocess_frame(env.reset()[0])
    state_stack = np.stack([state] * 4, axis=0)
    total_reward = 0

    while True:
        state_tensor = torch.FloatTensor(state_stack).unsqueeze(0).to(device)
        with torch.no_grad():
            action = policy_net(state_tensor).argmax().item()
        next_state, reward, done, _, _ = env.step(action)

        next_state = preprocess_frame(next_state)
        state_stack = np.append(state_stack[1:], next_state[np.newaxis, ...], axis=0)
        total_reward += reward

        if done:
            print("Partie terminée, Score : ", total_reward)
            break

    env.close()


if __name__ == "__main__":

    # Hyperparamètres
    batch_size = 32  # Number of samples per gradient step.
    gamma = 0.99
    epsilon = 0.995
    epsilon_min = 0.1
    epsilon_decay = 0.995
    target_update = 10000  # Target Network Update Frequency
    memory_size = 1000000  # Replay Buffer Size ( Capacity for storing past experiences)
    learning_rate = 1e-4  # Optimizer's step size
    replay_ratio = 8  # Number of gradient steps per environment interaction
    episodes = 10  # Number of training episodes

    # Entraînement du modèle
    model_path = "./Models/model_bbf.pth"
    train_bbf(batch_size, gamma, epsilon, epsilon_min, epsilon_decay, target_update, memory_size, learning_rate, replay_ratio, episodes, model_path)

    # Lancement du modèle entrainé
    run_bbf(model_path)
import gymnasium as gym
import numpy as np
import random
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from collections import deque
import ale_py
import time




# Charger l'environnement
gym.register_envs(ale_py)
env = gym.make("ALE/Assault-v5", render_mode="human")
num_actions = env.action_space.n
input_shape = (4, 84, 84)

# Hyperparamètres
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
episodes = 7000
minibatch_size = 32
replay_memory_size = 10000
alpha = 0.0001

replay_memory = deque(maxlen=replay_memory_size)

# Prétraitement des frames
def preprocessing(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized_frame = cv2.resize(gray_frame, (84, 84), interpolation=cv2.INTER_AREA)
    normalized_frame = resized_frame / 255.0
    return normalized_frame

# Réseau Dueling DQN
class DuelingDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DuelingDQN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(7 * 7 * 64, 512)

        # Réseaux séparés pour la valeur et l'avantage
        self.value_stream = nn.Linear(512, 1)
        self.advantage_stream = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc(x.view(x.size(0), -1)))

        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

# Initialisation des réseaux
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
online_dqn = DuelingDQN(input_shape, num_actions).to(device)
target_dqn = DuelingDQN(input_shape, num_actions).to(device)
target_dqn.load_state_dict(online_dqn.state_dict())

optimizer = optim.Adam(online_dqn.parameters(), lr=alpha)
loss_fn = nn.MSELoss()

# Fonction pour choisir une action (epsilon-greedy)
def choose_action(state, epsilon, num_actions):
    if np.random.rand() < epsilon:
        return random.randint(0, num_actions - 1)
    else:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = online_dqn(state_tensor)
        return torch.argmax(q_values).item()

# Fonction d'entraînement
def train_dqn():
    if len(replay_memory) < minibatch_size:
        return

    minibatch = random.sample(replay_memory, minibatch_size)
    states, actions, rewards, next_states, dones = zip(*minibatch)

    states = torch.FloatTensor(np.array(states)).to(device)
    actions = torch.LongTensor(actions).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    next_states = torch.FloatTensor(np.array(next_states)).to(device)
    dones = torch.FloatTensor(dones).to(device)

    # Double DQN: actions optimales sélectionnées par le réseau en ligne
    next_state_actions = online_dqn(next_states).argmax(dim=1, keepdim=True)

    # Valeurs cibles calculées avec le réseau cible
    target_q_values = target_dqn(next_states).gather(1, next_state_actions).squeeze(1)
    targets = rewards + gamma * target_q_values * (1 - dones)

    # Q-valeurs prédites
    q_values = online_dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # Calcul et optimisation de la perte
    loss = loss_fn(q_values, targets.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Temps de début
time_start = time.time()

# Boucle principale d'entraînement
for episode in range(episodes):
    state, _ = env.reset()
    state = preprocessing(state)
    state_stack = np.stack([state] * 4, axis=0)
    done = False
    score = 0

    while not done:
        action = choose_action(state_stack, epsilon, num_actions)
        next_state, reward, done, _, _ = env.step(action)
        next_state = preprocessing(next_state)
        next_state_stack = np.append(state_stack[1:], [next_state], axis=0)

        replay_memory.append((state_stack, action, reward, next_state_stack, done))
        state_stack = next_state_stack
        score += reward

        train_dqn()

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    # Mise à jour du réseau cible périodiquement
    if episode % 10 == 0:
        target_dqn.load_state_dict(online_dqn.state_dict())

    print(f"Épisode {episode + 1}/{episodes}, Score: {score}, Epsilon: {epsilon:.4f}")

env.close()

time_end = time.time()

time_elapsed = time_end - time_start
print("time elasped (in second)", time_elapsed)
import gymnasium as gym
import numpy as np
import random
import ale_py
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


# Preprocessing : On convertit les images en niveaux de gris et on les réduit à une résolution de 84x84
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84))
    return resized

# Classe ReplayBuffer qui permet de stocker les expériences (state, action, reward, next_state, done)
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    # Ajout d'une nouvelle expérience dans le buffer
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    # On choisit un batch aléatoirement (pour l'entraînement)
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), actions, rewards, np.array(next_states), dones

    def __len__(self):
        return len(self.buffer)



# Réseau de neuronnes Q-Network
class QNetwork(nn.Module):

    # Initialisation du réseau de neuronnes (à l'aide de l'architecture Impala-CNN)
    def __init__(self, action_size):
        super(QNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),  # 1ère couche de convolution (En entrée 4 images de taille 84x84)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # 2ème couche de convolution
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # 3ème couche de convolution
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, action_size)
        )

    def forward(self, x):
        return self.net(x)



# Choix des actions (Epsilon-greedy)
def select_action(state, steps_done, policy_net, device, action_size, epsilon, epsilon_min, epsilon_decay):

    # Epsilon diminue à chaque fois pour favoriser l'exploitation par rapport à l'exploration
    eps_threshold = epsilon_min + (epsilon - epsilon_min) * np.exp(-1. * steps_done / epsilon_decay)
    if random.random() > eps_threshold:
        with torch.no_grad():
            return policy_net(state.to(device)).argmax().item()
    else:
        return random.randrange(action_size)

# Entraînement
def train_bbf(batch_size, gamma, epsilon, epsilon_min, epsilon_decay, target_update, memory_size, learning_rate, replay_ratio, episodes, model_path, perf_path):

    # Charger l'environnement
    gym.register_envs(ale_py)
    env = gym.make("ALE/Assault-v5", render_mode="rgb_array")
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
            state_tensor = torch.tensor(state_stack, dtype=torch.float32, device=device).unsqueeze(0)
            action = select_action(state_tensor, steps_done, policy_net, device, action_size, epsilon, epsilon_min, epsilon_decay)
            next_state, reward, done, _, _ = env.step(action)


            next_state = preprocess_frame(next_state)
            next_state_stack = np.append(state_stack[1:], next_state[np.newaxis, ...], axis=0)

            memory.push(state_stack, action, reward, next_state_stack, done)
            state_stack = next_state_stack
            total_reward += reward
            steps_done += 1

            # Entraînement
            if len(memory) > batch_size:
                states, actions, rewards, next_states, dones = memory.sample(batch_size)
                states = torch.tensor(states, dtype=torch.float32, device=device)
                next_states = torch.tensor(next_states, dtype=torch.float32, device=device)
                actions = torch.tensor(actions, dtype=torch.long, device=device)
                rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
                dones = torch.tensor(dones, dtype=torch.float32, device=device)

                q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                next_q_values = target_net(next_states).max(1)[0].detach()
                expected_q_values = rewards + gamma * next_q_values * (1 - dones)

                loss = nn.SmoothL1Loss()(q_values, expected_q_values) # Initialement on avait la MSE loss, mais on a changé de loss et on en a favorisé une qui était moins sensible aux outliers (car le modèle n'apprenait pas)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Nettoyage Mémoire GPU
                torch.cuda.empty_cache()

            # Mise à jour du réseau cible
            if steps_done % 1000 == 0:
              target_net.load_state_dict(policy_net.state_dict())


            if done:
                print("Episode", episode+1 , "/", episodes, "Reward:", total_reward, ", Epsilon :", epsilon)
                with open(perf_path, 'a') as f:
                    f.write("Episode"+ str(episode+1) + "/"+ str(episodes)+ "Reward:"+ str(total_reward) + ", Epsilon :"+ str(epsilon) + "\n")

                epsilon = max(epsilon_min, epsilon * epsilon_decay)
                break

    # Sauvegarde du modèle
    torch.save(policy_net.state_dict(), model_path)
    print("Modèle sauvegardé avec succès")
    env.close()


def run_bbf(model_path):

    # Charger l'environnement
    env = gym.make("ALE/Assault-v5", render_mode="human")
    action_size = env.action_space.n

    # Récupération du modèle
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = QNetwork(action_size).to(device)
    policy_net.load_state_dict(torch.load(model_path, map_location=device))
    policy_net.eval()

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
            print("Partie terminée. Score obtenu : ", total_reward)
            break

    env.close()


if __name__ == "__main__":

    # Hyperparamètres
    batch_size = 32
    gamma = 0.99
    epsilon = 0.995
    epsilon_min = 0.1
    epsilon_decay = 0.995
    target_update = 1000
    memory_size = 50000
    learning_rate = 0.00025
    replay_ratio = 8
    episodes = 2500

    # Entraînement du modèle
    model_path = "../Models/model_bbf.pth"
    perf_path = "../Training performances/perf_bbf.txt"
    train_bbf(batch_size, gamma, epsilon, epsilon_min, epsilon_decay, target_update, memory_size, learning_rate, replay_ratio, episodes, model_path, perf_path)
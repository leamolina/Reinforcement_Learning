import gymnasium as gym
import numpy as np
import random
import cv2
import ale_py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from collections import deque


# Preprocessing : On convertit les images en niveaux de gris et on les réduit à une résolution de 84x84, et les valeurs des pixels sont normalisées en les divisant par 255.0,
def preprocessing(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized_frame = cv2.resize(gray_frame, (84, 84), interpolation=cv2.INTER_AREA)
    normalized_frame = resized_frame / 255.0
    return normalized_frame



# Réseau de neurones
class DQN(nn.Module):


    # Initialisation du réseau de neuronnes
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 16, kernel_size=8, stride=4) # Première couche
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2) # Deuxième couche
        self.fc1 = nn.Linear(32 * 9 * 9, 256)  # Calculé après les convolutions
        self.fc2 = nn.Linear(256, num_actions)

    # Mise en forme du réseau de neuronnes (activation ReLU pour les deux couches convolutionnelles)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # On applatit
        x = F.relu(self.fc1(x))
        return self.fc2(x)



# Choix d'une action (epsilon-greedy)
def choose_action(state, epsilon, num_actions, device, dqn):
    if np.random.rand() < epsilon:
        return random.randint(0, num_actions - 1)
    else:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = dqn(state_tensor)
        return torch.argmax(q_values).item()


# Etape d'entraînement du DQN
def train_step(device, dqn, optimizer, loss_fn, replay_memory):

    if len(replay_memory) < minibatch_size:
        return

    # Échantillonner un minibatch
    minibatch = random.sample(replay_memory, minibatch_size)
    states, actions, rewards, next_states, dones = zip(*minibatch)

    # Conversion en tenseurs
    states = torch.FloatTensor(np.array(states)).to(device)
    actions = torch.LongTensor(actions).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    next_states = torch.FloatTensor(np.array(next_states)).to(device)
    dones = torch.FloatTensor(dones).to(device)

    # Calcul des Q-valeurs cibles
    q_values = dqn(states)
    next_q_values = dqn(next_states)
    target_q_values = rewards + gamma * torch.max(next_q_values, dim=1)[0] * (1 - dones)

    # Calcul de la perte
    predicted_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    loss = loss_fn(predicted_q_values, target_q_values.detach())

    # Optimisation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# Entraînement à l'aide du Deep Q-Learning
def train_deep_q_learning(gamma, epsilon, epsilon_decay, episodes, minibatch_size, replay_memory, alpha, model_path):

    # Charger l'environnement
    gym.register_envs(ale_py)
    env = gym.make("ALE/Assault-v5", render_mode='rgb_array')
    num_actions = env.action_space.n
    input_shape = (4, 84, 84)

    # Instanciation du modèle
    dqn = DQN(input_shape, num_actions)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dqn.to(device)

    optimizer = optim.Adam(dqn.parameters(), lr=alpha)
    loss_fn = nn.MSELoss()

    for episode in range(episodes):
        state, _ = env.reset()
        state = preprocessing(state)
        state_stack = np.stack([state] * 4, axis=0)
        done = False
        score = 0

        while not done:

            # Choisir une action
            action = choose_action(state_stack, epsilon, num_actions, device, dqn)

            # Effectuer une action dans l'environnement
            next_state, reward, done, _, _ = env.step(action)
            next_state = preprocessing(next_state)
            next_state_stack = np.append(state_stack[1:], [next_state], axis=0)

            # Ajouter la transition à la mémoire de replay
            replay_memory.append((state_stack, action, reward, next_state_stack, done))

            # Mettre à jour l'état
            state_stack = next_state_stack
            score += reward

            # Entraîner le modèle
            train_step(device, dqn, optimizer, loss_fn, replay_memory)

        # Réduire epsilon : on exploite un peu plus
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        print("Épisode ", episode + 1, "/", episodes, ", Score: ", score, "Epsilon: ", epsilon)

    env.close()

    # Enregistrement du modèle
    torch.save(dqn.state_dict(), model_path)
    print("Modèle sauvegardé dans", model_path, "avec succès.")



def run_deep_q_learning(model_path):
    env = gym.make("ALE/Assault-v5", render_mode='human')
    num_actions = env.action_space.n
    input_shape = (4, 84, 84)

    dqn = DQN(input_shape, num_actions)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dqn.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    dqn.to(device)
    dqn.eval()

    state, _ = env.reset()
    state = preprocessing(state)
    state_stack = np.stack([state] * 4, axis=0)
    done = False
    score = 0

    while not done:
        action = choose_action(state_stack, 0.05, num_actions, device, dqn)  # Faible epsilon pour exploitation
        next_state, reward, done, _, _ = env.step(action)
        next_state = preprocessing(next_state)
        state_stack = np.append(state_stack[1:], [next_state], axis=0)
        score += reward

    print("Partie terminée. Score obtenu :", score)

    env.close()

if __name__ == "__main__":

    # Hyperparamètres
    gamma = 0.99
    epsilon = 1.0  # Taux d'exploration
    epsilon_min = 0.01  # Minimum pour epsilon
    epsilon_decay = 0.995  # Décroissance d'épsilon
    episodes = 2500  # Nombre d'épisodes d'entraînement
    minibatch_size = 32
    replay_memory_size = 10000
    alpha = 0.0001  # Taux d'apprentissage

    # Entrainement du modèle
    replay_memory = deque(maxlen=replay_memory_size)
    model_path = "../Models/model_deep_q_learning.pt"
    train_deep_q_learning(gamma, epsilon, epsilon_decay, episodes, minibatch_size, replay_memory, alpha,model_path)


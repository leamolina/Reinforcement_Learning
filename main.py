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

# Evaluation du modèle entraîné sur 2 épisodes
def evaluate_policy(env, model, episodes=2):
    model.eval()
    total_score = 0

    for episode in range(episodes):
        state, _ = env.reset()
        state = preprocessing(state)
        state_stack = np.stack([state] * 4, axis=0)
        done = False
        score = 0

        while not done:
            state_tensor = torch.FloatTensor(state_stack).unsqueeze(0).to(device)
            with torch.no_grad():
                action = torch.argmax(model(state_tensor)).item()

            next_state, reward, done, _, _ = env.step(action)
            next_state = preprocessing(next_state)
            state_stack = np.append(state_stack[1:], [next_state], axis=0)
            score += reward

        print(f"Épisode d'évaluation {episode + 1}, Score: {score}")
        total_score += score

    avg_score = total_score / episodes
    print(f"Score moyen sur {episodes} épisodes: {avg_score}")



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

from preprocessing import preprocessing
from DuelingDQN import DuelingDQN


# Evaluation du modèle entraîné sur 2 épisodes
def evaluate_policy(env, model, episodes=2):
    model.eval()
    total_score = 0

    for episode in range(episodes):
        state, _ = env.reset()
        state = preprocessing(state)
        state_stack = np.stack([state] * 4, axis=0)
        done = False
        score = 0

        while not done:
            state_tensor = torch.FloatTensor(state_stack).unsqueeze(0).to(device)
            with torch.no_grad():
                action = torch.argmax(model(state_tensor)).item()

            next_state, reward, done, _, _ = env.step(action)
            next_state = preprocessing(next_state)
            state_stack = np.append(state_stack[1:], [next_state], axis=0)
            score += reward

        print(f"Épisode d'évaluation {episode + 1}, Score: {score}")
        total_score += score

    avg_score = total_score / episodes
    print(f"Score moyen sur {episodes} épisodes: {avg_score}")




env = gym.make("ALE/Assault-v5", render_mode="human")

# Récupération du modèle entraîné et évaluation de ce modèle
input_shape = (4, 84, 84)
num_actions = env.action_space.n
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trained_model = DuelingDQN(input_shape, num_actions).to(device)
# Charger le modèle avec les poids uniquement
trained_model.load_state_dict(torch.load("model_deep_q_learning_double.pth"))
evaluate_policy(env, trained_model, episodes=5)
env.close()
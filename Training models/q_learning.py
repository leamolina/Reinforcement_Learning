import os  # Pour gérer les fichiers et dossiers
import pickle as pkl
import random

import cv2
import gymnasium as gym
import numpy as np

# La table Q est une variable globale
Q = {}

# Prétraitement pour Q-learning
def preprocessing_q_learning(obs, n_bins=10):
    gray_obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    resized_obs = cv2.resize(gray_obs, (84, 84))
    normalized_obs = resized_obs / 255.0
    quantized_obs = (normalized_obs * (n_bins - 1)).astype(int)
    state_hash = hash(quantized_obs.tobytes())
    return state_hash

# Récupération de Q
def get_q_value(state, action, n_actions):
    if state not in Q:
        Q[state] = np.zeros(n_actions)
    return Q[state][action]

# Mise à jour de Q
def update_q_value(state, action, value, n_actions):
    if state not in Q:
        Q[state] = np.zeros(n_actions)
    Q[state][action] = value

# Entraînement Q-learning
def train_q_learning(perf_path, model_path, alpha, gamma, epsilon, epsilon_decay, epsilon_min, episodes):
    env = gym.make("ALE/Assault-v5", render_mode="rgb_array")
    n_actions = env.action_space.n
    global Q

    with open(perf_path, "w")  as file:
        for episode in range(episodes):
            obs, info = env.reset()
            state = preprocessing_q_learning(obs)
            total_reward = 0
            done = False

            while not done:
                action = env.action_space.sample() if random.uniform(0, 1) < epsilon else np.argmax(Q.get(state, np.zeros(n_actions)))
                next_obs, reward, terminated, truncated, info = env.step(action)
                next_state = preprocessing_q_learning(next_obs)
                total_reward += reward
                done = terminated or truncated

                # Mise à jour Q-learning
                current_q = get_q_value(state, action, n_actions)
                max_next_q = np.max(Q.get(next_state, np.zeros(n_actions)))
                new_q = current_q + alpha * (reward + gamma * max_next_q - current_q)
                update_q_value(state, action, new_q, n_actions)

                state = next_state

            epsilon = max(epsilon_min, epsilon * epsilon_decay)

            print("Épisode", episode + 1, "/episodes | Récompense: ", total_reward, " | Epsilon: ", epsilon)
            file.write("Score : ", total_reward, "Epsilon : ", epsilon)


    # Sauvegarde du modèle
    with open(model_path, "wb") as f:
        pkl.dump(Q, f)
    env.close()


# Exécution d'un agent entraîné
def run_q_learning(pickle_path):
    global Q

    # Charger l'environnement
    env = gym.make("ALE/Assault-v5", render_mode="human")
    n_actions = env.action_space.n

    try:
        with open(pickle_path, "rb") as f:
            Q = pkl.load(f)
        print("Modèle chargé avec succès !")
    except FileNotFoundError:
        print("Aucun modèle trouvé, exécution impossible.")
        env.close()
        return

    obs, info = env.reset()
    state = preprocessing_q_learning(obs)
    done = False

    while not done:
        action = np.argmax(Q.get(state, np.zeros(n_actions)))
        next_obs, reward, terminated, truncated, info = env.step(action)
        state = preprocessing_q_learning(next_obs)
        done = terminated or truncated
        env.render()

    env.close()


if __name__ == "__main__":
    # Hyperparamètres
    alpha = 0.1
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01
    episodes = 2000

    # Entraîner le modèle
    perf_path = "./Results/perf_q_learning.txt"
    model_path = "./Models/model_q_learning.pkl"
    train_q_learning(perf_path, model_path, alpha, gamma, epsilon, epsilon_decay, epsilon_min, episodes)

    # Lancer l'agent entraîné
    run_q_learning(model_path)

import gymnasium as gym
import numpy as np
import random
import ale_py
import cv2

# Charger l'environnement
gym.register_envs(ale_py)
env = gym.make("ALE/Assault-v5", render_mode="human")

# Hyperparamètres
alpha = 0.1  # Taux d'apprentissage
gamma = 0.99  # Facteur de réduction
epsilon = 1.0  # Taux d'exploration
epsilon_decay = 0.995  # Décroissance d'épsilon
epsilon_min = 0.01  # Minimum pour epsilon
episodes = 5000  # Nombre d'épisodes d'entraînement


# Modification de la discrétisation
def discretize(obs, n_bins=10):
    # Convertir l'image en niveaux de gris
    gray_obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)

    # Réduire la résolution de l'image
    resized_obs = cv2.resize(gray_obs, (84, 84))

    # Normaliser et quantifier l'image
    normalized_obs = resized_obs / 255.0
    quantized_obs = (normalized_obs * (n_bins - 1)).astype(int)

    # Calculer un hash unique basé sur quelques caractéristiques de l'image
    state_hash = hash(quantized_obs.tobytes())

    return state_hash


# Initialiser la table Q
n_actions = env.action_space.n
Q = {}


def get_q_value(state, action):
    if state not in Q:
        Q[state] = np.zeros(n_actions)
    return Q[state][action]


def update_q_value(state, action, value):
    if state not in Q:
        Q[state] = np.zeros(n_actions)
    Q[state][action] = value


# Boucle d'entraînement
for episode in range(episodes):
    obs, info = env.reset()
    state = discretize(obs)
    total_reward = 0
    done = False
    steps = 0

    while not done:
        # Épsilon-greedy pour choisir une action
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Exploration
        else:
            # Choisir l'action avec la meilleure valeur Q
            action = np.argmax(Q.get(state, np.zeros(n_actions)))

        # Effectuer l'action et observer la récompense
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_state = discretize(next_obs)
        total_reward += reward
        done = terminated or truncated

        # Mise à jour de Q-learning
        current_q = get_q_value(state, action)
        max_next_q = np.max(Q.get(next_state, np.zeros(n_actions)))
        new_q = current_q + alpha * (reward + gamma * max_next_q - current_q)

        # Mettre à jour la valeur Q
        update_q_value(state, action, new_q)

        # Passer à l'état suivant
        state = next_state
        steps += 1

        # Arrêter si trop longtemps
        if steps > 1000:
            break

    # Décroissance d'épsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Affichage des progrès
    print(f"Épisode {episode + 1}/{episodes} | Récompense totale: {total_reward} | Epsilon: {epsilon:.2f}")

    # Sauvegarder périodiquement
    if (episode + 1) % 100 == 0:
        print(f"Sauvegarde de la table Q après {episode + 1} épisodes")

env.close()
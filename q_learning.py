import gymnasium as gym
import numpy as np
import random
import ale_py
import cv2
import pickle as pkl

# Charger l'environnement
gym.register_envs(ale_py)
env = gym.make("ALE/Assault-v5", render_mode="human")

# Hyperparamètres
alpha = 0.1  # Taux d'apprentissage
gamma = 0.99  # Facteur de réduction
epsilon = 1.0  # Epsilon de départ
epsilon_decay = 0.995  # Décroissance d'épsilon
epsilon_min = 0.01  # Minimum pour epsilon
episodes = 1000  # Nombre d'épisodes


# Modification de la discrétisation
def preprocessing_q_learning(obs, n_bins=10):
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


# Récupération de Q
def get_q_value(state, action):
    if state not in Q:
        Q[state] = np.zeros(n_actions)
    return Q[state][action]


# Modification de Q
def update_q_value(state, action, value):
    if state not in Q:
        Q[state] = np.zeros(n_actions)
    Q[state][action] = value


def train_q_learning():

    # Boucle d'entraînement
    for episode in range(episodes):
        obs, info = env.reset()
        state = preprocessing_q_learning(obs)
        total_reward = 0
        done = False
        steps = 0

        while not done:

            # Épsilon-greedy pour choisir une action
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Exploration
            else:
                # Choisir l'action avec la meilleure valeur Q (exploitation)
                action = np.argmax(Q.get(state, np.zeros(n_actions)))

            # On effectue l'action
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = preprocessing_q_learning(next_obs)
            total_reward += reward
            done = terminated or truncated

            # Mise à jour de Q-learning
            current_q = get_q_value(state, action)
            max_next_q = np.max(Q.get(next_state, np.zeros(n_actions)))
            new_q = current_q + alpha * (reward + gamma * max_next_q - current_q)

            # Mettre à jour la valeur de Q
            update_q_value(state, action, new_q)

            # On passe à l'état suivant
            state = next_state
            steps += 1


        # Décroissance d'épsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Affichage des progrès
        print(f"Épisode {episode + 1}/{episodes} | Récompense totale: {total_reward} | Epsilon: {epsilon:.2f}")

    # A la fin, charger le modèle entraîné dans un pickle
    with open("q_table.pkl", "wb") as f:
        pkl.dump(Q, f)

def run_q_learning(pickle_path):
    Q = {}
    # Récupération de la table dans le pickle
    try:
        with open("q_table.pkl", "rb") as f:
            Q = pkl.load(f)
        print("Modèle chargé avec succès !")
    except FileNotFoundError:
        print("Aucun modèle trouvé")

    obs, info = env.reset()
    state = preprocessing_q_learning(obs)
    done = False

    while not done:
        action = np.argmax(Q.get(state, np.zeros(n_actions)))  # Exploitation
        next_obs, reward, terminated, truncated, info = env.step(action)
        state = preprocessing_q_learning(next_obs)
        done = terminated or truncated
        env.render()


env.close()

if __name__ == "__main__":
    train_q_learning()
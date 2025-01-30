import matplotlib.pyplot as plt
import numpy as np

# Nom du fichier contenant les données
fichier = "Results/perf_q_learning.txt"

# Initialisation de la liste pour stocker les scores
rewards = []

# Lecture du fichier et extraction des scores
with open(fichier, "r", encoding="utf-8") as file:
    for line in file:
        if "Score :" in line:
            # Extraction du score en utilisant un découpage de la ligne
            parts = line.split(",")  # On sépare par des virgules
            for part in parts:
                if "Score :" in part:
                    # On récupère le score après "Score:"
                    score = float(part.split(":")[1].strip())
                    rewards.append(score)



# Générer les indices des épisodes
episodes = list(range(1, len(rewards) + 1))

# Calcul de la récompense moyenne glissante (sur une fenêtre de 50 épisodes)
window = 50
average_rewards = [np.mean(rewards[max(0, i - window):i + 1]) for i in range(len(rewards))]

# Tracer le graphique
plt.figure(figsize=(12, 6))
plt.plot(episodes, rewards, label="Récompense par épisode", alpha=0.5, color="blue")
plt.plot(episodes, average_rewards, label="Récompense moyenne (fenêtre=50)", color="orange", linewidth=2)
plt.title("Performance de l'algorithme DQN sur Assault Atari")
plt.xlabel("Épisodes")
plt.ylabel("Récompense")

plt.legend()
plt.grid()
plt.show()

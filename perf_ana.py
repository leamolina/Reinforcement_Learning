import numpy as np
import matplotlib.pyplot as plt

# Nom du fichier contenant les données
fichier = "data_ana.txt"

# Liste pour stocker les rewards
rewards = []

# Lecture et extraction des rewards
with open(fichier, "r", encoding="utf-8") as file:
    for line in file:
        if "Reward:" in line:
            # Extraction de la récompense après "Reward:"
            try:
                reward = float(line.split("Reward:")[1].strip())
                rewards.append(reward)
            except ValueError:
                print(f"Erreur lors de l'extraction de la ligne : {line.strip()}")



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


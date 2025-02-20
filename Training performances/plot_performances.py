import matplotlib.pyplot as plt
import numpy as np
import re  # Importation pour la gestion des expressions régulières

# Nom du fichier contenant les données
fichier = "./Training performances/perf_deep_q_learning_double_3000ep.txt"

# Initialisation de la liste pour stocker les scores
rewards = []

# Lecture du fichier et extraction des scores
with open(fichier, "r", encoding="utf-8") as file:
    for line in file:
        if "Score :" in line:
            try:
                # Nettoyage de la ligne pour éviter les erreurs
                line = line.strip().replace(",", ".")  # Remplace les virgules par des points si nécessaire

                # Utilisation d'une expression régulière pour extraire correctement le score
                match = re.search(r"Score\s*:\s*([\d.]+)", line)  # Capture le nombre après "Score :"
                if match:
                    reward_str = match.group(1).strip().rstrip(".")  # Supprime un éventuel point final
                    reward = float(reward_str)  # Convertir proprement en float
                    rewards.append(reward)
                else:
                    print("Erreur sur la ligne", line)

            except ValueError as e:
                print("Erreur de conversion en float")

# Vérifier si des scores ont été récupérés
# Générer les indices des épisodes
episodes = list(range(1, len(rewards) + 1))

# Calcul de la récompense moyenne sur une fenêtre de 50 épisodes
window = 50
average_rewards = [np.mean(rewards[max(0, i - window):i + 1]) for i in range(len(rewards))]

# Tracer le graphique
plt.figure(figsize=(12, 6))
plt.plot(episodes, rewards, label="Récompense par épisode", alpha=0.5, color="blue")
plt.plot(episodes, average_rewards, label="Récompense moyenne (fenêtre=50)", color="orange", linewidth=2)
plt.title("Performance de l'algorithme Dueling Double DQN sur Assault Atari")
plt.xlabel("Épisodes")
plt.ylabel("Récompense")

plt.legend()
plt.grid()
plt.show()

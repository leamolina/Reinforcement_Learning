# Projet d'apprentissage par renforcement

### Introduction

Ce projet contient plusieurs modèles d'apprentissage par renforcement pour entraîner des agents sur le jeu Assault. Les modèles entraînés sont :
- **Q-Learning**
- **Deep Q-Learning**
- **BBF**
- **Deep Q-Learning Double**


Avant d'exécuter ce projet, vous devez installer les dépendances nécessaires. Assurez-vous d'avoir Python installé sur votre machine. Pour installer les bibliothèques nécessaires, suivez les étapes suivante :

### Étapes d'installation

1. Téléchargez le projet sur votre machine.

2. Installez les dépendances :
   
   Vous devez installer les packages nécessaires avec `pip`. Pour ce faire : 
   - Ouvrez le terminal
   - À l'aide de la commande `cd`, allez sur le répertoire du projet
   - Exécutez la commande suivante dans le terminal : `pip install -r requirements.txt`



### Utilisation

Une fois que toutes les dépendances sont installées, vous pouvez exécuter le script principal pour choisir un modèle à utiliser. Les modèles ont déjà été préalablement entraînés et stockés, donc le but du code est uniquement de faire jouer l'agent en utilisant ces modèles déjà entraînés. Dans le terminal, exécutez le fichier Python contenant le code principal (`main.py`) à l'aide de la commande suivante : `python main.py`. 
Le script vous demandera de choisir un modèle parmi les options suivantes : Q-Learning (1), Deep Q-Learning (2), BBF (3), Deep Q-Learning Double (4).
Il faudra entrer un entier entre 1 et 4 pour sélectionner le modèle que vous souhaitez utiliser.


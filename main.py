from Training_models.bbf import run_bbf
from Training_models.deep_q_learning import run_deep_q_learning
from Training_models.deep_q_learning_double import run_deep_q_learning_double
from Training_models.q_learning import run_q_learning

if __name__ == "__main__":

    print ("Plusieurs modèles sont disponibles : ")
    print("1 : Q-Learning")
    print("2 : Deep Q-Learning")
    print("3 : BBF")
    print("4 : Deep Q-Learning Double")

    while True:

        try:
            modele_choisi = int(input("Quel modèle souhaitez-vous utiliser ? (choisir un entier entre 1 et 4) : "))

            # Q-Learning
            if modele_choisi == 1:
                run_q_learning("Models/model_q_learning.pkl")
                break

            # Deep Q-Learning
            elif modele_choisi == 2:
                run_deep_q_learning("Models/model_deep_q_learning.pt")
                break

            # BBF
            elif modele_choisi == 3:
                run_bbf("Models/model_bbf.pth")
                break

            # Deep Q-Learning Double
            elif modele_choisi == 4:
                run_deep_q_learning_double("Models/model_deep_q_learning_double_4500ep.pth")
                break

            else:
                print("Veuillez entrer un nombre entre 1 et 4.")


        except ValueError:
            print("Ce n'est pas un entier valide. Essayez encore.")
from math import *  # pour utiliser les fonctions math dans eval()
import pandas as pd
import matplotlib.pyplot as plt
from naive import Naive
from sarima import SARIMA
from lstm import LSTM
import numpy as np
from statsmodels.tsa.stattools import adfuller
from fonctions import enregistrer_plot

entrainement = 0.6  # 60% du jeu de données sert à l'entraînement
test = 0.2          # 20% du jeu de données sert aux tests (20 derniers % pour la validation)

class Serie:
    data = []       # Pandas dataframe
    modeles = []

    def __init__(self, nom, csv_file=None, formule=None, std=None):
        points = 500

        self.nom = nom
        self.nom_sauvegarde = nom.replace(" ", "_").lower().replace("é","e")
        self.modeles = [Naive(), SARIMA()]

        # charge fichier CSV
        if csv_file:
            self.data = pd.read_csv("datasets/" + csv_file)


        # créé des données à partir d'une formule
        elif formule:
            Y = []
            X = []
            for x in range(0, points):
                X.append(x)
                Y.append(float(eval(formule)))

            self.data = pd.DataFrame(Y, index=X,
                                     columns=['Série'])


            # ajout de bruit blanc (centré)
            mu, sigma = 0, std # moyenne et écart type du bruit
            self.data['Bruit'] = pd.DataFrame(
                np.random.normal(mu, sigma, size=points))

            self.data['Série'] = self.data['Série'] + \
                self.data['Bruit']

        self.stationnariser()

        self.index_fin_entrainement = floor(len(self.data) * entrainement)
        self.index_fin_test = self.index_fin_entrainement + floor(len(self.data) * test)

        # Découpage de la série en entrainement/test/validation
        self.data['Entraînement'] = self.data['Série'][:self.index_fin_entrainement]
        self.data['Test'] = self.data['Série'][self.index_fin_entrainement:self.index_fin_test]
        self.data['Validation'] = self.data['Série'][self.index_fin_test:]
    
    def stationnariser(self):
        """
            Permet de stationnariser la série
        """

        # Une différence première suffit
        self.data['Série stationnarisée'] = self.data['Série'].diff(periods=1)

        # Test de stationnarité par Duckey-Fuller augmenté
        result = adfuller(self.data['Série stationnarisée'].dropna().values)
        adf = result[0]
        seuil_5 = result[4].get("5%")
        # print(result)
  

        if adf > seuil_5:
            # Ne devrait pas arriver
            print("La série " + self.nom + " n'est pas stationnaire")
            raise Exception("Série non stationnaire. Arrêt du benchmark.")

        else:
            print("La série " + self.nom + " est stationnaire")


    def calculer_indicateurs(self):
        self.longueur = len(self.data)

    def graphique_prevision_one_step_ahead(self):
        """
            Montre la série complète avec les prévisions des modèles
        """

        donnees = pd.DataFrame(self.data['Série'])
        legende = list()
        legende.append("Série")

        for modele in self.modeles:
            donnees[modele.__class__.__name__] = modele.serie.data[modele.__class__.__name__]
            legende.append("Prévision " + str(modele.__class__.__name__))

        # ratio d'affichage carré 4:10 (en inch)
        plt.figure(figsize=(10, 4))
        
        previsions = 20 # 20 dernières prévisions
        plt.plot(donnees.tail(previsions))  
        plt.xticks(np.arange(len(donnees)-previsions, len(donnees), step=1))
        plt.gca().legend(legende)
        plt.title(str(previsions) + " dernières prévisions des modèles sur " + self.nom)
        plt.ylabel("Y")
        plt.xlabel("X")

        enregistrer_plot(plt, self.nom_sauvegarde + '_previsions_serie.pdf')

        plt.show()


    def graphique_prevision_dynamique(self):
        """
            Affiche la série prévue de façon dynamique pour les modèles
            (valeur prévue t = modele(valeur prévue par modele t-1)
        """

        donnees = pd.DataFrame(self.data['Série'])
        legende = list()
        legende.append("Série")

        for modele in self.modeles:
            donnees[modele.__class__.__name__] = modele.serie.data[modele.__class__.__name__ + "_dynamique"]
            legende.append("Prévision dynamique " + str(modele.__class__.__name__))

        # ratio d'affichage carré 4:10 (en inch)
        plt.figure(figsize=(10, 4))

        # Prévisions sur le jeu de test + validation
        previsions = int((1 - entrainement) * len(self.data))
        plt.plot(donnees.tail(previsions))
        # plt.xticks(np.arange(len(donnees)-previsions, len(donnees), step=1))
        plt.gca().legend(legende)
        plt.title("Prévisions dynamiques sur l'ensemble test + validation des modèles sur " + self.nom)
        plt.ylabel("Y")
        plt.xlabel("X")

        enregistrer_plot(plt, self.nom_sauvegarde + '_previsions_dynamique_serie.pdf')
        plt.show()

    def graphique_sous_serie(self):
        """
            Affiche uniquement la série, décomposée en 3 sous-séries
        """

        plt.figure(figsize=(10, 4))
        plt.plot(self.data['Entraînement'])
        plt.plot(self.data['Test'])
        plt.plot(self.data['Validation'])

        plt.gca().legend()
        plt.title(self.nom)
        plt.ylabel("Y")
        plt.xlabel("X")

        enregistrer_plot(plt, self.nom_sauvegarde + '_presentation_serie.pdf')

        plt.show()

    def graphique_serie_stationnarisee(self):
        """
            Affiche uniquement la série stationnarisée avec la distribution
            des résidus et le test de Dickey Fuller augmenté
        """

        fig = plt.figure(figsize=(10,10)) # ratio d'affichage carré 1:1 (en inch)


        ax1 = fig.add_subplot(2, 1, 1)
        ax1.set_title('Série stationnarisée')
        ax1.plot(self.data['Série stationnarisée'])


        ax2 = fig.add_subplot(2,2,3)
        ax2.set_title('Distribution de la série stationnarisée')
        plt.hist(self.data['Série stationnarisée'].dropna(), bins=30)

        ax3 = fig.add_subplot(2,2,4)
        ax3.set_title('Test de Dickey-Fuller augmenté', weight='bold')
        ax3.set_axis_off()

        result = adfuller(self.data['Série stationnarisée'].dropna())

        ax3.text(0, 0.9, 'Valeur ADF : %f' % result[0], fontsize=12)
        ax3.text(0, 0.7, 'p-value : %f' % result[1], fontsize=12)
        ax3.text(0, 0.5, 'Valeurs critiques', fontsize=12)
        y = 0.4
        for key, value in result[4].items():
            ax3.text(0, y, '%s : %.3f' % (key, value), fontsize=11)
            y -= 0.1

        enregistrer_plot(plt, self.nom_sauvegarde +
                         '_stationnarisation_serie.pdf')
        plt.show()
    

    def recapitulatif(self):
        """
            Affiche le resultat de chaque modèle sur la série
        """
        self.calculer_indicateurs()

        print("\n")
        print("======================================")
        print("====       " + self.nom  + "       ===")
        print("| Observations : " + str(self.longueur))
        for modele in self.modeles:
            modele.recapitulatif()
            modele.description_modele()
        print("======================================")

        self.graphique_sous_serie()
        self.graphique_serie_stationnarisee()
        self.graphique_prevision_one_step_ahead() # valeur prévue t = modele(valeur donnée t-1)
        self.graphique_prevision_dynamique()  # valeur prévue t = modele(valeur prévue par modele t-1)




        



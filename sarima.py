from modele import Modele
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt
from pandas import DataFrame

class SARIMA(Modele):
    """
        Cette classe implémente le modèle ARMA avec différenciation intégrée
        et gestion d'une saisonnalité simple (SARIMA).
    """

    def __init__(self):
        pass

    def trouver_hyperparametres(self):
        try:
            import joblib
            print("Entrainement SARIMA en parallèle.")
            parallel = True

        except ImportError:
            print("Il est recommandé d'installer joblib (pip3 install joblib)\
                pour profiter d'une amélioration des performances.")
            parallel = False

        config_list = list()
        # grid search
        for p in range(0, 3):
            for d in [0, 1]:
                for q in range(0, 3):
                    for P in range(0, 3):
                        for D in [0, 1]:
                            for Q in range(0, 3):
                                for m in [0,12]:
                                    for t in ['n', 'c', 't', 'ct']:
                                        config = ((p, d, q), (P, D, Q, m), t)
                                        config_list.append(config)
        config_list.clear()
        config_list.append(((2, 0, 1), (1, 0, 0, 12), 't'))
        config_list.append(((2, 0, 1), (1, 0, 0, 12), 't'))
        config_list.append(((2, 0, 1), (1, 0, 0, 12), 't'))
        config_list.append(((2, 0, 1), (1, 0, 0, 12), 't'))
        config_list.append(((2, 0, 1), (1, 0, 0, 12), 't'))
        print("Taille des combinaisons SARIMA : " + str(len(config_list)))

        if parallel:
            executor = joblib.Parallel(joblib.cpu_count(
            ), backend='multiprocessing', verbose=50, batch_size='auto')
            tasks = (joblib.delayed(self.fit_modele)(config)
                    for config in config_list)
        
            scores = executor(tasks)
        
        else:
            scores = [self.fit_modele(config) for config in config_list]

        # Enleve les scores vides
        scores = [r for r in scores if r[1] != None]

        # Trie par les configurations par erreur
        scores.sort(key=lambda tup: tup[1])

        # Affiche les 3 meilleures configurations
        print("Les 3 meilleurs configurations SARIMA sont : ")
        print(scores[0:3])

        self.config = eval(scores[0][0])


    def fit_modele(self, config, final=False):
        """
            Fait un fit d'un modèle SARIMA selon les paramètres contenus dans
            config (p,d,q,P,D,Q,s,t)
        """
        resultat = None
        key = str(config)

        print("Fit du modèle " + key + " en cours...")

        try:
            # Prédiction sur jeu de test + validation
            serie_predite = []

            # walk-forward validation
            longueur_test = len(self.serie.data['Test'].dropna())
            if final:
                longueur_test += len(self.serie.data['Validation'].dropna())

            for i in range(0, longueur_test):
                modele = SARIMAX(self.serie.data['Série'][0:self.serie.index_fin_entrainement+i],
                                order=config[0],
                                seasonal_order=config[1],
                                trend=config[2],
                                simple_differencing=False)

                modele_fit = modele.fit(disp=False)

                if modele_fit.mle_retvals['converged'] == False:
                    print("     > Convergence non atteinte pour " + str(config))
                    return (key, resultat)

                valeur_prevue = modele_fit.forecast()

                serie_predite.append(valeur_prevue.values[0])

            if not final:
                resultat = mean_squared_error(
                    serie_predite[0:len(self.serie.data['Test'].dropna())], self.serie.data['Test'].dropna())
            
            else:
                # on conserve le modele pour obtenir d'autres infos plus tard
                self.modele_fit = modele_fit

        except KeyboardInterrupt:
            print("Arrêt...")

        except:
            print("     > Configuration impossible : " + str(config))
        
        if final:
            # ajout d'un padding avec des nan
            a = np.empty(
                (1, len(self.serie.data['Entraînement'].dropna())))
            a[:] = np.nan
            serie_predite = np.concatenate((a[0], np.array(serie_predite)), axis=0)
            return serie_predite

        return (key, resultat)

    def fit(self):
        print("Configuration SARIMA retenue : " + str(self.config))
        self.serie.data[self.__class__.__name__] = self.fit_modele(self.config, True)

        print(self.modele_fit.summary())

        residuals = DataFrame(self.modele_fit.resid)
        print(residuals.describe())
    
    def description_modele(self):

        # affichage de la distribution des résidus
        plt.figure(figsize=(10, 4))
        plt.title("Résidus issus du modèle")
        plt.plot(self.modele_fit.resid)
        plt.show()

        plt.figure(figsize=(10, 4))
        plt.title("Distribution des résidus")
        plt.hist(self.modele_fit.resid, bins=30)
        plt.show()



from math import *
from modele import Modele
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
import pmdarima as pm
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt
from pandas import DataFrame
from fonctions import enregistrer_plot

class SARIMA(Modele):
    """
        Cette classe implémente le modèle ARMA avec différenciation intégrée
        et gestion d'une saisonnalité simple (SARIMA).
    """

    def __init__(self):
        pass

    def trouver_hyperparametres(self):
        # détermination de la fréquence de la saisonnalité

        # difference premiere
        t = np.diff(self.serie.data['Série'][0:self.serie.index_fin_entrainement], n=1)

        w = np.fft.fft(t)   # fft

        # on cherche les périodes de saisonnalité < 30 (et pas la premiere)
        tableau = np.absolute(w)[1:-(int(0.1*len(t)))]

        if max(tableau) < np.mean(tableau) + 4 * np.std(tableau):
            frequence = 1

        else:
            # on rajoute la premiere période
            repetition = max(range(len(tableau)), key=tableau.__getitem__) + 1
            frequence = round((len(t)+1)/repetition)  # +1 car différenciée une fois

        print("Période détectée de la saisonnalité : " + str(frequence))

        # détermination des ordres p, d, q, P, D, Q sur l'ensemble de train
        stepwise_fit = pm.auto_arima(self.serie.data['Série'][0:self.serie.index_fin_test],
                                     start_p=2, d=None, start_q=2, max_p=5, max_d=4, max_q=5,
                                     start_P=2, D=None, start_Q=1, max_P=5, max_D=4, max_Q=5,
                                     m=frequence, seasonal=True, seasonal_test='ocsb', trace=True,
                                     error_action='ignore',
                                     suppress_warnings=True,
                                     stepwise=True,
                                     out_of_sample_size=self.serie.index_fin_test-self.serie.index_fin_entrainement,
                                     scoring='mse',
                                     trend=None,
                                     with_intercept=False)

        print(stepwise_fit.summary())
        print(stepwise_fit.get_params())

        fig = stepwise_fit.plot_diagnostics(figsize=(10, 10), lags=12)
        fig.axes[0].set_title("Résidus normalisés")
        fig.axes[1].set_title("Comparaison de la densité estimée avec N(0,1)")
        fig.axes[2].set_title("Diagramme Quantile-Quantile")
        fig.axes[2].set_xlabel("Quantiles théoriques")
        fig.axes[2].set_ylabel("Quantiles échantillonnés")
        fig.axes[3].set_title("Corrélogramme")
        enregistrer_plot(plt, self.nom_sauvegarde + '_diagnostic.pdf')
        plt.show()
        
        params = stepwise_fit.get_params()
        ordres = params['order']
        ordres_saisonniers = params['seasonal_order']
    
        config = (ordres, ordres_saisonniers, None)
        self.config = config


    def fit_modele(self, config):
        """
            Fait un fit d'un modèle SARIMA selon les paramètres contenus dans
            config (p,d,q,P,D,Q,s,t)
        """

        key = str(config)

        print("Fit du modèle " + key + " en cours...")

        # Prédiction sur jeu de test + validation
        # walk-forward validation
        longueur_test = len(self.serie.data['Test'].dropna())
        longueur_test += len(self.serie.data['Validation'].dropna())

        # Prévisions one step ahead 
        serie_predite = []
        for i in range(longueur_test):
            print("Prévision one-step ahead : " + str(i) + " sur " + str(longueur_test))
            modele = SARIMAX(self.serie.data['Série'][0:self.serie.index_fin_entrainement+i],
                order=config[0],
                seasonal_order=config[1],
                trend=config[2],
                simple_differencing=False)
            try:
                modele_fit = modele.fit(disp=False)
                yhat = modele_fit.forecast(steps=1).values[0]
                
            except KeyboardInterrupt:
                break

            except:
                print("t+1 = t")

            serie_predite.append(yhat)

        # Prévisions dynamiques
        modele = SARIMAX(self.serie.data['Série'][0:self.serie.index_fin_entrainement],
                        order=config[0],
                        seasonal_order=config[1],
                        trend=config[2],
                        simple_differencing=False)

        modele_fit = modele.fit(disp=False)

        if modele_fit.mle_retvals['converged'] == False:
            print("     > Convergence non atteinte pour " + str(config))

        serie_predite_dynamique = modele_fit.predict(start=self.serie.index_fin_entrainement, end=len(self.serie.data['Série'])-1, dynamic=True)

        # on conserve le modele pour obtenir d'autres infos plus tard
        self.modele_fit = modele_fit
        
    
        # ajout d'un padding avec des nan
        a = np.empty(
            (1, len(self.serie.data['Entraînement'].dropna())))
        a[:] = np.nan
        serie_predite = np.concatenate((a[0], np.array(serie_predite)), axis=0)
        serie_predite_dynamique = np.concatenate((a[0], np.array(serie_predite_dynamique)), axis=0)

        self.serie.data[self.__class__.__name__] = serie_predite
        self.serie.data[self.__class__.__name__ + "_dynamique"] = serie_predite_dynamique


    def fit(self):
        print("Configuration SARIMA retenue : " + str(self.config))

        self.fit_modele(self.config)

        print(self.modele_fit.summary())

        residuals = DataFrame(self.modele_fit.resid)
        print(residuals.describe())
    
    def description_modele(self):
        # déjà décrit à la fin de la recherche des hyper paramètres
        pass



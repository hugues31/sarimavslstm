from keras.models import Sequential
from keras.layers import LSTM as kLSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras import optimizers
from fonctions import MSE

import numpy as np

from modele import Modele
from fonctions import decouper_serie_apprentissage_supervise



class LSTM(Modele):
    """
        Implémentation d'un réseau LSTM via Keras, entrainable sur CPU ou GPU
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
        for h in range(1,4):
            for i in [50, 20, 100]:
                for n in [10, 50, 200]:
                    for f in ['tanh', 'relu', 'sigmoid']:
                        for d in [0.0, 0.2]:
                            config = {
                                'nbre_couches': h,
                                'taille_entree': i,
                                'nbre_neurones': n,
                                'activation': f,
                                'dropout': d
                            }
                            config_list.append(config)

        print("Taille des combinaisons LSTM : " + str(len(config_list)))

        if parallel:
            executor = joblib.Parallel(joblib.cpu_count(
            ), backend='multiprocessing', verbose=50, batch_size='auto')
            tasks = (joblib.delayed(self.fit_modele)(config)
                    for config in config_list[0:1])
        
            scores = executor(tasks)
        
        else:
            scores = [self.fit_modele(config) for config in config_list]

        # Enleve les scores vides
        scores = [r for r in scores if r[1] != None]

        # Trie par les configurations par erreur
        scores.sort(key=lambda tup: tup[1])

        # Affiche les 3 meilleures configurations
        print("Les 3 meilleurs configurations LSTM sont : ")
        print(scores[0:3])

        self.config = eval(scores[0][0])
    
    def fit_modele(self, config, final=False):
        """
            Fait un fit rapide (20 itérations par défaut) d'un LSTM selon les
            paramètres contenus dans config (h, n, f, d)
        """
        resultat = None
        key = str(config)

        print("Fit du modèle " + key + " en cours...")

        iter = 80 if final else 20 # entraînement final du modèle retenu

        nbre_couches = config.get("nbre_couches")
        taille = config.get("taille_entree")
        nbre_neurones = config.get("nbre_neurones")
        activation = config.get("activation")
        dropout = config.get("dropout")

        # TODO: MinMaxScaler

        X_train, y_train = decouper_serie_apprentissage_supervise(
            self.serie.data['Série stationnarisée'][0:self.serie.index_fin_entrainement].dropna().values, taille)

        n_features = 1  # une variable explicative
        X_train = X_train.reshape(
            (X_train.shape[0], X_train.shape[1], n_features))

        # Création du réseau de neurones
        model = Sequential()

        # couche d'entrée
        model.add(kLSTM(nbre_neurones, activation=activation, return_sequences=True,
                            input_shape=(taille, n_features)))
        model.add(Dropout(dropout))  # ajout d'un dropout

        # couches cachées
        for i in range(0, nbre_couches):
            model.add(kLSTM(nbre_neurones, activation=activation))
            model.add(Dropout(dropout))  # ajout d'un dropout

        # couche de sortie (1 dimension)
        model.add(Dense(1))

        methode_optimisation = optimizers.Nadam()

        model.compile(optimizer=methode_optimisation, loss='mse')

        nbre_retards = np.count_nonzero(np.isnan(self.serie.data['Série stationnarisée']))

        # Fit du modèle
        model.fit(X_train, y_train, epochs=iter, verbose=final)

        # Prédiction sur jeu de test + validation
        serie_predite = []
        # walk-forward validation
        for i in range(0, len(self.serie.data['Test'].dropna())+len(self.serie.data['Validation'].dropna())):

            x_input = self.serie.data['Série stationnarisée'][self.serie.index_fin_entrainement -
                                                              taille+i:self.serie.index_fin_entrainement+i].values

            x_input = x_input.reshape((1, taille, n_features))
            yhat = model.predict(x_input, verbose=0)[0][0]

            # déstationnarisation
            yhat = yhat + \
                self.serie.data['Série'][self.serie.index_fin_entrainement+i-nbre_retards]
            serie_predite.append(yhat)

        # ajout d'un padding avec des nan
        a = np.empty((1, len(self.serie.data['Entraînement'].dropna())))
        a[:] = np.nan
        serie_predite = np.concatenate((a[0], np.array(serie_predite)), axis=0)

        resultat = MSE(serie_predite[self.serie.index_fin_entrainement:self.serie.index_fin_test], self.serie.data['Test'].dropna())

        if final:
            self.serie.data[self.__class__.__name__] = serie_predite
        
        return (key, resultat)

    def fit(self):
        # Création d'un jeu de données d'apprentissage supervisé
        print("Configuration LSTM retenue : " + str(self.config))

        print("Entraînement du modèle retenu...")
        self.fit_modele(self.config, True)
       



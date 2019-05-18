import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM as kLSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras import optimizers
from sklearn.metrics import mean_squared_error
from keras.callbacks import EarlyStopping
from keras.utils import plot_model

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
            print("Entrainement LSTM en parallèle.")
            parallel = True

        except ImportError:
            print("Il est recommandé d'installer joblib (pip3 install joblib)\
                pour profiter d'une amélioration des performances.")
            parallel = False
        parallel = False
        config_list = list()
        for h in range(1, 4):
            for i in [20, 20, 50]:
                for n in [20, 50, 200]:
                    for f in ['tanh', 'relu', 'sigmoid']:
                        for d in [0.1, 0.2]:
                            config = {
                                'nbre_couches': h,
                                'taille_entree': i,
                                'nbre_neurones': n,
                                'activation': f,
                                'dropout': d
                            }
                            config_list.append(config)

        config_list = config_list[:1]   # debug
        print("Taille des combinaisons LSTM : " + str(len(config_list)))

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

        # Trie les configurations par erreur
        scores.sort(key=lambda tup: tup[1])

        # Affiche les 3 meilleures configurations
        print("Les 3 meilleurs configurations LSTM sont : ")
        print(scores[0:3])

        self.config = eval(scores[0][0])

    def fit_modele(self, config, final=False):
        """
            Fait un fit rapide (50 itérations par défaut) d'un LSTM selon les
            paramètres contenus dans config (h, n, f, d)
        """
        resultat = None
        key = str(config)

        iter = 5 if final else 1  # entraînement final du modèle retenu

        nbre_couches = config.get("nbre_couches")
        taille = config.get("taille_entree")
        nbre_neurones = config.get("nbre_neurones")
        activation = config.get("activation")
        dropout = config.get("dropout")

        nbre_retards = np.count_nonzero(
            np.isnan(self.serie.data['Série stationnarisée']))

        # MinMaxScaler
        donnees_brutes = self.serie.data['Série stationnarisée'][0:self.serie.index_fin_entrainement].dropna(
        ).values
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler = scaler.fit(np.array(donnees_brutes).reshape(-1, 1))
        serie_reduite = scaler.transform(np.array(
            self.serie.data['Série stationnarisée'].dropna().values).reshape(-1, 1))
        a = np.empty((1, nbre_retards))
        a[:] = np.nan
        serie_reduite = np.concatenate((a, np.array(serie_reduite)), axis=0)

        self.serie.data['Série stationnarisée réduite'] = serie_reduite

        X_train, y_train = decouper_serie_apprentissage_supervise(
            self.serie.data['Série stationnarisée réduite'][0:self.serie.index_fin_entrainement].dropna().values, taille)

        X_test, y_test = decouper_serie_apprentissage_supervise(
            self.serie.data['Série stationnarisée réduite'][self.serie.index_fin_entrainement:self.serie.index_fin_test].dropna().values, taille)

        n_features = 1  # une variable explicative
        X_train = X_train.reshape(
            (X_train.shape[0], X_train.shape[1], n_features))
        X_test = X_test.reshape(
            (X_test.shape[0], X_test.shape[1], n_features))

        # Création du réseau de neurones
        model = Sequential()

        # couche d'entrée
        for i in range(0, nbre_couches):
            model.add(kLSTM(nbre_neurones, activation=activation, return_sequences=True,
                            input_shape=(taille, n_features)))
            model.add(Dropout(dropout))  # ajout d'un dropout

        # dernière couche (pas de retour)
        model.add(kLSTM(nbre_neurones, activation=activation))
        model.add(Dropout(dropout))  # ajout d'un dropout

        # couche de sortie (1 dimension)
        model.add(Dense(1))

        methode_optimisation = optimizers.Nadam()

        model.compile(optimizer=methode_optimisation,
                      loss='mse')

        # Critère d'arret prématuré, aucune amélioration sur le jeu de test
        # pendant plus de 30 itérations
        critere_stop = EarlyStopping(
            monitor='val_loss', min_delta=0, patience=30)

        # Fit du modèle
        historique = model.fit(
            X_train, y_train, validation_data=(X_test, y_test), epochs=iter, verbose=True, callbacks=[critere_stop], shuffle=False)

        if final:  # sauvegarde de l'historique d'entrainement si modèle final
            self.historique = historique

        # Prédiction sur jeu de test + validation
        serie_predite = []
        serie_predite_temp = []  # stock les prédictions réduites
        serie_predite_dynamique = []
        # walk-forward validation
        for i in range(0, len(self.serie.data['Test'].dropna())+len(self.serie.data['Validation'].dropna())):

            x_input = self.serie.data['Série stationnarisée réduite'][self.serie.index_fin_entrainement -
                                                                      taille+i:self.serie.index_fin_entrainement+i].values
            x_input = x_input.reshape((1, taille, n_features))
            yhat = model.predict(x_input, verbose=0)[0][0]
            serie_predite_temp.append(yhat)

            # inversion de la mise à l'échelle
            padding = np.zeros(taille-1).reshape(1, taille - 1)
            yhat = np.append(padding, [yhat]).reshape(1, -1)
            yhat = scaler.inverse_transform(yhat)
            yhat = yhat[0][-1]

            # déstationnarisation
            yhat = yhat + \
                self.serie.data['Série'][self.serie.index_fin_entrainement+i-nbre_retards]
            serie_predite.append(yhat)

        # prévision dynamique
        if final:
            anciennes_predictions = []
            for i in range(0, len(self.serie.data['Test'].dropna())+len(self.serie.data['Validation'].dropna())):
                decoupe = -taille + i

                if decoupe < 0:
                    x_input_dynamique = np.append(
                    self.serie.data['Série stationnarisée réduite'][decoupe:].values, anciennes_predictions)
                
                else:
                    x_input_dynamique = np.array(anciennes_predictions)[-taille:]

                x_input_dynamique = x_input_dynamique.reshape(
                    (1, taille, n_features))

                yhat_dynamique = model.predict(
                    x_input_dynamique, verbose=0)[0][0]

                anciennes_predictions.append(yhat_dynamique)

                # inversion de la mise à l'échelle
                padding = np.zeros(taille-1).reshape(1, taille - 1)
                yhat_dynamique = np.append(
                    padding, [yhat_dynamique]).reshape(1, -1)
                yhat_dynamique = scaler.inverse_transform(yhat_dynamique)
                yhat_dynamique = yhat_dynamique[0][-1]

                # déstationnarisation
                yhat_dynamique = yhat_dynamique + \
                    self.serie.data['Série'][self.serie.index_fin_entrainement+i-nbre_retards]
                serie_predite_dynamique.append(yhat_dynamique)

        # ajout d'un padding avec des nan
        a = np.empty((1, len(self.serie.data['Entraînement'].dropna())))
        a[:] = np.nan
        serie_predite = np.concatenate((a[0], np.array(serie_predite)), axis=0)

        # calcul du MSE uniquement sur le jeu de test
        resultat = mean_squared_error(
            serie_predite[self.serie.index_fin_entrainement:self.serie.index_fin_test], self.serie.data['Test'].dropna())

        if final:
            self.serie.data[self.__class__.__name__] = serie_predite

            # ajout d'un padding avec des nan
            a = np.empty((1, len(self.serie.data['Entraînement'].dropna())))
            a[:] = np.nan
            serie_predite_dynamique = np.concatenate(
                (a[0], np.array(serie_predite_dynamique)), axis=0)


            print(len(serie_predite_dynamique))
            print(len(self.serie.data))
            self.serie.data[self.__class__.__name__ +
                            "_dynamique"] = serie_predite_dynamique
            self.modele = model

        print("Fit du modèle " + key + " : " + str(resultat))

        return (key, resultat)

    def fit(self):
        # Création d'un jeu de données d'apprentissage supervisé
        print("Configuration LSTM retenue : " + str(self.config))

        print("Entraînement du modèle retenu...")
        self.fit_modele(self.config, True)

    def description_modele(self):
        # Evolution de la perte / perte sur ensemble de validation
        plt.figure(figsize=(10, 5))
        plt.plot(self.historique.history['loss'])
        plt.plot(self.historique.history['val_loss'])
        plt.title('Évolution de la perte du modèle')
        plt.ylabel('Perte')
        plt.xlabel('Itération')
        plt.legend(['Entraînement', 'Test'], loc='upper right')
        plt.savefig("outputs/" + self.nom_sauvegarde +
                    '_evolution_perte.pdf', dpi=300)
        plt.show()

        # Graphique du modèle retenu
        plot_model(self.modele, to_file="outputs/" + self.nom_sauvegarde +
                   "_schema.pdf", show_shapes=True, expand_nested=True, dpi=300)

from sklearn.metrics import mean_squared_error, median_absolute_error, explained_variance_score, r2_score

class Modele:

    def __init__(self):
        pass
    
    def definir_serie(self, serie):
        print("=== " + serie.nom + " | " + self.__class__.__name__ + " ===")
        self.serie = serie
        self.nom_sauvegarde = serie.nom_sauvegarde + "_" + self.__class__.__name__

    def trouver_hyperparametres(self):
        raise NotImplementedError # trouver_hyperparametres doit etre surcharge

    def fit(self):
        raise NotImplementedError  # fit doit etre surcharge

    def recapitulatif(self):
        y_reel = self.serie.data['Validation'].dropna()

        nom_previsions = ["Prévision one step ahead", "Prévision dynamique"]
        series = [self.serie.data[self.__class__.__name__]
                  [self.serie.index_fin_test:], self.serie.data[self.__class__.__name__ + "_dynamique"][self.serie.index_fin_test:]]

        print("|")
        print("|    --- " + self.__class__.__name__ + " ---")
        print("|")

        for i, serie in enumerate(series):
            y_pred = serie
            
            mse = mean_squared_error(y_reel, y_pred)
            medae = median_absolute_error(y_reel, y_pred)
            var_expliquee = explained_variance_score(y_reel, y_pred)
            r2 = r2_score(y_reel, y_pred)

            print("|    - " + nom_previsions[i] + " -")
            print("|    MSE       : " + str(mse))
            print("|    MedAE     : " + str(medae))
            print("|    Var. expl : " + str(var_expliquee))
            print("|    R^2       : " + str(r2))
            print("|")
        print("----------")


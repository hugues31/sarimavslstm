from fonctions import MSE

class Modele:

    def __init__(self):
        pass
    
    def definir_serie(self, serie):
        self.serie = serie

    def trouver_hyperparametres(self):
        raise NotImplementedError # trouver_hyperparametres doit etre surcharge

    def fit(self):
        raise NotImplementedError  # fit doit etre surcharge

    def recapitulatif(self):
        mse = MSE(self.serie.data[self.__class__.__name__],
                  self.serie.data['Série'])
        print("|")
        print("|    --- " + self.__class__.__name__ + " ---")
        print("|    MSE : " + str(mse))

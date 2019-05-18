from modele import Modele


class Naive(Modele):
    """
        Naive est le modèle de prévision naïf. La valeur prévue à l'instant
        t est égale à la valeur à l'instant t-1. Sert de point de comparaison
        aux autres modèles.
    """

    def __init__(self):
        pass

    def trouver_hyperparametres(self):
        # aucun hyperparametre pour un modele naïf
        pass

    def fit(self):
        # Prévision one step ahead
        self.serie.data[self.__class__.__name__] = self.serie.data['Série'][self.serie.index_fin_entrainement-1:].shift(
            periods=1)
        
        # Prévision dynamique
        self.serie.data[self.__class__.__name__ +
                        "_dynamique"] = self.serie.data['Série'].iloc[self.serie.index_fin_entrainement]
        

    def description_modele(self):
        pass

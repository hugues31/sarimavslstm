from numpy import array

def MSE(estimee, reelle):
    """
        Calcul le Mean Square Error entre la série estimée et la série réelle
    """

    assert len(estimee) == len(reelle)

    return round(float(((estimee - reelle) ** 2).mean() ** .5), 5)


def decouper_serie_apprentissage_supervise(serie, taille):
    """
        Découpe une série en n échantillons de taille 'taille'
        et n valeurs "cibles"
    """

    X, y = list(), list()
    for i in range(len(serie)):
            # borne supérieure
            end_ix = i + taille
            # arrêt si on dépasse la longueur de la série
            if end_ix > len(serie)-1:
                break
            # récupération des observations
            seq_x, seq_y = serie[i:end_ix], serie[end_ix]
            X.append(seq_x)
            y.append(seq_y)
    return array(X), array(y)

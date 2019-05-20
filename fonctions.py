from numpy import array

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


def tableau_en_latex(tableau):
    """
        Permet de créer un tableau Latex à partir d'un tableau Python à deux
        dimensions. La première ligne est l'en-tête du tableau.
    """
    pass

def enregistrer_plot(plot, nom):
    plot.savefig("outputs/" + nom, dpi=300, bbox_inches='tight', pad_inches=0)

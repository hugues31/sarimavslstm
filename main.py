import sys
from serie import Serie


def charger_series():
    # Tendance linéaire simple
    serie_1 = Serie("Série 1", formule="0.3*x", std=6)

    # Tendance linéaire + saisonnalité simple de période 12
    serie_2 = Serie("Série 2", formule="sin(x*pi/6)*4+0.3*x", std=2)

    # Tendance linéaire + saisonnalité de période 12 multiplicative
    serie_3 = Serie("Série 3", formule="sin(x*pi/6)*4*(0.01*x)+0.3*x", std=1.8)

    # Tendance non polynomiale + saisonnalités imbriquées non multiples 
    serie_4 = Serie(
        "Série 4", formule="(sin(x*pi/3.5)*8-sin(x*pi/6)*15)+sin(x*pi/40)*40+((4*x)**0.4)*30", std=2.8)

    # Série production de bières en Australie de 1956 à 1995
    serie_bieres = Serie("Production de bières", csv_file="production_biere_australie.csv")

    # Série financière très bruitée : le rendement du Bitcoin sur 4 heures
    serie_bitcoin = Serie("Rendement du Bitcoin", csv_file="rendements_bitcoin_4H.csv")

    return [serie_1, serie_2, serie_3, serie_4, serie_bieres, serie_bitcoin]

series = charger_series()   # synthétiques + données réelles

for serie in series:
    for modele in serie.modeles:
        modele.definir_serie(serie)
        modele.trouver_hyperparametres()
        modele.fit()

    serie.recapitulatif()

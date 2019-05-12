import sys
from serie import Serie


def charger_series():
    # Tendance linéaire simple
    serie_1 = Serie("Série 1", formule="0.3*x", std=5)
    return [serie_1]

    # Tendance + saisonnalité de période 12 multiplicative
    serie_2 = Serie("Série 2", formule="sin(x*pi/6)*4*(0.01*x)+0.3*x", std=1.8)

    serie_3 = Serie(
        "Série 3", formule="(sin(x/12)*15+sin(x/80)*100)*sin(x/200)*3", std=2.8)
    
    serie_bitcoin = Serie("Rendement horaire du Bitcoin", csv_file="rendements_bitcoin.csv")

    

    return [serie_1, serie_2, serie_3, serie_bitcoin]

series = charger_series()   # synthétiques + données réelles

for serie in series:
    for modele in serie.modeles:
        modele.definir_serie(serie)
        modele.trouver_hyperparametres()
        modele.fit()

    serie.recapitulatif()

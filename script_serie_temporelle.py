import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
import itertools
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import het_arch
from sklearn.metrics import mean_absolute_error, mean_squared_error

#import des données

data=pd.read_csv("velib_horaire.csv")
data["Departure_Time"]= pd.to_datetime(data["Departure_Time"])
data = data.sort_values("Departure_Time").set_index("Departure_Time")
Y_trajet=data["nombre_trajets"].dropna() # extration du processus stationnaire à étudier
Y_elec=data["proportion_velos_electriques"].dropna()
Y_duree=data["duree_moyenne_sec"].dropna()


def etude_complete(Y):
    #aperçue de Yt

    plt.figure(figsize=(12, 5))
    plt.plot(Y)
    plt.title(f"Graphique A: Série temporelle {Y.name}")
    plt.xlabel("Heure de départ")
    plt.ylabel(f"{Y.name}")
    plt.grid(True)
    plt.show()

    # test d'une tendance polynomiale de degré 3
    t = np.arange(len(Y))
    X = sm.add_constant(np.column_stack([t, t**2,t**3]))
    mod = sm.OLS(Y.values, X).fit()
    print(mod.summary())

    # retrait de la tendance polynomiamle de degrès 1 par différentiation
    y_sans_tendance = Y.diff(1).dropna()


    #fonction d'autocorrélation de Y pour recherche de saisonnalité
    fig, ax = plt.subplots(figsize=(10,4))
    plot_acf(Y, lags=24*21, ax=ax) 

    # graduations : tous les multiples de 24
    xticks = np.arange(0, 24*21 + 1, 24)
    ax.set_xticks(xticks)

    # forcer l'affichage avant de modifier les couleurs
    plt.draw()

    # couleur des labels
    for tick, val in zip(ax.get_xticklabels(), xticks):
        if val % 168 == 0 and val != 0:
            tick.set_color("green")
        elif val % 24 == 0 and val != 0:
            tick.set_color("red")

    plt.show()


    # retrait de la saisonnalité

    Y_diff24 = y_sans_tendance.diff(24)
    Y_diff168= Y_diff24.diff(168)
    Y_model= Y_diff168.dropna()


    #vérifier la stationnarité de la serie finale avec ACF

    plt.figure(figsize=(12, 5))
    plt.plot(Y_model)

    fig, ax = plt.subplots(figsize=(18, 6))
    plot_acf(Y_model, lags=24*21,ax=ax)

    # graduations : tous les multiples de 24
    xticks = np.arange(0, 24*21 + 1, 24)
    ax.set_xticks(xticks)

    # forcer l'affichage avant de modifier les couleurs
    plt.draw()

    # couleur des labels
    for tick, val in zip(ax.get_xticklabels(), xticks):
        if val % 168 == 0 and val != 0:
            tick.set_color("green")
        elif val % 24 == 0 and val != 0:
            tick.set_color("red")
    plt.show()


    #Recherche du meilleur modèle ARMA par comparaison des AIC 

    warnings.simplefilter("ignore", ConvergenceWarning)
    warnings.filterwarnings("ignore")

    p = range(0, 5)
    q = range(0, 5)

    resultats = []

    for i, j in itertools.product(p, q):
        try:
            mod = sm.tsa.ARIMA(Y_model, order=(i, 0, j))
            res = mod.fit()

            if res.mle_retvals.get("converged", False):
                resultats.append({
                    "p": i,
                    "q": j,
                    "modele": f"ARMA({i},{j})",
                    "AIC": res.aic
                })

        except Exception:
            continue

    df_resultats = pd.DataFrame(resultats)
    df_resultats = df_resultats.sort_values("AIC").reset_index(drop=True)

    print("Tableau des modèles classés par AIC :")
    print(df_resultats)

    p_opt = int(df_resultats.loc[0, "p"])
    q_opt = int(df_resultats.loc[0, "q"])
    modele_opt = df_resultats.loc[0, "modele"]
    aic_opt = df_resultats.loc[0, "AIC"]

    print("Meilleur modèle retenu :", modele_opt)
    print("AIC =", aic_opt)

    res_final = sm.tsa.ARIMA(Y_model, order=(p_opt, 0, q_opt)).fit()
    print(res_final.summary())


    #analyse des résidus

    residus = res_final.resid

    # tests
    ljung = acorr_ljungbox(residus, lags=[10], return_df=True)
    shap_stat, shap_p = shapiro(residus)

    #  tableau des résultats des tests
    df_tests = pd.DataFrame({
        "Test": ["Ljung-Box", "Shapiro-Wilk"],
        "Statistique": [
            round(ljung["lb_stat"].values[0], 2),
            round(shap_stat, 2)
        ],
        "p-value": [
            round(ljung["lb_pvalue"].values[0], 2),
            round(shap_p, 2)
        ]
    })

    print(df_tests)


    #performance du modèle en prévision

    n = len(Y)
    T=int(0.8*n)
    #séparation donnée entrainement et test
    train = Y_model.iloc[:T]
    test = Y_model.iloc[T:]

    #modélisation des données d'entrainement
    model_train = sm.tsa.ARIMA(train, order=(p_opt, 0, q_opt)).fit()

    #prédiction de la taille des données de test à comparer
    pred = model_train.forecast(steps=len(test))

    #comparaison avec les données de test
    mae = mean_absolute_error(test, pred)
    rmse = np.sqrt(mean_squared_error(test, pred))

    print("MAE :", mae)
    print("RMSE :", rmse)

    plt.figure(figsize=(12,5))
    plt.scatter(test.index, test, label="Valeurs observées")
    plt.plot(test.index, pred, label="Prévisions", linestyle="--")
    plt.legend()
    plt.title(f"Prévisions {modele_opt} des données de test")
    #plt.show()

etude_complete(Y_duree)
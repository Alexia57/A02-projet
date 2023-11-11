
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, confusion_matrix


#########################################
### Ingénéries des caractéristiques I ###
#########################################

def plot_shot_by_distance(data):
    """
    histogramme des tirs regroupés par distance
    """
    plt.figure(figsize=(10, 6))

    sns.histplot(data, x='distanceToNet', hue='isGoal', multiple='stack', bins=50, palette=['tomato', 'springgreen'], edgecolor='white')

    # titres et des légendes
    plt.title('Histogramme des tirs regroupés par distance')
    plt.xlabel('Distance au filet')
    plt.ylabel('Nombre de tirs')
    plt.legend(['Buts', 'Non-buts'])

    plt.show()


def plot_stacked_histogram(data, column, bins=70, alpha=0.5, title='', xlabel='', ylabel='', legend_labels=None):
    """
    histogramme des tirs regroupés par angle
    """
    plt.figure(figsize=(10, 6))

    sns.histplot(data, x=column, hue='isGoal', multiple='stack', bins=bins, palette=['red', 'green'], edgecolor='white', alpha=alpha)

    plt.xticks(np.arange(-100,101, 20))
    plt.xlim(-100, 100)

    # titres et des légendes

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(['Buts', 'Non-buts'])

    plt.show()


def plot_dist_angle(df, x_column, y_column):
    """
    Histogramme 2D de la distance et de l'angle des tirs
    """
    figure = plt.figure(figsize=(10, 8))
    
    # création d'un joint plot avec un histogramme 2D
    ax = sns.jointplot(data=df, x=x_column, y=y_column, kind="hist")
    
    ax.ax_joint.set_xlabel(f"{x_column.capitalize()}")
    ax.ax_joint.set_ylabel(f"{y_column.capitalize()}")
    
    # titre
    plt.subplots_adjust(top=0.9)
    plt.suptitle(f"Histogramme 2D de {x_column.capitalize()} et {y_column.capitalize()}")
    
    plt.show()


def plot_goal_rate_distance(df):
    """
    Taux de but en fonction de la distance
    """
    df_groupby_distance = df.groupby(["distanceToNet"])["isGoal"].mean().to_frame().reset_index()

    ax = sns.lineplot(data=df_groupby_distance, x='distanceToNet', y='isGoal')

    plt.xlabel('Distance au filet')
    plt.ylabel('Taux de but')
    plt.xticks(np.arange(0, 100, 10))
    plt.yticks(np.arange(0, 1.1, 0.2))
    plt.title('Taux de but en fonction de la distance au filet')

    plt.show()


def plot_but_distance(df):
    """
    Histogramme des buts classés par distance filet vide et non vide
    """
    but_events = df[df['isGoal'] == 1]

    # création de l'histogramme en séparant les événements de filet vide et non vide
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Distance au filet')
    ax1.set_ylabel('Nombre de buts - Filet non vide', color='green')
    ax1.hist(but_events[but_events['emptyNet'] == 0]['distanceToNet'], bins=20, alpha=0.5, label='Filet non vide', color='green', edgecolor='white')
    ax1.tick_params(axis='y', labelcolor='green')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()  # 2e axe y partageant le même x

    color = 'tab:blue'
    ax2.set_ylabel('Nombre de buts - Filet vide', color='red')
    ax2.hist(but_events[but_events['emptyNet'] == 1]['distanceToNet'], bins=20, alpha=0.5, label='Filet vide', color='red', edgecolor='white')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.legend(loc='upper right')

    # titre
    plt.title("Histogramme des buts classés par distance")

    plt.show()




#########################################
#### 4 plots demandés régulièrement  ####
#########################################

def plot_roc_auc(model, X, y, nom_model):
    """
    calcule les probabilités prédites, la courbe ROC et l'AUC, puis trace la courbe ROC.
    params:  model, X, y, nom_model
    return:  None (affiche la courbe ROC)
    """
    # Calculer les probabilités prédites
    y_prob = model.predict_proba(X)[:, 1]

    # Calculer la courbe ROC
    fpr, tpr, thresholds = roc_curve(y, y_prob)

    # Calculer l'AUC
    auc_value = roc_auc_score(y, y_prob)

    # Tracer la courbe ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label= nom_model + f' AUC = {auc_value:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('Taux de faux positifs')
    plt.ylabel('Taux de vrais positifs')
    plt.title('Courbe ROC')
    plt.legend(loc='lower right')
    plt.show()


def plot_taux_buts_par_centile(prob_pred, y_valide, model_name):
    """
    Cette fonction trace la courbe du taux de buts par rapport au centile de probabilité du modèle.
    params:  prob_pred, y_valide
    return:  fig_taux_buts
    """
    # Convertir les prédictions en une série 1D
    y_series = np.array(y_valide)
    y_series = np.reshape(y_series, (y_series.shape[0]))

    # Échelle des probabilités vraies (predict_proba() renvoie à la fois True et False)
    prob_vraies = pd.DataFrame()
    prob_vraies['cible_vraie'] = np.array(y_series)
    percentiles = [[np.percentile(prob_pred, i), np.percentile(prob_pred, i+5)] for i in range(0, 100, 5)]
    total_buts = np.sum(y_series)

    # Boucler sur les probabilités pour vérifier leurs percentiles avec leur statut (but/tir)
    taux_buts = []
    for i in range(len(percentiles)):
        # Vérifier l'intervalle de probabilité dans le centile et calculer le nombre de buts
        buts = prob_vraies[(prob_pred <= percentiles[i][1]) & (prob_pred > percentiles[i][0]) & (prob_vraies['cible_vraie'] == 1)].shape[0]
        # Vérifier l'intervalle de probabilité dans le centile et calculer le nombre de tirs (ou non-buts)
        non_buts = prob_vraies[(prob_pred <= percentiles[i][1]) & (prob_pred > percentiles[i][0]) & (prob_vraies['cible_vraie'] == 0)].shape[0]
        # Si pas de but, ne rien faire, calculer la formule si but
        if buts == 0:
            taux_buts.append(0)
        else:
            taux_buts.append((buts * 100) / (buts + non_buts))

    # Axe pour le centile
    centile_prob_modele = np.arange(0, 100, 5)

    # Tracé du taux de buts par rapport au centile de probabilité du modèle
    fig_taux_buts = plt.figure()
    sns.set()
    plt.plot(centile_prob_modele, taux_buts, label=model_name)
    plt.xlim(100, 0)
    plt.ylim(0, 100)
    plt.title("Taux de buts par rapport au centile de probabilité")
    plt.xlabel('Centile de probabilité', fontsize=14)
    plt.xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    plt.ylabel('Taux de buts', fontsize=14)
    y_axe = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    y_valeurs = ['0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%']
    plt.yticks(y_axe, y_valeurs)
    plt.legend()
    return fig_taux_buts



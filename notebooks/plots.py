
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, confusion_matrix
from sklearn.calibration import CalibrationDisplay, calibration_curve, CalibratedClassifierCV



#########################################
### Ingénéries des caractéristiques I ###
#########################################

def plot_shot_by_distance(data):
    """
    histogramme des tirs regroupés par distance
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(data, x='distanceToNet', hue='isGoal', multiple='stack', bins=50, palette=['tomato', 'springgreen'], edgecolor='white')
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
    
    plt.subplots_adjust(top=0.9)
    plt.suptitle(f"Histogramme 2D de {x_column.capitalize()} et {y_column.capitalize()}")
    
    plt.show()


'''def plot_goal_rate_distance(df, x_step=10):
    """
    Taux de but en fonction de la distance
    """
    df_groupby_distance = df.groupby(["distanceToNet"])["isGoal"].mean().to_frame().reset_index()

    plt.figure(figsize=(16,6))

    ax = sns.lineplot(data=df_groupby_distance, x='distanceToNet', y='isGoal')

    plt.xlabel('Distance au filet')
    plt.ylabel('Taux de but')
    
    plt.xticks(np.arange(0, 100, x_step))
    plt.yticks(np.arange(0, 1.1, 0.2))
    plt.title('Taux de but en fonction de la distance au filet')

    plt.show()

def plot_goal_rate_angle(df, x_step=10):
    """
    Taux de but en fonction de l'angle
    """
    df_groupby_angle = df.groupby([np.abs(df["relativeAngleToNet"])])["isGoal"].mean().to_frame().reset_index()

    plt.figure(figsize=(16, 6))
    
    ax = sns.lineplot(data=df_groupby_angle, x='relativeAngleToNet', y='isGoal')

    plt.xlabel('Angle au filet')
    plt.ylabel('Taux de but')
    
    plt.xticks(np.arange(0, 190, x_step))
    plt.yticks(np.arange(0, 1.1, 0.2))
    plt.title('Taux de but en fonction de l\'angle au filet (valeur absolue)')

    plt.show()'''

def plot_goal_rate_binned(df, column, bins, x_label, x_step=1):
    """
    Taux de but en fonction d'une colonne avec binning
    """
    df_copy = df.copy()
    df_copy['bin'] = pd.cut(df_copy[column], bins=bins)
    df_groupby_bin = df_copy.groupby('bin')['isGoal'].mean().to_frame().reset_index()

    plt.figure(figsize=(12, 6))
    
    ax = sns.barplot(data=df_groupby_bin, x='bin', y='isGoal')

    plt.xlabel(x_label)
    plt.ylabel('Taux de but')
    plt.xticks(rotation=45, ha='right')

    plt.title(f'Taux de but en fonction de {x_label} (Binned)')

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

    ax2.set_ylabel('Nombre de buts - Filet vide', color='red')
    ax2.hist(but_events[but_events['emptyNet'] == 1]['distanceToNet'], bins=20, alpha=0.5, label='Filet vide', color='red', edgecolor='white')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.legend(loc='upper right')

    plt.title("Histogramme des buts classés par distance")

    plt.show()




#########################################
#### 4 plots demandés régulièrement  ####
#########################################

def plot_roc_auc(models, X_list, y_list, nom_models):
    """
    Calcule les probabilités prédites, la courbe ROC et l'AUC, puis trace les courbes ROC pour plusieurs modèles.
    params:  models, X_list, y, nom_models
    return:  None (affiche les courbes ROC)
    """
    plt.figure(figsize=(8, 6))
    
    for model, X, y, nom_model in zip(models, X_list, y_list, nom_models):
        # probabilités prédites
        if model is None:  # cas random
            y = y_list[0]
            y_prob = np.random.rand(len(y))
        else:
            y_prob = model.predict_proba(X)[:, 1]

        # courbe ROC
        fpr, tpr, thresholds = roc_curve(y, y_prob)

        # AUC
        auc_value = roc_auc_score(y, y_prob)

        # Tracer la courbe ROC
        plt.plot(fpr, tpr, label=nom_model + f' AUC = {auc_value:.2f}', linewidth=2, linestyle='--')

    
    plt.plot([0, 1], [0, 1], 'k--') # ligne en pointillés diagonale
    plt.xlabel('Taux de faux positifs')
    plt.ylabel('Taux de vrais positifs')
    plt.title('Courbe ROC')
    plt.legend(loc='lower right')
    
    plt.show()


def plot_taux_buts_par_centile(models, X_list, y_list, nom_models):
    """
    Cette fonction trace la courbe du taux de buts par rapport au centile de probabilité pour plusieurs modèles.
    params:  models, X_list, y, nom_models
    return:  None (affiche les courbes)
    """
    plt.figure()
    sns.set()

    for model, X, y, nom_model in zip(models, X_list, y_list, nom_models):
        # probabilités prédites
        if model is None:  # cas random
            y = y_list[0]
            y_prob = np.random.rand(len(y))
        else:
            y_prob = model.predict_proba(X)[:, 1]

        y_series = np.array(y)
        y_series = np.reshape(y_series, (y_series.shape[0]))

        # Échelle des probabilités vraies (predict_proba() renvoie à la fois True et False)
        prob_vraies = pd.DataFrame()
        prob_vraies['cible_vraie'] = np.array(y_series)
        percentiles = [[np.percentile(y_prob, i), np.percentile(y_prob, i+5)] for i in range(0, 100, 5)]
        total_buts = np.sum(y_series)

        # On vérifie les percentiles des probas avec leur statut (but/tir)
        taux_buts = []
        for i in range(len(percentiles)):
            buts = prob_vraies[(y_prob <= percentiles[i][1]) & (y_prob > percentiles[i][0]) & (prob_vraies['cible_vraie'] == 1)].shape[0]
            non_buts = prob_vraies[(y_prob <= percentiles[i][1]) & (y_prob > percentiles[i][0]) & (prob_vraies['cible_vraie'] == 0)].shape[0]
            # Si pas de but, ne rien faire, calculer la formule si but
            if buts == 0:
                taux_buts.append(0)
            else:
                taux_buts.append((buts * 100) / (buts + non_buts))

        centile_prob_modele = np.arange(0, 100, 5)

        # Graphique
        plt.plot(centile_prob_modele, taux_buts, label=nom_model)

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

    plt.show()


def cumulative_goal_rate(models, X_list, y_list, model_names):
    """
    Proportion cumulée de buts vs percentile de probabilité pour plusieurs modèles
    """
    plt.figure(figsize=(8, 6))
    sns.set()

    for model, X, y, model_name in zip(models, X_list, y_list, model_names):
        # probabilités prédites
        if model is None:  # cas random
            y = y_list[0]
            prob_predict = np.random.rand(len(y))
        else:
            prob_predict = model.predict_proba(X)[:, 1]
        y_val = y.copy()
        goal_probas = prob_predict
        percentiles = np.arange(0, 101)  
        cumulative_goal = []
        for perc in percentiles:
            threshold = np.percentile(goal_probas, perc)
            predicted_goals = goal_probas >= threshold
            cumulative_goal_proportion = np.sum(y_val[predicted_goals])/ np.sum(y_val)
            cumulative_goal.append(cumulative_goal_proportion)

        percentiles = 100 - percentiles[::-1]

        # Graphique
        plt.plot(percentiles, cumulative_goal, linestyle='--', linewidth=2, label=model_name)

    plt.xlabel('Centile de probabilité du modèle')
    plt.ylabel('Proportion cumulée')
    plt.title('Proportion cumulée de buts vs percentile de probabilité')
    plt.xlim(100, 0)
    plt.ylim(0, 100)
    plt.ylim(top=1)
    plt.xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])    
    plt.grid()
    plt.legend()
    plt.show()


def plot_calibration_curve(models, X_list, y_list, model_names):
    """
    Courbe de calibration pour plusieurs modèles sur le même graphique.
    """
    colors = ['blue', 'red', 'orange', 'green']  # Ajoutez plus de couleurs si nécessaire
    plt.figure(figsize=(10, 6))
    
    for model, X, y, model_name, color in zip(models, X_list, y_list, model_names, colors):
        # probabilités prédites
        if model is None:  # cas random
            y = y_list[0]
            prob_predict = np.random.rand(len(y))
        else:
            prob_predict = model.predict_proba(X)[:, 1]

        # Courbe de calibration
        prob_true, prob_pred = calibration_curve(y, prob_predict, n_bins=10, strategy='uniform')

        # Créez l'objet CalibrationDisplay
        disp = CalibrationDisplay.from_predictions(y, prob_predict, n_bins=10, label=model_name, ax=plt.gca(), color=color)

    # Ajoutez des légendes et des titres
    plt.ylabel("Fraction de positives (Positive classe : 1)")
    plt.xlabel("Moyenne de proba prédite (Positive classe : 1)")
    plt.title('Courbe de calibration')
    plt.legend()
    plt.grid(linestyle='-')
    plt.show()

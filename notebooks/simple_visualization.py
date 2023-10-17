import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import math

################################# Question 1 #################################

def shot_type_analysis(df: pd.DataFrame, years: list = [], save_plot: bool = True):
    '''
    Returns a bar plot of the number of shots and goals with the ratio of the two grouped by shot type.
    You can specify the year(s) of the season you want
    '''
    # removing "fake" shots
    types_to_exclude = ["Deflected", "Tip-In", "Wrap-around"]

    # filter df given the specified years
    if years:
        df = df[df["year"].isin(years)]

    # we exclude those
    filtered_data = df[~df["typeDeTir"].isin(types_to_exclude)]

    # grouping by shot type
    grouped_data = filtered_data.groupby("typeDeTir")

    # nb of shots and goals for each shot type
    total_shots = grouped_data.size()
    total_goals = grouped_data["eventTypeId"].apply(lambda x: (x == "GOAL").sum())

    # success rate
    success_rate = (total_goals / total_shots).round(2)

    # creating the bar plot
    bar_width = 0.8
    index = range(len(total_shots))
    plt.figure(figsize=(12, 6))
    plt.bar(index, total_shots, bar_width, label="Total Shots", color="teal")
    plt.bar(index, total_goals, bar_width, label="Total Goals", bottom=total_shots, color="deeppink")  # "bottom" is used to stack the bars

    # we add here success rates above each bar
    for i, rate in enumerate(success_rate):
        plt.text(index[i], total_shots[i] + total_goals[i], f"{rate}", ha="center", va="bottom", fontsize=14)

    # title and labels
    plt.xlabel("Shot Type")
    plt.xticks(index, total_shots.index, rotation=45)
    plt.legend()
    plt.title("Comparison of Shot Types and Goals in a Hockey Season with Success Rate for year(s) : " + str(years) , fontsize=14)
    plt.ylabel("Number of Shots or Goals")

    # display the plot
    plt.tight_layout()
    plt.show()

    if save_plot:
        # save the figure in /figures 
            plt.savefig('shot_goal.png', format='png', dpi=300)

    return success_rate

################################# Question 2 #################################

def calculate_distances_hockey(df: pd.DataFrame):
    """
    Requirements : math & json
    fonction pour le calcul des distances des joueurs au camp (Utilisé dans les deux fonctions suivantes)
    """

    def calculate_distance(x, y, x_camp_min, x_camp_max, y_camp_min, y_camp_max):
        distance_x = max(x - x_camp_max, x_camp_min - x)
        distance_y = max(y - y_camp_max, y_camp_min - y)
        return math.sqrt(distance_x**2 + distance_y**2)

    #Extracting x and y
    def extract_x(coord):
        try:
            return float(coord['x'])
        except (KeyError, ValueError):
            return None

    def extract_y(coord):
        try:
            return float(coord['y'])
        except (KeyError, ValueError):
            return None

    # Convert JSON strings to dictionaries (with double quotes)
    df['coordinates'] = df['coordinates'].apply(lambda x: json.loads(x.replace("'", '"')))

    # Extract 'x' and 'y' values from 'coordinates' column and convert to float
    df['x'] = df['coordinates'].apply(extract_x)
    df['y'] = df['coordinates'].apply(extract_y)

    # Camp coordinates
    x_camp_min = -100
    x_camp_max = 100
    y_camp_min = -42.5
    y_camp_max = 42.5

    # Calculating distance to camps
    df['distance_to_camp'] = df.apply(lambda row: calculate_distance(row['x'], row['y'], x_camp_min, x_camp_max, y_camp_min, y_camp_max), axis=1)
    return df


def plot_goal_probability_by_distance(data : pd.DataFrame):
    """
    Requirementa : pandas as pd, numpy as mp, matplotlib.pyplot as plt
    fonction pour la creation du graphique representant la probabilite de buts en fonction de la distance pour les saisons 2018-19, 2019-20 et 2020-21
    """

    # Probability calculation function
    def calculate_goal_probability(data):
        data = data[(data['eventTypeId'] == 'SHOT') | (data['eventTypeId'] == 'GOAL')]
        grouped_data = data.groupby('distance_group')['eventTypeId'].apply(lambda x: (x == 'GOAL').mean()).reset_index()
        return grouped_data

    # The DataFrame must contain the columns 'distance_to_net' (for distance), 'eventTypeId' (if the shot is a goal), and 'dateTime' (for date).
    df = data.copy()
    df = calculate_distances_hockey(df)
    df['dateTime'] = pd.to_datetime(df['dateTime'])

    # Extracts year and month from 'dateTime' column
    df['year'] = df['dateTime'].dt.year
    df['month'] = df['dateTime'].dt.month

    # Filter for 2018-19, 2019-20 and 2020-21 seasons
    data_2018_2019 = df[(df['year'] == 2018) | ((df['year'] == 2019) & (df['month'] < 7))]
    data_2019_2020 = df[((df['year'] == 2019) & (df['month'] >= 7)) | ((df['year'] == 2020) & (df['month'] < 7))]
    data_2020_2021 = df[((df['year'] == 2020) & (df['month'] >= 7)) | ((df['year'] == 2021) & (df['month'] < 7))]

    # Group the data by distance intervals and calculate the goal probability
    distance_bins = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, np.inf]
    distance_labels = ['0-2', '2-4', '4-6', '6-8', '8-10', '10-12', '12-14', '14-16', '16-18', '18-20', '20-22', '22-24', '24-26', '26-28', '28-30', '30-32', '32-34', '34-36', '36-38', '38-40', '40-42', '42-44', '44-46', '50+']
    data_2018_2019['distance_group'] = pd.cut(data_2018_2019['distance_to_camp'], bins=distance_bins, labels=distance_labels)
    data_2019_2020['distance_group'] = pd.cut(data_2019_2020['distance_to_camp'], bins=distance_bins, labels=distance_labels)
    data_2020_2021['distance_group'] = pd.cut(data_2020_2021['distance_to_camp'], bins=distance_bins, labels=distance_labels)

    # Calculate the probability of a goal for each season
    goal_prob_2018_2019 = calculate_goal_probability(data_2018_2019)
    goal_prob_2019_2020 = calculate_goal_probability(data_2019_2020)
    goal_prob_2020_2021 = calculate_goal_probability(data_2020_2021)

    # Create a graph to represent the results
    fig, ax = plt.subplots(figsize=(20, 6))

    ax.plot(goal_prob_2018_2019['distance_group'], goal_prob_2018_2019['eventTypeId'], label='2018-2019')
    ax.plot(goal_prob_2019_2020['distance_group'], goal_prob_2019_2020['eventTypeId'], label='2019-2020')
    ax.plot(goal_prob_2020_2021['distance_group'], goal_prob_2020_2021['eventTypeId'], label='2020-2021')

    ax.set_xlabel('Distance au filet')
    ax.set_ylabel('Probabilité de but')
    ax.set_title('Probabilité de but en fonction de la distance au filet (Saisons 2018-2019 à 2020-2021)')
    ax.legend()
    ax.grid(True, linestyle='--', linewidth=0.2, color='gray')

    plt.show()


################################# Question 3 #################################

def plot_goal_percentage(data : pd.DataFrame, year:int):
    """
    Requirements : pandas as pd, numpy as np, matplotlib.pyplot as plt
    """
    df = data.copy()

    # Extraction de l'année à partir de la colonne 'dateTime' pour récupérer l'année voulue
    df['dateTime'] = pd.to_datetime(df['dateTime'])
    df['year'] = df['dateTime'].dt.year
    df = df[df['year'] == year]

    # Calcul des distances (depuis la fonction calculate_distances_hockey())
    df = calculate_distances_hockey(df)

    # Récupération des vrais tirs uniquement
    selected_types = ['Backhand', 'Slap Shot', 'Snap Shot', 'Wrist Shot']
    df = df[df['typeDeTir'].isin(selected_types)]

    # Calcul du pourcentage de but
    df['goal_percentage'] = np.where(df['eventTypeId'] == 'GOAL', 1, 0)

    # Regroupement des données par catégorie de types de tirs et par intervalles de distance
    distance_bins = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, np.inf]
    distance_labels = ['0-2', '2-4', '4-6', '6-8', '8-10', '10-12', '12-14', '14-16', '16-18', '18-20', '20-22', '22-24', '24-26', '26-28', '28-30', '30-32', '32-34', '34-36', '36-38', '38-40', '40-42', '42-44', '44-46', '50+']
    df['distance_group'] = pd.cut(df['distance_to_camp'], bins=distance_bins, labels=distance_labels)

    # Groupement et calcul du pourcentage de buts
    grouped_data = df.groupby(['typeDeTir', 'distance_group'])['goal_percentage'].mean().reset_index()
    grouped_data['goal_percentage'] = grouped_data['goal_percentage'] * 100

    # Création du graphique pour représenter les résultats
    fig, ax = plt.subplots(figsize=(20, 6))
    for t in grouped_data['typeDeTir'].unique():
        subset = grouped_data[grouped_data['typeDeTir'] == t]
        ax.plot(subset['distance_group'], subset['goal_percentage'], label=t)

    ax.set_xlabel('Distance au filet')
    ax.set_ylabel('Pourcentage de buts')
    ax.set_title(f'Pourcentage de buts en fonction de la distance par rapport au filet et du type de tir (Saison {year})')
    ax.legend()
    ax.grid(True, linestyle='--', linewidth=0.2, color='gray')

    plt.show()


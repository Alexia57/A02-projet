import streamlit as st
import pandas as pd
import numpy as np
import json
import ift6758
import requests

import matplotlib.pyplot as plt
import plotly.express as px

import sys
#sys.path.append('/Users/canelle/Documents/Automne_2023/Science_données/Milestone2/A02-projet/docker-project-milestone3/ift6758')

from game_client import GameClient
from serving_client import ServingClient


#from ift6758.ift6758.client import serving_client
#from ift6758.ift6758.client import game_client
#from client.serving_client import ServingClient

st.title("Hockey Visualisation APP")

sc = ServingClient(ip = "127.0.0.1", port = 8000)
gc = GameClient()


with st.sidebar:
    workspace = st.selectbox("Workspace", ["ift6758-a02"])

    model = st.selectbox("Model",["reg_logistique-distance", "reg_logistique_dist_angle"])

    version = st.selectbox("Version",["1.0.0"])

    if st.button('Get Model'):
        # Appeler la méthode download_registry_model du client de service
        response = sc.download_registry_model(workspace=workspace, model=model, version=version)
        
        # Vérifier la réponse et afficher un message approprié
        if response.get('status') == 'success':
            st.success(f'Model {model} version {version} downloaded successfully!')
        else:
            st.error(f"Failed to download model {model} version {version}.")
        

#with st.container():
#    # TODO: Add Game ID input
#    game_id = st.text_input("Game ID",2022030411)
#    pass
with st.container():
    game_id = st.text_input("Game ID", "2021020329")  # Utilisez une valeur par défaut ou laissez vide

    if st.button('Ping Game'):
        # Utilisez la méthode ping_game pour récupérer les données du jeu
        df , last_event, _ = gc.ping_game(game_id)

        # Préparer les données pour la prédiction
        features_for_prediction = df[sc.features] 

        # Appeler la méthode predict
        prediction_response = sc.predict(features_for_prediction)

        if prediction_response.get('status') == 'success':
            st.success("Prediction successful!")
        else:
            st.error("Prediction failed!")

        #print(prediction_response)
        predictions_df = pd.read_csv('out.csv')
        sum_predictions = predictions_df.iloc[:, -1].sum()

        # Reset the index for both DataFrames without keeping the old index
        features_for_prediction = features_for_prediction.reset_index(drop=True)
        predictions_df = predictions_df.reset_index(drop=True)

        # Concatenate the DataFrames by columns (side by side)
        result_df = pd.concat([features_for_prediction, predictions_df.iloc[:, -1].rename('Model output')], axis=1)

        home_team_name = last_event["homeTeamName"][0]
        away_team_name = last_event["awayTeamName"][0]
        home_team_xG = round(sum_predictions, 2)  # prédiction
        away_team_xG =  round(sum_predictions, 2)  # prédiction
        current_period = last_event["period"][0]
        time_left = last_event["time_left"][0] 
        current_score_home = last_event["homeTeamLatestScore"][0]
        current_score_away = last_event["awayTeamLatestScore"][0]

        st.subheader(f"Game {game_id}: {home_team_name} vs {away_team_name}")
        st.caption(f"Period {current_period} - {time_left} left")

        col1, col2 = st.columns(2)
        col1.metric(f"{home_team_name} xG (actual)", (f"{home_team_xG} ({current_score_home})"), f"{home_team_xG - current_score_home}")
        col2.metric(f"{away_team_name} xG (actual)", (f"{away_team_xG} ({current_score_away})"), f"{away_team_xG - current_score_away}")

        st.subheader("Data used for predictions (and predictions)")
        st.table(result_df)

    

with st.container():
    # TODO: Add Game info and predictions
    pass
        

with st.container():
    # TODO: Add data used for predictions
    st.title("Data used for predictions")
    if st.button('Ping Game'):
    # Utilisez la méthode grab_a_game pour récupérer les données du jeu
        game_data = gc.grab_a_game(game_id)
        print(game_data)
        st.subheader("Game Data")
        st.dataframe(game_data)


with st.container():
    

    # bonus a completer
    if st.button('Ping Game'):
    # Utiliser la méthode ping_game pour récupérer les données du jeu
        df, last_event, df_to_update = gc.ping_game(game_id)

        # Afficher les dernières données
        st.subheader("Last Event:")
        st.dataframe(last_event)

        # Afficher les données complètes
        st.subheader("Full Game Data:")
        st.dataframe(df)

        # Afficher uniquement les nouvelles données
        st.subheader("New Events:")
        st.dataframe(df_to_update)

        # Visualisation : Exemple de tracé d'un graphique simple
        st.subheader("Example Chart:")
        
        # Groupement des données par période et calcul de la somme des buts par période
        goals_by_period = df.groupby('period')['isGoal'].sum()

        # Tracé d'un graphique à barres
        st.bar_chart(goals_by_period)




with st.container():
    if st.button('Ping Game'):
    # Utiliser la méthode ping_game pour récupérer les données du jeu
        df, _, _ = gc.ping_game(game_id)

        # Carte de chaleur pour la répartition des tirs au but
        st.subheader("Shot Distribution Heatmap:")
        
        # Utilisation de Plotly Express pour créer une carte de chaleur
        fig = px.scatter(df, x='details.xCoord', y='details.yCoord', color='isGoal', size='isGoal',
                        hover_data=['details.xCoord', 'details.yCoord', 'typeDescKey'],
                        labels={'details.xCoord': 'X Coordinate', 'details.yCoord': 'Y Coordinate'},
                        title='Shot Distribution Heatmap')

        # Afficher la carte de chaleur
        st.plotly_chart(fig)
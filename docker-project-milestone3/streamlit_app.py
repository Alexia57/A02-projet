import streamlit as st
import pandas as pd
import numpy as np
import json
import ift6758
import requests

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

        print(last_event.columns)
        print(last_event)

        # Préparer les données pour la prédiction
        features_for_prediction = df[sc.features] 
        print(features_for_prediction)

        # Appeler la méthode predict
        prediction_response = sc.predict(features_for_prediction)

        if prediction_response.get('status') == 'success':
            #predictions = prediction_response['predictions']
            # Traiter les prédictions ici...
            st.success("Prediction successful!")
        else:
            st.error("Prediction failed!")

        print(prediction_response)

        home_team_name = last_event["homeTeamName"][0]
        away_team_name = last_event["awayTeamName"][0]
        home_team_xG = 3.2  # prédiction
        away_team_xG = 1.4  # prédiction
        current_period = last_event["period"][0]
        time_left = last_event["time_left"][0] 
        current_score_home = last_event["homeTeamLatestScore"][0]
        current_score_away = last_event["awayTeamLatestScore"][0]

        st.subheader(f"Game {game_id}: {home_team_name} vs {away_team_name}")
        st.caption(f"Period {current_period} - {time_left} left")

        col1, col2 = st.columns(2)
        col1.metric(f"{home_team_name} xG (actual)", home_team_xG, f"{home_team_xG - current_score_home}")
        col2.metric(f"{away_team_name} xG (actual)", away_team_xG, f"{away_team_xG - current_score_away}")



    

with st.container():
    # TODO: Add Game info and predictions
    pass
        

with st.container():
    # TODO: Add data used for predictions
    pass


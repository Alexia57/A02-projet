import streamlit as st
import pandas as pd
import numpy as np
import json
from ift6758.ift6758.client.game_client import ServingClient
from ift6758.ift6758.client.game_client import GameClient

"""
General template for your streamlit app. 
Feel free to experiment with layout and adding functionality!
Just make sure that the required functionality is included as well
"""

st.title("NHL data visualisation API")

sc = ServingClient.ServingClient()
gc = GameClient.GameClient()


with st.sidebar:
    st.header("Workspace")
    workspace = st.selectbox("Select a workspace", ["a02-ift6758-milestone-2"])

    st.header("Model")
    model = st.selectbox("Select a model", ["model_distance", "model_distance_angle"])

    st.header("Version")
    version = st.selectbox("Select a version", ["Choisissez une option", "1.0.0"])

    if st.button('Download Model'):
        print('workspace: ', workspace)
        print('model:', model)
        print('version:', version)
        try : 
            sc.download_registry_model(workspace=workspace, model=model, version=version)
            st.write('Model Downloaded')
        except:
            st.warning("Please, select the workspace, model and version")
        

with st.container():
    # TODO: Add Game ID input
    #game_id = st.text_input("Select a match", 2022030411)
    pass

with st.container():
    # TODO: Add Game info and predictions
    pass

with st.container():
    # TODO: Add data used for predictions
    pass


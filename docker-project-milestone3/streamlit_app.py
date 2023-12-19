import streamlit as st
import pandas as pd
import numpy as np
import json
import sys
#sys.path.append('./ift6758/ift6758/')

#from ift6758.ift6758.client import serving_client
#from ift6758.ift6758.client import game_client
#from client.serving_client import ServingClient
#from client.game_client import GameClient

"""
General template for your streamlit app. 
Feel free to experiment with layout and adding functionality!
Just make sure that the required functionality is included as well
"""

st.title("Hockey Visualisation APP")

#sc = ServingClient.ServingClient()
#gc = GameClient.GameClient()


with st.sidebar:
    workspace = st.selectbox("Workspace", ["a02-ift6758-milestone-2"])

    model = st.selectbox("Model",["model_distance_angle"])

    version = st.selectbox("Version",["", "1.0.0"])

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
    game_id = st.text_input("Game ID",2022030411)
    pass

with st.container():
    # TODO: Add Game info and predictions
    pass

with st.container():
    # TODO: Add data used for predictions
    pass


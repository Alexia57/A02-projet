import json
import requests
import pandas as pd
import logging


logger = logging.getLogger(__name__)


class ServingClient:
    def __init__(self, ip: str = "localhost", port: int = 8000, features=None):
        self.base_url = f"http://{ip}:{port}"
        logger.info(f"Initializing client; base URL: {self.base_url}")

        if features is None:
            #features = ["distance"]
            features = ["distanceToNet", "relativeAngleToNet"]
        self.features = features

        # any other potential initialization

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Formats the inputs into an appropriate payload for a POST request, and queries the
        prediction service. Retrieves the response from the server, and processes it back into a
        dataframe that corresponds index-wise to the input dataframe.
        
        Args:
            X (Dataframe): Input dataframe to submit to the prediction service.
        """

        #raise NotImplementedError("TODO: implement this function")

    
        try:
            # Ensure the DataFrame has a unique index
            X = X.reset_index(drop=True)
            # Convert the DataFrame to a JSON payload
            #json_payload = json.loads(X.to_json(orient='records'))
            
            # Send the POST request with the JSON payload
            #response = requests.post(f"{self.base_url}/predict", json=json_payload)
            response = requests.post(f"{self.base_url}/predict", json=X.to_json())
            if response.status_code == 200:
                logger.info(f"Predictions done successfully !")
                return response.json()
            else:
                logger.error(f"Prediction failed with status code {response.status_code}: {response.text}")
                return {'status': 'error', 'MMessage': response.text}
        except Exception as e:
            logger.error(f"Exception occurred: {e}")
            return {'status': 'error', 'MMMessage': str(e)}


    def logs(self) -> dict:
        """Get server logs"""

        #raise NotImplementedError("TODO: implement this function")
        request = requests.get(f"{self.base_url}/logs")
        logger.info(f"logs obtained successfully !")
        return request.json()

    def download_registry_model(self, workspace: str, model: str, version: str) -> dict:
        """
        Triggers a "model swap" in the service; the workspace, model, and model version are
        specified and the service looks for this model in the model registry and tries to
        download it. 

        See more here:

            https://www.comet.ml/docs/python-sdk/API/#apidownload_registry_model
        
        Args:
            workspace (str): The Comet ML workspace
            model (str): The model in the Comet ML registry to download
            version (str): The model version to download
        """

        #raise NotImplementedError("TODO: implement this function")

        self.workspace = workspace
        self.model = model
        self.version = version
        self.model_filename = f"{model}_{version}"

        request = requests.post(f"{self.base_url}/download_registry_model", json= {'workspace': workspace, 'model': model, 'version': version})

        logger.info(f"Model {model}-{version} download successfully !")

        return request.json()
    

















if __name__ == '__main__':

    from pathlib import Path
    from comet_ml import API
    import os
    import pickle
    import sklearn
    from io import StringIO
    from game_client import GameClient

    #importe localement les fonctions de app.py pour tester
    def add_version_to_name(model_name):
        if model_name == 'reg_logistique-distance_1.0.0':
            os.rename('logistic_distance.pkl', f'{model_name}.pkl')

        if model_name == 'reg_logistique_dist_angle_1.0.0':
            os.rename('logistic_distance_angle.pkl', f'{model_name}.pkl')

    def download_model(workspace, model_name, version):
    
        model_complete_name = f'{model_name}_{version}'
        model_file = Path(f"{model_complete_name}.pkl")
        api = API('2uE4OYhHYaZ4qtDmCGEJibulo')

        if not model_file.is_file():
            api.download_registry_model(workspace, model_name, version, output_path="./", expand=True)
            add_version_to_name(model_complete_name)
            model = pickle.load(open(f"{model_complete_name}.pkl", 'rb'))
            print(f'Model {model_complete_name} downloaded.')
        else:
            model = pickle.load(open(f"{model_complete_name}.pkl", 'rb'))
            print(f'Model {model_complete_name} already downloaded.')

        return model 
    
    def get_feature(model):
        if model == 'logistic_distance':
            feature = ['distanceToNet']
        if model == 'logistic_distance_angle':
            feature = ['distanceToNet', 'relativeAngleToNet']
        return feature
    
    workspace = "ift6758-a02"
    model_name = 'logistic_distance'
    model = 'reg_logistique-distance'
    version = "1.0.0"
    game_id = 2021020329

    #crée le game client pour obtenir la partie
    gc = GameClient()
    df , _, _ = gc.ping_game(game_id)

    # Préparer les données pour la prédiction
    features_for_prediction = df[['distanceToNet']] 
    features_for_prediction = features_for_prediction.reset_index(drop=True)
    X = features_for_prediction.to_json()
    Y = pd.read_json(StringIO(X))[['distanceToNet']]
    

    # Appeler la méthode predict
    model = download_model(workspace,model,version)
    pred = model.predict_proba(features_for_prediction)[:,1]
    print(pred)

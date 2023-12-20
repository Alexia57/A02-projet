"""
If you are in the same directory as this file (app.py), you can run run the app using gunicorn:
    
    $ gunicorn --bind 0.0.0.0:<PORT> app:app

gunicorn can be installed via:

    $ pip install gunicorn

"""
import os
from pathlib import Path
import logging
from flask import Flask, jsonify, request, abort
import sklearn
import pandas as pd
import joblib
from comet_ml import API
import pickle

from dotenv import load_dotenv
load_dotenv(r"../../notebooks/.env")

import sys
#import ift6758


LOG_FILE = os.environ.get("FLASK_LOG", "flask.log")


app = Flask(__name__)

try :
    LOG_FILE = os.environ.get("FLASK_LOG", "flask.log")
    COMET_API_KEY = os.environ.get("COMET_API_KEY")
    api = API(COMET_API_KEY)
    print("api key loaded")
except Exception as e:
    print(e)
    app.logger.info("Error while loading api key.")


def get_feature(model):
    if model == 'logistic_distance':
        feature = ['distanceToNet']
    if model == 'logistic_distance_angle':
        feature = ['distanceToNet', 'relativeAngleToNet']
    return feature

sys.path.append('../ift6758')

def download_model(workspace, model_name, version):
    
    model_complete_name = f'{model_name}_{version}'
    model_file = Path(f"{model_complete_name}.pkl")

    if not model_file.is_file():
        api.download_registry_model(workspace, model_name, version, output_path="./", expand=True)
        add_version_to_name(model_complete_name)
        model = pickle.load(open(f"{model_complete_name}.pkl", 'rb'))
        print(f'Model {model_complete_name} downloaded.')
        app.logger.info(f'Model {model_complete_name} downloaded.')
    else:
        model = pickle.load(open(f"{model_complete_name}.pkl", 'rb'))
        print(f'Model {model_complete_name} already downloaded.')
        app.logger.info(f'Model {model_complete_name} already downloaded.')

    return model 

def add_version_to_name(model_name):
    if model_name == 'reg_logistique-distance_1.0.0':
        os.rename('logistic_distance.pkl', f'{model_name}.pkl')

    if model_name == 'reg_logistique_dist_angle_1.0.0':
        os.rename('logistic_distance_angle.pkl', f'{model_name}.pkl')

#@app.before_first_request
#def before_first_request():
with app.app_context():
    """
    Hook to handle any initialization before the first request (e.g. load model,
    setup logging handler, etc.)
    """
    # TODO: setup basic logging configuration
    logging.basicConfig(filename=LOG_FILE, level=logging.INFO)
    logging.info('Server starting')

    # reset the log file
    with open(LOG_FILE, 'w'):
        pass

    # TODO: any other initialization before the first request (e.g. load default model)
    global model_name
    global model

    json_models = [
        {'workspace': 'ift6758-a02', 'model': 'reg_logistique-distance', 'version': '1.0.0'},
        {'workspace': 'ift6758-a02', 'model': 'reg_logistique_dist_angle', 'version': '1.0.0'}
    ]

    for json in json_models:
        model = download_model(json['workspace'], json['model'], json['version'])
        model_name = f'{json["model"]}_{json["version"]}'


@app.route("/logs", methods=["GET"])
def logs():
    """Reads data from the log file and returns them as the response"""
    
    # TODO: read the log file specified and return the data
    #raise NotImplementedError("TODO: implement this endpoint")
    with open(LOG_FILE) as file:
        response = file.readlines()

    return jsonify(response)  # response must be json serializable!


@app.route("/download_registry_model", methods=["POST"])
def download_registry_model():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/download_registry_model

    The comet API key should be retrieved from the ${COMET_API_KEY} environment variable.

    Recommend (but not required) json with the schema:

        {
            workspace: (required),
            model: (required),
            version: (required),
            ... (other fields if needed) ...
        }
    
    """
    # Get POST json data
    json = request.get_json()
    app.logger.info(json)

    # TODO: check to see if the model you are querying for is already downloaded

    # TODO: if yes, load that model and write to the log about the model change.  
    # eg: app.logger.info(<LOG STRING>)
    
    # TODO: if no, try downloading the model: if it succeeds, load that model and write to the log
    # about the model change. If it fails, write to the log about the failure and keep the 
    # currently loaded model

    # Tip: you can implement a "CometMLClient" similar to your App client to abstract all of this
    # logic and querying of the CometML servers away to keep it clean here

    #raise NotImplementedError("TODO: implement this endpoint")

    global model_name
    global model
    
    current_model = model_name

    #model_query = f'{json["workspace"]}_{json["model"]}_{json["version"]}'
    #model_query_path = Path(model_query)
    

    try : 
        model = download_model(json["workspace"],json["model"],json["version"])
        response = {"status": "success", "message": f"Model {model_name} downloaded successfully !"}
        app.logger.info(f'Model {model_name} downloaded successfully !')
    except Exception as e:
        response = {"status": "error", "message": f"Failed to download model {model_name}."}
        model_name = current_model
        app.logger.info(f'Failed to download model {model_name}. We keep the model {current_model}.')

    #if model_query_path.is_file():
    #    model_path = model_query_path
    #    model = pickle.load(open(model_path, 'rb'))
    #    app.logger.info(f'Model {model} already downloaded.')
    #else:
    #    try:
    #        api.download_registry_model(json['workspace'], json['model'], json['version'], output_path="./", expand=True)
    #        add_version_to_name(model_query)
    #        model = model_query
    #        model = pickle.load(open(model_path, 'rb'))
    #        app.logger.info(f'Model {model_query} downloaded.')
    #    except:
    #        app.logger.info(f'Model {model_query} couldn\'t be downloaded so we keep {current_model}.')

    app.logger.info(response)
    return jsonify(response)  # response must be json serializable!


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/predict

    Returns predictions
    """
    # Get POST json data
    json = request.get_json()
    app.logger.info(json)

    # TODO:
    #raise NotImplementedError("TODO: implement this enpdoint")

    try:
        X = pd.read_json(json)[get_feature(model_name)]
        response = {"status": "success", "predictions": model.predict_proba(X)[:,1]}
        app.logger.info(f'Predictions : {response}.')
    except:
        response = {"status": "failure", "predictions": None}
        app.logger.warning(f'Unable to calcul predictions.')
    
    app.logger.info(response)
    return jsonify(response)  # response must be json serializable!

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)
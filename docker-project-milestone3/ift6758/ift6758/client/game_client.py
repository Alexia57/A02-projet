import json
import requests
import pandas as pd
import logging
import os

class GameClient:
    def __init__(self):
        self.tracker = 0
        self.game_id = None
        self.game = None
        self.home_team = None
        self.away_team = None
        self.dashboard_time = float('inf')
        self.dashboard_period = 0
        
    def download_game_data(self, path, game_id):
        '''
        Retrieve match data for the provided game_id from the API and save it as a JSON file with the right path.
        '''
        self.game_id = game_id
        file_path = os.path.join(os.path.abspath(path), f"{self.game_id}.json")
        base_url = "https://api-web.nhle.com/v1/gamecenter/"

        if not file_path.is_file():
            data = requests.get(base_url)

            if data.status_code == 404:
                return None
            else:
                with open(file_path, 'w') as outfile:
                    json.dump(data.json(), outfile)

        return file_path
    
    def update_events(self):
        """
        Updates the match events based on the downloaded data.
        """
        if self.game_id is None:
            print("Please ensure that you first selected the game ID by using download_game_data.")
            return None

        file_path = self.download_game_data(self.game_id, ".")

        with open(file_path, 'r') as f:
            game_data = json.load(f)

        self.game = pd.DataFrame(game_data['plays'])
        self.home_team = game_data['homeTeam']['name']['default']
        self.away_team = game_data['awayTeam']['name']['default']

        self.team_home_id = game_data['homeTeam']['id']
        self.team_away_id = game_data['awayTeam']['id']

        # etc ...

        return self.game


    
    def clean_data(dir_path):
        pass




    def ping_game(self,file_path):
        #df_game_tidied = tidy_data(file_path)
        #df_game_features = feature_engineering(df_game_tidied)
        #last_event = df_game_features.iloc[-1]
        #self.game = df_game_features
        #self.update_model_df_length()
        #tracker = self.model_df_length
       
        #return df_game_features, last_event, tracker
        pass
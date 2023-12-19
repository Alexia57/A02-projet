import json
import requests
import pandas as pd
import logging
import os
import numpy as np

class GameClient:
    def __init__(self):
        self.tracker = 0
        self.game_data = None
        self.tracker = 0
        #self.game_id = None
        #self.game = None
        #self.home_team = None
        #self.away_team = None
        #self.dashboard_time = float('inf')
        #self.dashboard_period = 0
        
    #def download_game_data(self, path, game_id):
    #    '''
    #    Retrieve match data for the provided game_id from the API and save it as a JSON file with the right path.
    #    '''
    #    self.game_id = game_id
    #    file_path = os.path.join(os.path.abspath(path), f"{self.game_id}.json")
    #    base_url = "https://api-web.nhle.com/v1/gamecenter/"
#
    #    if not file_path.is_file():
    #        data = requests.get(base_url)
#
    #        if data.status_code == 404:
    #            return None
    #        else:
    #            with open(file_path, 'w') as outfile:
    #                json.dump(data.json(), outfile)
#
    #    return file_path

    def clean_a_game(self, df):
        """
        json_path: path vers le fichier json

        clean et sauvegarde un df clean
        """

        df['gameId'] = df.id
        df['homeTeamLatestScore'] = pd.json_normalize(df['homeTeam']).score
        df['awayTeamLatestScore'] = pd.json_normalize(df['awayTeam']).score
        df['homeTeamId'] = pd.json_normalize(df['homeTeam']).id
        df['awayTeamId'] = pd.json_normalize(df['awayTeam']).id
        df['homeTeamAbbrev'] = pd.json_normalize(df['homeTeam']).abbrev
        df['awayTeamAbbrev'] = pd.json_normalize(df['awayTeam']).abbrev
        df['homeTeamName'] = pd.json_normalize(df['homeTeam'])['name.default']
        df['awayTeamName'] = pd.json_normalize(df['awayTeam'])['name.default']

        dp = df.iloc[:,[-1, -2, -3, -4, -5, -6, -7, -8, -9, -10]]
        dp = dp.explode('plays')

        d = pd.json_normalize(dp.plays).set_index(dp.index)
        d['homeTeamLatestScore'] = dp['homeTeamLatestScore']
        d['awayTeamLatestScore'] = dp['awayTeamLatestScore']
        d['homeTeamId'] = dp['homeTeamId']
        d['awayTeamId'] = dp['awayTeamId']
        d['gameId'] = dp['gameId']
        d['homeTeamAbbrev'] = dp['homeTeamAbbrev']
        d['awayTeamAbbrev'] = dp['awayTeamAbbrev']
        d['homeTeamName'] = dp['homeTeamName']
        d['awayTeamName'] = dp['awayTeamName']

        d = d[d['details.shotType'].notna()]
        d = d[['gameId', 'period', 'timeInPeriod', 'timeRemaining', 'typeDescKey', 'details.xCoord', 'details.yCoord', 'situationCode', 'details.eventOwnerTeamId', 'homeTeamLatestScore', 'awayTeamLatestScore', 'homeTeamId', 'awayTeamId', 'homeTeamName', 'awayTeamName', 'homeTeamAbbrev', 'awayTeamAbbrev']]

        d['details.eventOwnerTeamId']=d['details.eventOwnerTeamId'].astype(int)

        # Coordonnées des camps gauche et droit
        coord_camp_gauche = (-90, 0)
        coord_camp_droit = (90, 0)

        d['distanceToNet'] = np.sqrt(np.minimum((d['details.xCoord'] - coord_camp_gauche[0])**2 + (d['details.yCoord'] - coord_camp_gauche[1])**2, (d['details.xCoord'] - coord_camp_droit[0])**2 + (d['details.yCoord'] - coord_camp_droit[1])**2))

        # Calculer l'angle relatif du joueur par rapport au filet (filet gauche)
        d['relativeAngleToNet'] = np.degrees(np.arctan2(d['details.yCoord'], d['details.xCoord'] - coord_camp_gauche[0]))

        d['isGoal'] = (d['typeDescKey']=='goal').astype(int)
        d['isHome'] = d['details.eventOwnerTeamId']==d['homeTeamId']
        d['filetVide'] = ((((d['situationCode'].astype(int)*d['isHome'])//1000+(d['situationCode'].astype(int)*(d['isHome']-1)*(-1))%10)-1)*(-1))
        d['time_left'] = d['timeRemaining']
        final = d[['gameId', 'homeTeamLatestScore', 'awayTeamLatestScore', 'period', 'time_left', 'timeInPeriod', 'typeDescKey', 'homeTeamId', 'awayTeamId', 'details.eventOwnerTeamId', 'homeTeamName', 'awayTeamName', 'homeTeamAbbrev', 'awayTeamAbbrev', 'distanceToNet', 'relativeAngleToNet', 'filetVide', 'isGoal']]
        return final

    def grab_a_game(self, game_id):
        base_url = "https://api-web.nhle.com/v1/gamecenter/"
        url = f"{base_url}{game_id}/play-by-play/"
        data = []  # Stocker les données de tous les matchs

        response = requests.get(url)

        if response.status_code == 200:
            game_data = response.json()
            data.append(game_data)
            df=pd.DataFrame(data)
            return self.clean_a_game(df)
        else:
            print(f"Failed to download the game {game_id}")
            #is_game_number = False
    
    #def update_tracker(self, df_live_game_features):
    #    if df_live_game_features is not None and not df_live_game_features.empty:
    #        self.tracker = df_live_game_features.shape[0]
#
    #def get_unprocessed_events(self):
    #    if self.live_game_data is not None:
    #        return self.live_game_data.iloc[self.tracker:]


    def update_model_df_length(self):
        self.model_df_length = self.game.shape[0]


    def ping_game(self,game_id):
        """
        game_id: le id du game voulu

        return: (df, dfToUpdate)
        df: le df de la game en entier
        dfToUpdate: le df contenant UNIQUEMENT les évènements des nouveaux évènement
        EN D'AUTRE TERME, la première utilisation devrait donner la même chose que df, la deuxième donne un df vide, puisque qu'il n'y aura pas de nouveaux events rajouté depuis le premier

        """
        df = self.grab_a_game(game_id)
        last_event = df.iloc[-1]
        self.game = df
        dfToUpdate = df.iloc[self.tracker:, :]
        self.update_model_df_length()
        self.tracker = self.model_df_length

        return (df, dfToUpdate)
    
if __name__ == '__main__':
    gc = GameClient()
    print(gc.ping_game(game_id=2022030411)[1])
    print(gc.model_df_length)

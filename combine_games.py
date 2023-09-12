import pandas as pd
import re

team_stats = pd.read_csv('./data/teams/Combined_Teams.csv')
combined_game_logs = pd.DataFrame()
game_logs_dir = './data/games/'
#combine all game logs into one dataframe and save to csv
unique_match_id_column = 'MatchID'
for team in team_stats['Team']:
    team_game_logs = pd.read_csv(game_logs_dir + team + '_22_23.csv')
    team_game_logs[unique_match_id_column] = team_game_logs['Date'].apply(lambda x: re.search(r'(\d{14})', x).group(1) if re.search(r'(\d{14})', x) else '')

    team_game_logs['Date'] = team_game_logs['Date'].apply(lambda x: re.search(r'>([^<]+)<', x).group(1) if pd.notnull(x) else x)
    # possible lines
    team_game_logs['Outcome'] = team_game_logs['PTS'].apply(lambda x: 1 if x == 3 else -1 if x == 0 else 0)
    team_game_logs['Moneyline'] = team_game_logs['PTS'].apply(lambda x: 1 if x == 3 or x == 2 else 0)
    # handicap bets
    # H1-1.5 = home team wins by 2 or more
    # H2+1.5 = away team wins or draws
    team_game_logs['H1-1.5'] = team_game_logs.apply(lambda row: 1 if row['GF']-1.5 > row['GA'] else 0, axis=1)
    team_game_logs['H1-2.5'] = team_game_logs.apply(lambda row: 1 if row['GF']-2.5 > row['GA'] else 0, axis=1)
    team_game_logs['H2+1.5'] = team_game_logs.apply(lambda row: 1 if row['GA']+1.5 > row['GF'] else 0, axis=1)
    team_game_logs['H2+2.5'] = team_game_logs.apply(lambda row: 1 if row['GA']+2.5 > row['GF'] else 0, axis=1)

    team_game_logs['Venue'] = team_game_logs['Venue'].apply(lambda x: 1 if x == "home" else 0)
    team_game_logs['Phase'] = team_game_logs['Phase'].apply(lambda x: 1 if x == "Regular Season" else 0 if x == "Playoffs" else -1)
    # total goals
    team_game_logs['TG'] = team_game_logs['GF'] + team_game_logs['GA']
    team_game_logs['OU4.5'] = team_game_logs['TG'].apply(lambda x: 1 if x > 4 else 0)
    team_game_logs['OU5.5'] = team_game_logs['TG'].apply(lambda x: 1 if x > 5 else 0)
    team_game_logs['OU6.5'] = team_game_logs['TG'].apply(lambda x: 1 if x > 6 else 0)
    team_game_logs['OU7.5'] = team_game_logs['TG'].apply(lambda x: 1 if x > 7 else 0)
    
    if unique_match_id_column not in team_game_logs.columns:
        print('No MatchID column in ' + team + ' game logs')
        continue
    combined_game_logs = pd.concat([combined_game_logs, team_game_logs], ignore_index=True)
    print('Added ' + team + ' game logs to combined game logs')
    
combined_game_logs = combined_game_logs.sort_values(by=['Date'])
combined_game_logs = combined_game_logs.drop_duplicates(subset=unique_match_id_column)
combined_game_logs.to_csv('./data/games/Combined_Games.csv', index=False)
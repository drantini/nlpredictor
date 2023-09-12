import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
team_stats = pd.read_csv('./data/teams/22_23-Teams.csv')
games = pd.read_csv('./data/games/Combined_Games.csv')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

features = ['pts_diff', 'shots_for_diff', 'leading_diff', 'h2h_scored_avg_left', 'h2h_conceded_avg_left', 'Venue', 'corsi_for_diff', 'corsi_against_diff', 'goals_for_diff', 'goals_against_diff', 'pp_percentage_diff', 'pk_percentage_diff', 'shots_against_diff', 'win_r5_left', 'draw_r5_left', 'lose_r5_left', 'scored_avg_r5_left', 'conceded_avg_r5_left', 'win_r5_right', 'draw_r5_right', 'lose_r5_right', 'scored_avg_r5_right', 'conceded_avg_r5_right', 'h2h_win_ratio_left', 'h2h_draw_ratio_left', 'h2h_lose_ratio_left']


def get_last_5_h2h(team1_name, team2_name, date): 
    h2h_last_5 = games[((games['Team'] == team1_name) & (games['Opp.'] == team2_name)) | ((games['Team'] == team2_name) & (games['Opp.'] == team1_name))]
    h2h_last_5 = h2h_last_5[h2h_last_5['Date'] < date]
    h2h_last_5 = h2h_last_5.tail(5)
    return h2h_last_5

def get_last_5_team(team_name, date):
    team_last_5 = games[(games['Team'] == team_name) & (games['Date'] < date)].tail(5);
    team_last_5_on_right = games[(games['Opp.'] == team_name) & (games['Date'] < date)].tail(5);
    team_last_5 = pd.concat([team_last_5, team_last_5_on_right])
    # get last 5 before matchup date
    team_last_5 = team_last_5[team_last_5['Date'] < date]
    # get last 5 from both results
    team_last_5 = team_last_5.tail(5)
    return team_last_5

def populate_matchup_data(matchup):
    shots_per_match_diff = matchup['SF/60_left']-matchup['SF/60_right']
    shots_against_per_match_diff = matchup['SA/60_left']-matchup['SA/60_right']
    matchup['shots_for_diff'] = shots_per_match_diff
    matchup['shots_against_diff'] = shots_against_per_match_diff
    goals_per_match_diff = matchup['GF_/60_left']-matchup['GF_/60_right']
    goals_against_per_match_diff = matchup['GA_/60_left']-matchup['GA_/60_right']
    pp_percentage_diff = matchup['PP%_left']-matchup['PP%_right']
    pk_percentage_diff = matchup['PK%_left']-matchup['PK%_right']
    matchup['pp_percentage_diff'] = pp_percentage_diff
    matchup['pk_percentage_diff'] = pk_percentage_diff

    matchup['goals_for_diff'] = goals_per_match_diff
    matchup['goals_against_diff'] = goals_against_per_match_diff
    points_per_match_diff = matchup['PTS/GP_left']-matchup['PTS/GP_right'];
    matchup['pts_diff'] = points_per_match_diff

    leading_diff = matchup['Leading%_left']-matchup['Leading%_right']
    matchup['leading_diff'] = leading_diff
    trailing_diff = matchup['Trailing%_left']-matchup['Trailing%_right']
    matchup['trailing_diff'] = trailing_diff

    corsi_per_match_diff = matchup['CF/60_left']-matchup['CF/60_right']
    corsi_against_per_match_diff = matchup['CA/60_left']-matchup['CA/60_right']
    matchup['corsi_for_diff'] = corsi_per_match_diff
    matchup['corsi_against_diff'] = corsi_against_per_match_diff

    eq_goals_diff = matchup['EQ G±_left']-matchup['EQ G±_right']
    matchup['eq_goals_diff'] = eq_goals_diff




team_stats.drop(['Team name', 'Season'], axis=1, inplace=True)
# add team stats to games dataframe
# Team = _left
# Opp. = _right
games = pd.merge(games, team_stats, left_on='Team', right_on='Team', suffixes=('', '_left'))
# merge again for opponent stats
games = pd.merge(games, team_stats, left_on='Opp.', right_on='Team', suffixes=('', '_right'))
# write to csv
# sort by date
games = games.sort_values(by=['Date'])
for index, row in games.iterrows():
    # get last 5 games where team was either Opp. or Team
    team_left_last_5 = get_last_5_team(row['Team'], row['Date'])
    left_last_len = len(team_left_last_5)
    
    team_right_last_5 = get_last_5_team(row['Opp.'], row['Date'])
    right_last_len = len(team_right_last_5)

    #get last h2h games
    h2h_last_5 = get_last_5_h2h(row['Team'], row['Opp.'], row['Date'])
    h2h_last_len = len(h2h_last_5)

    shots_per_match_diff = row['SF/60']-row['SF/60_right']
    shots_against_per_match_diff = row['SA/60']-row['SA/60_right']
    games.at[index, 'shots_for_diff'] = shots_per_match_diff
    games.at[index, 'shots_against_diff'] = shots_against_per_match_diff
    goals_per_match_diff = row['GF_/60']-row['GF_/60_right']
    goals_against_per_match_diff = row['GA_/60']-row['GA_/60_right']
    pp_percentage_diff = row['PP%_left']-row['PP%_right']
    pk_percentage_diff = row['PK%_left']-row['PK%_right']
    games.at[index, 'pp_percentage_diff'] = pp_percentage_diff
    games.at[index, 'pk_percentage_diff'] = pk_percentage_diff

    games.at[index, 'goals_for_diff'] = goals_per_match_diff
    games.at[index, 'goals_against_diff'] = goals_against_per_match_diff
    points_per_match_diff = row['PTS/GP']-row['PTS/GP_right'];
    games.at[index, 'pts_diff'] = points_per_match_diff
    
    leading_diff = row['Leading%_left']-row['Leading%_right']
    games.at[index, 'leading_diff'] = leading_diff
    trailing_diff = row['Trailing%_left']-row['Trailing%_right']
    games.at[index, 'trailing_diff'] = trailing_diff

    corsi_per_match_diff = row['CF/60']-row['CF/60_right']
    corsi_against_per_match_diff = row['CA/60']-row['CA/60_right']
    games.at[index, 'corsi_for_diff'] = corsi_per_match_diff
    games.at[index, 'corsi_against_diff'] = corsi_against_per_match_diff
    

    eq_goals_diff = row['EQ G±_left']-row['EQ G±_right']
    games.at[index, 'eq_goals_diff'] = eq_goals_diff


    win_ratio_left = 0
    draw_ratio_left = 0
    lose_ratio_left = 0
    scored_avg_left = 0
    conceded_avg_left = 0
    h2h_left_win_ratio = 0
    h2h_left_draw_ratio = 0
    h2h_left_lose_ratio = 0

    win_ratio_right = 0
    draw_ratio_right = 0
    lose_ratio_right = 0
    scored_avg_right = 0
    conceded_avg_right = 0
    scored_avg_h2h = 0
    conceded_avg_h2h = 0
    for index2, last_game in team_left_last_5.iterrows():
        is_team_left = last_game['Team'] == row['Team']
        if last_game['Outcome'] == 1 and is_team_left:
            win_ratio_left += 1
        elif last_game['Outcome'] == -1 and not is_team_left:
            win_ratio_left += 1
        if last_game['Outcome'] == 0:
            draw_ratio_left += 1
        if last_game['Outcome'] == -1 and is_team_left:
            lose_ratio_left += 1
        elif last_game['Outcome'] == 1 and not is_team_left:
            lose_ratio_left += 1
        scored_avg_left += last_game['GF'] if is_team_left else last_game['GA']
        conceded_avg_left += last_game['GA'] if is_team_left else last_game['GF']
    for index2, last_game in team_right_last_5.iterrows():
        is_team_left = last_game['Team'] == row['Team']
        if last_game['Outcome'] == 1 and is_team_left:
            win_ratio_right += 1
        elif last_game['Outcome'] == -1 and not is_team_left:
            win_ratio_right += 1
        if last_game['Outcome'] == 0:
            draw_ratio_right += 1
        if last_game['Outcome'] == -1 and is_team_left:
            lose_ratio_right += 1
        elif last_game['Outcome'] == 1 and not is_team_left:
            lose_ratio_right += 1
        scored_avg_right += last_game['GF'] if is_team_left else last_game['GA']
        conceded_avg_right += last_game['GA'] if is_team_left else last_game['GF']
    for index2, last_game in h2h_last_5.iterrows():
        is_team_left = last_game['Team'] == row['Team']
        if last_game['Outcome'] == 1 and is_team_left:
            h2h_left_win_ratio += 1
        elif last_game['Outcome'] == -1 and not is_team_left:
            h2h_left_win_ratio += 1
        if last_game['Outcome'] == 0:
            h2h_left_draw_ratio += 1
        if last_game['Outcome'] == -1 and is_team_left:
            h2h_left_lose_ratio += 1
        elif last_game['Outcome'] == 1 and not is_team_left:
            h2h_left_lose_ratio += 1
        scored_avg_h2h = last_game['GF'] if is_team_left else last_game['GA']
        conceded_avg_h2h = last_game['GA'] if is_team_left else last_game['GF']


    if left_last_len == 0:
        left_last_len = 1
    if right_last_len == 0:
        right_last_len = 1 
    win_ratio_left = win_ratio_left / left_last_len
    lose_ratio_left = lose_ratio_left / left_last_len 
    draw_ratio_left = draw_ratio_left/ left_last_len 
    scored_avg_left = scored_avg_left / left_last_len
    conceded_avg_left = conceded_avg_left / left_last_len 
    games.at[index, 'win_r5_left'] = win_ratio_left
    games.at[index, 'draw_r5_left'] = draw_ratio_left
    games.at[index, 'lose_r5_left'] = lose_ratio_left
    games.at[index, 'scored_avg_r5_left'] = scored_avg_left
    games.at[index, 'conceded_avg_r5_left'] = conceded_avg_left
    win_ratio_right = win_ratio_right / right_last_len
    lose_ratio_right = lose_ratio_right / right_last_len 
    draw_ratio_right = draw_ratio_right / right_last_len
    scored_avg_right = scored_avg_right / right_last_len
    conceded_avg_right = conceded_avg_right / right_last_len 
    games.at[index, 'win_r5_right'] = win_ratio_right
    games.at[index, 'draw_r5_right'] = draw_ratio_right
    games.at[index, 'lose_r5_right'] = lose_ratio_right
    games.at[index, 'scored_avg_r5_right'] = scored_avg_right
    games.at[index, 'conceded_avg_r5_right'] = conceded_avg_right
    if h2h_last_len == 0:
        h2h_last_len = 1
    h2h_left_win_ratio = h2h_left_win_ratio / h2h_last_len
    h2h_left_draw_ratio = h2h_left_draw_ratio / h2h_last_len
    h2h_left_lose_ratio = h2h_left_lose_ratio / h2h_last_len
    scored_avg_h2h = scored_avg_h2h / h2h_last_len
    conceded_avg_h2h = conceded_avg_h2h / h2h_last_len
    games.at[index, 'h2h_win_ratio_left'] = h2h_left_win_ratio
    games.at[index, 'h2h_draw_ratio_left'] = h2h_left_draw_ratio
    games.at[index, 'h2h_lose_ratio_left'] = h2h_left_lose_ratio
    games.at[index, 'h2h_scored_avg_left'] = scored_avg_h2h
    games.at[index, 'h2h_conceded_avg_left'] = conceded_avg_h2h
    

   
target_var = 'Outcome'
bookmaker_margin = 6.15
team1_name = 'SCB'
team2_name = 'LHC'
# create dataframe for matchups between team1 and team2
x = games[features]
y = games[target_var]

# train model
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1) 
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print('Accuracy: ' + str(accuracy_score(y_test, y_pred)))
print(classification_report(y_test, y_pred))

#real model train
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(x, y)
# feature importance graph
importance = model.feature_importances_

plt.figure(figsize=(10, 10))
plt.title("Feature importances")
plt.barh([x for x in range(len(importance))], importance, tick_label=features)
# plt.show()
plt.savefig('./data/feature_importance_' + target_var + '.png')


# predict

teams_stats = team_stats[(team_stats['Team'] == team1_name) | (team_stats['Team'] == team2_name)]
team1_stats = teams_stats[teams_stats['Team'] == team1_name]
team2_stats = teams_stats[teams_stats['Team'] == team2_name]
team1_stats = team1_stats.reset_index(drop=True)
team2_stats = team2_stats.reset_index(drop=True)
matchup = pd.DataFrame()
# add team stats to games dataframe
matchup = pd.concat([team1_stats.add_suffix('_left'), team2_stats.add_suffix('_right')], axis=1)
matchup = matchup.reset_index(drop=True)
# combine two rows into one
matchup['Venue'] = 1
# populate matchup data
populate_matchup_data(matchup)
team_left_last_5 = get_last_5_team(team1_name, '2023-10-01')
left_last_len = len(team_left_last_5)
team_right_last_5 = get_last_5_team(team2_name, '2023-10-01')
right_last_len = len(team_right_last_5)
h2h_last_5 = get_last_5_h2h(team1_name, team2_name, '2023-10-01')
h2h_last_len = len(h2h_last_5)
win_ratio_left = 0
draw_ratio_left = 0
lose_ratio_left = 0
scored_avg_left = 0
conceded_avg_left = 0
h2h_left_win_ratio = 0
h2h_left_draw_ratio = 0
h2h_left_lose_ratio = 0

win_ratio_right = 0
draw_ratio_right = 0
lose_ratio_right = 0
scored_avg_right = 0
conceded_avg_right = 0
scored_avg_h2h = 0
conceded_avg_h2h = 0
for index2, last_game in team_left_last_5.iterrows():
    is_team_left = last_game['Team'] == team1_name
    if last_game['Outcome'] == 1 and is_team_left:
        win_ratio_left += 1
    elif last_game['Outcome'] == -1 and not is_team_left:
        win_ratio_left += 1
    if last_game['Outcome'] == 0:
        draw_ratio_left += 1
    if last_game['Outcome'] == -1 and is_team_left:
        lose_ratio_left += 1
    elif last_game['Outcome'] == 1 and not is_team_left:
        lose_ratio_left += 1
    scored_avg_left += last_game['GF'] if is_team_left else last_game['GA']
    conceded_avg_left += last_game['GA'] if is_team_left else last_game['GF']
for index2, last_game in team_right_last_5.iterrows():
    is_team_left = last_game['Team'] == team1_name
    if last_game['Outcome'] == 1 and is_team_left:
        win_ratio_right += 1
    elif last_game['Outcome'] == -1 and not is_team_left:
        win_ratio_right += 1
    if last_game['Outcome'] == 0:
        draw_ratio_right += 1
    if last_game['Outcome'] == -1 and is_team_left:
        lose_ratio_right += 1
    elif last_game['Outcome'] == 1 and not is_team_left:
        lose_ratio_right += 1
    scored_avg_right += last_game['GF'] if is_team_left else last_game['GA']
    conceded_avg_right += last_game['GA'] if is_team_left else last_game['GF']
for index2, last_game in h2h_last_5.iterrows():
    is_team_left = last_game['Team'] == team1_name
    if last_game['Outcome'] == 1 and is_team_left:
        h2h_left_win_ratio += 1
    elif last_game['Outcome'] == -1 and not is_team_left:
        h2h_left_win_ratio += 1
    if last_game['Outcome'] == 0:
        h2h_left_draw_ratio += 1
    if last_game['Outcome'] == -1 and is_team_left:
        h2h_left_lose_ratio += 1
    elif last_game['Outcome'] == 1 and not is_team_left:
        h2h_left_lose_ratio += 1
    scored_avg_h2h = last_game['GF'] if is_team_left else last_game['GA']
    conceded_avg_h2h = last_game['GA'] if is_team_left else last_game['GF'] 

if left_last_len == 0:
    left_last_len = 1
if right_last_len == 0:
    right_last_len = 1
win_ratio_left = win_ratio_left / left_last_len
lose_ratio_left = lose_ratio_left / left_last_len
draw_ratio_left = draw_ratio_left/ left_last_len
scored_avg_left = scored_avg_left / left_last_len
conceded_avg_left = conceded_avg_left / left_last_len
matchup['win_r5_left'] = win_ratio_left
matchup['draw_r5_left'] = draw_ratio_left
matchup['lose_r5_left'] = lose_ratio_left
matchup['scored_avg_r5_left'] = scored_avg_left
matchup['conceded_avg_r5_left'] = conceded_avg_left
win_ratio_right = win_ratio_right / right_last_len
lose_ratio_right = lose_ratio_right / right_last_len
draw_ratio_right = draw_ratio_right / right_last_len
scored_avg_right = scored_avg_right / right_last_len
conceded_avg_right = conceded_avg_right / right_last_len
matchup['win_r5_right'] = win_ratio_right
matchup['draw_r5_right'] = draw_ratio_right
matchup['lose_r5_right'] = lose_ratio_right
matchup['scored_avg_r5_right'] = scored_avg_right
matchup['conceded_avg_r5_right'] = conceded_avg_right
if h2h_last_len == 0:
    h2h_last_len = 1
h2h_left_win_ratio = h2h_left_win_ratio / h2h_last_len
h2h_left_draw_ratio = h2h_left_draw_ratio / h2h_last_len
h2h_left_lose_ratio = h2h_left_lose_ratio / h2h_last_len
scored_avg_h2h = scored_avg_h2h / h2h_last_len
conceded_avg_h2h = conceded_avg_h2h / h2h_last_len
matchup['h2h_win_ratio_left'] = h2h_left_win_ratio
matchup['h2h_draw_ratio_left'] = h2h_left_draw_ratio
matchup['h2h_lose_ratio_left'] = h2h_left_lose_ratio
matchup['h2h_scored_avg_left'] = scored_avg_h2h
matchup['h2h_conceded_avg_left'] = conceded_avg_h2h
# predict
matchup = matchup[features]
y_pred = model.predict_proba(matchup)
# calculate odds
line = y_pred[0]

idx=0

print(team1_name + ' vs ' + team2_name + ' ' + target_var)
for prob in line:
    adjusted_prob = prob*100+(bookmaker_margin/len(line))
    adjusted_prob = adjusted_prob/100
    print(f'Outcome {idx}: {round(1/prob, 2)} - Adj: {round(1/adjusted_prob, 2)}')
    idx += 1

# remove duplicates
matchup = matchup.loc[:,~matchup.columns.duplicated()]
# write to csv

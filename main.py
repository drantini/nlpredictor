import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
team_stats = pd.read_csv('./data/teams/22_23-Teams.csv')
games = pd.read_csv('./data/games/Combined_Games.csv')
exits = pd.read_csv('./data/teams/exits.csv')
exits_denial = pd.read_csv('./data/teams/exits_denial.csv')
entries_denial = pd.read_csv('./data/teams/entries_denial.csv')
entries = pd.read_csv('./data/teams/entries.csv')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# TODO: Learn how to use HD passes and HD passes against to improve model
features = ['pts_diff', 'eq_goals_diff', 'exits_denial_diff', 'entries_denial_diff', 'exits_diff', 'entries_diff', 'exp_goals_left', 'exp_goals_right', 'attack_strength_diff', 'defence_strength_diff', 'shots_for_diff', 'leading_diff', 'h2h_scored_avg_left', 'h2h_conceded_avg_left', 'Phase', 'Venue', 'corsi_for_diff', 'corsi_against_diff', 'goals_for_diff', 'goals_against_diff', 'pp_percentage_diff', 'pk_percentage_diff', 'shots_against_diff', 'win_r5_left', 'draw_r5_left', 'lose_r5_left', 'scored_avg_r5_left', 'conceded_avg_r5_left', 'win_r5_right', 'draw_r5_right', 'lose_r5_right', 'scored_avg_r5_right', 'conceded_avg_r5_right', 'h2h_win_ratio_left', 'h2h_draw_ratio_left', 'h2h_lose_ratio_left']

target_var = 'Outcome'
league_stats = pd.read_csv('./data/22_23_league.csv')
# choose first row of league stats
league_stats = league_stats.iloc[0]
def get_dist(feature):
    mean = games[feature].mean()
    std = games[feature].std()
    dist = np.random.normal(mean, std, 10000)
    return dist

def get_samples(dist):
    samples = np.random.choice(dist, 10000)
    return samples

def monte_carlo_sim(model, its=5000):
    samples = []
    bankroll = 1000
    bet_size = 0.01
    for feature in features:
        dist = get_dist(feature)
        sample = get_samples(dist)
        samples.append(sample)
    
    simulated_data = pd.DataFrame(np.column_stack(samples), columns=features)
    simulated_outcomes = []
    avg_odds = 0
    for i in range(its):
        # randomly select row from simulated data
        row = simulated_data.sample(n=1, replace=True)
        probs = model.predict_proba(row)
        bet_amount = bankroll * bet_size
        team1_win_prob = probs[0][1]
        team2_win_prob = probs[0][0]
        
        team1_win_prob_percent = (team1_win_prob * 100)+(6.20/2)
        team2_win_prob_percent = (team2_win_prob * 100)+(6.20/2)
        team1_win_odds = 1 / (team1_win_prob_percent / 100)
        team2_win_odds = 1 / (team2_win_prob_percent / 100)
        team1_win_odds = round(team1_win_odds, 2)
        team2_win_odds = round(team2_win_odds, 2)
        ev_team1 = (team1_win_prob * bet_amount * team1_win_odds) - (bet_amount * (1-team1_win_prob))
        ev_team2 = (team2_win_prob * bet_amount * team2_win_odds) - (bet_amount * (1-team2_win_prob))

        winner = np.random.choice([1, 0], p=[team1_win_prob, team2_win_prob])
        if ev_team1 > ev_team2:
            avg_odds += team1_win_odds
            if winner == 1:
                bankroll += bet_amount * team1_win_odds
            else:
                bankroll -= bet_amount
        else:
            avg_odds += team2_win_odds
            if winner == 0:
                bankroll += bet_amount * team2_win_odds
            else:
                bankroll -= bet_amount
        simulated_outcomes.append(bankroll)
    
    avg_odds = avg_odds / its
    print('Average odds: ' + str(avg_odds))
    return simulated_outcomes


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

    #attack strength
    league_avg_goals = league_stats['GF']/league_stats['GP'];
    league_avg_goals_conceded = league_stats['GA']/league_stats['GP'];
    avg_goals_left = row['GF_left']/row['GP'];
    avg_goals_conceded_left = row['GA_left']/row['GP'];
    attack_strength_left = avg_goals_left/league_avg_goals;
    defence_strength_left = avg_goals_conceded_left/league_avg_goals_conceded;
    games.at[index, 'attack_strength_left'] = attack_strength_left
    games.at[index, 'defence_strength_left'] = defence_strength_left

    avg_goals_right = row['GF_right']/row['GP_right'];
    avg_goals_conceded_right = row['GA_right']/row['GP_right'];
    attack_strength_right = avg_goals_right/league_avg_goals;
    defence_strength_right = avg_goals_conceded_right/league_avg_goals_conceded;
    games.at[index, 'attack_strength_right'] = attack_strength_right
    games.at[index, 'defence_strength_right'] = defence_strength_right

    exp_left_goals = attack_strength_left * defence_strength_right * league_avg_goals;
    exp_right_goals = attack_strength_right * defence_strength_left * league_avg_goals;
    games.at[index, 'attack_strength_diff'] = attack_strength_left - attack_strength_right
    games.at[index, 'defence_strength_diff'] = defence_strength_left - defence_strength_right
    games.at[index, 'exp_goals_left'] = exp_left_goals
    games.at[index, 'exp_goals_right'] = exp_right_goals


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

    team_left_exits = exits[exits['Team'] == row['Team']]
    team_left_exits = team_left_exits['ControlledExits %'].mean()
    team_right_exits = exits[exits['Team'] == row['Opp.']]
    team_right_exits = team_right_exits['ControlledExits %'].mean()
    exits_diff = team_left_exits - team_right_exits
    games.at[index, 'exits_diff'] = exits_diff

    team_left_exits_denial = exits_denial[exits_denial['Team'] == row['Team']]
    team_left_exits_denial = team_left_exits_denial['ControlledExits %against'].mean()
    team_right_exits_denial = exits_denial[exits_denial['Team'] == row['Opp.']]
    team_right_exits_denial = team_right_exits_denial['ControlledExits %against'].mean()
    exits_denial_diff = team_left_exits_denial - team_right_exits_denial
    games.at[index, 'exits_denial_diff'] = exits_denial_diff


    team_left_entries = entries[entries['Team'] == row['Team']]
    team_left_entries = team_left_entries['ControlledEntries %'].mean()
    team_right_entries = entries[entries['Team'] == row['Opp.']]
    team_right_entries = team_right_entries['ControlledEntries %'].mean()
    entries_diff = team_left_entries - team_right_entries
    games.at[index, 'entries_diff'] = entries_diff

    team_left_entries_denial = entries_denial[entries_denial['Team'] == row['Team']]
    team_left_entries_denial = team_left_entries_denial['ControlledEntries %against'].mean()
    team_right_entries_denial = entries_denial[entries_denial['Team'] == row['Opp.']]
    team_right_entries_denial = team_right_entries_denial['ControlledEntries %against'].mean()
    entries_denial_diff = team_left_entries_denial - team_right_entries_denial
    games.at[index, 'entries_denial_diff'] = entries_denial_diff 

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
    

   
x = games[features]
y = games[target_var]

# train model
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)
model = RandomForestClassifier(n_estimators=400, min_samples_split=5, min_samples_leaf=2, max_depth=10, random_state=10)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print('Accuracy: ' + str(accuracy_score(y_test, y_pred)))
print(classification_report(y_test, y_pred))

#real model train
model = RandomForestClassifier(n_estimators=400, min_samples_split=5, min_samples_leaf=2, max_depth=10, random_state=10)
model.fit(x, y)

#save model
import pickle
filename = './data/models/' + target_var + '_model.sav'
pickle.dump(model, open(filename, 'wb'))

# 10 monte carlo simulations
# bankrolls = []
# for i in range(10):
#     bankrolls.append(monte_carlo_sim(model, 500))
# plt.figure(figsize=(10, 10))
# plt.title("Bankroll")
# plt.ylabel("Bankroll")
# plt.xlabel("Bets")
# for bankroll in bankrolls:
#     plt.plot(bankroll)
# # show line over 1000
# plt.axhline(y=1000, color='r', linestyle='-')
# plt.show()

# feature importance graph
importance = model.feature_importances_

plt.figure(figsize=(10, 10))
plt.title("Feature importances")
plt.barh([x for x in range(len(importance))], importance, tick_label=features)
# plt.show()
plt.savefig('./data/feature_importance_' + target_var + '.png')


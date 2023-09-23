import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import json
team_stats = pd.read_csv('./data/teams/Combined_Teams.csv')
games = pd.read_csv('./data/games/Combined_Games.csv')
exits = pd.read_csv('./data/teams/exits.csv')
hd_passes = pd.read_csv('./data/teams/hd_passes.csv')
exits_denial = pd.read_csv('./data/teams/exits_denial.csv')
entries_denial = pd.read_csv('./data/teams/entries_denial.csv')
entries = pd.read_csv('./data/teams/entries.csv')
config = json.load(open('./config.json'))
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

features = config['default_features'] 

target_var = 'Moneyline'
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
    team_last_5_on_away = games[(games['Opp.'] == team_name) & (games['Date'] < date)].tail(5);
    team_last_5 = pd.concat([team_last_5, team_last_5_on_away])
    # get last 5 before matchup date
    team_last_5 = team_last_5[team_last_5['Date'] < date]
    # get last 5 from both results
    team_last_5 = team_last_5.tail(5)
    return team_last_5




# add team stats to games dataframe
# Team = _home
# Opp. = _away
games = pd.merge(games, team_stats, left_on='Team', right_on='Team', suffixes=('', '_home'))
# merge again for opponent stats
games = pd.merge(games, team_stats, left_on='Opp.', right_on='Team', suffixes=('', '_away'))
# write to csv
# sort by date
games = games.sort_values(by=['Date'])
for index, row in games.iterrows():
    # get last 5 games where team was either Opp. or Team
    team_home_last_5 = get_last_5_team(row['Team'], row['Date'])
    left_last_len = len(team_home_last_5)
    
    team_away_last_5 = get_last_5_team(row['Opp.'], row['Date'])
    right_last_len = len(team_away_last_5)

    #get last h2h games
    h2h_last_5 = get_last_5_h2h(row['Team'], row['Opp.'], row['Date'])
    h2h_last_len = len(h2h_last_5)

    #attack strength
    league_avg_goals = league_stats['GF']/league_stats['GP'];
    league_avg_goals_conceded = league_stats['GA']/league_stats['GP'];
    avg_goals_home = row['GF_home']/row['GP'];
    avg_goals_conceded_home = row['GA_home']/row['GP'];
    attack_strength_home = avg_goals_home/league_avg_goals;
    defence_strength_home = avg_goals_conceded_home/league_avg_goals_conceded;
    games.at[index, 'attack_strength_home'] = attack_strength_home
    games.at[index, 'defence_strength_home'] = defence_strength_home

    avg_goals_away = row['GF_away']/row['GP_away'];
    avg_goals_conceded_away = row['GA_away']/row['GP_away'];
    attack_strength_away = avg_goals_away/league_avg_goals;
    defence_strength_away = avg_goals_conceded_away/league_avg_goals_conceded;
    games.at[index, 'attack_strength_away'] = attack_strength_away
    games.at[index, 'defence_strength_away'] = defence_strength_away

    exp_home_goals = attack_strength_home * defence_strength_away * league_avg_goals;
    exp_away_goals = attack_strength_away * defence_strength_home * league_avg_goals;
    games.at[index, 'attack_strength_diff'] = attack_strength_home - attack_strength_away
    games.at[index, 'defence_strength_diff'] = defence_strength_home - defence_strength_away
    games.at[index, 'exp_goals_home'] = exp_home_goals
    games.at[index, 'exp_goals_away'] = exp_away_goals


    shots_per_match_diff = row['SF/60']-row['SF/60_away']
    shots_against_per_match_diff = row['SA/60']-row['SA/60_away']
    games.at[index, 'shots_for_diff'] = shots_per_match_diff
    games.at[index, 'shots_against_diff'] = shots_against_per_match_diff
    goals_per_match_diff = row['GF_/60']-row['GF_/60_away']
    goals_against_per_match_diff = row['GA_/60']-row['GA_/60_away']
    pp_percentage_diff = row['PP%_home']-row['PP%_away']
    pk_percentage_diff = row['PK%_home']-row['PK%_away']
    games.at[index, 'pp_percentage_diff'] = pp_percentage_diff
    games.at[index, 'pk_percentage_diff'] = pk_percentage_diff

    games.at[index, 'goals_for_diff'] = goals_per_match_diff
    games.at[index, 'goals_against_diff'] = goals_against_per_match_diff
    points_per_match_diff = row['PTS/GP']-row['PTS/GP_away'];
    games.at[index, 'pts_diff'] = points_per_match_diff
    

    corsi_per_match_diff = row['CF/60']-row['CF/60_away']
    corsi_against_per_match_diff = row['CA/60']-row['CA/60_away']
    games.at[index, 'corsi_for_diff'] = corsi_per_match_diff
    games.at[index, 'corsi_against_diff'] = corsi_against_per_match_diff
    

    eq_goals_diff = row['EQ G±_home']-row['EQ G±_away']
    games.at[index, 'eq_goals_diff'] = eq_goals_diff

    team_home_exits = exits[exits['Team'] == row['Team']]
    team_home_exits = team_home_exits['ControlledExits %'].mean()
    team_away_exits = exits[exits['Team'] == row['Opp.']]
    team_away_exits = team_away_exits['ControlledExits %'].mean()
    exits_diff = team_home_exits - team_away_exits
    games.at[index, 'exits_diff'] = exits_diff

    team_home_exits_denial = exits_denial[exits_denial['Team'] == row['Team']]
    team_home_exits_denial = team_home_exits_denial['ControlledExits %against'].mean()
    team_away_exits_denial = exits_denial[exits_denial['Team'] == row['Opp.']]
    team_away_exits_denial = team_away_exits_denial['ControlledExits %against'].mean()
    exits_denial_diff = team_home_exits_denial - team_away_exits_denial
    games.at[index, 'exits_denial_diff'] = exits_denial_diff


    team_home_entries = entries[entries['Team'] == row['Team']]
    team_home_entries = team_home_entries['ControlledEntries %'].mean()
    team_away_entries = entries[entries['Team'] == row['Opp.']]
    team_away_entries = team_away_entries['ControlledEntries %'].mean()
    entries_diff = team_home_entries - team_away_entries
    games.at[index, 'entries_diff'] = entries_diff

    team_home_entries_denial = entries_denial[entries_denial['Team'] == row['Team']]
    team_home_entries_denial = team_home_entries_denial['ControlledEntries %against'].mean()
    team_away_entries_denial = entries_denial[entries_denial['Team'] == row['Opp.']]
    team_away_entries_denial = team_away_entries_denial['ControlledEntries %against'].mean()
    entries_denial_diff = team_home_entries_denial - team_away_entries_denial
    games.at[index, 'entries_denial_diff'] = entries_denial_diff 

    hdp_passes_home = hd_passes[hd_passes['Team'] == row['Team']]
    hdp_passes_away = hd_passes[hd_passes['Team'] == row['Opp.']]
    team_home_hdp_percent = hdp_passes_home['SuccessfulHDP %'].mean()
    team_away_hdp_percent = hdp_passes_away['SuccessfulHDP %'].mean()
    hdp_percent_diff = team_home_hdp_percent - team_away_hdp_percent
    games.at[index, 'hdp_percent_diff'] = hdp_percent_diff 





    win_ratio_home = 0
    draw_ratio_home = 0
    lose_ratio_home = 0
    scored_avg_home = 0
    conceded_avg_home = 0
    h2h_home_win_ratio = 0
    h2h_home_draw_ratio = 0
    h2h_home_lose_ratio = 0

    win_ratio_away = 0
    draw_ratio_away = 0
    lose_ratio_away = 0
    scored_avg_away = 0
    conceded_avg_away = 0
    scored_avg_h2h = 0
    conceded_avg_h2h = 0
    for index2, last_game in team_home_last_5.iterrows():
        is_team_home = last_game['Team'] == row['Team']
        if last_game['Outcome'] == 1 and is_team_home:
            win_ratio_home += 1
        elif last_game['Outcome'] == -1 and not is_team_home:
            win_ratio_home += 1
        if last_game['Outcome'] == 0:
            draw_ratio_home += 1
        if last_game['Outcome'] == -1 and is_team_home:
            lose_ratio_home += 1
        elif last_game['Outcome'] == 1 and not is_team_home:
            lose_ratio_home += 1
        scored_avg_home += last_game['GF'] if is_team_home else last_game['GA']
        conceded_avg_home += last_game['GA'] if is_team_home else last_game['GF']
    for index2, last_game in team_away_last_5.iterrows():
        is_team_home = last_game['Team'] == row['Team']
        if last_game['Outcome'] == 1 and is_team_home:
            win_ratio_away += 1
        elif last_game['Outcome'] == -1 and not is_team_home:
            win_ratio_away += 1
        if last_game['Outcome'] == 0:
            draw_ratio_away += 1
        if last_game['Outcome'] == -1 and is_team_home:
            lose_ratio_away += 1
        elif last_game['Outcome'] == 1 and not is_team_home:
            lose_ratio_away += 1
        scored_avg_away += last_game['GF'] if is_team_home else last_game['GA']
        conceded_avg_away += last_game['GA'] if is_team_home else last_game['GF']
    for index2, last_game in h2h_last_5.iterrows():
        is_team_home = last_game['Team'] == row['Team']
        if last_game['Outcome'] == 1 and is_team_home:
            h2h_home_win_ratio += 1
        elif last_game['Outcome'] == -1 and not is_team_home:
            h2h_home_win_ratio += 1
        if last_game['Outcome'] == 0:
            h2h_home_draw_ratio += 1
        if last_game['Outcome'] == -1 and is_team_home:
            h2h_home_lose_ratio += 1
        elif last_game['Outcome'] == 1 and not is_team_home:
            h2h_home_lose_ratio += 1
        scored_avg_h2h = last_game['GF'] if is_team_home else last_game['GA']
        conceded_avg_h2h = last_game['GA'] if is_team_home else last_game['GF']


    if left_last_len == 0:
        left_last_len = 1
    if right_last_len == 0:
        right_last_len = 1 
    win_ratio_home = win_ratio_home / left_last_len
    lose_ratio_home = lose_ratio_home / left_last_len 
    draw_ratio_home = draw_ratio_home/ left_last_len 
    scored_avg_home = scored_avg_home / left_last_len
    conceded_avg_home = conceded_avg_home / left_last_len 
    games.at[index, 'win_r5_home'] = win_ratio_home
    games.at[index, 'draw_r5_home'] = draw_ratio_home
    games.at[index, 'lose_r5_home'] = lose_ratio_home
    games.at[index, 'scored_avg_r5_home'] = scored_avg_home
    games.at[index, 'conceded_avg_r5_home'] = conceded_avg_home
    win_ratio_away = win_ratio_away / right_last_len
    lose_ratio_away = lose_ratio_away / right_last_len 
    draw_ratio_away = draw_ratio_away / right_last_len
    scored_avg_away = scored_avg_away / right_last_len
    conceded_avg_away = conceded_avg_away / right_last_len 
    games.at[index, 'win_r5_away'] = win_ratio_away
    games.at[index, 'draw_r5_away'] = draw_ratio_away
    games.at[index, 'lose_r5_away'] = lose_ratio_away
    games.at[index, 'scored_avg_r5_away'] = scored_avg_away
    games.at[index, 'conceded_avg_r5_away'] = conceded_avg_away
    if h2h_last_len == 0:
        h2h_last_len = 1
    h2h_home_win_ratio = h2h_home_win_ratio / h2h_last_len
    h2h_home_draw_ratio = h2h_home_draw_ratio / h2h_last_len
    h2h_home_lose_ratio = h2h_home_lose_ratio / h2h_last_len
    scored_avg_h2h = scored_avg_h2h / h2h_last_len
    conceded_avg_h2h = conceded_avg_h2h / h2h_last_len
    games.at[index, 'h2h_win_ratio_home'] = h2h_home_win_ratio
    games.at[index, 'h2h_draw_ratio_home'] = h2h_home_draw_ratio
    games.at[index, 'h2h_lose_ratio_home'] = h2h_home_lose_ratio
    games.at[index, 'h2h_scored_avg_home'] = scored_avg_h2h
    games.at[index, 'h2h_conceded_avg_home'] = conceded_avg_h2h
    

   
x = games[features]
y = games[target_var]

# train model
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)
model_log = LogisticRegression(random_state=15, max_iter=5000, solver='liblinear')
model_log.fit(x_train, y_train)
y_pred = model_log.predict(x_test)
y_pred_train = model_log.predict(x_train)
print('Logistic Regression - ' + target_var)
print('Accuracy: ' + str(accuracy_score(y_test, y_pred)))
print('Accuracy train: ' + str(accuracy_score(y_train, y_pred_train)))
print(classification_report(y_test, y_pred))
scores = cross_val_score(model_log, x, y, cv=5)
print('Cross validation scores: ' + str(scores))
print('Mean cross validation score: ' + str(scores.mean()))
print('Standard deviation of cross validation scores: ' + str(scores.std()))

#real model train
model = LogisticRegression(random_state=15, max_iter=5000, solver='liblinear') 
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
importance = model.coef_[0]
abs_importance = np.abs(importance)
importance = pd.DataFrame({
    'Feature': x.columns,
    'Importance': abs_importance
})
graph = importance.plot(x='Feature', y='Importance', kind='barh', figsize=(10, 10))
# save to png
fig = graph.get_figure()
fig.savefig('./data/feature_importance_' + target_var + '.png')

#correlation matrix
corr = games[features].corr()
plt.figure(figsize=(10, 10))
sns.heatmap(corr, annot=True)
plt.savefig('./data/correlation_matrix_' + target_var + '.png')

print(f'Model for {target_var} saved.')
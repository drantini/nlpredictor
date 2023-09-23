import os
import pandas as pd
import pickle
import json
model_dir = './data/models/'
team_stats = pd.read_csv('./data/teams/Combined_Teams.csv')
games = pd.read_csv('./data/games/Combined_Games_With_Stats.csv')
exits = pd.read_csv('./data/teams/exits.csv')
exits_denial = pd.read_csv('./data/teams/exits_denial.csv')
entries = pd.read_csv('./data/teams/entries.csv')
league_shots = pd.read_csv('./data/teams/shots_league.csv')
hd_passes = pd.read_csv('./data/teams/hd_passes.csv')
entries_denial = pd.read_csv('./data/teams/entries_denial.csv')
league_stats = pd.read_csv('./data/22_23_league.csv')
# choose first row of league stats
league_stats = league_stats.iloc[0]
config = json.load(open('./config.json'))
bookmaker_margin = config['bookmaker_margin'] 
ceiling_odd = config['ceiling_odd'] 
# get models in dir
models = {}
target_vars = []

features = config['default_features'] 
for model in os.listdir(model_dir): 
    if model.endswith('.sav'):
        type = model.split('_')[0]
        target_vars.append(type)
        models[type] = pickle.load(open(model_dir + model, 'rb'))


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

def populate_matchup_data(matchup, team1, team2):
    shots_per_match_diff = matchup['SF/60_home']-matchup['SF/60_away']
    shots_against_per_match_diff = matchup['SA/60_home']-matchup['SA/60_away']
    matchup['shots_for_diff'] = shots_per_match_diff
    matchup['shots_against_diff'] = shots_against_per_match_diff
    goals_per_match_diff = matchup['GF_/60_home']-matchup['GF_/60_away']
    goals_against_per_match_diff = matchup['GA_/60_home']-matchup['GA_/60_away']
    pp_percentage_diff = matchup['PP%_home']-matchup['PP%_away']
    pk_percentage_diff = matchup['PK%_home']-matchup['PK%_away']
    matchup['pp_percentage_diff'] = pp_percentage_diff
    matchup['pk_percentage_diff'] = pk_percentage_diff

    matchup['goals_for_diff'] = goals_per_match_diff
    matchup['goals_against_diff'] = goals_against_per_match_diff
    points_per_match_diff = matchup['PTS/GP_home']-matchup['PTS/GP_away']
    matchup['pts_diff'] = points_per_match_diff

    #attack strength
    league_avg_goals = league_stats['GF']/league_stats['GP']
    league_avg_goals_conceded = league_stats['GA']/league_stats['GP']
    avg_goals_home = matchup['GF_home']/matchup['GP_home']
    avg_goals_conceded_home = matchup['GA_home']/matchup['GP_home']
    attack_strength_home = avg_goals_home/league_avg_goals
    defence_strength_home = avg_goals_conceded_home/league_avg_goals_conceded
    matchup['attack_strength_home'] = attack_strength_home
    matchup['defence_strength_home'] = defence_strength_home

    avg_goals_away = matchup['GF_away']/matchup['GP_away']
    avg_goals_conceded_away = matchup['GA_away']/matchup['GP_away']
    attack_strength_away = avg_goals_away/league_avg_goals
    defence_strength_away = avg_goals_conceded_away/league_avg_goals_conceded
    matchup['attack_strength_away'] = attack_strength_away
    matchup['defence_strength_away'] = defence_strength_away

    exp_home_goals = attack_strength_home * defence_strength_away * league_avg_goals
    exp_away_goals = attack_strength_away * defence_strength_home * league_avg_goals
    matchup['attack_strength_diff'] = attack_strength_home - attack_strength_away
    matchup['defence_strength_diff'] = defence_strength_home - defence_strength_away
    matchup['exp_goals_home'] = exp_home_goals
    matchup['exp_goals_away'] = exp_away_goals


    corsi_per_match_diff = matchup['CF/60_home']-matchup['CF/60_away']
    corsi_against_per_match_diff = matchup['CA/60_home']-matchup['CA/60_away']
    matchup['corsi_for_diff'] = corsi_per_match_diff
    matchup['corsi_against_diff'] = corsi_against_per_match_diff

    eq_goals_diff = matchup['EQ G±_home']-matchup['EQ G±_away']
    matchup['eq_goals_diff'] = eq_goals_diff

    team_home_exits = exits[exits['Team'] == team1]
    team_home_exits = team_home_exits['ControlledExits %'].mean()
    team_away_exits = exits[exits['Team'] == team2]
    team_away_exits = team_away_exits['ControlledExits %'].mean()
    exits_diff = team_home_exits - team_away_exits
    matchup['exits_diff'] = exits_diff


    team_home_entries = entries[entries['Team'] == team1]
    team_home_entries = team_home_entries['ControlledEntries %'].mean()
    team_away_entries = entries[entries['Team'] == team2]
    team_away_entries = team_away_entries['ControlledEntries %'].mean()
    entries_diff = team_home_entries - team_away_entries
    matchup['entries_diff'] = entries_diff
    team_home_exits_denial = exits_denial[exits_denial['Team'] == team1]
    team_home_exits_denial = team_home_exits_denial['ControlledExits %against'].mean()
    team_away_exits_denial = exits_denial[exits_denial['Team'] == team2]
    team_away_exits_denial = team_away_exits_denial['ControlledExits %against'].mean()
    exits_denial_diff = team_home_exits_denial - team_away_exits_denial
    matchup['exits_denial_diff'] = exits_denial_diff

    team_home_entries_denial = entries_denial[entries_denial['Team'] == team1]
    team_home_entries_denial = team_home_entries_denial['ControlledEntries %against'].mean()
    team_away_entries_denial = entries_denial[entries_denial['Team'] == team2]
    team_away_entries_denial = team_away_entries_denial['ControlledEntries %against'].mean()
    entries_denial_diff = team_home_entries_denial - team_away_entries_denial
    matchup['entries_denial_diff'] = entries_denial_diff


    hdp_passes_home = hd_passes[hd_passes['Team'] == team1]
    hdp_passes_away = hd_passes[hd_passes['Team'] == team2]
    team_home_hdp_percent = hdp_passes_home['SuccessfulHDP %'].mean()
    team_away_hdp_percent = hdp_passes_away['SuccessfulHDP %'].mean()
    hdp_percent_diff = team_home_hdp_percent - team_away_hdp_percent
    matchup['hdp_percent_diff'] = hdp_percent_diff 
    



# write to csv
# sort by date
# read odds from bookmaker from json
odds_dir = './data/odds/nike_odds.json'
odds = json.load(open(odds_dir))
odds_accuracy = 0
odds_amount = 0
date = ''
for match in odds:
    team1_name = match['Matchup'].split('/')[0]
    team2_name = match['Matchup'].split('/')[1]
    teams_stats = team_stats[(team_stats['Team'] == team1_name) | (team_stats['Team'] == team2_name)]
    team1_stats = teams_stats[teams_stats['Team'] == team1_name]
    team2_stats = teams_stats[teams_stats['Team'] == team2_name]
    team1_stats = team1_stats.reset_index(drop=True)
    team2_stats = team2_stats.reset_index(drop=True)
    matchup = pd.DataFrame()
    # add team stats to games dataframe
    matchup = pd.concat([team1_stats.add_suffix('_home'), team2_stats.add_suffix('_away')], axis=1)
    matchup = matchup.reset_index(drop=True)
    matchup['Venue'] = 1
    matchup['Phase'] = 1
    # populate matchup data
    populate_matchup_data(matchup, team1_name, team2_name)
    team_home_last_5 = get_last_5_team(team1_name, '2023-10-01')
    left_last_len = len(team_home_last_5)
    team_away_last_5 = get_last_5_team(team2_name, '2023-10-01')
    right_last_len = len(team_away_last_5)
    h2h_last_5 = get_last_5_h2h(team1_name, team2_name, '2023-10-01')
    h2h_last_len = len(h2h_last_5)
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
        is_team_home = last_game['Team'] == team1_name
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
        is_team_home = last_game['Team'] == team1_name
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
        is_team_home = last_game['Team'] == team1_name
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
    matchup['win_r5_home'] = win_ratio_home
    matchup['draw_r5_home'] = draw_ratio_home
    matchup['lose_r5_home'] = lose_ratio_home
    matchup['scored_avg_r5_home'] = scored_avg_home
    matchup['conceded_avg_r5_home'] = conceded_avg_home
    win_ratio_away = win_ratio_away / right_last_len
    lose_ratio_away = lose_ratio_away / right_last_len
    draw_ratio_away = draw_ratio_away / right_last_len
    scored_avg_away = scored_avg_away / right_last_len
    conceded_avg_away = conceded_avg_away / right_last_len
    matchup['win_r5_away'] = win_ratio_away
    matchup['draw_r5_away'] = draw_ratio_away
    matchup['lose_r5_away'] = lose_ratio_away
    matchup['scored_avg_r5_away'] = scored_avg_away
    matchup['conceded_avg_r5_away'] = conceded_avg_away
    if h2h_last_len == 0:
        h2h_last_len = 1
    h2h_home_win_ratio = h2h_home_win_ratio / h2h_last_len
    h2h_home_draw_ratio = h2h_home_draw_ratio / h2h_last_len
    h2h_home_lose_ratio = h2h_home_lose_ratio / h2h_last_len
    scored_avg_h2h = scored_avg_h2h / h2h_last_len
    conceded_avg_h2h = conceded_avg_h2h / h2h_last_len
    matchup['h2h_win_ratio_home'] = h2h_home_win_ratio
    matchup['h2h_draw_ratio_home'] = h2h_home_draw_ratio
    matchup['h2h_lose_ratio_home'] = h2h_home_lose_ratio
    matchup['h2h_scored_avg_home'] = scored_avg_h2h
    matchup['h2h_conceded_avg_home'] = conceded_avg_h2h
    # predict
    matchup = matchup.loc[:,~matchup.columns.duplicated()]
    matchup = matchup[features]
    if date != match['Date']:
        print('===============================')
        print(match['Date'])
    
    date = match['Date']

    print(team1_name + ' vs ' + team2_name)
    for target_var in target_vars:
        model = models[target_var]
        y_pred = model.predict_proba(matchup)
        # calculate odds
        line = y_pred[0]

        idx=0

        bookmaker_line = match.get(target_var)
        print(target_var)
        for prob in line:
            adjusted_prob = prob*100+(bookmaker_margin/len(line))
            adjusted_prob = adjusted_prob/100
            # reverse bookmaker odds
            bookmaker_odd = bookmaker_line[idx]
            if bookmaker_odd < 1:
                continue
            ev = (prob*bookmaker_odd)-1
            ev = round(ev*100, 2)
            outcome_str = ''
            if target_var == 'Moneyline':
                if idx == 0:
                    outcome_str = team2_name + ' wins'
                elif idx == 1:
                    outcome_str = team1_name + ' wins'
            if target_var == 'Outcome':
                if idx == 0:
                    outcome_str = team2_name + ' wins in regulation'
                elif idx == 1:
                    outcome_str = 'Draw'
                elif idx == 2:
                    outcome_str = team1_name + ' wins in regulation'
            if target_var == 'OU5.5':
                if idx == 0:
                    outcome_str = 'Under 5.5'
                elif idx == 1:
                    outcome_str = 'Over 5.5'
            if target_var == 'Both2Goals':
                if idx == 0:
                    outcome_str = 'No'
                elif idx == 1:
                    outcome_str = 'Yes'

            # 30% is max EV,reason: possible error in model
            if ev>config['min_ev'] and bookmaker_odd<ceiling_odd and bookmaker_odd > 1:
                bankroll = config['bankroll']
                kelly_fraction = config['kelly_fraction']
                kelly_bet = (prob*bookmaker_odd-1)/(bookmaker_odd-1)
                kelly_bet = round(kelly_bet, 2)
                kelly_bet = kelly_bet*kelly_fraction
                print(f'Detected value bet: {ev}% - {target_var} {outcome_str} - Bookmaker: {bookmaker_odd}(Correct: {round(1/adjusted_prob, 2)}) - Prob: {round(prob*100, 2)}%') 
                print(f'Recommended bet: {round(kelly_bet*bankroll, 2)}€')
            
            odds_accuracy += (1-abs(1/adjusted_prob-bookmaker_odd)/bookmaker_odd)*100
            odds_amount += 1
            # print(f'Outcome {idx}: - Adj: {round(1/adjusted_prob, 2)} - Prob: {round(prob*100, 2)}% - Bookmaker: {bookmaker_odd}')
            idx += 1

        # print('Conclusion')
        # prediction = model.predict(matchup)
        # if target_var == 'Moneyline':

        #     if prediction[0] == 1:
        #         print(team1_name + ' wins')
        #     elif prediction[0] == 0:
        #         print(team2_name + ' wins')

        # if target_var == "OU5.5":
        #     if prediction[0] == 1:
        #         print('Over 5.5')
        #     elif prediction[0] == 0:
        #         print('Under 5.5')
        
        # if target_var == "H1-1.5":
        #     if prediction[0] == 1:
        #         print(team1_name + '-1.5 - Yes')
        #     elif prediction[0] == 0:
        #         print(team1_name + '-1.5 - No')

        # if target_var == "Outcome":
        #     if prediction[0] == 1:
        #         print(team1_name + ' wins in regulation')
        #     elif prediction[0] == 0:
        #         print('Draw')
        #     elif prediction[0] == -1:
        #         print(team2_name + ' wins in regulation')

        # if target_var == "Both2Goals":
        #     if prediction[0] == 1:
        #         print('Both teams to score 2 goals - Yes')
        #     elif prediction[0] == 0:
        #         print('Both teams to score 2 goals - No')
    print('---------------------------')

print(f'Model-Bookmaker similiarity: {round(odds_accuracy/odds_amount, 2)}%')
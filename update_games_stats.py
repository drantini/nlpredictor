
import pandas as pd
team_stats = pd.read_csv('./data/teams/22_23-Teams.csv')
games = pd.read_csv('./data/games/Combined_Games.csv')
# add team stats to games dataframe
# Team = _home
# Opp. = _away
league_stats = pd.read_csv('./data/22_23_league.csv')
# choose first row of league stats
league_stats = league_stats.iloc[0]
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
games = pd.merge(games, team_stats, left_on='Team', right_on='Team', suffixes=('', '_home'))
# merge again for opponent stats
games = pd.merge(games, team_stats, left_on='Opp.', right_on='Team', suffixes=('', '_away'))
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
    
    leading_diff = row['Leading%_home']-row['Leading%_away']
    games.at[index, 'leading_diff'] = leading_diff
    trailing_diff = row['Trailing%_home']-row['Trailing%_away']
    games.at[index, 'trailing_diff'] = trailing_diff

    corsi_per_match_diff = row['CF/60']-row['CF/60_away']
    corsi_against_per_match_diff = row['CA/60']-row['CA/60_away']
    games.at[index, 'corsi_for_diff'] = corsi_per_match_diff
    games.at[index, 'corsi_against_diff'] = corsi_against_per_match_diff
    

    eq_goals_diff = row['EQ G±_home']-row['EQ G±_away']
    games.at[index, 'eq_goals_diff'] = eq_goals_diff


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

# write games with stats to csv
games.to_csv('./data/games/Combined_Games_Stats.csv', index=False)
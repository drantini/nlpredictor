import pandas as pd
season_last = pd.read_csv('./data/teams/22_23-Teams.csv')
season_new = pd.read_csv('./data/teams/23_24-Teams.csv')
season_last_shots = pd.read_csv('./data/teams/22_23-shots.csv')
season_new_shots = pd.read_csv('./data/teams/23_24-shots.csv')

include_new_season = True

combined = pd.DataFrame()
season_last = season_last.drop(columns=['Season'])
season_new = season_new.drop(columns=['Season'])
season_last_shots = season_last_shots.drop(columns=['Season'])
season_new_shots = season_new_shots.drop(columns=['Season'])
weight = season_new['GP'].sum() / season_last['GP'].sum();
weight = 0 if not include_new_season else weight * 1.5;
print(f'Weight: {weight}')
cols_to_average = ['GP', 'PTS', 'PTS/GP', 'TOI', 'Leading', 'Tied', 'Trailing', 'GF', 'GA', 'G±', 'GF_', 'GA_', 'G_±', 
                      'GF_/60', 'GA_/60', 'EQ TOI', 'EQ GF', 'EQ GA', 'EQ G±', 'EQ GF/60', 'EQ GA/60', 'PP TOI', 'PP OP', 
                      'PP%', 'PP GF', 'PP GA', 'PP G±', 'PP GF/60', 'SH TOI', 'SH SI', 'PK%', 'SH GF', 'SH GA', 'SH G±', 
                      'SH GA/60', 'GF EN', 'GA EN', 'GF SO', 'GA SO', 'SH%', 'SV%', 'PDO', 'FOW', 'FOL', 'FO%', '# 0\'', 
                      '# 2\'', '# 5\'', '# 10\'', '# 20\'', 'PIM', 'Spect. / GP', 'Spect. / GP - Home', 'Spect. / GP - Away']
shots_cols_to_average = season_new_shots.columns;
# drop columns Team and Season
shots_cols_to_average = shots_cols_to_average.drop(['Team']);
# drop columns that are not in both seasons

season_last[cols_to_average] = season_last[cols_to_average].apply(lambda x: x * (1-weight))
season_new[cols_to_average] = season_new[cols_to_average].apply(lambda x: x * weight)
season_last_shots[shots_cols_to_average] = season_last_shots[shots_cols_to_average].apply(lambda x: x * (1-weight))
season_new_shots[shots_cols_to_average] = season_new_shots[shots_cols_to_average].apply(lambda x: x * weight)
combined_shots = pd.concat([season_last_shots, season_new_shots]).groupby(['Team'])[shots_cols_to_average].mean().reset_index()
# add shots data to team data
if include_new_season:
    combined = pd.concat([season_last, season_new]).groupby(['Team'])[cols_to_average].mean().reset_index()
else:
    combined = season_last
combined = pd.concat([combined, combined_shots], axis=1)
combined.to_csv('./data/teams/Combined_Teams.csv', index=False)




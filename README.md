# NLPredictor

NLPredictor is my personal Machine Learning project with intent to learn machine learning
NLPredictor works to predict upcoming National League (Swiss Hockey league) matches outcomes using past data with Logistic Regression (found to have best results)
NLPredictor also detects value bets (not a financial advice!)


## Workflow
Add new match data into one of the team CSV's in data/games -> 
run 'combine_games.py' -> 
run 'main.py' (train model) ->
run 'nike_odds.py' (grab upcoming games and odds) ->
run 'predict.py'

## Credits
Data provided by [NL Ice Data](https://nlicedata.com)

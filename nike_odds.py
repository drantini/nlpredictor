import requests
import json
cookies = "Nike.LOCALE=sk; Trace-Id=9c888f25-5762-4704-b581-7a90858a061f; Nike.UUID=ff30d60c-8ca6-47ad-9cc6-1c67094ceaab; Nike.bitfa=745df340-119f-4aab-b400-d70a8dfdc50b; SESSION=YTBjYzVkMDAtNWZiYi00ZmFkLWEwMTEtZWZjYjg5Y2QwMTFi; TS016649c9=010743573e4edfc379d081d3340a1405c1ea40937a42099bf7af6045f6af0263ca72a9ded71d2bcf76909b0a7669d14d055e402bc3f28215cce5712a90dc2cd8530c7d91eb387d25d747648a37fed78f56ce57a2779f47c8801d4b0165674d77f9d2a16fba; Nike.SESSION_ID_TE=12828fe9-353f-1d45-7de7-17f0b9a2bf33; BetslipSessionId=NjA3YzZhMzctNDU3Zi00N2U0LThjMDMtMzc2ZmJmZWQ0ODI2; JSESSIONID=node0yfbkxcdd8pmjikssge4ikus91371350.node0; TS01750538=010743573ea70932475bbf9c609022583c8bf1e21436c9479607fbc9672e88a9bcd375ea3d8853d99c574d45e79e5fa3b2074194c0a7e92191a235b430c4327ff4ac11492898686a79ccf4067610196b1470f9ca6b9bee1f6f49ddc824ffebab10c80c518a"
# parse cookies
cookies = {c.split('=')[0]: c.split('=')[1] for c in cookies.split('; ')}
headers = {
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-GB,en;q=0.6',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Content-Language': 'sk',
    'Origin': 'https://www.nike.sk',
    'Referer': 'https://www.nike.sk/',
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Site': 'same-site',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/116.0.1938.81',
    'sec-ch-ua': '"Chromium";v="116", "Not)A;Brand";v="24", "Microsoft Edge";v="116"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
    'Sec-Gpc': '1'
}
def get_game_ids():

    response = requests.get(
    'https://nike.sk/api-gw/nikeone/v1/boxes/search/portal?betNumbers&date&live=true&menu=%2Fhokej%2Fsvajciarsko%2Fsvajciarsko&minutes&order&prematch=true&results=false',
    cookies=cookies,
    headers=headers,
    )
    boxes = response.json()['boxes']
    # filter out superchance box id
    boxes = [box for box in boxes if box['boxId'] != 'superchance']
    boxes = [box for box in boxes if box['boxId'] != 'superoffer']
    # filer sportEventIds with liveSportEventIds
    sportEventIds = boxes[0]['sportEventIds']
    liveSportEventIds = boxes[0]['liveSportEventIds']
    boxes[0]['sportEventIds'] = [id for id in sportEventIds if id not in liveSportEventIds]
    sportEventIds = boxes[0]['sportEventIds']
    return sportEventIds 

def get_game_odds(id):
    response = requests.get(
    'https://nike.sk/api-gw/nikeone/v1/boxes/extended/sport-event-id',
    params={
        'boxId': 'bi-3-22-6',
        'sportEventId': id,
    },
    cookies=cookies,
    headers=headers,
    )
    return response.json()['bets']

teams = {
    "Ambri Piotta": "HCAP",
    "Rapperswil": "SCRJ",
    "EV Zug": "EVZ",
    "EHC Kloten": "EHCK",
    "HC Davos": "HCD",
    "HC Lugano": "HCL",
    "Fribourg": "HCFG",
    "Lausanne HC": "LHC",
    "SC Bern": "SCB",
    "Zürich Lions": "ZSC",
    "Langnau Tigers": "SCLT",
    "HC Ajoie": "HCA",
    "Servette": "GSHC",
    "EHC Biel-Bienne": "EHCB",
}
def format_odds(odds):
    # get match outcome: betHeaderDetail = 'Zapas'
    moneyline = [x for x in odds if x['headerDetail'] == 'Víťaz zápasu'][0];
    date = moneyline['expirationTime'].split('T')[0];
    moneyline_first_team = moneyline['selectionGrid'][0][0]['odds'];
    moneyline_second_team = moneyline['selectionGrid'][0][1]['odds'];
    outcome = [x for x in odds if x['headerDetail'] == 'Zápas'][0]; 
    first_team_name = outcome['participants'][0];
    second_team_name = outcome['participants'][1];
    first_team_odds = outcome['selectionGrid'][0][0]['odds'];
    draw_odds = outcome['selectionGrid'][0][1]['odds'];
    second_team_odds = outcome['selectionGrid'][0][2]['odds'];
    both_2_goals_line = [x for x in odds if x['headerDetail'] == 'Každý z tímov strelí aspoň 2 góly']
    both_2_goals_yes, both_2_goals_no = 0, 0;
    if len(both_2_goals_line) > 0:
        both_2_goals_line = both_2_goals_line[0];
        both_2_goals_yes = both_2_goals_line['selectionGrid'][0][0]['odds']
        both_2_goals_no = both_2_goals_line['selectionGrid'][0][1]['odds']
    over_5_5_goals_line = [x for x in odds if x['headerDetail'] == 'Počet gólov do rozhodnutia']
    over_5_5_goals_odds, under_5_5_goals_odds = 0, 0;
    for line in over_5_5_goals_line:
        selec_grid = line['selectionGrid'][0];
        if selec_grid[0]['name'] == 'menej ako 5.5':
            under_5_5_goals_odds = selec_grid[0]['odds'];
        if selec_grid[1]['name'] == 'viac ako 5.5':
            over_5_5_goals_odds = selec_grid[1]['odds'];
    
    handicap_lines = [x for x in odds if x['headerDetail'] == 'Handicap']
    handicaps = {}
    for line in handicap_lines:
        selec_grid = line['selectionGrid'][0];
        if selec_grid[0]['name'] == f'{first_team_name} -1.5':
            handicaps['H1-1.5'] = selec_grid[0]['odds'];
        if selec_grid[1]['name'] == f'{second_team_name} +1.5':
            handicaps['H2+1.5'] = selec_grid[1]['odds'];
        if selec_grid[1]['name'] == f'{second_team_name} -1.5':
            handicaps['H2-1.5'] = selec_grid[1]['odds'];
    
    first_team_name = teams[first_team_name];
    second_team_name = teams[second_team_name];
    matchup_name = first_team_name + '/' + second_team_name;
    print(f'Found matchup: {matchup_name}');
    return {
        'Date': date,
        'Matchup': matchup_name,
        'Outcome': [second_team_odds, draw_odds, first_team_odds],
        'Moneyline': [moneyline_second_team, moneyline_first_team],
        'Both2Goals': [both_2_goals_no, both_2_goals_yes],
        'OU5.5': [under_5_5_goals_odds, over_5_5_goals_odds],
        'H1-1.5': [1, handicaps['H1-1.5'] if 'H1-1.5' in handicaps else 0],
        'H2+1.5': [1, handicaps['H2+1.5'] if 'H2+1.5' in handicaps else 0],
        'H2-1.5': [1, handicaps['H2-1.5'] if 'H2-1.5' in handicaps else 0],
    }

ids = get_game_ids()
matches = [] 
for id in ids:
    odds = get_game_odds(id);
    odds = format_odds(odds);
    if odds is not None:
        matches.append(odds);

#write to json
print(f'Found {len(matches)} matches.')
with open('./data/odds/nike_odds.json', 'w') as outfile:
    json.dump(matches, outfile)
print('Updated odds.')
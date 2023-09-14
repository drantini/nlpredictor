import requests
import json
cookies = {
    'Trace-Id': '11526b41-304e-4713-ba4f-840de4c4d114',
    'Nike.UUID': 'dfaafbe1-9e04-4f75-b2eb-c2675d270b5c',
    'Nike.bitfa': '34ce5c1f-bb15-4423-932f-67cf72b77132',
    '_gcl_au': '1.1.326794301.1692441962',
    '_fbp': 'fb.1.1692441963125.1736939095',
    '_gid': 'GA1.2.824419695.1694555686',
    'csrfToken': 'q1SXYI2r9%2BKaMY0BKP3elDViZDc5MTFkNDQ5Y2RhMGY1MjRmY2QzYjdiMjk0ZWVkNWM4ZDllYWE%3D',
    'Nike.bttfa': 'drantini:61e87d84-46da-447b-9967-2d51012dd5ae',
    'Contract-Id': '14171707',
    'TS016649c9': '010743573edcbad31c6f53d3f8814c33dff22168356a8055a8f1de53cd1baecf990c409e24fb77af075c0be27ae1f929c6dff11d7d2d97b60ba4061ed188cb4534766db33074ec71002ccd7c02a47aac02efb7bfaa2ed91b0071273cef8d8cd753e903334c4f61438c07ceae3e1ebd87781d628962',
    '_sp_srt_id.2d0c': '902edbc0-44cf-4ab2-93e1-87794be453bd.1692441964.27.1694689903.1694687659.4ca525ec-f632-47a9-8e52-3d73104c478b.a6f02247-3346-45fb-9c8b-23294ea44c96...0',
    '_ga': 'GA1.2.895249329.1692441955',
    '_ga_DE6QBHRM6X': 'GS1.1.1694689900.30.1.1694691686.58.0.0',
    'TS337d42a9029': '0807006ba9ab2800664311e7cb17bd95de1c72c119289beafc579c0d86cdeec6b6675d358678a732ed56fe6f7c750bea',
    'TSfabc11ac027': '0807006ba9ab2000ebfc6e47cb49bf1a76a12bc3ac758df1cdde6b132289ecda3d8a4e3cd743805b080d4dd5fb113000f118edcb7202e5cd0573be732c929f4ff5b8e6698be4ee50a9c378729c24ae24fdd9c26aa72274c6667972d9037ad338',
}
headers = {
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-GB,en;q=0.9,en-US;q=0.8',
    'Connection': 'keep-alive',
    'Content-Language': 'sk',
    # 'Cookie': 'Trace-Id=11526b41-304e-4713-ba4f-840de4c4d114; Nike.UUID=dfaafbe1-9e04-4f75-b2eb-c2675d270b5c; Nike.bitfa=34ce5c1f-bb15-4423-932f-67cf72b77132; _gcl_au=1.1.326794301.1692441962; _fbp=fb.1.1692441963125.1736939095; _gid=GA1.2.824419695.1694555686; csrfToken=q1SXYI2r9%2BKaMY0BKP3elDViZDc5MTFkNDQ5Y2RhMGY1MjRmY2QzYjdiMjk0ZWVkNWM4ZDllYWE%3D; Nike.bttfa=drantini:61e87d84-46da-447b-9967-2d51012dd5ae; Contract-Id=14171707; TS016649c9=010743573edcbad31c6f53d3f8814c33dff22168356a8055a8f1de53cd1baecf990c409e24fb77af075c0be27ae1f929c6dff11d7d2d97b60ba4061ed188cb4534766db33074ec71002ccd7c02a47aac02efb7bfaa2ed91b0071273cef8d8cd753e903334c4f61438c07ceae3e1ebd87781d628962; _sp_srt_id.2d0c=902edbc0-44cf-4ab2-93e1-87794be453bd.1692441964.27.1694689903.1694687659.4ca525ec-f632-47a9-8e52-3d73104c478b.a6f02247-3346-45fb-9c8b-23294ea44c96...0; _ga=GA1.2.895249329.1692441955; _ga_DE6QBHRM6X=GS1.1.1694689900.30.1.1694691686.58.0.0; TS337d42a9029=0807006ba9ab2800664311e7cb17bd95de1c72c119289beafc579c0d86cdeec6b6675d358678a732ed56fe6f7c750bea; TSfabc11ac027=0807006ba9ab2000ebfc6e47cb49bf1a76a12bc3ac758df1cdde6b132289ecda3d8a4e3cd743805b080d4dd5fb113000f118edcb7202e5cd0573be732c929f4ff5b8e6698be4ee50a9c378729c24ae24fdd9c26aa72274c6667972d9037ad338',
    'Origin': 'https://www.nike.sk',
    'Referer': 'https://www.nike.sk/',
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Site': 'same-site',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36 Edg/116.0.1938.81',
    'sec-ch-ua': '"Chromium";v="116", "Not)A;Brand";v="24", "Microsoft Edge";v="116"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
}
def get_game_ids():

    response = requests.get(
    'https://api.nike.sk/api/nikeone/v1/boxes/search/portal?betNumbers&date&live=true&menu=%2Fhokej%2Fsvajciarsko%2Fsvajciarsko&minutes&order&prematch=true&results=false',
    cookies=cookies,
    headers=headers,
    )
    boxes = response.json()['boxes']
    # filter out superchance box id
    boxes = [box for box in boxes if box['boxId'] != 'superchance']
    return boxes[0]['sportEventIds']

def get_game_odds(id):
    response = requests.get(
    'https://api.nike.sk/api/nikeone/v1/boxes/extended/sport-event-id',
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
    moneyline = odds[0];
    moneyline_first_team = moneyline['selectionGrid'][0][0]['odds'];
    moneyline_second_team = moneyline['selectionGrid'][0][1]['odds'];
    outcome = odds[1];
    first_team_name = outcome['participants'][0];
    second_team_name = outcome['participants'][1];
    first_team_odds = outcome['selectionGrid'][0][0]['odds'];
    draw_odds = outcome['selectionGrid'][0][1]['odds'];
    second_team_odds = outcome['selectionGrid'][0][2]['odds'];
    both_2_goals_line = [x for x in odds if x['headerDetail'] == 'Každý z tímov strelí aspoň 2 góly'][0];
    both_2_goals_yes = both_2_goals_line['selectionGrid'][0][0]['odds'];
    both_2_goals_no = both_2_goals_line['selectionGrid'][0][1]['odds'];
    over_5_5_goals_line = [x for x in odds if x['headerDetail'] == 'Počet gólov do rozhodnutia'];
    over_5_5_goals_odds, under_5_5_goals_odds = 0, 0;
    for line in over_5_5_goals_line:
        selec_grid = line['selectionGrid'][0];
        if selec_grid[0]['name'] == 'menej ako 5.5':
            under_5_5_goals_odds = selec_grid[0]['odds'];
        if selec_grid[1]['name'] == 'viac ako 5.5':
            over_5_5_goals_odds = selec_grid[1]['odds'];
    
    first_team_name = teams[first_team_name];
    second_team_name = teams[second_team_name];
    matchup_name = first_team_name + '/' + second_team_name;
    return {
        'Matchup': matchup_name,
        'Outcome': [second_team_odds, draw_odds, first_team_odds],
        'Moneyline': [moneyline_second_team, moneyline_first_team],
        'Both2Goals': [both_2_goals_no, both_2_goals_yes],
        'OU5.5': [under_5_5_goals_odds, over_5_5_goals_odds]
    }

ids = get_game_ids()
matches = [] 
for id in ids:
    odds = get_game_odds(id);
    odds = format_odds(odds);
    matches.append(odds);

#write to json
with open('./data/odds/nike_odds.json', 'w') as outfile:
    json.dump(matches, outfile)
print('Updated odds.')
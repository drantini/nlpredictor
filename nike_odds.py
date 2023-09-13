import requests
import json
cookies = {
    'Trace-Id': '11526b41-304e-4713-ba4f-840de4c4d114',
    'Nike.UUID': 'dfaafbe1-9e04-4f75-b2eb-c2675d270b5c',
    'Nike.bitfa': '34ce5c1f-bb15-4423-932f-67cf72b77132',
    '_gcl_au': '1.1.326794301.1692441962',
    '_fbp': 'fb.1.1692441963125.1736939095',
    '_gid': 'GA1.2.824419695.1694555686',
    '_sp_srt_ses.2d0c': '*',
    'csrfToken': 'TSqIWRDGZeOX1sFhEcUrAGExYzIyYzRhNTY2YzQ3ODgxYTM3NDIwNmNiYjM2NDViMDZhZDk0OWY%3D',
    'Nike.bttfa': 'drantini:6d2acd33-7264-45a4-8492-c037bf041144',
    'Contract-Id': '14171707',
    'TS016649c9': '010743573e353191f7e4b55cf78302c06e3d70d6701e3938bf5f9f784b14593dd58c24cac4f890018dc2500ca62072bcc8a2420977a8fae8e50ba3bfeba48e8700bc606055317aa2065eae9fa35a2ce1f916d6e2cdec09a344d3f0a19c2bc51015a4bed93e91b9c0b6940e7c0c0b6f8f6d3ed6e409',
    '_sp_srt_id.2d0c': '902edbc0-44cf-4ab2-93e1-87794be453bd.1692441964.23.1694638347.1694620488.1b0e758c-fd71-4190-a88f-ac44abb7b628.2fd17d22-d3cb-4165-b304-a0b2cddb510f...0',
    '_ga': 'GA1.2.895249329.1692441955',
    '_gat_UA-166255788-2': '1',
    '_gat_UA-166255788-4': '1',
    '_ga_DE6QBHRM6X': 'GS1.1.1694637310.28.1.1694640055.58.0.0',
    'TS337d42a9029': '0807006ba9ab28007efea59d889633b08b8b51e29b0793e12da61da95ddc95398c621a86abc8d4422e8f09c4db81b044',
    'TSfabc11ac027': '0807006ba9ab2000f243ba49b893e15e3a4783bcb018047ea9ce790ae7be9c5e14e2511708077a3108d5be0edf1130007434e864150bb4eff7d428055c63e02a6952d299f9649fd8e9f1151d00b6b5b68e41062ba01030a0e403947364a2878f',
    }
headers = {
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-GB,en;q=0.9,en-US;q=0.8',
    'Connection': 'keep-alive',
    'Content-Language': 'sk',
    # 'Cookie': 'Trace-Id=11526b41-304e-4713-ba4f-840de4c4d114; Nike.UUID=dfaafbe1-9e04-4f75-b2eb-c2675d270b5c; Nike.bitfa=34ce5c1f-bb15-4423-932f-67cf72b77132; _gcl_au=1.1.326794301.1692441962; _fbp=fb.1.1692441963125.1736939095; _gid=GA1.2.824419695.1694555686; _sp_srt_ses.2d0c=*; csrfToken=TSqIWRDGZeOX1sFhEcUrAGExYzIyYzRhNTY2YzQ3ODgxYTM3NDIwNmNiYjM2NDViMDZhZDk0OWY%3D; Nike.bttfa=drantini:6d2acd33-7264-45a4-8492-c037bf041144; Contract-Id=14171707; TS016649c9=010743573e353191f7e4b55cf78302c06e3d70d6701e3938bf5f9f784b14593dd58c24cac4f890018dc2500ca62072bcc8a2420977a8fae8e50ba3bfeba48e8700bc606055317aa2065eae9fa35a2ce1f916d6e2cdec09a344d3f0a19c2bc51015a4bed93e91b9c0b6940e7c0c0b6f8f6d3ed6e409; _sp_srt_id.2d0c=902edbc0-44cf-4ab2-93e1-87794be453bd.1692441964.23.1694638347.1694620488.1b0e758c-fd71-4190-a88f-ac44abb7b628.2fd17d22-d3cb-4165-b304-a0b2cddb510f...0; _ga=GA1.2.895249329.1692441955; _gat_UA-166255788-2=1; _gat_UA-166255788-4=1; _ga_DE6QBHRM6X=GS1.1.1694637310.28.1.1694640055.58.0.0; TS337d42a9029=0807006ba9ab28007efea59d889633b08b8b51e29b0793e12da61da95ddc95398c621a86abc8d4422e8f09c4db81b044; TSfabc11ac027=0807006ba9ab2000f243ba49b893e15e3a4783bcb018047ea9ce790ae7be9c5e14e2511708077a3108d5be0edf1130007434e864150bb4eff7d428055c63e02a6952d299f9649fd8e9f1151d00b6b5b68e41062ba01030a0e403947364a2878f',
    'Origin': 'https://www.nike.sk',
    'Referer': 'https://www.nike.sk/',
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Site': 'same-site',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36 Edg/116.0.1938.76',
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
    return response.json()['boxes'][0]['sportEventIds']

def get_game_odds(id):
    response = requests.get(
    f'https://api.nike.sk/api/stats2/v1/market-statistics/{id}?gameType=Prematch&lang=sk',
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
    outcome = odds[0];
    first_team_name = outcome['selections'][0]['selectionName'];
    second_team_name = outcome['selections'][2]['selectionName'];
    first_team_odds = outcome['selections'][0]['odds'];
    draw_odds = outcome['selections'][1]['odds'];
    second_team_odds = outcome['selections'][2]['odds'];
    moneyline = odds[1];
    moneyline_first_team = moneyline['selections'][0]['odds'];
    moneyline_second_team = moneyline['selections'][1]['odds'];
    first_team_name = teams[first_team_name];
    second_team_name = teams[second_team_name];
    matchup_name = first_team_name + '/' + second_team_name;
    return {
        'Matchup': matchup_name,
        'Outcome': [first_team_odds, draw_odds, second_team_odds],
        'Moneyline': [moneyline_first_team, moneyline_second_team],
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
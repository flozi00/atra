import requests
import time
import random
URLs = ["http://127.0.0.1:7860"]
langs = ["german", "english"]

while True:
    for URL in URLs:
        for lang in langs:
            headers = {
                'Accept': '*/*',
                'Accept-Language': 'de-DE,de;q=0.9,en-US;q=0.8,en;q=0.7',
                'Connection': 'keep-alive',
                # Already added when you pass json=
                # 'Content-Type': 'application/json',
                'Origin': URL,
                'Referer': URL,
                'Sec-Fetch-Dest': 'empty',
                'Sec-Fetch-Mode': 'cors',
                'Sec-Fetch-Site': 'same-origin',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.36',
                'sec-ch-ua': '"Google Chrome";v="105", "Not)A;Brand";v="8", "Chromium";v="105"',
                'sec-ch-ua-mobile': '?0',
                'sec-ch-ua-platform': '"Windows"',
            }

            json_data = {
                'fn_index': 0,
                'data': [
                    {
                        'data': 'data:audio/wav;base64,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',
                        'name': 'audio',
                    },
                    lang,
                    [],
                ],
                'session_hash': 'ka0jkywdr5p',
            }

            response = requests.post('https://asr.a-ware.io/api/predict/', headers=headers, json=json_data)
            time.sleep(random.randint(3,5))
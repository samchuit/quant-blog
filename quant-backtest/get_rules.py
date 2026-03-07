"""获取交易规则"""
import requests
import json

PROXIES = {'http': 'http://192.168.0.45:7897', 'https': 'http://192.168.0.45:7897'}

# 获取SOLUSDT交易规则
url = 'https://testnet.binance.vision/api/v3/exchangeInfo?symbol=SOLUSDT'
r = requests.get(url, proxies=PROXIES)
data = r.json()

if 'symbols' in data:
    for s in data['symbols']:
        print("SOLUSDT Trading Rules:")
        for f in s['filters']:
            if f['filterType'] in ['LOT_SIZE', 'MIN_NOTIONAL', 'NOTIONAL', 'PRICE_FILTER']:
                print(f"  {f['filterType']}: {f}")
        
        print(f"\n  Base Asset: {s['baseAsset']}")
        print(f"  Quote Asset: {s['quoteAsset']}")
        print(f"  Status: {s['status']}")

"""测试新API Key"""
import requests
import hmac
import hashlib
import time

API_KEY = "W1dAcyGFmPloyATJIxs2xwhGqEDhRIqZA7JesyTSSNV4ILYLJ3pHw39Je406bUWC"
API_SECRET = "NnvDpodpaQ27VKJwArDZ9IWuZkyYPmzBaL7K2IUYBKdSgQaMZUdrh8MP0jh9tFI6"
PROXIES = {'http': 'http://192.168.0.45:7897', 'https': 'http://192.168.0.45:7897'}

print("="*50)
print("Testing new API Key...")
print("="*50)

# Test account
base_url = "https://testnet.binance.vision"
endpoint = "/api/v3/account"
timestamp = int(time.time() * 1000)
query_string = f"timestamp={timestamp}"
signature = hmac.new(API_SECRET.encode(), query_string.encode(), hashlib.sha256).hexdigest()

url = f"{base_url}{endpoint}?{query_string}&signature={signature}"
headers = {'X-MBX-APIKEY': API_KEY}

r = requests.get(url, headers=headers, proxies=PROXIES, timeout=15)
data = r.json()

if 'balances' in data:
    print("\nSUCCESS! Account connected!")
    print("\nBalances:")
    for b in data['balances']:
        free = float(b['free'])
        if free > 0:
            print(f"  {b['asset']}: {free}")
else:
    print(f"\nError: {data}")

print("\n" + "="*50)

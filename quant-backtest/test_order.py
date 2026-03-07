"""测试下单功能"""
import requests
import hmac
import hashlib
import time
import json

API_KEY = "W1dAcyGFmPloyATJIxs2xwhGqEDhRIqZA7JesyTSSNV4ILYLJ3pHw39Je406bUWC"
API_SECRET = "NnvDpodpaQ27VKJwArDZ9IWuZkyYPmzBaL7K2IUYBKdSgQaMZUdrh8MP0jh9tFI6"
BASE_URL = "https://testnet.binance.vision"
PROXIES = {'http': 'http://192.168.0.45:7897', 'https': 'http://192.168.0.45:7897'}

print("="*50)
print("测试下单功能")
print("="*50)

# 1. 先获取账户余额
print("\n[1] 获取账户余额...")
session = requests.Session()
session.proxies = PROXIES

timestamp = int(time.time() * 1000)
query_string = f"timestamp={timestamp}"
signature = hmac.new(API_SECRET.encode(), query_string.encode(), hashlib.sha256).hexdigest()

url = f"{BASE_URL}/api/v3/account?{query_string}&signature={signature}"
headers = {'X-MBX-APIKEY': API_KEY}

r = session.get(url, headers=headers)
data = r.json()

if 'balances' in data:
    print("账户余额:")
    for b in data['balances']:
        if b['asset'] in ['USDT', 'BTC', 'ETH', 'BNB', 'SOL']:
            free = float(b['free'])
            if free > 0:
                print(f"  {b['asset']}: {free}")
    usdt_balance = next((float(b['free']) for b in data['balances'] if b['asset'] == 'USDT'), 0)
else:
    print(f"错误: {data}")
    usdt_balance = 0

# 2. 获取当前价格
print("\n[2] 获取SOL价格...")
url = f"{BASE_URL}/api/v3/ticker/price?symbol=SOLUSDT"
r = session.get(url)
price_data = r.json()
price = float(price_data['price'])
print(f"  SOL价格: ${price:.2f}")

# 3. 尝试下单（小额买入SOL）
print("\n[3] 尝试下单买入SOL...")

if usdt_balance >= 10:
    # 计算数量 - 需要符合LOT_SIZE规则
    # stepSize = 0.001, 所以数量需要是0.001的倍数
    invest = 50  # 买入$50的SOL
    quantity = invest / price
    # 按stepSize取整
    quantity = round(quantity, 3)  # 保留3位小数（0.001精度）
    
    print(f"  计划买入: {quantity} SOL (~${invest})")
    
    # 构建下单参数
    params = {
        'symbol': 'SOLUSDT',
        'side': 'BUY',
        'type': 'MARKET',
        'quantity': quantity,
        'timestamp': int(time.time() * 1000)
    }
    
    # 按字母顺序排序参数
    sorted_params = dict(sorted(params.items()))
    query_string = '&'.join([f"{k}={v}" for k, v in sorted_params.items()])
    
    # 签名
    signature = hmac.new(API_SECRET.encode(), query_string.encode(), hashlib.sha256).hexdigest()
    
    # 完整URL
    url = f"{BASE_URL}/api/v3/order?{query_string}&signature={signature}"
    headers = {'X-MBX-APIKEY': API_KEY}
    
    print(f"  请求URL: {url[:100]}...")
    
    r = session.post(url, headers=headers)
    result = r.json()
    
    if 'orderId' in result:
        print(f"\n  SUCCESS! Order placed!")
        print(f"  Order ID: {result['orderId']}")
        print(f"  Symbol: {result['symbol']}")
        print(f"  Side: {result['side']}")
        print(f"  Status: {result['status']}")
    else:
        print(f"\n  FAILED: {result}")
else:
    print("  余额不足，无法下单")

print("\n" + "="*50)

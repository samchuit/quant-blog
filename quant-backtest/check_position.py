"""持仓 vs 信号对比检查"""
import requests
import numpy as np
from stable_baselines3 import PPO
import hmac
import hashlib
import time

PROXIES = {'http': 'http://192.168.0.45:7897', 'https': 'http://192.168.0.45:7897'}
API_KEY = "W1dAcyGFmPloyATJIxs2xwhGqEDhRIqZA7JesyTSSNV4ILYLJ3pHw39Je406bUWC"
API_SECRET = "NnvDpodpaQ27VKJwArDZ9IWuZkyYPmzBaL7K2IUYBKdSgQaMZUdrh8MP0jh9tFI6"

print("="*60)
print("持仓 vs 策略信号 对比")
print("="*60)

# 获取账户余额
timestamp = int(time.time() * 1000)
query_string = f"timestamp={timestamp}"
sig = hmac.new(API_SECRET.encode(), query_string.encode(), hashlib.sha256).hexdigest()
url = f"https://testnet.binance.vision/api/v3/account?{query_string}&signature={sig}"
headers = {"X-MBX-APIKEY": API_KEY}
r = requests.get(url, headers=headers, proxies=PROXIES)
balances = {b["asset"]: float(b["free"]) for b in r.json()["balances"] if float(b["free"]) > 0}

# 获取价格
def get_price(symbol):
    url = f"https://testnet.binance.vision/api/v3/ticker/price?symbol={symbol}"
    r = requests.get(url, proxies=PROXIES)
    return float(r.json()["price"])

# 获取模型信号
def get_signal(symbol):
    try:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1d&limit=100"
        r = requests.get(url, proxies=PROXIES)
        data = r.json()
        close = np.array([float(d[4]) for d in data], dtype=np.float32)
        
        ma5 = np.mean(close[-5:])
        ma20 = np.mean(close[-20:])
        
        obs = np.zeros(25, dtype=np.float32)
        obs[0] = ma5 / 1000
        obs[1] = ma20 / 1000
        obs[12] = (ma5 - ma20) / ma20
        obs[18] = 1
        
        model = PPO.load(f"ppo_v7_{symbol}")
        action, _ = model.predict(obs)
        actions = ["HOLD", "BUY_33%", "BUY_66%", "SELL_50%", "SELL_100%"]
        return actions[min(action, 4)]
    except Exception as e:
        return f"ERROR: {e}"

print("\n[持仓情况 vs 策略信号]")
print("-"*60)

symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
usdt = balances.get("USDT", 0)

for sym in symbols:
    asset = sym.replace("USDT", "")
    price = get_price(sym)
    qty = balances.get(asset, 0)
    
    # 计算持仓比例
    if qty > 0:
        pos_value = qty * price
        total = pos_value + usdt
        pos_pct = pos_value / total * 100
        pos_status = f"{qty:.4f} ({pos_pct:.0f}%)"
    else:
        pos_status = "空仓"
    
    # 获取信号
    signal = get_signal(sym)
    
    # 判断一致性
    if "BUY" in signal:
        if qty > 0:
            match = "OK (已买入)"
        else:
            match = "NO (应买但空仓)"
    elif "SELL" in signal:
        if qty > 0:
            match = "NO (应卖但持仓)"
        else:
            match = "OK (已卖出)"
    else:  # HOLD
        match = "OK"
    
    print(f"\n{sym}:")
    print(f"  价格: ${price:,.2f}")
    print(f"  持仓: {pos_status}")
    print(f"  信号: {signal}")
    print(f"  一致: {match}")

print(f"\n" + "-"*60)
print(f"USDT余额: ${usdt:.2f}")
print("="*60)

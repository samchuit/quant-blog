"""
测试币安API连接 - 详细诊断
"""
import requests
import hmac
import hashlib
import time
import json

API_KEY = "exvBy9sEElgpGcKdG3b7EY1ZWwbwxQJFsiNjJrCYjwdd5YvJoshgNWE5OMxRUO9U"
API_SECRET = "XDUkeJ1EM30SohX8C90bbtgs3lAlpXAHtTh4YIgVHZCow7HBe0H8HDCjHJUmo9hM"

PROXIES = {'http': 'http://192.168.0.45:7897', 'https': 'http://192.168.0.45:7897'}

print("="*60)
print("币安测试网API诊断")
print("="*60)

# 测试1: 公开接口（不需要签名）
print("\n[测试1] 公开接口 - 获取交易对信息")
try:
    url = "https://testnet.binance.vision/api/v3/exchangeInfo"
    r = requests.get(url, proxies=PROXIES, timeout=15)
    data = r.json()
    if 'symbols' in data:
        print(f"  成功! 共有 {len(data['symbols'])} 个交易对")
    else:
        print(f"  响应: {list(data.keys())}")
except Exception as e:
    print(f"  失败: {e}")

# 测试2: 公开接口 - 获取价格
print("\n[测试2] 公开接口 - 获取BTC价格")
try:
    url = "https://testnet.binance.vision/api/v3/ticker/price?symbol=BTCUSDT"
    r = requests.get(url, proxies=PROXIES, timeout=15)
    data = r.json()
    print(f"  成功! BTC价格: ${float(data['price']):,.2f}")
except Exception as e:
    print(f"  失败: {e}")

# 测试3: 私有接口 - 获取账户信息
print("\n[测试3] 私有接口 - 获取账户信息")
try:
    base_url = "https://testnet.binance.vision"
    endpoint = "/api/v3/account"
    
    timestamp = int(time.time() * 1000)
    query_string = f"timestamp={timestamp}"
    
    signature = hmac.new(
        API_SECRET.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    url = f"{base_url}{endpoint}?{query_string}&signature={signature}"
    headers = {'X-MBX-APIKEY': API_KEY}
    
    print(f"  请求URL: {url[:80]}...")
    print(f"  API Key: {API_KEY[:20]}...")
    
    r = requests.get(url, headers=headers, proxies=PROXIES, timeout=15)
    data = r.json()
    
    if 'balances' in data:
        print(f"  成功! 账户余额:")
        for b in data['balances']:
            free = float(b['free'])
            if free > 0:
                print(f"    {b['asset']}: {free}")
    else:
        print(f"  错误响应: {data}")
        
except Exception as e:
    print(f"  失败: {e}")

# 测试4: 尝试_spot接口
print("\n[测试4] 尝试Spot测试网地址")
try:
    # 尝试不同的测试网地址
    urls = [
        "https://testnet.binance.vision/api/v3/account",
        "https://api.binance.com/api/v3/account",  # 主网（会失败但可看错误信息）
    ]
    
    for url_base in urls:
        print(f"\n  尝试: {url_base}")
        timestamp = int(time.time() * 1000)
        query_string = f"timestamp={timestamp}"
        signature = hmac.new(
            API_SECRET.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        full_url = f"{url_base}?{query_string}&signature={signature}"
        headers = {'X-MBX-APIKEY': API_KEY}
        
        r = requests.get(full_url, headers=headers, proxies=PROXIES, timeout=15)
        data = r.json()
        
        if 'balances' in data:
            print(f"    成功! 找到账户")
            for b in data['balances']:
                free = float(b['free'])
                if free > 0:
                    print(f"    {b['asset']}: {free}")
            break
        else:
            print(f"    错误: {data.get('msg', data)}")
            
except Exception as e:
    print(f"  失败: {e}")

print("\n" + "="*60)
print("诊断完成")
print("="*60)

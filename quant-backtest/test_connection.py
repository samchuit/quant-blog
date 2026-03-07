"""测试币安测试网连接"""
from live_trading import BinanceTrading

print("="*50)
print("测试币安测试网连接")
print("="*50)

exchange = BinanceTrading()

# 测试账户
print("\n1. 获取账户信息...")
account = exchange.get_account()

if 'error' in account:
    print(f"错误: {account['error']}")
elif 'code' in account:
    print(f"API错误码: {account.get('code')}")
    print(f"错误信息: {account.get('msg')}")
elif 'canTrade' in account:
    print("连接成功!")
    print(f"可以交易: {account.get('canTrade', False)}")
    
    # 显示余额
    print("\n账户余额:")
    balances = account.get('balances', [])
    for b in balances:
        free = float(b['free'])
        if free > 0:
            print(f"  {b['asset']}: {free:.4f}")
else:
    print(f"响应: {list(account.keys())}")

# 测试获取价格
print("\n2. 获取当前价格...")
for sym in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']:
    price = exchange.get_price(sym)
    print(f"  {sym}: ${price:,.2f}")

print("\n测试完成!")

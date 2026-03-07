"""
币安模拟盘交易接口
==================
接入币安Testnet进行模拟交易
"""
import numpy as np
import requests
import json
import hmac
import hashlib
import time
from datetime import datetime
from stable_baselines3 import PPO
import gymnasium as gym
import warnings
warnings.filterwarnings('ignore')

# ============ 配置 ============
API_KEY = "W1dAcyGFmPloyATJIxs2xwhGqEDhRIqZA7JesyTSSNV4ILYLJ3pHw39Je406bUWC"
API_SECRET = "NnvDpodpaQ27VKJwArDZ9IWuZkyYPmzBaL7K2IUYBKdSgQaMZUdrh8MP0jh9tFI6"
# 测试网地址 - 尝试不同的endpoint
BASE_URL = "https://testnet.binance.vision/api"

# 代理设置
PROXIES = {'http': 'http://192.168.0.45:7897', 'https': 'http://192.168.0.45:7897'}


class BinanceTrading:
    """币安模拟盘交易类"""
    
    def __init__(self, api_key=None, api_secret=None, testnet=True):
        self.api_key = api_key or API_KEY
        self.api_secret = api_secret or API_SECRET
        self.base_url = BASE_URL if testnet else "https://api.binance.com/api"
        self.session = requests.Session()
        self.session.proxies = PROXIES
    
    def _sign(self, params):
        """生成签名"""
        # 参数需要排序
        sorted_params = dict(sorted(params.items()))
        query_string = '&'.join([f"{k}={v}" for k, v in sorted_params.items()])
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _request(self, method, endpoint, params=None, signed=False):
        """发起请求"""
        url = f"{self.base_url}{endpoint}"
        headers = {'X-MBX-APIKEY': self.api_key}
        
        if params is None:
            params = {}
        
        if signed:
            params['timestamp'] = int(time.time() * 1000)
            params['signature'] = self._sign(params)
        
        try:
            if method == 'GET':
                r = self.session.get(url, params=params, headers=headers, timeout=15)
            elif method == 'POST':
                # POST请求，参数放在URL query string中
                r = self.session.post(url, params=params, headers=headers, timeout=15)
            elif method == 'DELETE':
                r = self.session.delete(url, params=params, headers=headers, timeout=15)
            return r.json()
        except Exception as e:
            return {'error': str(e)}
    
    def get_account(self):
        """获取账户信息"""
        return self._request('GET', '/v3/account', signed=True)
    
    def get_balance(self, asset='USDT'):
        """获取余额"""
        account = self.get_account()
        if 'balances' in account:
            for b in account['balances']:
                if b['asset'] == asset:
                    return float(b['free'])
        return 0
    
    def get_price(self, symbol):
        """获取当前价格"""
        r = self._request('GET', '/v3/ticker/price', {'symbol': symbol})
        return float(r.get('price', 0))
    
    def place_order(self, symbol, side, quantity, order_type='MARKET'):
        """下单 - 直接构建请求"""
        timestamp = int(time.time() * 1000)
        
        # 构建参数（不包含signature）
        params = {
            'symbol': symbol,
            'side': side,
            'type': order_type,
            'quantity': float(quantity),  # 确保是float
            'timestamp': timestamp
        }
        
        # 按字母顺序排序并构建query string
        sorted_params = dict(sorted(params.items()))
        query_string = '&'.join([f"{k}={v}" for k, v in sorted_params.items()])
        
        # 计算签名
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        # 构建完整URL
        url = f"{self.base_url}/v3/order?{query_string}&signature={signature}"
        headers = {'X-MBX-APIKEY': self.api_key}
        
        try:
            r = self.session.post(url, headers=headers, timeout=15)
            return r.json()
        except Exception as e:
            return {'error': str(e)}
    
    def get_open_orders(self, symbol=None):
        """获取未成交订单"""
        params = {'symbol': symbol} if symbol else {}
        return self._request('GET', '/v3/openOrders', params, signed=True)
    
    def cancel_order(self, symbol, order_id):
        """取消订单"""
        params = {'symbol': symbol, 'orderId': order_id}
        return self._request('DELETE', '/v3/order', params, signed=True)


def calculate_indicators(data):
    """计算技术指标"""
    close = data[:, 3]
    high = data[:, 1]
    low = data[:, 2]
    volume = data[:, 5]
    n = len(data)
    ind = np.zeros((n, 20), dtype=np.float32)
    
    for i in range(5, n): ind[i, 0] = np.mean(close[max(0,i-5):i])
    for i in range(20, n): ind[i, 1] = np.mean(close[max(0,i-20):i])
    for i in range(60, n): ind[i, 2] = np.mean(close[max(0,i-60):i])
    
    for i in range(14, n):
        gains = np.diff(close[i-14:i+1])
        gains = gains[gains > 0]
        losses = -gains[gains < 0]
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0
        rs = avg_gain / (avg_loss + 1e-8)
        ind[i, 3] = 100 - (100 / (1 + rs))
    
    for i in range(20, n):
        ma, std = ind[i, 1], np.std(close[i-20:i])
        ind[i, 4] = (close[i] - ma) / (std + 1e-8)
    
    for i in range(26, n):
        e12 = np.mean(close[max(0,i-12):i])*(2/13) + ind[i-1,7]*(11/13) if i>12 else np.mean(close[:i])
        e26 = np.mean(close[max(0,i-26):i])*(2/27) + ind[i-1,8]*(25/27) if i>26 else np.mean(close[:i])
        macd = e12 - e26
        ind[i, 7] = macd / (close[i] + 1e-8)
        ind[i, 9] = 0.5  # 简化
    
    for i in range(20, n):
        returns = np.diff(close[i-20:i]) / (close[i-20:i-1] + 1e-8)
        ind[i, 11] = np.std(returns) * np.sqrt(365) if len(returns) > 0 else 0
    
    for i in range(20, n): ind[i, 12] = (ind[i,0] - ind[i,1]) / (ind[i,1] + 1e-8)
    for i in range(20, n): ind[i, 13] = (volume[i] - np.mean(volume[i-20:i])) / (np.mean(volume[i-20:i]) + 1e-8)
    
    return ind


def get_klines(symbol, limit=100):
    """获取K线数据"""
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1d&limit={limit}"
    r = requests.get(url, proxies=PROXIES, timeout=15)
    data = json.loads(r.text)
    return np.array([[float(d[1]),float(d[2]),float(d[3]),float(d[4]),float(d[5]),float(d[7])] for d in data], dtype=np.float32)


def get_observation(data, position=0, balance=10000, entry_price=0):
    """获取当前观察"""
    ind = calculate_indicators(data)
    i = len(data) - 1
    c = data[i, 3]
    
    total = balance + position * c
    pos_ratio = (position * c) / (total + 1e-8)
    
    obs = np.array([
        ind[i,0]/1000, ind[i,1]/1000, ind[i,2]/1000, ind[i,3]/100,
        ind[i,4], 0, 0, ind[i,7]*10, 0, 0,
        0, ind[i,11], ind[i,12]*10, ind[i,13], 0,
        0, 0, 0,
        balance / 10000,
        pos_ratio,
        (total - 10000) / 10000,
        0, 0, 0, 1 if position > 0 else 0
    ], dtype=np.float32)
    
    return obs, c


class LiveTrader:
    """实盘/模拟盘交易机器人"""
    
    # 各币种精度配置
    PRECISION = {
        'BTCUSDT': 5,   # 0.00001 BTC
        'ETHUSDT': 4,   # 0.0001 ETH
        'BNBUSDT': 3,   # 0.001 BNB
        'SOLUSDT': 3,   # 0.001 SOL
        'XRPUSDT': 1,   # 0.1 XRP
        'ADAUSDT': 1,   # 0.1 ADA
        'DOGEUSDT': 0,  # 1 DOGE
        'DOTUSDT': 2,   # 0.01 DOT
        'MATICUSDT': 1, # 0.1 MATIC
        'LINKUSDT': 2,  # 0.01 LINK
    }
    
    def __init__(self, symbol, model_path, initial_balance=10000):
        self.symbol = symbol
        self.model = PPO.load(model_path)
        self.exchange = BinanceTrading()
        self.initial_balance = initial_balance
        
        self.position = 0
        self.balance = initial_balance
        self.entry_price = 0
        self.trade_log = []
        
        # 获取精度
        self.precision = self.PRECISION.get(symbol, 3)
    
    def check_position(self):
        """检查当前持仓"""
        asset = self.symbol.replace('USDT', '')
        self.balance = self.exchange.get_balance('USDT')
        # 获取持仓（简化版，实际需要查询账户）
        return self.balance, self.position
    
    def run(self, dry_run=False):  # 默认开启真实下单模式
        """运行交易"""
        # 获取数据
        data = get_klines(self.symbol, limit=100)
        obs, current_price = get_observation(data, self.position, self.balance, self.entry_price)
        
        # 模型预测
        action, _ = self.model.predict(obs)
        
        # 执行动作
        action_names = ['HOLD', 'BUY_33%', 'BUY_66%', 'SELL_50%', 'SELL_100%']
        action_name = action_names[min(action, 4)]
        
        if dry_run:
            # 模拟执行
            if 'BUY' in action_name and self.position == 0:
                pct = 0.33 if '33' in action_name else 0.66
                invest = self.balance * pct
                self.position = invest / current_price
                self.balance -= invest
                self.entry_price = current_price
                self.trade_log.append({
                    'time': datetime.now().strftime('%Y-%m-%d %H:%M'),
                    'action': action_name,
                    'price': current_price,
                    'result': 'SUCCESS (dry run)'
                })
            elif 'SELL' in action_name and self.position > 0:
                pct = 0.5 if '50' in action_name else 1.0
                self.balance += self.position * pct * current_price
                self.position *= (1 - pct)
                if self.position == 0:
                    self.entry_price = 0
                self.trade_log.append({
                    'time': datetime.now().strftime('%Y-%m-%d %H:%M'),
                    'action': action_name,
                    'price': current_price,
                    'result': 'SUCCESS (dry run)'
                })
            else:
                self.trade_log.append({
                    'time': datetime.now().strftime('%Y-%m-%d %H:%M'),
                    'action': 'HOLD',
                    'price': current_price,
                    'result': 'NO ACTION'
                })
        else:
            # 真实下单模式
            asset = self.symbol.replace('USDT', '')
            
            if 'BUY' in action_name:
                pct = 0.33 if '33' in action_name else 0.66
                usdt_balance = self.exchange.get_balance('USDT')
                invest_amount = usdt_balance * pct
                
                if invest_amount > 10:  # 最小下单金额$10
                    # 计算下单数量（使用正确精度）
                    quantity = round(invest_amount / current_price, self.precision)
                    
                    if quantity > 0:
                        # 下买单
                        result = self.exchange.place_order(
                            self.symbol, 'BUY', quantity, 'MARKET'
                        )
                        
                        if 'orderId' in result:
                            self.trade_log.append({
                                'time': datetime.now().strftime('%Y-%m-%d %H:%M'),
                                'action': action_name,
                                'price': current_price,
                                'quantity': quantity,
                                'result': f"SUCCESS - Order ID: {result['orderId']}"
                            })
                            self.entry_price = current_price
                        else:
                            self.trade_log.append({
                                'time': datetime.now().strftime('%Y-%m-%d %H:%M'),
                                'action': action_name,
                                'price': current_price,
                                'result': f"FAILED - {result}"
                            })
                        
            elif 'SELL' in action_name:
                # 获取当前持仓
                asset_balance = self.exchange.get_balance(asset)
                
                if asset_balance > 0:
                    pct = 0.5 if '50' in action_name else 1.0
                    quantity = round(asset_balance * pct, self.precision)
                    
                    if quantity > 0:
                        result = self.exchange.place_order(
                            self.symbol, 'SELL', quantity, 'MARKET'
                        )
                        
                        if 'orderId' in result:
                            self.trade_log.append({
                                'time': datetime.now().strftime('%Y-%m-%d %H:%M'),
                                'action': action_name,
                                'price': current_price,
                                'quantity': quantity,
                                'result': f"SUCCESS - Order ID: {result['orderId']}"
                            })
                        else:
                            self.trade_log.append({
                                'time': datetime.now().strftime('%Y-%m-%d %H:%M'),
                                'action': action_name,
                                'price': current_price,
                                'result': f"FAILED - {result}"
                            })
            else:
                self.trade_log.append({
                    'time': datetime.now().strftime('%Y-%m-%d %H:%M'),
                    'action': 'HOLD',
                    'price': current_price,
                    'result': 'NO ACTION'
                })
        
        total_value = self.balance + self.position * current_price
        pnl = (total_value - self.initial_balance) / self.initial_balance * 100
        
        return {
            'symbol': self.symbol,
            'price': current_price,
            'action': action_name,
            'position': self.position,
            'balance': self.balance,
            'total_value': total_value,
            'pnl_pct': pnl
        }


def main():
    """主程序"""
    print("="*50)
    print("币安测试网交易机器人 - 真实下单模式")
    print("="*50)
    
    # 配置要交易的币种
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
    
    print("\n当前模式: 真实下单 (dry_run=False)")
    print("警告: 这将在测试网真实下单!\n")
    
    results = []
    for sym in symbols:
        model_path = f"ppo_v7_{sym}"
        trader = LiveTrader(sym, model_path)
        result = trader.run(dry_run=False)  # 真实下单
        results.append(result)
        
        print(f"\n{sym}:")
        print(f"  当前价格: ${result['price']:.2f}")
        print(f"  建议动作: {result['action']}")
        print(f"  持仓: {result['position']:.4f}")
        print(f"  余额: ${result['balance']:.2f}")
        print(f"  总资产: ${result['total_value']:.2f}")
        print(f"  收益: {result['pnl_pct']:+.2f}%")
        
        # 显示交易日志
        if trader.trade_log:
            print(f"  交易记录:")
            for log in trader.trade_log[-1:]:
                print(f"    {log}")
    
    return results


if __name__ == "__main__":
    main()

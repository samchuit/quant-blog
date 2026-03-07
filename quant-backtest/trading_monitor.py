"""
交易监控系统
============
定时运行 + 日志记录 + 结果报告
"""
import os
import json
import numpy as np
import requests
from datetime import datetime
import schedule
import time
from stable_baselines3 import PPO
import hmac
import hashlib
import warnings
warnings.filterwarnings('ignore')

# 配置
WORK_DIR = r"C:\Users\Administrator\.openclaw\workspace\quant-backtest"
os.chdir(WORK_DIR)

API_KEY = "W1dAcyGFmPloyATJIxs2xwhGqEDhRIqZA7JesyTSSNV4ILYLJ3pHw39Je406bUWC"
API_SECRET = "NnvDpodpaQ27VKJwArDZ9IWuZkyYPmzBaL7K2IUYBKdSgQaMZUdrh8MP0jh9tFI6"
BASE_URL = "https://testnet.binance.vision"
PROXIES = {'http': 'http://192.168.0.45:7897', 'https': 'http://192.168.0.45:7897'}

# 精度配置
PRECISION = {
    'BTCUSDT': 5, 'ETHUSDT': 4, 'BNBUSDT': 3, 'SOLUSDT': 3,
    'XRPUSDT': 1, 'ADAUSDT': 1, 'DOGEUSDT': 0, 'DOTUSDT': 2,
}


class TradingMonitor:
    """交易监控系统"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.proxies = PROXIES
        self.log_file = "trading_log.json"
        self.daily_report_file = "daily_report.txt"
        
        # 加载历史日志
        self.trade_history = self._load_log()
    
    def _load_log(self):
        """加载日志"""
        try:
            with open(self.log_file, 'r') as f:
                return json.load(f)
        except:
            return []
    
    def _save_log(self):
        """保存日志"""
        with open(self.log_file, 'w') as f:
            json.dump(self.trade_history, f, indent=2, ensure_ascii=False)
    
    def _sign(self, query_string):
        """签名"""
        return hmac.new(API_SECRET.encode(), query_string.encode(), hashlib.sha256).hexdigest()
    
    def _request(self, endpoint, params=None, signed=False):
        """请求"""
        url = f"{BASE_URL}{endpoint}"
        headers = {'X-MBX-APIKEY': API_KEY}
        
        if params is None:
            params = {}
        
        if signed:
            params['timestamp'] = int(time.time() * 1000)
            sorted_params = dict(sorted(params.items()))
            query_string = '&'.join([f"{k}={v}" for k, v in sorted_params.items()])
            signature = self._sign(query_string)
            url = f"{url}?{query_string}&signature={signature}"
        
        try:
            r = self.session.get(url, headers=headers, timeout=15)
            return r.json()
        except Exception as e:
            return {'error': str(e)}
    
    def get_balance(self, asset='USDT'):
        """获取余额"""
        data = self._request('/api/v3/account', signed=True)
        if 'balances' in data:
            for b in data['balances']:
                if b['asset'] == asset:
                    return float(b['free'])
        return 0
    
    def get_all_balances(self):
        """获取所有余额"""
        data = self._request('/api/v3/account', signed=True)
        balances = {}
        if 'balances' in data:
            for b in data['balances']:
                free = float(b['free'])
                if free > 0:
                    balances[b['asset']] = free
        return balances
    
    def get_price(self, symbol):
        """获取价格 - 公开接口不需要签名"""
        try:
            url = f"{BASE_URL}/api/v3/ticker/price?symbol={symbol}"
            r = self.session.get(url, timeout=15)
            data = r.json()
            if isinstance(data, dict) and 'price' in data:
                return float(data['price'])
        except:
            pass
        return 0
    
    def place_order(self, symbol, side, quantity):
        """下单"""
        timestamp = int(time.time() * 1000)
        precision = PRECISION.get(symbol, 3)
        quantity = round(float(quantity), precision)
        
        params = {
            'symbol': symbol,
            'side': side,
            'type': 'MARKET',
            'quantity': quantity,
            'timestamp': timestamp
        }
        
        sorted_params = dict(sorted(params.items()))
        query_string = '&'.join([f"{k}={v}" for k, v in sorted_params.items()])
        signature = self._sign(query_string)
        
        url = f"{BASE_URL}/api/v3/order?{query_string}&signature={signature}"
        headers = {'X-MBX-APIKEY': API_KEY}
        
        try:
            r = self.session.post(url, headers=headers, timeout=15)
            return r.json()
        except Exception as e:
            return {'error': str(e)}
    
    def get_model_action(self, symbol):
        """获取模型建议"""
        try:
            # 获取K线数据
            url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1d&limit=100"
            r = requests.get(url, proxies=PROXIES, timeout=15)
            data = r.json()
            
            # 简化特征
            close = np.array([float(d[4]) for d in data], dtype=np.float32)
            
            # 计算简单指标
            ma5 = np.mean(close[-5:])
            ma20 = np.mean(close[-20:])
            rsi = self._calc_rsi(close)
            
            # 构建观察
            obs = np.zeros(25, dtype=np.float32)
            obs[0] = ma5 / 1000
            obs[1] = ma20 / 1000
            obs[3] = rsi / 100
            obs[12] = (ma5 - ma20) / ma20
            obs[18] = 1  # balance ratio
            
            # 加载模型
            model = PPO.load(f"ppo_v7_{symbol}")
            action, _ = model.predict(obs)
            
            actions = ['HOLD', 'BUY_33%', 'BUY_66%', 'SELL_50%', 'SELL_100%']
            return actions[min(action, 4)]
        except Exception as e:
            return f"ERROR: {e}"
    
    def _calc_rsi(self, close, period=14):
        """计算RSI"""
        if len(close) < period + 1:
            return 50
        gains = np.diff(close[-period-1:])
        up = np.mean(gains[gains > 0]) if len(gains[gains > 0]) > 0 else 0
        down = -np.mean(gains[gains < 0]) if len(gains[gains < 0]) > 0 else 0
        if down == 0:
            return 100
        rs = up / down
        return 100 - (100 / (1 + rs))
    
    def execute_trade(self, symbol, action, price):
        """执行交易"""
        asset = symbol.replace('USDT', '')
        result = {'symbol': symbol, 'action': action, 'price': price}
        
        if 'BUY' in action:
            pct = 0.33 if '33' in action else 0.66
            usdt = self.get_balance('USDT')
            invest = usdt * pct
            
            if invest > 10:
                qty = invest / price
                order = self.place_order(symbol, 'BUY', qty)
                
                if 'orderId' in order:
                    result['status'] = 'SUCCESS'
                    result['orderId'] = order['orderId']
                    result['quantity'] = qty
                else:
                    result['status'] = 'FAILED'
                    result['error'] = order
            else:
                result['status'] = 'SKIPPED'
                result['reason'] = 'Insufficient balance'
                
        elif 'SELL' in action:
            pct = 0.5 if '50' in action else 1.0
            balance = self.get_balance(asset)
            
            if balance > 0:
                qty = balance * pct
                order = self.place_order(symbol, 'SELL', qty)
                
                if 'orderId' in order:
                    result['status'] = 'SUCCESS'
                    result['orderId'] = order['orderId']
                    result['quantity'] = qty
                else:
                    result['status'] = 'FAILED'
                    result['error'] = order
            else:
                result['status'] = 'SKIPPED'
                result['reason'] = 'No position'
        else:
            result['status'] = 'HOLD'
        
        return result
    
    def run_trading_cycle(self):
        """运行一次交易周期"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n{'='*60}")
        print(f"交易监控 - {timestamp}")
        print(f"{'='*60}")
        
        symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
        results = []
        
        # 获取账户余额
        print("\n[账户余额]")
        balances = self.get_all_balances()
        for asset in ['USDT', 'BTC', 'ETH', 'SOL']:
            if asset in balances:
                print(f"  {asset}: {balances[asset]:.4f}")
        
        # 交易执行
        print("\n[交易执行]")
        for symbol in symbols:
            price = self.get_price(symbol)
            action = self.get_model_action(symbol)
            
            print(f"\n  {symbol}:")
            print(f"    价格: ${price:,.2f}")
            print(f"    建议: {action}")
            
            # 执行交易
            result = self.execute_trade(symbol, action, price)
            result['timestamp'] = timestamp
            results.append(result)
            
            print(f"    状态: {result.get('status', 'N/A')}")
            if 'orderId' in result:
                print(f"    订单ID: {result['orderId']}")
            
            # 记录日志
            self.trade_history.append(result)
        
        # 保存日志
        self._save_log()
        
        # 生成每日报告
        self._generate_daily_report(results, balances)
        
        return results
    
    def _generate_daily_report(self, results, balances):
        """生成每日报告"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
        
        report = f"""
{'='*60}
交易日报 - {timestamp}
{'='*60}

[账户余额]
"""
        for asset, amount in sorted(balances.items()):
            report += f"  {asset}: {amount:.4f}\n"
        
        report += "\n[交易记录]\n"
        for r in results:
            report += f"  {r['symbol']}: {r['action']} @ ${r['price']:,.2f} - {r.get('status', 'N/A')}\n"
            if 'orderId' in r:
                report += f"    订单ID: {r['orderId']}\n"
        
        report += f"\n[历史统计]\n"
        report += f"  总交易次数: {len(self.trade_history)}\n"
        success = sum(1 for t in self.trade_history if t.get('status') == 'SUCCESS')
        report += f"  成功交易: {success}\n"
        
        with open(self.daily_report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n[报告已保存] {self.daily_report_file}")


def main():
    """主程序"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['once', 'schedule', 'daily'], default='once')
    parser.add_argument('--interval', type=int, default=60, help='运行间隔(分钟)')
    parser.add_argument('--daily-time', type=str, default='08:00', help='每日运行时间(北京时间)')
    args = parser.parse_args()
    
    monitor = TradingMonitor()
    
    if args.mode == 'once':
        # 运行一次
        monitor.run_trading_cycle()
    elif args.mode == 'daily':
        # 每日定时运行（日线策略）
        print(f"启动日线交易监控 (每天 {args.daily_time} 执行)")
        print("策略说明: 使用日线数据，与回测完全匹配")
        print("按 Ctrl+C 停止")
        
        schedule.every().day.at(args.daily_time).do(monitor.run_trading_cycle)
        
        # 显示下次运行时间
        import datetime
        now = datetime.datetime.now()
        print(f"\n当前时间: {now.strftime('%Y-%m-%d %H:%M')}")
        print(f"下次执行: 明天 {args.daily_time}")
        
        while True:
            schedule.run_pending()
            time.sleep(60)
    else:
        # 每隔X分钟运行
        print(f"启动定时交易监控 (每{args.interval}分钟)")
        print("注意: 此模式可能与日线数据不匹配")
        print("按 Ctrl+C 停止")
        
        schedule.every(args.interval).minutes.do(monitor.run_trading_cycle)
        
        # 首次运行
        monitor.run_trading_cycle()
        
        while True:
            schedule.run_pending()
            time.sleep(10)


if __name__ == "__main__":
    main()

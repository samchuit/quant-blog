"""
自动重训练脚本
===============
每月自动重新训练模型，保持模型与市场同步
"""
import os
import sys
import json
import time
import schedule
import numpy as np
import requests
from datetime import datetime, timedelta
from stable_baselines3 import PPO
import gymnasium as gym
import warnings
warnings.filterwarnings('ignore')

# 工作目录
WORK_DIR = r"C:\Users\Administrator\.openclaw\workspace\quant-backtest"
os.chdir(WORK_DIR)

PROXIES = {'http': 'http://192.168.0.45:7897', 'https': 'http://192.168.0.45:7897'}


def get_data(symbol, limit=730):
    """获取数据"""
    try:
        url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1d&limit={limit}'
        r = requests.get(url, proxies=PROXIES, timeout=15)
        data = json.loads(r.text)
        return np.array([[float(d[1]),float(d[2]),float(d[3]),float(d[4]),float(d[5]),float(d[7])] for d in data], dtype=np.float32)
    except Exception as e:
        print(f"获取{symbol}数据失败: {e}")
        return None


def calculate_indicators(data):
    """计算指标"""
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
        ind[i, 8] = (macd * 0.8 + ind[i-1,9]*0.2) / (close[i] + 1e-8) if i>26 else ind[i,7]
        ind[i, 9] = (macd - ind[i,8]) / (abs(macd - ind[i,8]) + 1e-8)
    
    for i in range(14, n):
        tr = np.maximum(high[i]-low[i], max(abs(high[i]-close[i-1]), abs(low[i]-close[i-1])))
        ind[i, 10] = tr / (close[i] + 1e-8)
    
    for i in range(20, n):
        returns = np.diff(close[i-20:i]) / (close[i-20:i-1] + 1e-8)
        ind[i, 11] = np.std(returns) * np.sqrt(365) if len(returns) > 0 else 0
    
    for i in range(20, n): ind[i, 12] = (ind[i,0] - ind[i,1]) / (ind[i,1] + 1e-8)
    for i in range(20, n): ind[i, 13] = (volume[i] - np.mean(volume[i-20:i])) / (np.mean(volume[i-20:i]) + 1e-8)
    
    return ind


class TradingEnv(gym.Env):
    """训练环境"""
    def __init__(self, data, commission=0.001, slippage=0.0005):
        super().__init__()
        self.data = data
        self.commission = commission
        self.slippage = slippage
        self.indicators = calculate_indicators(data)
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(-10, 10, (25,), dtype=np.float32)
        self.reset()
    
    def reset(self, seed=None, options=None):
        self.idx = 60
        self.balance = 10000
        self.position = 0
        self.total_profit = 10000
        self.peak_profit = 10000
        self.trade_count = 0
        self.entry_price = 0
        return self._obs(), {}
    
    def _obs(self):
        i = min(self.idx, len(self.data)-1)
        c = self.data[i, 3]
        ind = self.indicators[i]
        pos_ratio = (self.position * c) / (self.total_profit + 1e-8)
        return np.array([
            ind[0]/1000, ind[1]/1000, ind[2]/1000, ind[3]/100,
            ind[4], 0, 0, ind[7]*10, 0, 0,
            0, ind[11], ind[12]*10, ind[13], 0,
            0, 0, 0,
            self.balance / 10000, pos_ratio,
            (self.total_profit - 10000) / 10000, 0, 0, 0, 1 if self.position > 0 else 0
        ], dtype=np.float32)
    
    def step(self, action):
        price = self.data[self.idx, 0]
        
        if self.position > 0 and self.entry_price > 0:
            pnl = (price - self.entry_price) / self.entry_price
            if pnl >= 0.15 or pnl <= -0.08:
                action = 3
        
        if action == 1 and self.balance > 0 and self.position == 0:
            eff_price = price * (1 + self.slippage)
            self.position = self.balance * 0.7 / eff_price * (1 - self.commission)
            self.balance *= 0.3
            self.entry_price = eff_price
            self.trade_count += 1
        elif action == 2 and self.position > 0:
            sell_price = price * (1 - self.slippage)
            self.balance += self.position * 0.5 * sell_price * (1 - self.commission)
            self.position *= 0.5
            self.trade_count += 1
        elif action == 3 and self.position > 0:
            sell_price = price * (1 - self.slippage)
            self.balance += self.position * sell_price * (1 - self.commission)
            self.position = 0
            self.entry_price = 0
            self.trade_count += 1
        
        self.idx += 1
        price = self.data[min(self.idx, len(self.data)-1), 0]
        self.total_profit = self.balance + self.position * price
        self.peak_profit = max(self.peak_profit, self.total_profit)
        
        reward = (self.total_profit - 10000) / 10000
        done = self.idx >= len(self.data) - 1
        
        return self._obs(), reward, done, False, {}


def train_model(symbol, steps=150000):
    """训练单个模型"""
    print(f"\n{'='*50}")
    print(f"训练 {symbol}")
    print(f"{'='*50}")
    
    data = get_data(symbol)
    if data is None:
        print(f"跳过 {symbol}")
        return None
    
    print(f"数据: {len(data)}天")
    
    env = TradingEnv(data)
    model = PPO('MlpPolicy', env, learning_rate=3e-4, n_steps=2048,
                batch_size=64, n_epochs=10, gamma=0.99, ent_coef=0.01, verbose=0)
    
    print(f"开始训练 ({steps//1000}k步)...")
    model.learn(steps, progress_bar=False)
    
    # 评估
    env2 = TradingEnv(data)
    obs, _ = env2.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, _, done, _, _ = env2.step(action)
    
    final_return = (env2.total_profit / 10000 - 1) * 100
    bh_return = (data[-1, 3] / data[0, 3] - 1) * 100
    
    print(f"结果: RL {final_return:+.1f}% BH {bh_return:+.1f}% 超额 {final_return-bh_return:+.1f}%")
    
    # 保存
    timestamp = datetime.now().strftime('%Y%m%d')
    save_path = f"ppo_v7_{symbol}_{timestamp}"
    model.save(save_path)
    print(f"保存: {save_path}.zip")
    
    return {
        'symbol': symbol,
        'rl_return': final_return,
        'bh_return': bh_return,
        'excess': final_return - bh_return,
        'save_path': save_path
    }


def monthly_retrain():
    """每月重训练"""
    print("\n" + "="*60)
    print(f"月度重训练 - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*60)
    
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
               'ADAUSDT', 'DOGEUSDT', 'DOTUSDT', 'MATICUSDT', 'LINKUSDT']
    
    results = []
    for sym in symbols:
        try:
            r = train_model(sym, steps=150000)
            if r:
                results.append(r)
        except Exception as e:
            print(f"{sym} 训练失败: {e}")
    
    # 保存日志
    log = {
        'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'results': results,
        'avg_excess': np.mean([r['excess'] for r in results]) if results else 0
    }
    
    with open('retrain_log.json', 'w') as f:
        json.dump(log, f, indent=2)
    
    print(f"\n平均超额收益: {log['avg_excess']:+.1f}%")
    print("日志已保存: retrain_log.json")
    
    return results


def schedule_monthly():
    """设置定时任务"""
    print("设置每月1号凌晨3点重训练...")
    schedule.every().day.at("03:00").do(check_and_retrain)
    
    while True:
        schedule.run_pending()
        time.sleep(3600)  # 每小时检查一次


def check_and_retrain():
    """检查是否需要重训练"""
    today = datetime.now()
    if today.day == 1:  # 每月1号
        print("今天是月初，开始重训练...")
        monthly_retrain()
    else:
        print(f"今天{today.day}号，不是重训练日")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['now', 'schedule'], default='now')
    args = parser.parse_args()
    
    if args.mode == 'now':
        # 立即执行一次
        monthly_retrain()
    else:
        # 启动定时任务
        schedule_monthly()

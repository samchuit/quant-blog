"""
量化交易模型 v8 - 滚动测试版
==================
改进: 滚动窗口验证
- 每90天重训练一次
- 保持模型与市场同步
- 验证模型在不同时期的稳健性
"""
import numpy as np
import requests
import json
from stable_baselines3 import PPO
import gymnasium as gym
import warnings
warnings.filterwarnings('ignore')

proxies = {'http': 'http://192.168.0.45:7897', 'https': 'http://192.168.0.45:7897'}

def get_data(symbol, interval='1d', limit=1000):
    """获取更多数据用于滚动测试"""
    for i in range(3):
        try:
            url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}'
            r = requests.get(url, proxies=proxies, timeout=15)
            data = json.loads(r.text)
            ohlcv = [[float(d[1]),float(d[2]),float(d[3]),float(d[4]),float(d[5]),float(d[7])] for d in data]
            return np.array(ohlcv, dtype=np.float32)
        except:
            import time
            time.sleep(1)
    return None


def calculate_indicators(data):
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
        ind[i, 5] = (ma + 2*std - close[i]) / (2*std + 1e-8)
        ind[i, 6] = (close[i] - (ma - 2*std)) / (2*std + 1e-8)
    
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
    for i in range(10, n): ind[i, 14] = (close[i] - close[i-10]) / (close[i-10] + 1e-8)
    for i in range(20, n):
        h20, l20 = np.max(high[i-20:i]), np.min(low[i-20:i])
        ind[i, 15] = 1 if close[i] > h20 else (-1 if close[i] < l20 else 0)
    for i in range(60, n):
        m5, m20, m60 = ind[i,0], ind[i,1], ind[i,2]
        ind[i, 16] = 1 if m5 > m20 > m60 else (-1 if m5 < m20 < m60 else 0)
    for i in range(20, n):
        h20, l20 = np.max(high[i-20:i]), np.min(low[i-20:i])
        ind[i, 17] = (close[i] - l20) / (h20 - l20 + 1e-8)
    
    return ind


class TradingEnv(gym.Env):
    def __init__(self, data, commission=0.001, slippage=0.0005,
                 max_drawdown=0.20, take_profit=0.15, stop_loss=0.08):
        super().__init__()
        self.data = data
        self.commission = commission
        self.slippage = slippage
        self.max_drawdown = max_drawdown
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        
        self.indicators = calculate_indicators(data)
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(-10, 10, (25,), dtype=np.float32)
        self.total_cost = 0
        self.reset()
    
    def reset(self, seed=None, options=None):
        self.idx = min(60, len(self.data) // 2)  # 从中间开始，确保有足够历史
        self.balance = 10000
        self.position = 0
        self.total_profit = 10000
        self.peak_profit = 10000
        self.trade_count = 0
        self.win_count = 0
        self.entry_price = 0
        self.total_cost = 0
        return self._obs(), {}
    
    def _obs(self):
        i = min(self.idx, len(self.data)-1)
        c = self.data[i, 3]
        ind = self.indicators[i]
        pos_ratio = (self.position * c) / (self.total_profit + 1e-8)
        drawdown = (self.peak_profit - self.total_profit) / (self.peak_profit + 1e-8)
        trend = (c - ind[1]) / (ind[1] + 1e-8) if ind[1] > 0 else 0
        
        return np.array([
            ind[0]/1000, ind[1]/1000, ind[2]/1000, ind[3]/100,
            ind[4], ind[5], ind[6], ind[7]*10, ind[8]*10, ind[9],
            ind[10]*100, ind[11], ind[12]*10, ind[13], ind[14]*10,
            ind[15], ind[16], ind[17],
            self.balance / 10000,
            pos_ratio,
            (self.total_profit - 10000) / 10000,
            drawdown,
            self.trade_count / 100,
            trend,
            1 if self.position > 0 else 0
        ], dtype=np.float32)
    
    def step(self, action):
        price = self.data[self.idx, 0]
        
        # 止盈止损
        if self.position > 0 and self.entry_price > 0:
            pnl = (price - self.entry_price) / self.entry_price
            if pnl >= self.take_profit or pnl <= -self.stop_loss:
                action = 3
        
        # 回撤强平
        if self.position > 0:
            drawdown = (self.peak_profit - self.total_profit) / self.peak_profit
            if drawdown > self.max_drawdown:
                action = 3
        
        # 执行
        if action == 1 and self.balance > 0 and self.position == 0:
            eff_price = price * (1 + self.slippage)
            invest = self.balance * 0.7
            self.position = invest / eff_price * (1 - self.commission)
            self.balance -= invest
            self.total_cost += invest * (self.commission + self.slippage)
            self.entry_price = eff_price
            self.trade_count += 1
        elif action == 2 and self.position > 0:
            sell_price = price * (1 - self.slippage)
            self.balance += self.position * 0.5 * sell_price * (1 - self.commission)
            self.position *= 0.5
            self.total_cost += self.position * price * (self.commission + self.slippage)
            self.trade_count += 1
        elif action == 3 and self.position > 0:
            sell_price = price * (1 - self.slippage)
            if sell_price > self.entry_price:
                self.win_count += 1
            self.balance += self.position * sell_price * (1 - self.commission)
            self.total_cost += self.position * price * (self.commission + self.slippage)
            self.position = 0
            self.entry_price = 0
            self.trade_count += 1
        
        self.idx += 1
        price = self.data[min(self.idx, len(self.data)-1), 0]
        self.total_profit = self.balance + self.position * price
        self.peak_profit = max(self.peak_profit, self.total_profit)
        
        reward = (self.total_profit - 10000) / 10000 - self.total_cost / 10000 * 0.1
        done = self.idx >= len(self.data) - 1
        
        return self._obs(), reward, done, False, {
            'total_profit': self.total_profit,
            'total_cost': self.total_cost,
            'trade_count': self.trade_count
        }


def rolling_test(sym, data, train_window=240, test_window=120, step=120):
    """
    滚动测试
    train_window: 训练窗口(天) - 8个月
    test_window: 测试窗口(天) - 4个月
    step: 滚动步长(天) - 4个月
    """
    results = []
    n = len(data)
    
    # 计算滚动次数
    start_idx = 0
    round_num = 0
    
    while start_idx + train_window + test_window <= n:
        round_num += 1
        train_start = start_idx
        train_end = start_idx + train_window
        test_start = train_end
        test_end = test_start + test_window
        
        if test_end > n:
            break
        
        train_data = data[train_start:train_end]
        test_data = data[test_start:test_end]
        
        # 训练
        env_train = TradingEnv(train_data)
        model = PPO('MlpPolicy', env_train, learning_rate=3e-4, n_steps=1024,
                    batch_size=64, n_epochs=8, gamma=0.99, ent_coef=0.01, verbose=0)
        model.learn(80000, progress_bar=False)
        
        # 测试
        env_test = TradingEnv(test_data)
        obs, _ = env_test.reset()
        done = False
        while not done:
            a, _ = model.predict(obs)
            obs, _, done, _, _ = env_test.step(a)
        
        rl = (env_test.total_profit / 10000 - 1) * 100
        bh = (test_data[-1,3] / test_data[0,3] - 1) * 100
        
        results.append({
            'round': round_num,
            'train_days': train_window,
            'test_days': test_window,
            'rl': rl,
            'bh': bh,
            'excess': rl - bh,
            'trades': env_test.trade_count,
            'cost': env_test.total_cost
        })
        
        print(f"    Round {round_num}: RL {rl:+.1f}% BH {bh:+.1f}% Excess {rl-bh:+.1f}%")
        
        # 滚动
        start_idx += step
    
    return results


if __name__ == "__main__":
    print("="*60)
    print("量化交易模型 v8 - 滚动测试版")
    print("="*60)
    print("配置: 训练240天(8个月) -> 测试120天(4个月) -> 滚动120天")
    print("="*60)
    
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT']
    
    print("\n[1] 获取数据 (最多1000天)...")
    all_data = {}
    for s in symbols:
        data = get_data(s, limit=1000)
        if data is not None:
            all_data[s] = data
            years = len(data) / 365
            print(f"  {s}: {len(data)}天 (~{years:.1f}年)")
    
    print("\n[2] 滚动测试...")
    
    all_results = {}
    for sym, data in all_data.items():
        print(f"\n  === {sym} ===")
        results = rolling_test(sym, data, train_window=240, test_window=120, step=120)
        all_results[sym] = results
        
        # 统计
        avg_excess = np.mean([r['excess'] for r in results])
        win_rate = sum(1 for r in results if r['excess'] > 0) / len(results) * 100
        print(f"    平均超额: {avg_excess:+.1f}% 胜率: {win_rate:.0f}%")
    
    # 总汇总
    print("\n" + "="*60)
    print("v8 滚动测试汇总")
    print("="*60)
    
    for sym, results in all_results.items():
        avg_excess = np.mean([r['excess'] for r in results])
        avg_rl = np.mean([r['rl'] for r in results])
        win_rate = sum(1 for r in results if r['excess'] > 0) / len(results) * 100
        rounds = len(results)
        print(f"{sym}: 平均超额 {avg_excess:+.1f}% 平均RL {avg_rl:+.1f}% 胜率 {win_rate:.0f}% ({rounds}轮)")
    
    total_avg_excess = np.mean([r['excess'] for results in all_results.values() for r in results])
    total_win_rate = sum(1 for results in all_results.values() for r in results if r['excess'] > 0) / sum(len(r) for r in all_results.values()) * 100
    
    print(f"\n总平均超额: {total_avg_excess:+.1f}%")
    print(f"总胜率: {total_win_rate:.0f}%")

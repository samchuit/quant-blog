"""
量化交易模型 v7 - 风控增强版
==================
改进:
1. 交易成本优化 - 真实手续费+滑点
2. 极端行情风控 - 止损/熔断/强平
3. 样本外测试 - 时间段分割验证
"""
import numpy as np
import requests
import json
from stable_baselines3 import PPO
import gymnasium as gym
import warnings
warnings.filterwarnings('ignore')

proxies = {'http': 'http://192.168.0.45:7897', 'https': 'http://192.168.0.45:7897'}

def get_data(symbol, interval='1d', limit=730):
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
    
    # 合并波动率比到现有列
    for i in range(60, n):
        vs = np.std(np.diff(close[i-20:i]) / (close[i-20:i-1] + 1e-8))
        vl = np.std(np.diff(close[i-60:i]) / (close[i-60:i-1] + 1e-8))
        ind[i, 18] = vs / (vl + 1e-8) if vl > 0 else 1
    
    return ind


def get_market_type(data, idx):
    if idx < 60:
        return 0, 0.7
    close = data[:, 3]
    ma20 = np.mean(close[max(0,idx-20):idx])
    ma60 = np.mean(close[max(0,idx-60):idx])
    current = close[idx-1] if idx > 0 else close[0]
    trend_20 = (current - ma20) / (ma20 + 1e-8)
    trend_60 = (current - ma60) / (ma60 + 1e-8)
    returns = np.diff(close[max(0,idx-60):idx]) / (close[max(0,idx-60):idx-1] + 1e-8)
    vol = np.std(returns) * np.sqrt(365) if len(returns) > 1 else 0
    
    if trend_20 > 0.08 and trend_60 > 0.05 and vol < 1.5:
        return 1, 1.0
    elif trend_20 < -0.08 and trend_60 < -0.05:
        return -1, 0.4
    elif abs(trend_20) < 0.08 and vol > 0.8:
        return 0, 0.5
    elif trend_20 > 0:
        return 0, 0.7
    else:
        return 0, 0.5


class TradingEnvV7(gym.Env):
    """v7 风控增强环境"""
    
    def __init__(self, data, initial_balance=10000,
                 # 改进1: 真实交易成本
                 commission=0.001,      # 手续费0.1%
                 slippage=0.0005,       # 滑点0.05%
                 # 改进2: 极端行情风控
                 max_daily_loss=0.05,   # 单日最大亏损5%
                 max_drawdown=0.20,     # 最大回撤20%强平
                 min_profit_threshold=0.008,  # 最小交易收益阈值0.8%
                 # 训练/测试模式
                 mode='train'
                 ):
        super().__init__()
        self.data = data
        self.initial_balance = initial_balance
        self.commission = commission
        self.slippage = slippage
        self.max_daily_loss = max_daily_loss
        self.max_drawdown = max_drawdown
        self.min_profit_threshold = min_profit_threshold
        self.mode = mode
        
        self.indicators = calculate_indicators(data)
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(-10, 10, (25,), dtype=np.float32)
        
        # 记录交易成本
        self.total_cost = 0
        self.daily_pnl_list = []
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        self.idx = 60
        self.balance = self.initial_balance
        self.position = 0
        self.total_profit = self.initial_balance
        self.peak_profit = self.initial_balance
        self.trade_count = 0
        self.win_count = 0
        self.entry_price = 0
        self.daily_start_profit = self.initial_balance
        self.total_cost = 0
        self.daily_pnl_list = []
        self.circuit_breaker = False  # 熔断标志
        self.trades_today = 0
        return self._obs(), {}
    
    def step(self, action):
        current_price = self.data[self.idx, 0]
        
        # ========== 改进2: 极端行情风控 ==========
        # 1. 单日亏损检查
        daily_pnl = (self.total_profit - self.daily_start_profit) / self.daily_start_profit
        if daily_pnl < -self.max_daily_loss:
            action = 3  # 强制平仓
            self.circuit_breaker = True
        
        # 2. 最大回撤检查
        drawdown = (self.peak_profit - self.total_profit) / self.peak_profit
        if drawdown > self.max_drawdown:
            action = 3  # 强平
            self.circuit_breaker = True
        
        # 3. 止盈止损
        if self.position > 0 and self.entry_price > 0:
            pnl = (current_price - self.entry_price) / self.entry_price
            if pnl >= 0.15:  # 15%止盈
                action = 3
            elif pnl <= -0.08:  # 8%止损
                action = 3
        
        # ========== 改进1: 交易成本 ==========
        cost_multiplier = 1 + self.commission + self.slippage
        
        # 执行动作
        if action == 1 and self.balance > 0 and self.position == 0:
            # 买入 - 计算真实成本
            effective_price = current_price * (1 + self.slippage)
            invest = self.balance * 0.7
            self.position = invest / effective_price * (1 - self.commission)
            self.balance -= invest
            
            # 记录成本
            self.total_cost += invest * (self.commission + self.slippage)
            self.entry_price = effective_price
            self.trade_count += 1
            self.trades_today += 1
            
        elif action == 2 and self.position > 0:
            # 卖出50%
            sell_price = current_price * (1 - self.slippage)
            self.balance += self.position * 0.5 * sell_price * (1 - self.commission)
            self.position *= 0.5
            self.total_cost += self.position * 0.5 * current_price * (self.commission + self.slippage)
            self.trade_count += 1
            self.trades_today += 1
            
        elif action == 3 and self.position > 0:
            # 全部卖出
            sell_price = current_price * (1 - self.slippage)
            if self.position * sell_price * (1 - self.commission) > self.entry_price * self.position:
                self.win_count += 1
            self.balance += self.position * sell_price * (1 - self.commission)
            self.total_cost += self.position * current_price * (self.commission + self.slippage)
            self.position = 0
            self.entry_price = 0
            self.trade_count += 1
            self.trades_today += 1
        
        self.idx += 1
        current_price = self.data[min(self.idx, len(self.data)-1), 0]
        self.total_profit = self.balance + self.position * current_price
        
        # 更新峰值
        if self.total_profit > self.peak_profit:
            self.peak_profit = self.total_profit
        
        # 每日重置
        if self.idx % 24 == 0:
            self.daily_pnl_list.append(daily_pnl)
            self.daily_start_profit = self.total_profit
            self.trades_today = 0
        
        # Reward - 扣除真实成本
        reward = (self.total_profit - self.initial_balance) / self.initial_balance
        reward -= self.total_cost / self.initial_balance * 0.1  # 成本惩罚
        
        # 回撤惩罚
        drawdown = (self.peak_profit - self.total_profit) / self.peak_profit
        if drawdown > 0.1: reward -= drawdown * 0.3
        if drawdown > 0.15: reward -= drawdown * 0.5
        
        # 胜率奖励
        if action in [2, 3] and self.total_profit > self.initial_balance:
            reward += 0.02
        
        # 熔断惩罚
        if self.circuit_breaker:
            reward -= 0.1
        
        done = self.idx >= len(self.data) - 1
        
        return self._obs(), reward, done, False, {
            'total_profit': self.total_profit,
            'total_cost': self.total_cost,
            'trade_count': self.trade_count,
            'circuit_breaker': self.circuit_breaker
        }
    
    def _obs(self):
        i = min(self.idx, len(self.data) - 1)
        current_price = self.data[i, 0]
        ind = self.indicators[i]
        market_type, position_scale = get_market_type(self.data, self.idx)
        
        pos_ratio = (self.position * current_price) / (self.total_profit + 1e-8)
        drawdown = (self.peak_profit - self.total_profit) / (self.peak_profit + 1e-8)
        
        obs = np.array([
            ind[0]/1000, ind[1]/1000, ind[2]/1000, ind[3]/100,
            ind[4], ind[5], ind[6], ind[7]*10, ind[8]*10, ind[9],
            ind[10]*100, ind[11], ind[12]*10, ind[13], ind[14]*10,
            ind[15], ind[16], ind[17],
            self.balance / self.initial_balance,
            pos_ratio,
            (self.total_profit - self.initial_balance) / self.initial_balance,
            drawdown,
            self.trade_count / 100,
            position_scale,
            market_type
        ], dtype=np.float32)
        
        return np.clip(obs, -10, 10)


def train_and_test(sym, data, train_ratio=0.7):
    """改进3: 样本外测试"""
    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    print(f"  {sym}: 训练{len(train_data)}天, 测试{len(test_data)}天")
    
    # 训练
    env = TradingEnvV7(train_data, mode='train')
    model = PPO('MlpPolicy', env, learning_rate=3e-4, n_steps=2048,
                batch_size=64, n_epochs=10, gamma=0.99, ent_coef=0.01, verbose=0)
    model.learn(150000, progress_bar=False)
    
    # 样本内评估
    env_in = TradingEnvV7(train_data, mode='test')
    obs, _ = env_in.reset()
    done = False
    while not done:
        a, _ = model.predict(obs)
        obs, _, done, _, _ = env_in.step(a)
    in_sample = (env_in.total_profit/10000-1)*100
    
    # 样本外评估
    env_out = TradingEnvV7(test_data, mode='test')
    obs, _ = env_out.reset()
    done = False
    while not done:
        a, _ = model.predict(obs)
        obs, _, done, _, _ = env_out.step(a)
    out_sample = (env_out.total_profit/10000-1)*100
    bh_out = (test_data[-1,3]/test_data[0,3]-1)*100
    
    return {
        'symbol': sym,
        'in_sample': in_sample,
        'out_sample': out_sample,
        'bh_out': bh_out,
        'excess_out': out_sample - bh_out,
        'trade_count': env_out.trade_count,
        'total_cost': env_out.total_cost,
        'model': model
    }


if __name__ == "__main__":
    print("="*60)
    print("量化交易模型 v7 - 风控增强版")
    print("="*60)
    print("改进:")
    print("  1. 真实交易成本 (手续费+滑点)")
    print("  2. 极端行情风控 (止损/熔断/强平)")
    print("  3. 样本外测试 (70%训练+30%测试)")
    print("="*60)
    
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
               'ADAUSDT', 'DOGEUSDT', 'DOTUSDT', 'MATICUSDT', 'LINKUSDT']
    
    print("\n[1] 获取数据...")
    all_data = {}
    for s in symbols:
        data = get_data(s)
        if data is not None:
            all_data[s] = data
            print(f"  {s}: {len(data)}天")
    
    print("\n[2] 样本外训练+测试...")
    results = []
    
    for sym, data in all_data.items():
        r = train_and_test(sym, data, train_ratio=0.7)
        results.append(r)
        r['model'].save(f"ppo_v7_{sym}")
        print(f"    样本内:{r['in_sample']:+.1f}% 样本外:{r['out_sample']:+.1f}% BH:{r['bh_out']:+.1f}% 超额:{r['excess_out']:+.1f}% 成本:${r['total_cost']:.0f}")
    
    print("\n" + "="*60)
    print("v7 风控增强结果")
    print("="*60)
    
    avg_out = np.mean([r['out_sample'] for r in results])
    avg_bh = np.mean([r['bh_out'] for r in results])
    avg_excess = np.mean([r['excess_out'] for r in results])
    avg_cost = np.mean([r['total_cost'] for r in results])
    
    print(f"\n样本外平均: RL {avg_out:+.1f}% BH {avg_bh:+.1f}% 超额 {avg_excess:+.1f}%")
    print(f"平均交易成本: ${avg_cost:.0f}")
    print("\n[*] 模型已保存: ppo_v7_*.zip")

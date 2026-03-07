"""
v7模型 - 5年数据回测
生成资金曲线图和详细报告
"""
import numpy as np
import requests
import json
from stable_baselines3 import PPO
import gymnasium as gym
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 无GUI模式

proxies = {'http': 'http://192.168.0.45:7897', 'https': 'http://192.168.0.45:7897'}

def get_all_data(symbol, start_year=2021):
    """获取多年数据"""
    all_data = []
    
    # Binance API限制每次1000根K线，需要分批获取
    # 从2021年开始获取，每年获取一次
    for year in range(start_year, 2027):
        start_time = int(f"{year}0101") * 1000  # 年初时间戳
        try:
            url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1d&limit=365&startTime={start_time}000'
            r = requests.get(url, proxies=proxies, timeout=15)
            data = json.loads(r.text)
            if isinstance(data, list) and len(data) > 0:
                ohlcv = [[float(d[1]),float(d[2]),float(d[3]),float(d[4]),float(d[5]),float(d[7])] for d in data]
                all_data.extend(ohlcv)
                print(f"  {year}: {len(data)}天")
        except Exception as e:
            print(f"  {year}: 获取失败 - {e}")
    
    if len(all_data) > 0:
        return np.array(all_data, dtype=np.float32)
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
        self.equity_curve = []
        self.trade_log = []
        self.reset()
    
    def reset(self, seed=None, options=None):
        self.idx = 60
        self.balance = 10000
        self.position = 0
        self.total_profit = 10000
        self.peak_profit = 10000
        self.trade_count = 0
        self.win_count = 0
        self.entry_price = 0
        self.entry_idx = 0
        self.total_cost = 0
        self.equity_curve = [10000]
        self.trade_log = []
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
            self.entry_idx = self.idx
            self.trade_count += 1
        elif action == 2 and self.position > 0:
            sell_price = price * (1 - self.slippage)
            self.balance += self.position * 0.5 * sell_price * (1 - self.commission)
            self.position *= 0.5
            self.total_cost += self.position * price * (self.commission + self.slippage)
            self.trade_count += 1
        elif action == 3 and self.position > 0:
            sell_price = price * (1 - self.slippage)
            pnl_pct = (sell_price - self.entry_price) / self.entry_price * 100
            if sell_price > self.entry_price:
                self.win_count += 1
            
            # 记录交易
            self.trade_log.append({
                'entry_idx': self.entry_idx,
                'exit_idx': self.idx,
                'entry_price': self.entry_price,
                'exit_price': sell_price,
                'pnl_pct': pnl_pct
            })
            
            self.balance += self.position * sell_price * (1 - self.commission)
            self.total_cost += self.position * price * (self.commission + self.slippage)
            self.position = 0
            self.entry_price = 0
            self.trade_count += 1
        
        self.idx += 1
        price = self.data[min(self.idx, len(self.data)-1), 0]
        self.total_profit = self.balance + self.position * price
        self.peak_profit = max(self.peak_profit, self.total_profit)
        
        # 记录资金曲线
        self.equity_curve.append(self.total_profit)
        
        reward = (self.total_profit - 10000) / 10000 - self.total_cost / 10000 * 0.1
        done = self.idx >= len(self.data) - 1
        
        return self._obs(), reward, done, False, {
            'total_profit': self.total_profit,
            'total_cost': self.total_cost,
            'trade_count': self.trade_count,
            'equity_curve': self.equity_curve
        }


def run_backtest(symbol, model_path, data):
    """运行回测"""
    model = PPO.load(model_path)
    env = TradingEnv(data)
    
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, _, done, _, info = env.step(action)
    
    # 计算统计
    equity = np.array(env.equity_curve)
    returns = np.diff(equity) / equity[:-1]
    
    # 最大回撤
    running_max = np.maximum.accumulate(equity)
    drawdown = (running_max - equity) / running_max
    max_dd = np.max(drawdown) * 100
    
    # 夏普比率
    sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(365)
    
    # 胜率
    win_rate = env.win_count / max(env.trade_count, 1) * 100
    
    # 收益
    total_return = (equity[-1] / equity[0] - 1) * 100
    bh_return = (data[-1, 3] / data[60, 3] - 1) * 100
    
    return {
        'symbol': symbol,
        'equity_curve': equity,
        'total_return': total_return,
        'bh_return': bh_return,
        'excess': total_return - bh_return,
        'max_drawdown': max_dd,
        'sharpe': sharpe,
        'trade_count': env.trade_count,
        'win_rate': win_rate,
        'total_cost': env.total_cost,
        'trade_log': env.trade_log,
        'days': len(data)
    }


def plot_equity_curves(results, save_path):
    """绘制资金曲线图"""
    plt.figure(figsize=(16, 10))
    
    # 子图1: 所有币种资金曲线
    plt.subplot(2, 2, 1)
    for r in results:
        plt.plot(r['equity_curve'], label=f"{r['symbol']} ({r['total_return']:+.1f}%)")
    plt.axhline(y=10000, color='gray', linestyle='--', alpha=0.5)
    plt.title('Equity Curves - All Coins', fontsize=14)
    plt.xlabel('Trading Days')
    plt.ylabel('Portfolio Value ($)')
    plt.legend(loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # 子图2: 平均资金曲线
    plt.subplot(2, 2, 2)
    min_len = min(len(r['equity_curve']) for r in results)
    avg_equity = np.mean([r['equity_curve'][:min_len] for r in results], axis=0)
    plt.plot(avg_equity, 'b-', linewidth=2, label=f"Average ({(avg_equity[-1]/10000-1)*100:+.1f}%)")
    plt.axhline(y=10000, color='gray', linestyle='--', alpha=0.5)
    plt.title('Average Equity Curve', fontsize=14)
    plt.xlabel('Trading Days')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图3: 收益对比柱状图
    plt.subplot(2, 2, 3)
    symbols = [r['symbol'] for r in results]
    rl_returns = [r['total_return'] for r in results]
    bh_returns = [r['bh_return'] for r in results]
    x = np.arange(len(symbols))
    width = 0.35
    plt.bar(x - width/2, rl_returns, width, label='RL Strategy', color='green', alpha=0.7)
    plt.bar(x + width/2, bh_returns, width, label='Buy & Hold', color='blue', alpha=0.7)
    plt.xticks(x, symbols, rotation=45, ha='right')
    plt.title('Returns Comparison', fontsize=14)
    plt.ylabel('Return (%)')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    # 子图4: 风险指标
    plt.subplot(2, 2, 4)
    metrics = ['Excess Return (%)', 'Max Drawdown (%)', 'Sharpe Ratio', 'Win Rate (%)']
    avg_values = [
        np.mean([r['excess'] for r in results]),
        np.mean([r['max_drawdown'] for r in results]),
        np.mean([r['sharpe'] for r in results]) * 10,  # 放大
        np.mean([r['win_rate'] for r in results])
    ]
    colors = ['green', 'red', 'blue', 'purple']
    plt.bar(metrics, avg_values, color=colors, alpha=0.7)
    plt.title('Average Risk Metrics', fontsize=14)
    plt.ylabel('Value')
    plt.xticks(rotation=15, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"图表已保存: {save_path}")


if __name__ == "__main__":
    print("="*60)
    print("v7模型 - 多年数据回测")
    print("="*60)
    
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT']
    
    print("\n[1] 获取历史数据...")
    all_data = {}
    for sym in symbols:
        print(f"\n{sym}:")
        data = get_all_data(sym, start_year=2021)
        if data is not None and len(data) > 0:
            all_data[sym] = data
            years = len(data) / 365
            bh = (data[-1, 3] / data[0, 3] - 1) * 100
            print(f"  总计: {len(data)}天 (~{years:.1f}年), BH: {bh:+.1f}%")
    
    print("\n[2] 回测中...")
    results = []
    
    for sym, data in all_data.items():
        model_path = f"ppo_v7_{sym}"
        try:
            r = run_backtest(sym, model_path, data)
            results.append(r)
            print(f"  {sym}: RL {r['total_return']:+.1f}% BH {r['bh_return']:+.1f}% 超额 {r['excess']:+.1f}% 回撤 {r['max_drawdown']:.1f}% 夏普 {r['sharpe']:.2f}")
        except Exception as e:
            print(f"  {sym}: 回测失败 - {e}")
    
    # 绘制资金曲线
    print("\n[3] 生成资金曲线图...")
    plot_equity_curves(results, "backtest_5year_equity_curves.png")
    
    # 生成详细报告
    print("\n" + "="*60)
    print("回测详细报告")
    print("="*60)
    
    avg_return = np.mean([r['total_return'] for r in results])
    avg_bh = np.mean([r['bh_return'] for r in results])
    avg_excess = np.mean([r['excess'] for r in results])
    avg_dd = np.mean([r['max_drawdown'] for r in results])
    avg_sharpe = np.mean([r['sharpe'] for r in results])
    avg_win = np.mean([r['win_rate'] for r in results])
    avg_trades = np.mean([r['trade_count'] for r in results])
    avg_cost = np.mean([r['total_cost'] for r in results])
    
    print(f"\n数据范围: 2021-2026 (~{np.mean([r['days'] for r in results]):.0f}天)")
    print(f"测试币种: {len(results)}个")
    
    print("\n各币种详细结果:")
    print("-" * 80)
    print(f"{'币种':<10} {'RL收益':>10} {'BH收益':>10} {'超额':>10} {'最大回撤':>10} {'夏普':>8} {'胜率':>8} {'交易次数':>8}")
    print("-" * 80)
    for r in results:
        print(f"{r['symbol']:<10} {r['total_return']:>+10.1f}% {r['bh_return']:>+10.1f}% {r['excess']:>+10.1f}% {r['max_drawdown']:>10.1f}% {r['sharpe']:>8.2f} {r['win_rate']:>7.0f}% {r['trade_count']:>8}")
    print("-" * 80)
    print(f"{'平均':<10} {avg_return:>+10.1f}% {avg_bh:>+10.1f}% {avg_excess:>+10.1f}% {avg_dd:>10.1f}% {avg_sharpe:>8.2f} {avg_win:>7.0f}% {avg_trades:>8.0f}")
    
    # 保存报告
    report = f"""
============================================
v7模型多年回测报告
============================================

回测周期: 2021-2026 (~{np.mean([r['days'] for r in results]):.0f}天)
策略版本: v7 (风控增强版)
初始资金: $10,000

--------------------------------------------
汇总指标
--------------------------------------------
平均RL收益: {avg_return:+.1f}%
平均BH收益: {avg_bh:+.1f}%
平均超额收益: {avg_excess:+.1f}%
平均最大回撤: {avg_dd:.1f}%
平均夏普比率: {avg_sharpe:.2f}
平均胜率: {avg_win:.0f}%
平均交易次数: {avg_trades:.0f}
平均交易成本: ${avg_cost:.0f}

--------------------------------------------
各币种详情
--------------------------------------------
"""
    for r in results:
        report += f"""
{r['symbol']}:
  - RL收益: {r['total_return']:+.1f}%
  - BH收益: {r['bh_return']:+.1f}%  
  - 超额收益: {r['excess']:+.1f}%
  - 最大回撤: {r['max_drawdown']:.1f}%
  - 夏普比率: {r['sharpe']:.2f}
  - 胜率: {r['win_rate']:.0f}%
  - 交易次数: {r['trade_count']}
  - 交易成本: ${r['total_cost']:.0f}
"""
    
    report += f"""
--------------------------------------------
风险提示
--------------------------------------------
1. 历史表现不代表未来收益
2. 实盘需考虑滑点和手续费
3. 极端行情可能导致超预期亏损
4. 建议小资金实盘验证后再扩大规模

报告生成时间: 2026-03-07
"""
    
    with open("backtest_5year_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    
    print("\n报告已保存: backtest_5year_report.txt")
    print("资金曲线图已保存: backtest_5year_equity_curves.png")

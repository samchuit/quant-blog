# 量化交易完整指南

> 更新时间: 2026-03-08
> 整理: Jarvis 🤖

---

# 第一部分: 策略归因分析

## 1. 收益归因

### 1.1 收益分解

```python
def returns_attribution(portfolio_returns, factors):
    """
    收益归因分析
    
    收益 = Alpha + Beta1*因子1 + Beta2*因子2 + ...
    """
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LinearRegression
    
    # 准备数据
    X = factors.values  # 因子收益率
    y = portfolio_returns.values  # 组合收益率
    
    # 回归
    model = LinearRegression()
    model.fit(X, y)
    
    # 结果
    attribution = pd.DataFrame({
        'factor': factors.columns,
        'beta': model.coef_,
        'contribution': model.coef_ * factors.mean()
    })
    
    # Alpha
    alpha = model.intercept_
    
    return {
        'alpha': alpha,
        'r_squared': model.score(X, y),
        'attribution': attribution
    }
```

### 1. Brinson 归因

```python
def brinson_attribution(portfolio_weights, benchmark_weights, sector_returns):
    """
    Brinson 归因
    分解行业配置和选股收益
    """
    # 权重差异
    weight_diff = portfolio_weights - benchmark_weights
    
    # 行业收益差异
    sector_allocation = weight_diff * sector_returns
    
    # 选择效应 (简化的)
    stock_selection = portfolio_weights * (portfolio_returns - sector_returns)
    
    # 交互效应
    interaction = weight_diff * (portfolio_returns - sector_returns)
    
    return {
        'allocation_effect': sector_allocation.sum(),
        'selection_effect': stock_selection.sum(),
        'interaction_effect': interaction.sum(),
        'total_active_return': sector_allocation.sum() + stock_selection.sum()
    }
```

---

## 2. 风格分析

### 2.1 Barra 风格因子

```python
def style_analysis(portfolio_returns, style_factors):
    """
    风格分析
    """
    from sklearn.linear_model import Ridge
    
    # 风格因子: 市值/价值/成长/动量/波动率
    X = style_factors.values
    y = portfolio_returns.values
    
    model = Ridge(alpha=1.0)
    model.fit(X, y)
    
    styles = pd.DataFrame({
        'style': style_factors.columns,
        'exposure': model.coef_
    })
    
    return styles
```

### 2.2 滚动风格

```python
def rolling_style(returns, factors, window=252):
    """
    滚动风格分析
    """
    styles_list = []
    
    for i in range(window, len(returns)):
        y = returns.iloc[i-window:i]
        X = factors.iloc[i-window:i]
        
        model = Ridge(alpha=1.0)
        model.fit(X, y)
        
        styles_list.append({
            'date': returns.index[i],
            **dict(zip(factors.columns, model.coef_))
        })
    
    return pd.DataFrame(styles_list).set_index('date')
```

---

# 第二部分: 基准对比

## 3. 基准分析

### 3.1 对比指标

```python
def benchmark_comparison(portfolio_returns, benchmark_returns):
    """
    基准对比分析
    """
    import numpy as np
    
    # 超额收益
    excess_returns = portfolio_returns - benchmark_returns
    
    # 信息比率 (超额收益/跟踪误差)
    tracking_error = excess_returns.std() * np.sqrt(252)
    info_ratio = excess_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0
    
    # Alpha
    beta = np.cov(portfolio_returns, benchmark_returns)[0,1] / np.var(benchmark_returns)
    alpha = (portfolio_returns.mean() - beta * benchmark_returns.mean()) * 252
    
    # R²
    correlation = np.corrcoef(portfolio_returns, benchmark_returns)[0,1]
    
    # 最大回撤对比
    def max_drawdown(returns):
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        return (cumulative - running_max) / running_max
    
    return {
        'alpha': alpha,
        'beta': beta,
        'r_squared': correlation ** 2,
        'tracking_error': tracking_error,
        'info_ratio': info_ratio,
        'correlation': correlation,
        'portfolio_max_dd': max_drawdown(portfolio_returns).min(),
        'benchmark_max_dd': max_drawdown(benchmark_returns).min()
    }
```

### 3.2 常用基准

```python
BENCHMARKS = {
    'BTC': 'BTC/USDT',      # 比特币
    'ETH': 'ETH/USDT',      # 以太坊
    'SP500': '^GSPC',       # 标普500
    'NASD': '^IXIC',        # 纳斯达克
    'DOW': '^DJI',          # 道琼斯
    'ALLSI': '000300.XSHG',  # 沪深300
    'CSI500': '000905.XSHG', # 中证500
}
```

---

## 4. 资金曲线管理

### 4.1 动态仓位

```python
class DynamicPosition:
    def __init__(self, initial_capital, max_position=1.0):
        self.capital = initial_capital
        self.max_position = max_position
        self.peak = initial_capital
    
    def calculate_position(self, current_capital, signal_strength=1.0):
        """
        根据资金曲线调整仓位
        """
        # 当前回撤
        drawdown = (self.peak - current_capital) / self.peak
        
        # 回撤越大, 仓位越小
        if drawdown < 0.05:
            position_pct = self.max_position
        elif drawdown < 0.10:
            position_pct = self.max_position * 0.8
        elif drawdown < 0.15:
            position_pct = self.max_position * 0.6
        elif drawdown < 0.20:
            position_pct = self.max_position * 0.4
        else:
            position_pct = self.max_position * 0.2
        
        # 乘以信号强度
        position_pct *= signal_strength
        
        # 更新峰值
        if current_capital > self.peak:
            self.peak = current_capital
        
        return position_pct
```

### 4.2 加仓减仓

```python
def pyramid_position(entry_price, current_price, base_size, max_ladder=3):
    """
    金字塔加仓
    价格越低, 仓位越大
    """
    ladders = []
    
    for i in range(max_ladder):
        # 每次加仓间隔
        step = entry_price * 0.02 * (i + 1)  # 2%间隔
        
        if current_price <= entry_price - step:
            ladder_size = base_size * (1 + i * 0.5)  # 递增50%
            ladders.append({
                'level': i + 1,
                'price': entry_price - step,
                'size': ladder_size
            })
    
    return ladders
```

### 4.3 回撤控制

```python
class DrawdownController:
    def __init__(self, max_drawdown=0.20):
        self.max_drawdown = max_drawdown
        self.peak = 0
    
    def should_stop(self, current_value):
        """
        是否应该停止交易
        """
        if current_value > self.peak:
            self.peak = current_value
        
        drawdown = (self.peak - current_value) / self.peak
        
        if drawdown >= self.max_drawdown:
            return True, drawdown
        
        return False, drawdown
    
    def reduce_exposure(self, drawdown):
        """
        根据回撤减少仓位
        """
        if drawdown < 0.05:
            return 1.0
        elif drawdown < 0.10:
            return 0.75
        elif drawdown < 0.15:
            return 0.50
        else:
            return 0.25
```

---

# 第三部分: 策略生命周期

## 5. 开发流程

### 5.1 策略开发阶段

```
阶段1: 想法形成 (1-2周)
  - 市场观察
  - 初步假设
  - 文献调研

阶段2: 原型开发 (2-4周)
  - 核心逻辑实现
  - 简单回测
  - 参数敏感性分析

阶段3: 完善优化 (4-8周)
  - 特征工程
  - 参数优化
  - 组合优化

阶段4: 模拟测试 (4-12周)
  - 样本外测试
  - 模拟盘运行
  - 稳定性验证

阶段5: 实盘运行 (持续)
  - 小资金实盘
  - 监控优化
  - 定期评估
```

### 5.2 策略评估清单

```python
STRATEGY_CHECKLIST = {
    '回测指标': [
        '年化收益 > 15%',
        '夏普比率 > 1.0',
        '最大回撤 < 20%',
        '胜率 > 45%',
        '盈亏比 > 1.5'
    ],
    '稳健性': [
        '样本外收益 > 样本内收益 * 0.7',
        '多周期有效',
        '多品种有效',
        '参数不敏感'
    ],
    '实盘可行性': [
        '交易频率合理',
        '手续费占比 < 10%',
        '滑点可控',
        '容量足够'
    ]
}
```

---

## 6. 模拟盘与实盘

### 6.1 模拟盘监控

```python
class PaperTradingMonitor:
    def __init__(self):
        self.trades = []
        self.equity_curve = []
    
    def log_trade(self, signal, execution, timestamp):
        self.trades.append({
            'signal': signal,
            'execution': execution,
            'timestamp': timestamp,
            'slippage': abs(execution - signal.price) / signal.price
        })
    
    def calculate_metrics(self):
        import pandas as pd
        
        df = pd.DataFrame(self.trades)
        
        return {
            'total_trades': len(df),
            'avg_slippage': df['slippage'].mean() if len(df) > 0 else 0,
            'execution_delay': (df['execution.time'] - df['signal.time']).mean()
        }
```

### 6.2 实盘监控

```python
class LiveTradingMonitor:
    def __init__(self, alerts):
        self.alerts = alerts
    
    def check_positions(self, positions, limits):
        """检查仓位限制"""
        for symbol, pos in positions.items():
            if pos > limits.get(symbol, 0):
                self.alerts.send(f'警告: {symbol} 仓位超限!')
    
    def check_daily_loss(self, daily_pnl, limit=-0.03):
        """检查日亏损"""
        if daily_pnl < limit:
            self.alerts.send(f'警告: 日亏损 {daily_pnl*100}%, 达到限制!')
            return False
        return True
    
    def weekly_report(self, weekly_stats):
        """周报"""
        return f"""
        周收益: {weekly_stats['return']*100:.2f}%
        交易次数: {weekly_stats['trades']}
        胜率: {weekly_stats['win_rate']*100:.1f}%
        最大回撤: {weekly_stats['max_drawdown']*100:.2f}%
        """
```

---

# 第四部分: 多时间框架

## 7. 多时间框架策略

### 7.1 框架设计

```python
class MultiTimeFrame:
    def __init__(self):
        self.timeframes = {
            'daily': {'data': None, 'signal': None},
            '4h': {'data': None, 'signal': None},
            '1h': {'data': None, 'signal': None},
            '15m': {'data': None, 'signal': None}
        }
    
    def update_data(self, timeframe, data):
        self.timeframes[timeframe]['data'] = data
        self.timeframes[timeframe]['signal'] = self.generate_signal(data)
    
    def generate_signal(self, data):
        """生成信号"""
        # 实现具体策略
        return 'buy' if data['trend'] > 0 else 'sell'
    
    def combined_signal(self):
        """
        多时间框架信号确认
        
        规则:
        - 大周期确认趋势
        - 小周期选择入场
        """
        # 大周期 (日线) 确认方向
        daily = self.timeframes['daily']['signal']
        h4 = self.timeframes['4h']['signal']
        
        # 方向一致才交易
        if daily == h4:
            # 小周期寻找入场点
            h1 = self.timeframes['1h']['signal']
            
            if h1 == daily:
                return daily  # 顺势交易
        
        return 'hold'
```

### 7.2 实战示例

```python
def multi_tf_strategy():
    """
    多时间框架策略示例
    
    日线: 确认趋势 (MA200)
    4小时: 过滤噪音
    1小时: 入场时机
    """
    
    # 日线: 趋势判断
    daily_ma200 = df_daily['close'].rolling(200).mean()
    trend_up = df_daily['close'].iloc[-1] > daily_ma200.iloc[-1]
    
    # 4小时: 均线多头
    h4_ma20 = df_h4['close'].rolling(20).mean()
    h4_trend = df_h4['close'].iloc[-1] > h4_ma20.iloc[-1]
    
    # 1小时: RSI超卖
    h1_rsi = calculate_rsi(df_h1)
    oversold = h1_rsi < 30
    
    # 入场条件
    if trend_up and h4_trend and oversold:
        return 'buy'
    
    return 'hold'
```

---

# 第五部分: 统计套利

## 8. 配对交易

### 8.1 协整检验

```python
def cointegration_test(series1, series2):
    """
    协整性检验
    """
    from statsmodels.tsa.stattools import coint
    
    score, pvalue, _ = coint(series1, series2)
    
    return {
        'score': score,
        'pvalue': pvalue,
        'cointegrated': pvalue < 0.05
    }
```

### 8.2 配对交易策略

```python
class PairsTrading:
    def __init__(self, lookback=60, entry_threshold=2, exit_threshold=0.5):
        self.lookback = lookback
        self.entry_threshold = entry_threshold  # z-score入阈值
        self.exit_threshold = exit_threshold    # z-score出阈值
    
    def calculate_spread(self, price1, price2, hedge_ratio):
        """计算价差"""
        return price1 - hedge_ratio * price2
    
    def calculate_zscore(self, spread, lookback):
        """计算z分数"""
        mean = spread.rolling(lookback).mean()
        std = spread.rolling(lookback).std()
        return (spread - mean) / std
    
    def generate_signals(self, price1, price2):
        """生成信号"""
        # 计算对冲比率
        hedge_ratio = price1.rolling(20).mean() / price2.rolling(20).mean()
        
        # 计算价差
        spread = self.calculate_spread(price1, price2, hedge_ratio)
        
        # 计算z分数
        zscore = self.calculate_zscore(spread, self.lookback)
        
        current_z = zscore.iloc[-1]
        
        # 交易信号
        if current_z > self.entry_threshold:
            return 'short_spread'  # 做空价差 (预期回归)
        elif current_z < -self.entry_threshold:
            return 'long_spread'   # 做多价差
        elif abs(current_z) < self.exit_threshold:
            return 'close'          # 平仓
        
        return 'hold'
```

---

## 9. 因子套利

### 9.1因子有效性套利

```python
def factor_arbitrage(df, factor_name, long_top=20, short_bottom=20):
    """
    因子套利
    做多因子强的, 做空因子弱的
    """
    # 因子排序
    df['rank'] = df[factor_name].rank()
    
    # 做多top, 做空bottom
    n = len(df)
    
    long_signals = df[df['rank'] > n - long_top].index
    short_signals = df[df['rank'] < short_bottom].index
    
    return {
        'long': long_signals,
        'short': short_signals
    }
```

---

# 第六部分: 量化平台

## 10. 聚宽平台

### 10.1 基础使用

```python
# 聚宽 API (需要注册获取token)
# import jqdata

# 获取数据
# stocks = get_all_securities('stock')
# price = get_price('000001.XSHE', start_date='2020-01-01', end_date='2024-12-31')

# 因子计算
# df['ma20'] = price['close'].rolling(20).mean()

# 回测
# def initialize(context):
#     context.stock = '000001.XSHE'
#
# def handle_data(context, data):
#     if data[context.stock].close < context.portfolio.positions[context.stock].avg_cost * 0.95:
#         order_target(context.stock, 0)
```

### 10.2 因子研究

```python
# 因子分析
# from jqf import Alpha

# alpha = Alpha('pe_ratio')
# results = alpha.analyze_factor('pe_ratio', start_date='2020-01-01')
# print(results.ic_mean, results.rank_ic_mean)
```

---

## 11. Backtrader 进阶

### 11.1 完整策略模板

```python
import backtrader as bt

class CompleteStrategy(bt.Strategy):
    params = (
        ('maperiod', 20),
        ('printlog', False),
    )
    
    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None
        self.buyprice = None
        self.buycomm = None
        
        # 指标
        self.sma = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.maperiod)
        
        # 交叉信号
        self.crossover = bt.indicators.CrossOver(
            self.datas[0], self.sma)
    
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            elif order.issell():
                pass
        
        self.order = None
    
    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        if self.params.printlog:
            print(f'收益: {trade.pnl:.2f}, 佣金: {trade.pnlcomm:.2f}')
    
    def next(self):
        if self.order:
            return
        
        # 买入信号
        if not self.position:
            if self.crossover > 0:
                self.order = self.buy()
        # 卖出信号
        elif self.crossover < 0:
            self.order = self.sell()
    
    def stop(self):
        if self.params.printlog:
            print(f'({self.params.maperiod}) 结束: {self.datas[0].close[0]:.2f}')
```

---

# 第七部分: 量化面试

## 12. 常见面试题

### 12.1 编程题

```python
# 1. 反转链表
def reverse_linked_list(head):
    prev = None
    current = head
    
    while current:
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node
    
    return prev

# 2. 快速排序
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quick_sort(left) + middle + quick_sort(right)

# 3. 最大子数组和
def max_subarray(nums):
    max_sum = nums[0]
    current_sum = nums[0]
    
    for i in range(1, len(nums)):
        current_sum = max(nums[i], current_sum + nums[i])
        max_sum = max(max_sum, current_sum)
    
    return max_sum
```

### 12.2 策略题

```
1. 什么是动量效应? 如何利用?
2. 什么是均值回归? 何时有效?
3. 解释夏普比率和卡玛比率
4. 过拟合怎么办?
5. 如何处理幸存者偏差?
6. 描述你最喜欢的策略
7. 如何确定样本外测试的可靠性?
8. 什么是套利机会? 举例
9. 解释期权 Greeks
10. VaR 和 CVaR 的区别
```

### 12.3 概率题

```python
# 硬币游戏: 连续正面则继续, 出现反面停止
# 期望收益?

# 答案: 2
# E = 0.5 * 1 + 0.5 * (1 + E)
# 0.5E = 1
# E = 2
```

---

# 第八部分: 策略文档

## 13. 策略文档模板

```markdown
# 策略名称

## 1. 策略概述
- 策略类型: 
- 目标市场:
- 预期收益:
- 风险等级:

## 2. 核心逻辑
```
描述策略的核心逻辑
```

## 3. 策略参数
| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| period | 20 | 10-50 | 均线周期 |

## 4. 风险特征
- 最大回撤: XX%
- 波动率: XX%
- 胜率: XX%
- 盈亏比: XX

## 5. 适用市场
- 牛市:
- 熊市:
- 震荡:

## 6. 局限性
- 滑点影响
- 容量限制
- 失效条件

## 7. 改进方向
- 
```

---

# 第九部分: 监管合规

## 14. 量化合规

### 14.1 主要监管机构

```
中国: CSRC (证监会), AMAC (基金业协会)
美国: SEC, FINRA, CFTC
欧盟: ESMA, MiFID II
```

### 14.2 合规要点

```python
COMPLIANCE_CHECKLIST = {
    '信息披露': [
        '策略逻辑说明',
        '风险提示',
        '历史业绩真实'
    ],
    '风控要求': [
        '止损机制',
        '仓位限制',
        '日亏损限制'
    ],
    '禁止行为': [
        '操纵市场',
        '内幕交易',
        '虚假宣传'
    ]
}
```

---

*持续更新中...*

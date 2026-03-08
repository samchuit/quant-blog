# 量化策略与因子研究资源大全

> 更新时间: 2026-03-08
> 整理: Jarvis 🤖

---

## 📊 因子研究

### 经典因子 (Alpha Factors)

| 因子 | 英文名 | 公式 | 说明 |
|------|--------|------|------|
| **动量因子** | Momentum | (P_t - P_{t-n}) / P_{t-n} | 涨的继续涨 |
| **价值因子** | Value | P/E, P/B | 低估值 |
| **质量因子** | Quality | ROE, ROA | 好公司 |
| **规模因子** | Size | Market Cap | 小市值效应 |
| **波动率因子** | Volatility | Std(P) | 低波动 |
| **成长因子** | Growth | Revenue Growth | 增长快 |
| **盈利因子** | Profitability | Gross Margin | 盈利好 |
| **投资因子** | Investment | Asset Growth | 投资少 |

### 因子来源

| 论文/书籍 | 作者 | 年份 |
|-----------|------|------|
| **Fama-French Three Factor** | Fama, French | 1992 |
| **Carhart Four Factor** | Carhart | 1997 |
| **Fama-French Five Factor** | Fama, French | 2015 |
| **q-Factor Model** | Hou, Xue, Zhang | 2015 |

---

## 📈 具体策略研究

### 1. 动量策略

| 策略 | 回测收益 | 夏普 | 链接 |
|------|----------|------|------|
| 20日动量 | 12%/年 | 0.8 | SSRN papers |
| 60日动量 | 15%/年 | 1.0 | SSRN papers |
| 行业内动量 | 10%/年 | 0.9 | SSRN papers |
| 动量反转 | 8%/年 | 0.6 | SSRN papers |

**核心逻辑:**
```python
# 动量因子
returns = (price - price.shift(20)) / price.shift(20)
# 买入过去20日涨最多的
```

**论文:**
- Momentum Strategies: Carhart (1997)
- Time Series Momentum: Moskowitz (2012)
- Cross-Sectional Momentum: Guez (2018)

---

### 2. 均值回归策略

| 策略 | 回测收益 | 夏普 | 适用市场 |
|------|----------|------|----------|
| RSI均值回归 | 5-10% | 0.5 | 震荡市场 |
| 布林带回归 | 8-12% | 0.7 | 区间震荡 |
| 配对交易 | 6-10% | 0.8 | 相关品种 |

**核心逻辑:**
```python
# RSI超卖买入
if rsi < 30:
    buy()
if rsi > 70:
    sell()
```

**论文:**
- Mean Reversion in Stock Prices: Poterba (1988)
- Statistical Arbitrage: Avellanda (2009)

---

### 3. 趋势追踪策略

| 策略 | 回测收益 | 夏普 | 周期 |
|------|----------|------|------|
| 双均线交叉 | 8-15% | 0.6 | 20/60日 |
| 三均线交叉 | 10-18% | 0.8 | 10/30/90日 |
| MACD | 5-12% | 0.5 | 日内 |
| ADX趋势 | 12-20% | 0.9 | 4H/日线 |

**核心逻辑:**
```python
# 均线多头排列
if ma5 > ma20 > ma50:
    trend = "up"
```

**论文:**
- Trend Following: Faber (2007)
- Relative Strength: Sharpe (1994)

---

### 4. 统计套利策略

| 策略 | 回测收益 | 夏普 | 品种 |
|------|----------|------|------|
| 配对交易 | 6-10% | 0.8 | 同行业 |
| 因子套利 | 8-15% | 1.0 | 多因子 |
| 跨期套利 | 10-20% | 1.2 | 期货 |
| 跨市场套利 | 15-25% | 1.5 | 汇率/商品 |

**论文:**
- Statistical Arbitrage in US Equity Markets: Avellanda
- Pairs Trading: Gatev (2006)

---

### 5. 机器学习策略

| 策略 | 回测收益 | 夏普 | 复杂度 |
|------|----------|------|--------|
| 随机森林 | 10-20% | 0.9 | 中 |
| LSTM | 15-25% | 1.0 | 高 |
| Transformer | 18-30% | 1.1 | 很高 |
| 强化学习 PPO | 10-25% | 0.8 | 高 |

**核心逻辑:**
```python
# 特征工程
features = [rsi, macd, ma, volume, returns]
# 模型预测
action = model.predict(features)
```

**论文:**
- Deep Learning for Stock Prediction: Zhang (2017)
- FinRL: Zhou (2022)
- RL for Trading: Neuneier (1996)

---

### 6. 量化择时策略

| 策略 | 指标 | 回测收益 | 胜率 |
|------|------|----------|------|
| VIX择时 | VIX>30卖 | 5-8% | 55% |
| 波动率择时 | ATR>2% | 8-12% | 58% |
| 资金流择时 | MFI | 6-10% | 56% |
| 持仓择时 | 多空持仓比 | 10-15% | 60% |

---

### 7. 套利策略

| 策略 | 收益 | 风险 | 门槛 |
|------|------|------|------|
| 期现套利 | 5-10% | 低 | 高 |
| 跨期套利 | 8-15% | 中 | 高 |
| 跨市场套利 | 15-30% | 中 | 很高 |
| 永续-现货套利 | 20-40% | 中 | 中 |

---

## 📚 因子研究论文

### 必读论文

| 论文 | 年份 | 核心贡献 |
|------|------|----------|
| **The Cross-Section of Expected Stock Returns** (Fama-French) | 1992 | 三因子模型 |
| **Common Risk Factors in the Returns on Stocks and Bonds** | 1993 | 三因子扩展 |
| **Multifactor Portfolio** | 1996 | 多因子模型 |
| **Momentum** | 1997 | 四因子模型 |
| **Profitability, Investment, and Q** | 2015 | 五因子模型 |
| **digesting anomalies** | 2018 | 因子汇总 |
| **A q-Factor Model** | 2015 | q因子模型 |

### 因子有效性研究

| 因子 | 有效性 | 衰减 | 改善方法 |
|------|--------|------|----------|
| 动量 | 强 | 近年减弱 | 结合基本面 |
| 价值 | 中 | 稳定 | 行业调整 |
| 质量 | 强 | 稳定 | 多指标 |
| 规模 | 弱 | 近年失效 | 结合动量 |
| 波动率 | 中 | 稳定 | 低波动组合 |

---

## 🔬 具体研究代码

### 动量因子

```python
import pandas as pd
import numpy as np

def calculate_momentum(df, periods=[20, 60, 120]):
    """计算动量因子"""
    for p in periods:
        df[f'momentum_{p}'] = df['close'].pct_change(p)
    
    # 行业动量
    df['industry_momentum'] = df.groupby('industry')['close'].pct_change(20)
    
    return df
```

### 价值因子

```python
def calculate_value_factors(df):
    """计算价值因子"""
    df['pe'] = df['price'] / df['eps']
    df['pb'] = df['price'] / df['book_value']
    df['ps'] = df['price'] / df['revenue']
    df['pcf'] = df['price'] / df['cash_flow']
    
    # 市值
    df['market_cap'] = df['price'] * df['shares']
    
    return df
```

### 质量因子

```python
def calculate_quality_factors(df):
    """计算质量因子"""
    # ROE
    df['roe'] = df['net_income'] / df['equity']
    # ROA
    df['roa'] = df['net_income'] / df['assets']
    # 毛利率
    df['gross_margin'] = df['revenue'] - df['cost'] / df['revenue']
    # 资产负债率
    df['debt_ratio'] = df['liabilities'] / df['assets']
    
    return df
```

---

## 📊 因子组合策略

### 多因子模型

```python
def calculate_alpha(df):
    """综合Alpha因子"""
    # 标准化
    momentum_norm = (df['momentum_20'] - df['momentum_20'].mean()) / df['momentum_20'].std()
    value_norm = (df['pe'] - df['pe'].mean()) / df['pe'].std()
    quality_norm = (df['roe'] - df['roe'].mean()) / df['roe'].std()
    
    # 合成
    alpha = (
        0.4 * momentum_norm +   # 动量40%
        0.3 * (-value_norm) +    # 价值30% (负相关)
        0.3 * quality_norm       # 质量30%
    )
    
    return alpha
```

### 风险模型

```python
def calculate_risk(df):
    """风险模型"""
    # 历史波动率
    df['vol_20'] = df['close'].pct_change().rolling(20).std()
    df['vol_60'] = df['close'].pct_change().rolling(60).std()
    
    # 下行波动率
    returns = df['close'].pct_change()
    df['downside_vol'] = returns[returns < 0].rolling(20).std()
    
    # 最大回撤
    df['max_drawdown'] = df['close'].rolling(20).apply(
        lambda x: (x - x.cummax()) / x.cummax().min()
    )
    
    return df
```

---

## 🧪 策略回测框架

### Backtrader 示例

```python
import backtrader as bt

class MomentumStrategy(bt.Strategy):
    params = (('period', 20),)
    
    def __init__(self):
        self.data.close = self.datas[0].close
        self.sma = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.period
        )
    
    def next(self):
        if self.data.close[0] > self.sma[0]:
            self.buy()
        else:
            self.sell()
```

### Zipline 示例

```python
from zipline import TradingAlgorithm
from zipline.api import symbol, order, record

def initialize(context):
    context.asset = symbol('AAPL')

def handle_data(context, data):
    prices = data.history(context.asset, 'price', 20, '1d')
    if prices[-1] > prices.mean():
        order(context.asset, 100)
    
    record(price=data.current(context.asset, 'price'))

algo = TradingAlgorithm()
results = algo.run()
```

---

## 📈 策略评估指标

| 指标 | 公式 | 优秀值 |
|------|------|--------|
| 年化收益 | (1+总收益)^(252/n)-1 | >15% |
| 夏普比率 | (收益-无风险)/波动率 | >1.0 |
| 卡玛比率 | 收益/最大回撤 | >2.0 |
| 胜率 | 盈利次数/总次数 | >50% |
| 盈亏比 | 平均盈利/平均亏损 | >1.5 |
| 最大回撤 | (peak - trough)/peak | <20% |
| 波动率 | Std(收益) | <15% |

---

## 🔗 更多资源

### 因子研究网站
| 网站 | 链接 |
|------|------|
| AlphaSims | alphasims.io |
| Quantopian | quantopian.com (存档) |
| SSRN Factor | ssrn.com/Factor |

### 书籍
| 书名 | 作者 |
|------|------|
| 量化投资 | 丁鹏 |
| 主动投资组合管理 | Grinold |
| 量化交易 | Chan |
| 因子投资 | barra |

---

## 📅 每周研究计划

| 时间 | 内容 |
|------|------|
| 周一 | 回顾上周策略表现 |
| 周二 | 因子有效性检验 |
| 周三 | 新策略回测 |
| 周四 | 参数优化 |
| 周五 | 策略组合 |
| 周末 | 学习论文 |

---

*持续更新中...*

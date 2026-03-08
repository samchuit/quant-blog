# 量化交易高级专题

> 更新时间: 2026-03-08
> 整理: Jarvis 🤖

---

# 第一部分: 加密货币特有策略

## 1. 资金费率套利

### 1.1 永续合约基础

```python
class FundingArbitrage:
    def __init__(self, exchange):
        self.exchange = exchange
    
    def get_funding_rate(self, symbol):
        """获取资金费率"""
        funding = self.exchange.fetch_funding_rate(symbol)
        return {
            'rate': funding['fundingRate'],
            'next_funding': funding['nextFundingTime'],
            'predicted_rate': self.predict_funding(symbol)
        }
    
    def predict_funding(self, symbol):
        """
        预测资金费率方向
        基于: 持仓量/资金费率历史/现货溢价
        """
        # 简化的预测逻辑
        ohlcv = self.exchange.fetch_ohlcv(symbol, '1h', limit=24)
        
        # 简单判断: 最近价格涨 -> 资金费率可能为正
        recent = [x[4] for x in ohlcv[-8:]]  # 最近8小时
        older = [x[4] for x in ohlcv[-24:-8]]
        
        avg_recent = sum(recent) / len(recent)
        avg_older = sum(older) / len(older)
        
        # 上涨趋势 -> 资金费率正
        if avg_recent > avg_older * 1.01:
            return 'positive'
        elif avg_recent < avg_older * 0.99:
            return 'negative'
        return 'neutral'
```

### 1.2 资金费率套利策略

```python
def funding_arb_strategy(self, symbol, threshold=0.0001):
    """
    资金费率套利
    
    原理:
    - 资金费率为正: 做多合约, 做空现货 (收资金费)
    - 资金费率为负: 做空合约, 做多现货 (付资金费但获得补贴)
    
    收益 = 资金费 + 价差收益 - 手续费
    """
    funding = self.get_funding_rate(symbol)
    
    if funding['rate'] > threshold:
        # 正资金费率 -> 做多合约收钱, 做空现货对冲
        return {
            'action': 'long_contract_short_spot',
            'funding_rate': funding['rate'],
            'expected_daily_return': funding['rate'] * 3  # 每天3次计算
        }
    
    elif funding['rate'] < -threshold:
        # 负资金费率 -> 做空合约, 做多现货
        return {
            'action': 'short_contract_long_spot',
            'funding_rate': funding['rate'],
            'expected_daily_return': abs(funding['rate']) * 3
        }
    
    return {'action': 'hold'}
```

---

## 2. 现货-合约价差套利

### 2.1 期现价差策略

```python
class SpotFutureArbitrage:
    def __init__(self, exchange, symbols):
        self.exchange = exchange
        self.symbols = symbols
    
    def calculate_basis(self, symbol):
        """计算基差 (现货-合约)"""
        # 现货价格
        spot = self.exchange.fetch_ticker(symbol)['last']
        
        # 合约价格 (币安用 marked price)
        future = self.exchange.fetch_ticker(f"{symbol.replace('USDT', 'USDT_PERP')}")['last']
        
        basis = (spot - future) / future
        
        return {
            'spot': spot,
            'future': future,
            'basis': basis,
            'basis_pct': basis * 100
        }
    
    def trading_signal(self, symbol, lookback=20):
        """交易信号"""
        # 计算历史基差
        bases = []
        for i in range(lookback):
            # 简化实现
            basis_info = self.calculate_basis(symbol)
            bases.append(basis_info['basis'])
        
        mean_basis = sum(bases) / len(bases)
        std_basis = (sum((x - mean_basis) ** 2 for x in bases) / len(bases)) ** 0.5
        
        current = self.calculate_basis(symbol)
        
        # 基差过大时套利
        z_score = (current['basis'] - mean_basis) / (std_basis + 1e-8)
        
        if z_score > 2:
            # 基差高估 -> 买入现货, 卖出合约
            return 'short_basis'
        elif z_score < -2:
            # 基差低估 -> 卖出现货, 买入合约
            return 'long_basis'
        
        return 'hold'
```

---

## 3. 流动性挖矿

### 3.1 DEX 套利

```python
class DEXArbitrage:
    def __init__(self):
        self.uniswap = '0x7a250d5630B4cF539739dB2E8D43B1'
        self.sushiswap = '0xd9e1cE17f10c5dD4aB0'
    
    def find_arbitrage(self, token_in, token_out, amount):
        """
        寻找DEX套利机会
        
        原理: 不同DEX价格不同, 低买高卖
        """
        # 获取uniswap价格
        uniswap_price = self.get_price_uniswap(token_in, token_out, amount)
        
        # 获取sushiswap价格
        sushiswap_price = self.get_price_sushiswap(token_in, token_out, amount)
        
        if sushiswap_price < uniswap_price * 0.99:
            # sushiswap便宜, 买入sushiswap, 卖出uniswap
            return {
                'buy_dex': 'sushiswap',
                'sell_dex': 'uniswap',
                'profit': (uniswap_price - sushiswap_price) / sushiswap_price
            }
        elif uniswap_price < sushiswap_price * 0.99:
            return {
                'buy_dex': 'uniswap',
                'sell_dex': 'sushiswap',
                'profit': (sushiswap_price - uniswap_price) / uniswap_price
            }
        
        return None
    
    def get_price_uniswap(self, token_in, token_out, amount):
        """获取Uniswap价格 (简化)"""
        # 实际需要调用合约
        return 1.0
    
    def get_price_sushiswap(self, token_in, token_out, amount):
        """获取Sushiswap价格 (简化)"""
        return 1.0
```

---

# 第二部分: 进阶统计方法

## 4. GARCH 波动率模型

### 4.1 GARCH(1,1)

```python
from arch import arch_model
import numpy as np

def garch_forecast(returns):
    """
    GARCH(1,1) 波动率预测
    """
    # 拟合GARCH(1,1)模型
    model = arch_model(returns * 100, vol='Garch', p=1, q=1)
    result = model.fit(disp='off')
    
    # 预测下一期波动率
    forecast = result.forecast(horizon=1)
    predicted_vol = forecast.variance.iloc[-1].values[0] / 100
    
    return predicted_vol

def volatility_strategy(returns, threshold=0.02):
    """
    基于GARCH波动率的策略
    """
    # 计算当前波动率
    current_vol = returns.std()
    
    # 预测下一期波动率
    predicted_vol = garch_forecast(returns)
    
    # 波动率放大时增加仓位
    if predicted_vol > current_vol * 1.2:
        return 'increase_position'
    elif predicted_vol < current_vol * 0.8:
        return 'decrease_position'
    
    return 'hold'
```

### 4.2 多变量 GARCH

```python
def dcc_garch(returns_df):
    """
    DCC-GARCH: 动态条件相关
    用于多资产波动率
    """
    from arch import arch_model
    from arch.cov import DCC
    
    # 标准化收益
    std_returns = (returns_df - returns_df.mean()) / returns_df.std()
    
    # DCC模型
    dcc = DCC(std_returns, p=1, q=1)
    result = dcc.fit()
    
    # 条件相关矩阵
    correlation = result.conditional_correlation
    
    return correlation
```

---

## 5. 状态机模型

### 5.1 市场状态识别

```python
class MarketStateDetector:
    def __init__(self):
        self.states = ['bull', 'bear', 'sideways', 'volatile']
    
    def detect_state(self, prices, volumes):
        """
        识别市场状态
        """
        returns = prices.pct_change()
        
        # 趋势
        ma20 = prices.rolling(20).mean()
        ma60 = prices.rolling(60).mean()
        
        if ma20.iloc[-1] > ma60.iloc[-1] * 1.05:
            trend = 'up'
        elif ma20.iloc[-1] < ma60.iloc[-1] * 0.95:
            trend = 'down'
        else:
            trend = 'sideways'
        
        # 波动率
        vol = returns.rolling(20).std()
        avg_vol = returns.std()
        
        if vol.iloc[-1] > avg_vol * 1.5:
            volatility = 'high'
        elif vol.iloc[-1] < avg_vol * 0.7:
            volatility = 'low'
        else:
            volatility = 'normal'
        
        # 成交量
        vol_ratio = volumes.iloc[-1] / volumes.rolling(20).mean().iloc[-1]
        
        # 综合状态
        if trend == 'up' and volatility != 'high':
            return 'bull'
        elif trend == 'down' and volatility != 'high':
            return 'bear'
        elif volatility == 'high':
            return 'volatile'
        else:
            return 'sideways'
    
    def state_based_strategy(self, state):
        """
        根据状态选择策略
        """
        strategies = {
            'bull': {
                'position': 1.0,
                'strategy': 'momentum'
            },
            'bear': {
                'position': 0.2,
                'strategy': 'short_or_hold'
            },
            'sideways': {
                'position': 0.5,
                'strategy': 'mean_reversion'
            },
            'volatile': {
                'position': 0.3,
                'strategy': 'short_volatility'
            }
        }
        
        return strategies.get(state, {'position': 0.5, 'strategy': 'default'})
```

---

## 6. 贝叶斯方法

### 6.1 贝叶斯回归

```python
import pymc3 as pm
import numpy as np

def bayesian_regression(X, y):
    """
    贝叶斯回归
    得到参数的后验分布
    """
    with pm.Model() as model:
        # 先验
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta = pm.Normal('beta', mu=0, sigma=10, shape=X.shape[1])
        sigma = pm.HalfNormal('sigma', sigma=1)
        
        # 似然
        mu = alpha + pm.math.dot(X, beta)
        likelihood = pm.Normal('y', mu=mu, sigma=sigma, observed=y)
        
        # 采样
        trace = pm.sample(1000, tune=1000)
    
    return trace

def bayesian_update(prior_mean, prior_std, new_data):
    """
    贝叶斯更新
    """
    # 新数据的均值和标准差
    new_mean = np.mean(new_data)
    new_std = np.std(new_data)
    
    # 后验
    posterior_mean = (prior_mean / prior_std**2 + new_mean / new_std**2) / (1/prior_std**2 + 1/new_std**2)
    posterior_std = 1 / np.sqrt(1/prior_std**2 + 1/new_std**2)
    
    return posterior_mean, posterior_std
```

---

## 7. 协整与VAR

### 7.1 协整检验

```python
from statsmodels.tsa.stattools import coint, adfuller

def cointegration_test_matrix(price_data):
    """
    多个时间序列的协整矩阵
    """
    n = price_data.shape[1]
    cointegration_matrix = np.zeros((n, n))
    pvalue_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                score, pvalue, _ = coint(price_data.iloc[:, i], price_data.iloc[:, j])
                cointegration_matrix[i, j] = score
                pvalue_matrix[i, j] = pvalue
    
    return {
        'score': cointegration_matrix,
        'pvalue': pvalue_matrix,
        'significant': pvalue_matrix < 0.05
    }
```

### 7.2 VAR模型

```python
from statsmodels.tsa.api import VAR

def var_forecast(price_data, steps=5):
    """
    VAR 向量自回归
    用于多变量时间序列预测
    """
    # 拟合VAR
    model = VAR(price_data)
    result = model.fit(maxlags=15, ic='aic')
    
    # 预测
    forecast = result.forecast(price_data.values[-result.k_ar:], steps=steps)
    
    return forecast
```

---

# 第三部分: 交易执行

## 8. 订单类型选择

### 8.1 市价单 vs 限价单

```python
def order_type_choice(signal_strength, liquidity, spread):
    """
    选择订单类型
    
    参数:
    - signal_strength: 信号强度 (0-1)
    - liquidity: 流动性 (0-1)
    - spread: 价差 (小数)
    """
    # 流动性高 + 信号强 -> 市价单
    if liquidity > 0.7 and signal_strength > 0.7:
        return 'market'
    
    # 流动性低 + 信号弱 -> 限价单
    if liquidity < 0.3 or signal_strength < 0.3:
        return 'limit'
    
    # 价差大 -> 限价单
    if spread > 0.001:  # 0.1%
        return 'limit'
    
    # 默认 -> 市价单
    return 'market'
```

### 8.2 冰山订单

```python
class IcebergOrder:
    def __init__(self, side, total_size, slice_size, price):
        self.side = side  # 'buy' or 'sell'
        self.total_size = total_size
        self.slice_size = slice_size  # 每笔显示数量
        self.price = price
        self.remaining = total_size
        self.filled = 0
    
    def next_slice(self):
        """下一 slice"""
        if self.remaining <= 0:
            return None
        
        # 显示数量
        display_size = min(self.slice_size, self.remaining)
        
        # 更新
        self.remaining -= display_size
        self.filled += display_size
        
        return {
            'side': self.side,
            'price': self.price,
            'size': display_size,
            'remaining': self.remaining
        }
    
    def is_complete(self):
        return self.remaining <= 0
```

---

## 9. 拆单算法

### 9.1 TWAP

```python
def twap_order(execute_price, total_quantity, num_slices, time_interval):
    """
    TWAP: 时间加权平均价格
    将大单拆分为等时间间隔的小单
    """
    slice_quantity = total_quantity / num_slices
    
    orders = []
    for i in range(num_slices):
        orders.append({
            'slice_id': i + 1,
            'quantity': slice_quantity,
            'price': execute_price,  # 可用执行价格
            'time_interval': time_interval
        })
    
    return orders
```

### 9.2 VWAP

```python
def vwap_order(historical_volumes, total_quantity):
    """
    VWAP: 成交量加权平均价格
    根据历史成交量分布拆单
    """
    # 权重 = 历史成交量 / 总成交量
    weights = historical_volumes / historical_volumes.sum()
    
    # 拆单
    slices = weights * total_quantity
    
    orders = []
    for i, (time, quantity) in enumerate(slices.items()):
        orders.append({
            'time': time,
            'quantity': quantity,
            'vwap_weight': weights.iloc[i]
        })
    
    return orders
```

### 9.3 POV

```python
def pov_order(target_pct, total_quantity, historical_volumes):
    """
    POV: 成交量占比
    按成交量的一定比例下单
    """
    avg_volume = historical_volumes.mean()
    
    orders = []
    for volume in historical_volumes:
        # 按目标比例
        order_size = volume * target_pct
        
        # 不能超过总量
        order_size = min(order_size, total_quantity * 0.1)
        
        orders.append(order_size)
        total_quantity -= order_size
        
        if total_quantity <= 0:
            break
    
    return orders
```

---

## 10. 执行算法

### 10.1 执行算法选择

```python
EXECUTION_ALGORITHMS = {
    'twap': {
        'description': '时间加权平均',
        '适用': '流动性好/大单'
    },
    'vwap': {
        'description': '成交量加权平均',
        '适用': '成交量稳定'
    },
    'pov': {
        'description': '成交量比例',
        '适用': '成交活跃'
    },
    'is': {
        'description': '实现波动率',
        '适用': '波动率预测'
    },
    ' liquidation': {
        'description': '快速清仓',
        '适用': '紧急情况'
    }
}

def choose_algorithm(order_size, urgency, liquidity):
    """
    选择执行算法
    """
    if urgency == 'high':
        # 紧急 -> 快速清仓
        return 'liquidation'
    
    if order_size > 1000000:  # 大单
        if liquidity > 0.7:
            return 'twap'
        else:
            return 'is'
    
    if liquidity > 0.5:
        return 'vwap'
    
    return 'twap'
```

---

# 第四部分: 其他专题

## 11. 税务计算

### 11.1 交易税计算

```python
def calculate_trading_tax(trades, tax_rate=0.001):
    """
    计算交易税 (币安为例子)
    
    中国大陆: 交易佣金 + 提现费
    """
    total_commission = 0
    
    for trade in trades:
        if trade['type'] == 'buy':
            # 买入不收税
            continue
        elif trade['type'] == 'sell':
            # 卖出收税
            tax = trade['value'] * tax_rate
            total_commission += tax
    
    return total_commission

def calculate_crypto_tax(income, region='CN'):
    """
    计算加密货币所得税
    """
    tax_rules = {
        'CN': {
            'rate': 0.20,  # 20%
            'threshold': 36000  # 免税额
        },
        'US': {
            'rate': 0.15,  # 长期资本利得
            'short_rate': 0.25  # 短期
        }
    }
    
    rules = tax_rules.get(region, {'rate': 0})
    
    if income <= rules.get('threshold', 0):
        return 0
    
    return (income - rules.get('threshold', 0)) * rules['rate']
```

---

## 12. 策略诊断

### 12.1 问题诊断

```python
def diagnose_strategy(returns, trades, params):
    """
    策略问题诊断
    """
    issues = []
    
    # 1. 过拟合检测
    if len(trades) > 0 and len(trades) < 30:
        issues.append('交易次数太少, 可能过拟合')
    
    # 2. 胜率异常
    win_rate = (returns > 0).sum() / len(returns)
    if win_rate > 0.8:
        issues.append('胜率过高, 可能存在未来数据泄露')
    
    # 3. 收益不稳定
    if returns.std() > returns.mean() * 2:
        issues.append('收益波动过大')
    
    # 4. 手续费侵蚀
    commission_ratio = abs(returns).sum() / (1 + returns).prod()
    if commission_ratio > 0.3:
        issues.append('手续费占比过高')
    
    # 5. 幸存者偏差
    # 需要对比历史数据和当前数据
    
    return issues

def check_data_quality(data):
    """
    数据质量检查
    """
    issues = []
    
    # 缺失值
    missing = data.isnull().sum()
    if missing.sum() > 0:
        issues.append(f'存在缺失值: {missing[missing > 0].to_dict()}')
    
    # 异常值
    z_scores = (data - data.mean()) / data.std()
    outliers = (z_scores.abs() > 5).sum()
    if outliers.sum() > 0:
        issues.append(f'存在异常值: {outliers[outliers > 0].to_dict()}')
    
    # 价格跳变
    returns = data.pct_change()
    large_moves = (returns.abs() > 0.5).sum()
    if large_moves > 0:
        issues.append(f'存在价格跳变: {large_moves}')
    
    return issues
```

---

## 13. 组合再平衡

### 13.1 定期再平衡

```python
def rebalance_portfolio(target_weights, current_weights, threshold=0.05):
    """
    定期再平衡
    
    只有当权重偏离超过阈值时才调仓
    """
    trades = []
    
    for asset in target_weights:
        target = target_weights[asset]
        current = current_weights.get(asset, 0)
        
        diff = abs(target - current)
        
        if diff > threshold:
            trades.append({
                'asset': asset,
                'action': 'buy' if target > current else 'sell',
                'amount': diff
            })
    
    return trades
```

### 13.2 阈值再平衡

```python
def threshold_rebalance(portfolio_value, positions, max_deviation=0.1):
    """
    阈值再平衡
    
    当某个资产偏离目标超过阈值时再平衡
    """
    trades = []
    
    for asset, position in positions.items():
        current_pct = position['value'] / portfolio_value
        target_pct = position['target_pct']
        
        deviation = current_pct - target_pct
        
        if abs(deviation) > max_deviation:
            # 调仓
            new_value = target_pct * portfolio_value
            trade_value = position['value'] - new_value
            
            trades.append({
                'asset': asset,
                'current_pct': current_pct,
                'target_pct': target_pct,
                'deviation': deviation,
                'trade_value': trade_value
            })
    
    return trades
```

---

*持续更新中...*

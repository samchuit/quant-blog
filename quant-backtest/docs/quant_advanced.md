# 量化交易高级进阶

> 更新时间: 2026-03-08
> 整理: Jarvis 🤖

---

# 第一部分: 组合优化

## 1. Mean-Variance 优化

### 1.1 马科维茨组合理论

```python
import numpy as np
import pandas as pd

def mean_variance_optimization(returns, risk_aversion=1.0):
    """
    Mean-Variance 组合优化
    
    目标: 最大化 (期望收益 - 风险惩罚)
    约束: 权重和为1, 权重 >= 0
    """
    n = len(returns.columns)
    
    # 期望收益
    expected_returns = returns.mean()
    
    # 协方差矩阵
    cov_matrix = returns.cov()
    
    # 优化
    from scipy.optimize import minimize
    
    def objective(weights):
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
        # 最大化收益, 最小化风险
        return -(portfolio_return - risk_aversion * portfolio_vol ** 2)
    
    # 约束
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # 权重和为1
    ]
    
    # 边界
    bounds = tuple((0, 1) for _ in range(n))  # 不允许卖空
    
    # 初始权重
    initial_weights = np.array([1/n] * n)
    
    result = minimize(
        objective, 
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    return result.x
```

### 1.2 最大夏普组合

```python
def max_sharpe_portfolio(returns):
    """
    最大夏普比率组合
    """
    n = len(returns.columns)
    
    expected_returns = returns.mean() * 252  # 年化
    cov_matrix = returns.cov() * 252  # 年化
    
    from scipy.optimize import minimize
    
    def objective(weights):
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
        return -portfolio_return / portfolio_vol  # 最大化夏普
    
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = tuple((0, 1) for _ in range(n))
    
    result = minimize(objective, [1/n]*n, method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result.x
```

### 1.3 最小方差组合

```python
def min_variance_portfolio(returns):
    """
    最小方差组合
    """
    n = len(returns.columns)
    cov_matrix = returns.cov() * 252
    
    from scipy.optimize import minimize
    
    def objective(weights):
        return np.dot(weights, np.dot(cov_matrix, weights))
    
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = tuple((0, 1) for _ in range(n))
    
    result = minimize(objective, [1/n]*n, method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result.x
```

---

## 2. Black-Litterman 模型

### 2.1 核心思想

```
BL模型 = 市场均衡收益 + 主观观点

预期收益 = π + τ * Σ * P' * (V + τ*P*Σ*P')^-1 * (Q - P*π)

π: 市场均衡收益
τ: 置信度参数
Σ: 协方差矩阵
P: 观点矩阵
Q: 主观收益观点
V: 观点不确定性
```

### 2.2 实现

```python
def black_litterman(market_caps, returns, views, view_confidence):
    """
    Black-Litterman 模型
    """
    # 市场权重 (从市值计算)
    weights = market_caps / market_caps.sum()
    
    # 市场均衡收益 (反向优化)
    cov_matrix = returns.cov() * 252
    risk_aversion = 0.03  # 风险厌恶系数
    equilibrium_returns = risk_aversion * np.dot(cov_matrix, weights)
    
    # 合并观点
    # 简化实现
    adjusted_returns = equilibrium_returns.copy()
    
    for view in views:
        asset_idx = view['asset_index']
        view_return = view['return']
        confidence = view['confidence']
        
        # 调整收益
        adjusted_returns[asset_idx] = (
            confidence * view_return + 
            (1 - confidence) * equilibrium_returns[asset_idx]
        )
    
    # 组合优化
    from scipy.optimize import minimize
    
    n = len(weights)
    cov_matrix_annual = returns.cov() * 252
    
    def objective(w):
        ret = np.dot(w, adjusted_returns)
        vol = np.sqrt(np.dot(w, np.dot(cov_matrix_annual, w)))
        return -(ret - 0.03 * vol ** 2)
    
    result = minimize(objective, weights, method='SLSQP',
                     bounds=tuple((0, 1) for _ in range(n)),
                     constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}])
    
    return result.x
```

---

## 3. 风险平价策略

### 3.1 风险平价组合

```python
def risk_parity_portfolio(returns):
    """
    风险平价组合
    每个资产贡献相同的风险
    """
    n = len(returns.columns)
    cov_matrix = returns.cov() * 252
    
    from scipy.optimize import minimize
    
    def risk_contribution(weights):
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
        marginal_contrib = np.dot(cov_matrix, weights)
        risk_contrib = weights * marginal_contrib / portfolio_vol
        
        # 目标是所有风险贡献相等
        target_risk = portfolio_vol / n
        return np.sum((risk_contrib - target_risk) ** 2)
    
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = tuple((0.01, 1) for _ in range(n))
    
    result = minimize(risk_contribution, [1/n]*n, method='SLSQP',
                     bounds=bounds, constraints=constraints)
    
    return result.x
```

---

# 第二部分: 期权策略

## 4. 基础期权策略

### 4.1 期权 Greeks

```python
from scipy.stats import norm

def black_scholes(S, K, T, r, sigma, option_type='call'):
    """
    Black-Scholes 期权定价
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
    
    # Greeks
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - 
             r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    
    return {
        'price': price,
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'theta': theta,
        'rho': rho
    }
```

### 4.2 备兑看涨期权 (Covered Call)

```python
def covered_call(spot_price, strike_price, premium, target_sell_price):
    """
    备兑看涨策略
    持有现货 + 卖出看涨期权
    """
    # 情景分析
    scenarios = {
        '价格下跌': spot_price * 0.9,
        '价格不变': spot_price,
        '价格上涨': spot_price * 1.1,
        '大涨超过行权价': spot_price * 1.2
    }
    
    results = {}
    for name, price in scenarios.items():
        if price <= strike_price:
            # 期权不被行权
            profit = (price - spot_price) + premium
        else:
            # 期权被行权
            profit = (strike_price - spot_price) + premium
        
        results[name] = profit / spot_price * 100  # 收益率%
    
    return results
```

### 4.3 保护性看跌 (Protective Put)

```python
def protective_put(spot_price, strike_price, premium):
    """
    保护性看跌策略
    持有现货 + 买入看跌期权
    """
    # 成本
    cost = premium
    
    # 收益
    if spot_price * 0.9 >= strike_price:
        # 看跌期权不被行权
        return -cost / spot_price * 100
    else:
        # 看跌期权被行权
        payoff = (strike_price - spot_price * 0.9) - cost
        return payoff / spot_price * 100
```

### 4.4 牛市价差 (Bull Call Spread)

```python
def bull_call_spread(spot_price, strike1, strike2, premium1, premium2):
    """
    牛市价差策略
    买入低行权价看涨 + 卖出高行权价看涨
    """
    # 净成本
    net_premium = premium1 - premium2
    
    # 收益分析
    max_profit = (strike2 - strike1) - net_premium
    max_loss = net_premium
    
    # 盈亏平衡点
    breakeven = strike1 + net_premium
    
    return {
        'max_profit': max_profit,
        'max_loss': max_loss,
        'breakeven': breakeven
    }
```

---

## 5. 期权量化策略

### 5.1 波动率交易

```python
def volatility_trading(implied_vol, historical_vol, option_premium):
    """
    波动率交易
    当隐含波动率 > 历史波动率时, 卖出波动率
    当隐含波动率 < 历史波动率时, 买入波动率
    """
    if implied_vol > historical_vol * 1.2:
        # 隐含波动率高估, 卖出期权
        return 'sell_volatility'
    elif implied_vol < historical_vol * 0.8:
        # 隐含波动率低估, 买入期权
        return 'buy_volatility'
    return 'hold'
```

### 5.2 Delta 对冲

```python
class DeltaHedge:
    def __init__(self, option_position):
        self.position = option_position  # 期权持仓
    
    def hedge_quantity(self, spot_price, delta):
        """
        计算对冲数量
        """
        # 需要卖出 delta 份标的资产
        return -delta * self.position
    
    def rebalance(self, spot_price, new_delta):
        """
        重新平衡对冲
        """
        current_hedge = self.hedge_quantity(spot_price, self.current_delta)
        new_hedge = self.hedge_quantity(spot_price, new_delta)
        
        # 调整仓位
        adjustment = new_hedge - current_hedge
        return adjustment
```

---

# 第三部分: 因子挖掘

## 6. 遗传编程因子挖掘

### 6.1 基础框架

```python
import random
import numpy as np

class GeneticFactor:
    """遗传编程因子"""
    
    def __init__(self):
        self.operators = ['+', '-', '*', '/', 'abs', 'log']
        self.features = ['close', 'open', 'high', 'low', 'volume']
    
    def generate_tree(self, depth=3):
        """生成随机表达式树"""
        if depth == 0:
            # 叶子节点: 特征或常数
            return random.choice(self.features + [random.uniform(-1, 1)])
        
        # 内部节点
        operator = random.choice(self.operators)
        left = self.generate_tree(depth - 1)
        right = self.generate_tree(depth - 1)
        
        return (operator, left, right)
    
    def evaluate_tree(self, tree, data):
        """计算表达式"""
        if isinstance(tree, str):
            if tree in data.columns:
                return data[tree]
            else:
                return tree
        
        op, left, right = tree
        left_val = self.evaluate_tree(left, data)
        right_val = self.evaluate_tree(right, data)
        
        if op == '+':
            return left_val + right_val
        elif op == '-':
            return left_val - right_val
        elif op == '*':
            return left_val * right_val
        elif op == '/':
            return left_val / (right_val + 1e-8)
        elif op == 'abs':
            return abs(self.evaluate_tree(left, data))
        elif op == 'log':
            return np.log(abs(self.evaluate_tree(left, data)) + 1e-8)
    
    def crossover(self, tree1, tree2):
        """交叉"""
        # 简化实现: 随机交换子树
        return random.choice([tree1, tree2])
    
    def mutate(self, tree, mutation_rate=0.1):
        """变异"""
        if random.random() < mutation_rate:
            return self.generate_tree(random.randint(1, 3))
        return tree
```

### 6.2 因子挖掘流程

```python
def factor_mining(data, target, population=100, generations=50):
    """
    因子挖掘主流程
    """
    # 初始化种群
    population = [GeneticFactor().generate_tree() for _ in range(population)]
    
    best_factors = []
    
    for gen in range(generations):
        # 评估
        fitness = []
        for tree in population:
            factor = GeneticFactor().evaluate_tree(tree, data)
            correlation = factor.corr(data[target])
            fitness.append(abs(correlation))
        
        # 选择
        sorted_pop = [x for _, x in sorted(zip(fitness, population), reverse=True)]
        population = sorted_pop[:population // 2]
        
        # 交叉变异
        new_population = []
        while len(new_population) < 100:
            parent1 = random.choice(population)
            parent2 = random.choice(population)
            child = GeneticFactor().crossover(parent1, parent2)
            child = GeneticFactor().mutate(child)
            new_population.append(child)
        
        population = new_population
        
        # 记录最佳
        best_factors.append(max(fitness))
    
    return best_factors
```

---

## 7. 机器学习因子挖掘

### 7.1 特征重要性选择

```python
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

def ml_factor_importance(data, target, features):
    """
    用机器学习找重要特征
    """
    X = data[features]
    y = data[target]
    
    # 训练随机森林
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # 特征重要性
    importance = pd.DataFrame({
        'feature': features,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return importance
```

### 7.2 自动特征工程

```python
def auto_feature_engineering(df):
    """
    自动生成特征
    """
    features = []
    
    # 时间特征
    if 'date' in df.columns:
        df['hour'] = df['date'].dt.hour
        df['dayofweek'] = df['date'].dt.dayofweek
        features.extend(['hour', 'dayofweek'])
    
    # 滚动统计
    for col in ['close', 'volume']:
        for window in [5, 10, 20, 60]:
            # 移动平均
            df[f'{col}_ma{window}'] = df[col].rolling(window).mean()
            features.append(f'{col}_ma{window}')
            
            # 标准差
            df[f'{col}_std{window}'] = df[col].rolling(window).std()
            features.append(f'{col}_std{window}')
            
            # 变化率
            df[f'{col}_pct{window}'] = df[col].pct_change(window)
            features.append(f'{col}_pct{window}')
    
    # 交叉特征
    df['close_volume_ratio'] = df['close'] / df['volume']
    features.append('close_volume_ratio')
    
    return df, features
```

---

# 第四部分: 风险指标

## 8. VaR 与 CVaR

### 8.1 历史 VaR

```python
def historical_var(returns, confidence=0.95):
    """
    历史模拟法 VaR
    """
    return np.percentile(returns, (1 - confidence) * 100)

def historical_cvar(returns, confidence=0.95):
    """
    历史模拟法 CVaR (Expected Shortfall)
    """
    var = historical_var(returns, confidence)
    return returns[returns <= var].mean()
```

### 8.2 Parametric VaR

```python
def parametric_var(returns, confidence=0.95):
    """
    参数法 VaR (正态分布假设)
    """
    mu = returns.mean()
    sigma = returns.std()
    z = 1.96 if confidence == 0.95 else 2.33  # 95% or 99%
    
    return mu - z * sigma

def parametric_cvar(returns, confidence=0.95):
    """
    参数法 CVaR
    """
    mu = returns.mean()
    sigma = returns.std()
    z = 1.96 if confidence == 0.95 else 2.33
    
    # CVaR 近似
    return mu - sigma * (norm.pdf(z) / (1 - confidence))
```

### 8.3 Monte Carlo VaR

```python
def monte_carlo_var(returns, simulations=10000, confidence=0.95):
    """
    Monte Carlo VaR
    """
    mu = returns.mean()
    sigma = returns.std()
    
    # 模拟
    simulated_returns = np.random.normal(mu, sigma, simulations)
    
    return np.percentile(simulated_returns, (1 - confidence) * 100)
```

---

## 9. 压力测试

### 9.1 情景分析

```python
def stress_test(portfolio, scenarios):
    """
    压力测试
    scenarios: 字典, 如 {'crash': -0.3, 'market_drop': -0.2}
    """
    results = {}
    
    for name, shock in scenarios.items():
        # 假设组合对市场敏感度为1
        portfolio_loss = portfolio * shock
        results[name] = {
            'shock': shock * 100,
            'loss': portfolio_loss,
            'loss_pct': shock * 100
        }
    
    return results
```

### 9.2 历史极端回撤

```python
def historical_stress(returns):
    """
    历史极端情景
    """
    # 最大单日亏损
    max_loss = returns.min()
    
    # 最大连续亏损
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # 波动率最高时期
    high_vol_period = returns.rolling(20).std().idxmax()
    
    return {
        'max_daily_loss': max_loss,
        'max_drawdown': max_drawdown,
        'high_vol_date': high_vol_period
    }
```

---

## 10. 风险指标综合

### 10.1 完整风险仪表盘

```python
def risk_dashboard(returns, portfolio_value):
    """
    综合风险仪表盘
    """
    import numpy as np
    
    # 基本指标
    total_return = returns.sum()
    annual_return = returns.mean() * 252
    annual_vol = returns.std() * np.sqrt(252)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0
    
    # 风险指标
    var_95 = historical_var(returns, 0.95)
    var_99 = historical_var(returns, 0.99)
    cvar_95 = historical_cvar(returns, 0.95)
    
    # 回撤
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # 胜率
    win_rate = (returns > 0).sum() / len(returns)
    
    return {
        '收益指标': {
            '总收益': f'{total_return*100:.2f}%',
            '年化收益': f'{annual_return*100:.2f}%',
            '夏普比率': f'{sharpe:.2f}'
        },
        '风险指标': {
            'VaR(95%)': f'{var_95*100:.2f}%',
            'VaR(99%)': f'{var_99*100:.2f}%',
            'CVaR(95%)': f'{cvar_95*100:.2f}%',
            '最大回撤': f'{max_drawdown*100:.2f}%'
        },
        '交易指标': {
            '胜率': f'{win_rate*100:.2f}%',
            '交易次数': len(returns)
        }
    }
```

---

# 第五部分: 市场微观结构

## 11. 订单簿分析

### 11.1 订单簿特征

```python
def order_book_features(bids, asks):
    """
    订单簿特征
    bids/asks: 价格-数量对
    """
    # 买卖价差
    spread = asks[0][0] - bids[0][0]
    
    # 订单簿深度
    bid_depth = sum([b[1] for b in bids[:5]])
    ask_depth = sum([a[1] for a in asks[:5]])
    
    # 订单不平衡
    imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth)
    
    # 加权中间价
    vwap = (sum([b[0]*b[1] for b in bids[:5]]) + 
            sum([a[0]*a[1] for a in asks[:5]])) / (bid_depth + ask_depth)
    
    return {
        'spread': spread,
        'bid_depth': bid_depth,
        'ask_depth': ask_depth,
        'imbalance': imbalance,
        'vwap': vwap
    }
```

### 11.2 流动性分析

```python
def liquidity_analysis(trades, order_book):
    """
    流动性分析
    """
    # 成交量
    volume = trades['volume'].sum()
    
    # 成交速度
    trade_count = len(trades)
    
    # 市场深度
    depth = order_book['bid_depth'] + order_book['ask_depth']
    
    # 流动性比率
    liquidity_ratio = volume / (depth + 1)
    
    return {
        'volume': volume,
        'trade_count': trade_count,
        'depth': depth,
        'liquidity_ratio': liquidity_ratio
    }
```

---

## 12. 交易成本模型

### 12.1 滑点估算

```python
def estimate_slippage(order_size, average_volume, market_impact=0.1):
    """
    估算滑点
    简化的流动性冲击模型
    """
    # 订单量占成交量的比例
    participation_rate = order_size / average_volume
    
    # 滑点 = k * (participation_rate)^2
    slippage = market_impact * participation_rate ** 2
    
    return slippage
```

### 12.2 综合成本

```python
def total_transaction_cost(order_value, commission_rate=0.001, slippage=0.0005):
    """
    总交易成本
    """
    commission = order_value * commission_rate
    slippage_cost = order_value * slippage
    
    total = commission + slippage_cost
    
    return {
        'commission': commission,
        'slippage': slippage_cost,
        'total': total,
        'cost_pct': total / order_value * 100
    }
```

---

*持续更新中...*

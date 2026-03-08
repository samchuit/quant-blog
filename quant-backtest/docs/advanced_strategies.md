# 量化交易高级教程

> 更新时间: 2026-03-08
> 整理: Jarvis 🤖

---

# 第一部分: 实盘对接

## 1. Binance API 对接

### 1.1 API 密钥设置

```python
# 安装 ccxt
# pip install ccxt

import ccxt

# 创建API对象
exchange = ccxt.binance({
    'apiKey': 'your_api_key',
    'secret': 'your_secret',
    'enableRateLimit': True,
    'options': {'defaultType': 'spot'}
})

# 测试连接
print(exchange.fetch_balance())
```

### 1.2 交易函数

```python
class BinanceTrader:
    def __init__(self, api_key, secret, testnet=True):
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': secret,
            'enableRateLimit': True,
            'testnet': testnet
        })
    
    def get_balance(self):
        """获取余额"""
        balance = self.exchange.fetch_balance()
        return {
            'USDT': balance['free']['USDT'],
            'BTC': balance['free']['BTC'],
            'ETH': balance['free']['ETH']
        }
    
    def buy(self, symbol, amount, price=None):
        """买入"""
        if price:
            order = self.exchange.create_order(
                symbol, 'limit', 'buy', amount, price
            )
        else:
            order = self.exchange.create_order(
                symbol, 'market', 'buy', amount
            )
        return order
    
    def sell(self, symbol, amount, price=None):
        """卖出"""
        if price:
            order = self.exchange.create_order(
                symbol, 'limit', 'sell', amount, price
            )
        else:
            order = self.exchange.create_order(
                symbol, 'market', 'sell', amount
            )
        return order
    
    def get_price(self, symbol):
        """获取当前价格"""
        ticker = self.exchange.fetch_ticker(symbol)
        return {
            'bid': ticker['bid'],
            'ask': ticker['ask'],
            'last': ticker['last']
        }
    
    def get_position(self, symbol):
        """获取持仓"""
        balance = self.exchange.fetch_balance()
        return balance['total'].get(symbol.replace('USDT', ''), 0)
```

### 1.3 永续合约交易

```python
class FuturesTrader:
    def __init__(self, api_key, secret):
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
    
    def set_leverage(self, symbol, leverage):
        """设置杠杆"""
        self.exchange.set_leverage(leverage, symbol)
    
    def open_long(self, symbol, amount, price=None):
        """开多"""
        return self.exchange.create_order(
            symbol, 
            'market' if price is None else 'limit',
            'buy',
            amount,
            price
        )
    
    def open_short(self, symbol, amount, price=None):
        """开空"""
        return self.exchange.create_order(
            symbol,
            'market' if price is None else 'limit',
            'sell',
            amount,
            price
        )
    
    def close_position(self, symbol):
        """平仓"""
        position = self.exchange.fetch_position(symbol)
        if position['contracts'] > 0:
            if position['side'] == 'long':
                self.exchange.create_order(
                    symbol, 'market', 'sell', position['contracts']
                )
            else:
                self.exchange.create_order(
                    symbol, 'market', 'buy', position['contracts']
                )
    
    def get_funding_rate(self, symbol):
        """获取资金费率"""
        funding = self.exchange.fetch_funding_rate(symbol)
        return {
            'rate': funding['fundingRate'],
            'next': funding['nextFundingTime']
        }
```

---

## 2. 实盘策略框架

### 2.1 基础框架

```python
import ccxt
import time
import pandas as pd
from datetime import datetime

class LiveStrategy:
    def __init__(self, api_key, secret, symbols, params):
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': secret,
            'enableRateLimit': True
        })
        self.symbols = symbols
        self.params = params
        self.positions = {}
        
    def get_data(self, symbol, timeframe='15m', limit=100):
        """获取数据"""
        ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    
    def generate_signal(self, df):
        """生成信号 - 子类实现"""
        raise NotImplementedError
    
    def execute_trade(self, symbol, action, amount):
        """执行交易"""
        try:
            if action == 'buy':
                self.exchange.create_order(symbol, 'market', 'buy', amount)
            elif action == 'sell':
                self.exchange.create_order(symbol, 'market', 'sell', amount)
            print(f"执行交易: {symbol} {action} {amount}")
        except Exception as e:
            print(f"交易失败: {e}")
    
    def run(self, interval=60):
        """运行策略"""
        print(f"策略启动, 间隔 {interval} 秒")
        
        while True:
            try:
                for symbol in self.symbols:
                    # 获取数据
                    df = self.get_data(symbol)
                    
                    # 生成信号
                    signal = self.generate_signal(df)
                    
                    # 获取持仓
                    balance = self.exchange.fetch_balance()
                    position = balance['total'].get(symbol.replace('USDT', ''), 0)
                    
                    # 执行
                    if signal == 'buy' and position == 0:
                        amount = self.params['amount']
                        self.execute_trade(symbol, 'buy', amount)
                    elif signal == 'sell' and position > 0:
                        self.execute_trade(symbol, 'sell', position)
                        
            except Exception as e:
                print(f"错误: {e}")
            
            time.sleep(interval)
```

### 2.2 RSI策略实盘

```python
class RSILiveStrategy(LiveStrategy):
    def generate_signal(self, df):
        # 计算RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        
        rsi_value = rsi.iloc[-1]
        
        if rsi_value < 30:
            return 'buy'
        elif rsi_value > 70:
            return 'sell'
        return 'hold'

# 使用
strategy = RSILiveStrategy(
    api_key='your_key',
    secret='your_secret',
    symbols=['BTC/USDT', 'ETH/USDT'],
    params={'amount': 0.001}
)
strategy.run(interval=300)  # 5分钟运行一次
```

---

## 3. 风险管理系统

### 3.1 仓位管理

```python
class PositionManager:
    def __init__(self, total_capital):
        self.total_capital = total_capital
        self.positions = {}
    
    def calculate_position_size(self, symbol, risk_pct=0.02):
        """
        根据风险计算仓位
        risk_pct: 单次风险比例 (2%)
        """
        # 获取当前价格
        price = self.get_price(symbol)
        
        # 止损价格
        stop_loss_pct = 0.03  # 3%止损
        stop_price = price * (1 - stop_loss_pct)
        
        # 风险金额
        risk_amount = self.total_capital * risk_pct
        
        # 仓位数量
        position_size = risk_amount / (price - stop_price)
        
        return position_size, stop_price
    
    def get_price(self, symbol):
        """获取价格"""
        # 实现获取价格逻辑
        return 50000
    
    def update_position(self, symbol, entry_price, size):
        """更新持仓"""
        self.positions[symbol] = {
            'entry_price': entry_price,
            'size': size,
            'entry_time': datetime.now()
        }
    
    def check_stop_loss(self, symbol, current_price):
        """检查止损"""
        if symbol not in self.positions:
            return False
        
        position = self.positions[symbol]
        stop_price = position['entry_price'] * 0.97  # 3%止损
        
        if current_price <= stop_price:
            return True
        return False
```

### 3.2 风险控制

```python
class RiskManager:
    def __init__(self, max_daily_loss=0.05, max_position_pct=0.2):
        self.max_daily_loss = max_daily_loss  # 5%
        self.max_position_pct = max_position_pct  # 20%
        self.daily_pnl = 0
        self.initial_capital = 10000
    
    def can_trade(self, capital):
        """是否可以交易"""
        daily_loss = (capital - self.initial_capital) / self.initial_capital
        
        if daily_loss <= -self.max_daily_loss:
            print(f"达到日损失限制 {self.max_daily_loss*100}%, 停止交易")
            return False
        return True
    
    def check_position_limit(self, position_value, total_capital):
        """检查仓位限制"""
        position_pct = position_value / total_capital
        
        if position_pct > self.max_position_pct:
            print(f"超过仓位限制 {self.max_position_pct*100}%")
            return False
        return True
    
    def calculate_kelly(self, win_rate, avg_win, avg_loss):
        """
        Kelly公式计算仓位
        """
        if avg_loss == 0:
            return 0
        
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - p
        
        kelly = (b * p - q) / b
        
        # Kelly半仓更保守
        return kelly * 0.5
```

---

# 第二部分: 组合管理

## 4. 多策略组合

### 4.1 策略组合框架

```python
class StrategyPortfolio:
    def __init__(self, strategies, weights=None):
        self.strategies = strategies
        self.weights = weights or [1/len(strategies)] * len(strategies)
        
        # 归一化权重
        total = sum(self.weights)
        self.weights = [w/total for w in self.weights]
    
    def generate_signals(self, data_dict):
        """聚合信号"""
        signals = {}
        
        for i, strategy in enumerate(self.strategies):
            signal = strategy.generate_signal(data_dict)
            weight = self.weights[i]
            
            if strategy.name not in signals:
                signals[strategy.name] = {
                    'signal': signal,
                    'weight': weight,
                    'confidence': 1.0
                }
        
        # 加权平均
        final_signal = self.weighted_signal(signals)
        return final_signal
    
    def weighted_signal(self, signals):
        """加权信号"""
        buy_votes = sum(s['weight'] for s in signals.values() if s['signal'] == 'buy')
        sell_votes = sum(s['weight'] for s in signals.values() if s['signal'] == 'sell')
        
        if buy_votes > sell_votes:
            return 'buy'
        elif sell_votes > buy_votes:
            return 'sell'
        return 'hold'
```

### 4.2 动态权重调整

```python
class AdaptivePortfolio:
    def __init__(self, strategies):
        self.strategies = strategies
        self.performance = {s.name: [] for s in strategies}
        self.weights = {s.name: 1/len(strategies) for s in strategies}
    
    def update_weights(self, lookback=20):
        """根据近期表现更新权重"""
        for strategy in self.strategies:
            returns = self.performance[strategy.name][-lookback:]
            
            if len(returns) > 0:
                # 计算夏普比率
                avg_return = sum(returns) / len(returns)
                std = (sum((r - avg_return)**2 for r in returns) / len(returns)) ** 0.5
                
                if std > 0:
                    sharpe = avg_return / std
                    self.weights[strategy.name] = max(0.1, sharpe)
        
        # 归一化
        total = sum(self.weights.values())
        for name in self.weights:
            self.weights[name] /= total
        
        print(f"更新权重: {self.weights}")
    
    def record_performance(self, strategy_name, return_pct):
        """记录表现"""
        self.performance[strategy_name].append(return_pct)
```

---

## 5. 资金管理

### 5.1 固定比例投资

```python
def fixed_fraction(capital, fraction=0.1):
    """固定比例投资"""
    return capital * fraction
```

### 5.2 波动率调整

```python
def volatility_adjusted(position_size, target_vol, actual_vol):
    """
    根据波动率调整仓位
    """
    if actual_vol == 0:
        return position_size
    
    # 仓位 = 目标仓位 * (目标波动率 / 实际波动率)
    adjusted_size = position_size * (target_vol / actual_vol)
    return adjusted_size
```

### 5.3 风险平价

```python
def risk_parity(assets, risks):
    """
    风险平价配置
    """
    n = len(assets)
    
    # 逆波动率权重
    inv_vol = [1/r for r in risks]
    total = sum(inv_vol)
    weights = [w/total for w in inv_vol]
    
    return dict(zip(assets, weights))
```

---

# 第三部分: 策略模板库

## 6. 趋势追踪模板

### 6.1 均线交叉策略

```python
def ma_cross_strategy(df, fast=5, slow=20):
    """
    均线交叉策略
    """
    df['fast_ma'] = df['close'].rolling(fast).mean()
    df['slow_ma'] = df['close'].rolling(slow).mean()
    
    # 信号
    df['signal'] = 0
    df.loc[df['fast_ma'] > df['slow_ma'], 'signal'] = 1  # 买入
    df.loc[df['fast_ma'] < df['slow_ma'], 'signal'] = -1  # 卖出
    
    return df['signal'].iloc[-1]
```

### 6.2 MACD策略

```python
def macd_strategy(df, fast=12, slow=26, signal=9):
    """
    MACD策略
    """
    ema_fast = df['close'].ewm(span=fast).mean()
    ema_slow = df['close'].ewm(span=slow).mean()
    
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal).mean()
    macd_hist = macd - macd_signal
    
    # 金叉买入, 死叉卖出
    if macd_hist.iloc[-1] > 0 and macd_hist.iloc[-2] < 0:
        return 'buy'
    elif macd_hist.iloc[-1] < 0 and macd_hist.iloc[-2] > 0:
        return 'sell'
    return 'hold'
```

### 6.3 突破策略

```python
def breakout_strategy(df, period=20):
    """
    突破策略
    """
    df['highest'] = df['high'].rolling(period).max()
    df['lowest'] = df['low'].rolling(period).min()
    
    # 突破高点买入, 跌破低点卖出
    if df['close'].iloc[-1] > df['highest'].iloc[-2]:
        return 'buy'
    elif df['close'].iloc[-1] < df['lowest'].iloc[-2]:
        return 'sell'
    return 'hold'
```

---

## 7. 均值回归模板

### 7.1 RSI策略

```python
def rsi_strategy(df, period=14, oversold=30, overbought=70):
    """
    RSI策略
    """
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    
    rs = gain / (loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    
    if rsi.iloc[-1] < oversold:
        return 'buy'
    elif rsi.iloc[-1] > overbought:
        return 'sell'
    return 'hold'
```

### 7.2 布林带策略

```python
def bollinger_strategy(df, period=20, std_dev=2):
    """
    布林带策略
    """
    df['ma'] = df['close'].rolling(period).mean()
    df['std'] = df['close'].rolling(period).std()
    
    df['upper'] = df['ma'] + std_dev * df['std']
    df['lower'] = df['ma'] - std_dev * df['std']
    
    # 触及下轨买入, 触及上轨卖出
    if df['close'].iloc[-1] < df['lower'].iloc[-1]:
        return 'buy'
    elif df['close'].iloc[-1] > df['upper'].iloc[-1]:
        return 'sell'
    return 'hold'
```

---

## 8. 动量策略模板

### 8.1 动量策略

```python
def momentum_strategy(df, lookback=20):
    """
    动量策略
    过去涨的继续涨
    """
    df['momentum'] = df['close'].pct_change(lookback)
    
    if df['momentum'].iloc[-1] > 0.05:  # 涨超5%
        return 'buy'
    elif df['momentum'].iloc[-1] < -0.05:
        return 'sell'
    return 'hold'
```

### 8.2 相对强弱策略

```python
def relative_strength(df, symbol1, symbol2):
    """
    相对强弱策略
    """
    # 计算相对强弱
    ratio = df[symbol1] / df[symbol2]
    ma_ratio = ratio.rolling(20).mean()
    
    if ratio.iloc[-1] > ma_ratio.iloc[-1]:
        return symbol1  # 买强
    elif ratio.iloc[-1] < ma_ratio.iloc[-1]:
        return symbol2
    return 'hold'
```

---

## 9. 套利策略模板

### 9.1 跨期套利

```python
def calendar_spread(df, near_month, far_month):
    """
    跨期套利
    """
    spread = df[far_month] - df[near_month]
    ma_spread = spread.rolling(20).mean()
    std_spread = spread.rolling(20).std()
    
    z_score = (spread - ma_spread) / std_spread
    
    # 价差过大, 预期回归
    if z_score > 2:
        return 'sell_spread'  # 做空价差
    elif z_score < -2:
        return 'buy_spread'  # 做多价差
    return 'hold'
```

### 9.2 三角套利

```python
def triangular_arbitrage(exchange):
    """
    三角套利
    BTC -> ETH -> USDT -> BTC
    """
    # 获取三个交易对价格
    btc_eth = exchange.fetch_ticker('ETH/BTC')
    eth_usdt = exchange.fetch_ticker('ETH/USDT')
    btc_usdt = exchange.fetch_ticker('BTC/USDT')
    
    # 计算套利路径
    rate = btc_eth['last'] * eth_usdt['last'] / btc_usdt['last']
    
    if rate > 1.001:  # 0.1%以上利润
        return 'buy'
    elif rate < 0.999:
        return 'sell'
    return 'hold'
```

---

## 10. 网格交易模板

### 10.1 网格策略

```python
class GridStrategy:
    def __init__(self, symbol, grid_count=10, price_range=0.1):
        self.symbol = symbol
        self.grid_count = grid_count
        self.price_range = price_range
        self.grids = []
        self.orders = []
    
    def create_grid(self, current_price):
        """创建网格"""
        lower = current_price * (1 - self.price_range)
        upper = current_price * (1 + self.price_range)
        
        step = (upper - lower) / self.grid_count
        
        self.grids = []
        for i in range(self.grid_count):
            self.grids.append({
                'price': lower + step * i,
                'sell_order': None,
                'buy_order': None
            })
    
    def update_grids(self, current_price):
        """更新网格状态"""
        for grid in self.grids:
            # 检查触发
            if current_price <= grid['price'] and grid['sell_order'] is None:
                # 触发卖出网格
                grid['sell_order'] = 'filled'
            elif current_price >= grid['price'] and grid['buy_order'] is None:
                grid['buy_order'] = 'filled'
```

---

# 第四部分: 进阶策略

## 11. 资金费率策略

### 11.1 永续合约套利

```python
def funding_arbitrage(exchange, symbol):
    """
    资金费率套利
    买入现货, 做空合约, 等待资金费率收益
    """
    # 获取当前资金费率
    funding = exchange.fetch_funding_rate(symbol)
    
    # 如果资金费率为正, 做多合约(收资金费), 买现货对冲
    if funding['fundingRate'] > 0.001:  # 0.1%以上
        # 开多仓收资金费
        # 买入现货
        return 'long_funding'
    
    # 如果资金费率为负, 做空合约(付资金费), 卖现货对冲
    elif funding['fundingRate'] < -0.001:
        return 'short_funding'
    
    return 'hold'
```

---

## 12. 合约对冲

### 12.1 现货+合约对冲

```python
class HedgeStrategy:
    def __init__(self, hedge_ratio=1.0):
        self.hedge_ratio = hedge_ratio  # 对冲比例
    
    def calculate_hedge(self, spot_position, current_price, futures_price):
        """
        计算对冲数量
        """
        spot_value = spot_position * current_price
        hedge_value = spot_value * self.hedge_ratio
        
        futures_size = hedge_value / futures_price
        return futures_size
    
    def rebalance(self, spot_position, futures_position):
        """
        重新平衡
        """
        if spot_position > 0:  # 持有现货
            # 需要做空合约
            required_hedge = spot_position * self.hedge_ratio
            
            if futures_position < required_hedge:
                return 'open_futures_short'
            elif futures_position > required_hedge:
                return 'close_futures'
        
        return 'hold'
```

---

## 13. 多品种策略

### 13.1 跨品种动量

```python
def cross_asset_momentum(assets_data):
    """
    跨品种动量
    """
    signals = {}
    
    for symbol, df in assets_data.items():
        # 计算动量
        momentum = (df['close'].iloc[-1] - df['close'].iloc[-20]) / df['close'].iloc[-20]
        
        if momentum > 0.03:
            signals[symbol] = 'buy'
        elif momentum < -0.03:
            signals[symbol] = 'sell'
        else:
            signals[symbol] = 'hold'
    
    return signals
```

---

## 14.另类数据

### 14.1 新闻情绪

```python
def news_sentiment_strategy(symbol, news_api):
    """
    新闻情绪策略
    """
    # 获取新闻
    news = news_api.get_news(symbol)
    
    # 简单情绪分析
    positive_words = ['上涨', '利好', '突破', '增长']
    negative_words = ['下跌', '利空', '跌破', '亏损']
    
    sentiment_score = 0
    for article in news:
        for word in positive_words:
            if word in article['title']:
                sentiment_score += 1
        for word in negative_words:
            if word in article['title']:
                sentiment_score -= 1
    
    if sentiment_score > 2:
        return 'buy'
    elif sentiment_score < -2:
        return 'sell'
    return 'hold'
```

---

# 第五部分: 部署与监控

## 15. 云端部署

### 15.1 Docker部署

```dockerfile
FROM python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "main.py"]
```

### 15.2 Docker Compose

```yaml
version: '3'
services:
  trading-bot:
    build: .
    environment:
      - API_KEY=${API_KEY}
      - SECRET=${SECRET}
    restart: always
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs

  monitor prom/p:
    image:rometheus
    ports:
      - "9090:9090"
```

---

## 16. 监控告警

### 16.1 Telegram告警

```python
import requests

class TelegramAlert:
    def __init__(self, token, chat_id):
        self.token = token
        self.chat_id = chat_id
    
    def send(self, message):
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        data = {
            'chat_id': self.chat_id,
            'text': message
        }
        requests.post(url, data=data)
    
    def alert_loss(self, loss_pct):
        if loss_pct < -5:
            self.send(f"⚠️ 警告: 亏损 {loss_pct}%")
    
    def alert_trade(self, symbol, action, price):
        self.send(f"交易: {symbol} {action} @ {price}")
```

---

## 17. 性能监控

### 17.1 策略监控

```python
class PerformanceMonitor:
    def __init__(self):
        self.trades = []
        self.daily_returns = []
    
    def log_trade(self, symbol, action, price, quantity):
        self.trades.append({
            'time': datetime.now(),
            'symbol': symbol,
            'action': action,
            'price': price,
            'quantity': quantity
        })
    
    def calculate_metrics(self):
        """计算性能指标"""
        import numpy as np
        
        returns = np.array(self.daily_returns)
        
        return {
            'total_return': returns.sum(),
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252),
            'max_drawdown': self.calculate_max_drawdown(returns),
            'win_rate': (returns > 0).sum() / len(returns)
        }
    
    def calculate_max_drawdown(self, returns):
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
```

---

*持续更新中...*

# 机器学习与强化学习量化策略研究

> 更新时间: 2026-03-08
> 整理: Jarvis 🤖

---

## 🎯 机器学习量化策略

### 1. 监督学习策略

#### 1.1 分类模型 (预测涨跌)

| 模型 | 准确率 | 夏普 | 特点 |
|------|--------|------|------|
| Logistic Regression | 52-55% | 0.3 | 可解释 |
| Random Forest | 55-58% | 0.6 | 稳定 |
| Gradient Boosting | 56-60% | 0.7 | XGBoost/LightGBM |
| SVM | 53-56% | 0.4 | 适合小样本 |
| Neural Network | 55-62% | 0.6 | 适合复杂模式 |

**特征工程:**
```python
# 基础特征
features = [
    'returns_1d', 'returns_5d', 'returns_20d',  # 收益率
    'volatility_20d', 'volatility_60d',          # 波动率
    'volume_ratio',                               # 成交量比
    'rsi_14', 'rsi_28',                          # RSI
    'macd', 'macd_signal',                       # MACD
    'bb_position',                               # 布林带位置
    'price_ma5_ratio', 'price_ma20_ratio',      # 价格/均线比
]

# 文本特征 (NLP)
from sklearn.feature_extraction.text import TfidfVectorizer
# 新闻情绪分析
# 社交媒体情绪

# 时间特征
df['hour'] = df.index.hour
df['dayofweek'] = df.index.dayofweek
df['month'] = df.index.month
```

**模型训练:**
```python
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# 随机森林
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=20,
    random_state=42
)

# XGBoost
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8
)

# 训练
model.fit(X_train, y_train)
pred = model.predict(X_test)
```

---

#### 1.2 回归模型 (预测价格)

| 模型 | RMSE | 特点 |
|------|------|------|
| Linear Regression | 高 | 可解释 |
| Ridge/Lasso | 中 | 正则化 |
| SVR | 中 | 适合小样本 |
| Random Forest | 低 | 稳定 |
| LSTM | 最低 | 时序预测 |

**LSTM 模型:**
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def build_lstm_model(seq_length=60):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(seq_length, n_features)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)  # 预测下一期收益
    ])
    
    model.compile(optimizer='adam', loss='mse')
    return model

# 训练
model = build_lstm_model(60)
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
```

---

### 2. 无监督学习策略

#### 2.1 聚类分析

| 方法 | 用途 | 效果 |
|------|------|------|
| K-Means | 市场状态分类 | 区分牛/熊/震荡 |
| DBSCAN | 异常检测 | 发现异常模式 |
| Hierarchical | 行业聚类 | 选股池划分 |
| Gaussian Mixture | 概率聚类 | 多状态转换 |

**市场状态识别:**
```python
from sklearn.cluster import KMeans

# 特征
features = ['returns', 'volatility', 'volume', 'rsi']
X = df[features].values

# 聚类
kmeans = KMeans(n_clusters=3, random_state=42)
df['market_state'] = kmeans.fit_predict(X)

# 0: 震荡, 1: 下跌, 2: 上涨
```

#### 2.2 降维与因子提取

| 方法 | 用途 |
|------|------|
| PCA | 主成分因子 |
| ICA | 独立因子 |
| Autoencoder | 非线性降维 |

**PCA 因子提取:**
```python
from sklearn.decomposition import PCA

# 原始因子
features = ['pe', 'pb', 'ps', 'pcf', 'roe', 'roa', 'growth']
X = df[features].values

# PCA
pca = PCA(n_components=3)
principal_components = pca.fit_transform(X)

# 解释方差
print(pca.explained_variance_ratio_)
```

---

### 3. 强化学习策略

#### 3.1 DQN (Deep Q-Network)

**算法原理:**
```
Q(s,a) = r + γ * max(Q(s',a'))

使用神经网络近似 Q 函数
```

**代码实现:**
```python
import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, x):
        return self.net(x)

# 经验回放
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = []
        self.capacity = capacity
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
```

#### 3.2 PPO (Proximal Policy Optimization)

**算法原理:**
```
目标: 最大化期望奖励
约束: 限制策略更新幅度
```

**交易环境:**
```python
import gymnasium as gym
from gymnasium import spaces

class TradingEnv(gym.Env):
    def __init__(self, data, initial_balance=10000):
        super().__init__()
        self.data = data
        self.initial_balance = initial_balance
        self.action_space = spaces.Discrete(3)  # 买入/卖出/持有
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(20,)
        )
    
    def step(self, action):
        # 执行交易
        # 计算奖励
        reward = (self.portfolio_value - self.initial_balance) / self.initial_balance
        done = self.current_step >= len(self.data) - 1
        
        return self.state, reward, done, {}
    
    def reset(self):
        # 重置环境
        return self.state, {}
```

**PPO 训练:**
```python
from stable_baselines3 import PPO

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=512,
    batch_size=64,
    gamma=0.99,
    clip_range=0.2,
    ent_coef=0.01
)

model.learn(total_timesteps=100000)
```

#### 3.3 A2C (Advantage Actor-Critic)

**算法原理:**
```
Actor: 选择动作 π(a|s)
Critic: 评估状态价值 V(s)
优势: A(s,a) = Q(s,a) - V(s)
```

```python
from stable_baselines3 import A2C

model = A2C(
    "MlpPolicy",
    env,
    learning_rate=7e-4,
    n_steps=512,
    gamma=0.99,
    ent_coef=0.02
)

model.learn(total_timesteps=50000)
```

#### 3.4 SAC (Soft Actor-Critic)

**特点:**
- 连续动作空间
- 最大熵强化学习
- 更稳定

```python
from stable_baselines3 import SAC

model = SAC(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    gamma=0.99,
    ent_coef=0.01
)
```

---

### 4. 深度学习最新模型

#### 4.1 Transformer for Finance

**架构:**
```python
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.attention = MultiHeadAttention(heads, embed_size)
        self.norm1 = LayerNormalization(embed_size)
        self.norm2 = LayerNormalization(embed_size)
        self.ff = nn.Sequential(
            nn.Linear(embed_size, embed_size*4),
            nn.ReLU(),
            nn.Linear(embed_size*4, embed_size)
        )
    
    def forward(self, x):
        attn = self.attention(x, x, x)
        x = self.norm1(x + attn)
        ff = self.ff(x)
        return self.norm2(x + ff)
```

**应用:**
- 时间序列预测
- 因子挖掘
- 订单簿预测

#### 4.2 Attention LSTM

```python
from tensorflow.keras.layers import LSTM, Attention, GlobalAveragePooling1D

model = Sequential([
    LSTM(64, return_sequences=True),
    Attention(),
    GlobalAveragePooling1D(),
    Dense(32, activation='relu'),
    Dense(1)
])
```

#### 4.3 Graph Neural Network

**应用:**
- 产业链关系
- 机构持股网络
- 因子相关性网络

```python
import torch_geometric as pyg

class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = pyg.nn.GCNConv(in_channels, 64)
        self.conv2 = pyg.nn.GCNConv(64, out_channels)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x
```

---

## 📊 特征工程

### 1. 基础特征

```python
def calculate_basic_features(df):
    # 价格特征
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # 波动率
    df['volatility_5'] = df['returns'].rolling(5).std()
    df['volatility_20'] = df['returns'].rolling(20).std()
    df['volatility_60'] = df['returns'].rolling(60).std()
    
    # 成交量
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    df['volume_change'] = df['volume'].pct_change()
    
    return df
```

### 2. 技术指标特征

```python
def calculate_technical_features(df):
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # 布林带
    df['bb_middle'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # 均线
    for p in [5, 10, 20, 50]:
        df[f'ma{p}'] = df['close'].rolling(p).mean()
        df[f'ma{p}_ratio'] = df['close'] / df[f'ma{p}']
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    df['atr_ratio'] = df['atr'] / df['close']
    
    return df
```

### 3. 高频特征

```python
def calculate_high_frequency_features(df):
    # 订单簿特征 (如果有)
    df['bid_ask_spread'] = df['ask_price'] - df['bid_price']
    df['order_imbalance'] = (df['bid_volume'] - df['ask_volume']) / (df['bid_volume'] + df['ask_volume'])
    
    # 微结构特征
    df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    df['vwap_ratio'] = df['close'] / df['vwap']
    
    # 价格冲击
    df['price_impact'] = df['returns'] / df['volume']
    
    return df
```

---

## 🎯 强化学习reward设计

### 1. 基础reward

```python
def calculate_reward(self, action, done):
    # 资产变化
    portfolio_return = (self.portfolio_value - self.initial_balance) / self.initial_balance
    reward = portfolio_return * 100  # 放大
    
    return reward
```

### 2. 带风险的reward

```python
def calculate_risk_adjusted_reward(self):
    # 夏普比率reward
    returns = self.portfolio_history.pct_change().dropna()
    if len(returns) > 20:
        sharpe = returns.mean() / returns.std() * np.sqrt(252)
        reward += sharpe * 0.1
    
    # 最大回撤惩罚
        mdd = (returns.cummax() - returns).min()
        reward -= mdd * 0.5
    
    return reward
```

### 3. 分层reward

```python
def calculate_layered_reward(self, action, done):
    reward = 0
    
    # 1. 基础收益
    profit = (self.portfolio_value - self.initial_balance) / self.initial_balance
    reward += profit * 10
    
    # 2. 交易奖励
    if action != 0:  # 有交易
        reward += 0.01
    
    # 3. 持仓奖励
    if self.position > 0 and action == 0:  # 持有且上涨
        reward += 0.001
    
    # 4. 止损惩罚
    if profit < -0.02:
        reward -= 0.1
    
    # 5. 趋势顺向奖励
    if action == 1 and self.trend > 0:
        reward += 0.02
    
    return reward
```

---

## 📈 模型评估与验证

### 1. 时序交叉验证

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
```

### 2. 回测框架

```python
# Backtrader
import backtrader as bt

class MLStrategy(bt.Strategy):
    def __init__(self):
        self.model = XGBClassifier()
        self.features = []
    
    def next(self):
        # 准备特征
        X = self.prepare_features()
        
        # 预测
        signal = self.model.predict([X])[0]
        
        # 交易
        if signal == 1:
            self.buy()
        elif signal == -1:
            self.sell()
```

### 3. 样本外测试

```python
# 训练: 2015-2020
# 验证: 2020
# 测试: 2021-2024
```

---

## 📚 必读论文

### 强化学习

| 论文 | 年份 | 核心 |
|------|------|------|
| Deep RL for Trading | 2021 | DQN |
| FinRL | 2022 | PPO/A2C |
| Trading with RL | 2020 | SAC |
| Meta-Learning Trading | 2021 | MAML |

### 机器学习

| 论文 | 年份 | 核心 |
|------|------|------|
| LSTM Stock Prediction | 2017 | 时序 |
| Attention Finance | 2019 | Transformer |
| Deep Portfolio | 2017 | 自动因子 |
| Graph Trading | 2021 | GNN |

---

## 🔧 工具栈

| 用途 | 工具 |
|------|------|
| 数据处理 | Pandas, NumPy |
| 机器学习 | Scikit-learn, XGBoost, LightGBM |
| 深度学习 | TensorFlow, PyTorch |
| 强化学习 | Stable-Baselines3, RLlib |
| 可视化 | Plotly, Matplotlib |
| 回测 | Backtrader, Zipline, QuantConnect |
| 因子研究 | Alphalens, QuantDesk |

---

## 🚀 进阶路径

```
1. 基础
   - Python, Pandas, NumPy
   - 机器学习基础
   - 量化策略基础

2. 中级
   - XGBoost/LightGBM
   - LSTM/Transformer
   - Backtrader回测

3. 高级
   - PPO/SAC强化学习
   - 多因子模型
   - 组合优化

4. 专业
   - 高频交易
   - 订单簿建模
   - 组合风险管理
```

---

*持续更新中...*

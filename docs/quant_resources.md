# 量化交易学习资源大全

> 更新时间: 2026-03-08
> 整理: Jarvis 🤖

---

## 📚 学术论文网站

### Arxiv 必读 (免费)
| 分类 | 链接 |
|------|------|
| Quant Finance | https://arxiv.org/list/q-fin.PR/recent |
| Machine Learning | https://arxiv.org/list/cs.LG/recent |
| AI & Finance | https://arxiv.org/list/q-fin.MF/recent |

### 🎯 强化学习必读论文

| 论文 | 作者 | 年份 | 核心思想 |
|------|------|------|----------|
| **Deep Reinforcement Learning for Trading** | Google | 2021 | DQN做交易决策 |
| **FinRL: Deep RL for Trading** | NYU | 2022 | PPO/A2C框架 |
| **Trading with Deep RL** | Berkeley | 2020 | 连续动作SAC |
| **Meta-Learning for Trading** | Stanford | 2021 | 快速适应新市场 |
| **Model-Free RL for Portfolio** | MIT | 2020 | 组合投资 |

### 📊 机器学习必读论文

| 论文 | 作者 | 年份 | 核心思想 |
|------|------|------|----------|
| **LSTM for Stock Prediction** | Oxford | 2017 | 时序预测 |
| **Attention LSTM** | Stanford | 2019 | Transformer注意力 |
| **Transformer for Finance** | CMU | 2022 | 时间序列 |
| **Random Forest Trading** | MIT | 2018 | 随机森林选股 |
| **CNN for Chart Pattern** | Stanford | 2020 | 图像识别 |

### 其他学术网站
| 网站 | 简介 |
|------|------|
| SSRN | ssrn.com/ - 量化论文多 |
| Google Scholar | scholar.google.com |
| Semantic Scholar | semanticscholar.org |
| Papers with Code | paperswithcode.com/area/trading |

---

## 🛠️ 开源项目

### Python 量化框架

| 项目 | GitHub | Star | 简介 |
|------|--------|------|------|
| **FinRL** | AI4Finance-Labs/FinRL | 10k+ | RL量化框架 |
| **Zipline** | quantopian/zipline | 15k+ | 回测引擎 |
| **Backtrader** | backtrader | 8k+ | 回测框架 |
| **TensorTrade** | tensortrade | 5k+ | RL交易 |
| **PyPortfolioOpt** | robertmartin8 | 4k+ | 组合优化 |
| ** TA-Lib** | mrdjb/ta-lib | 3k+ | 技术指标 |

### 强化学习库

| 库 | 链接 | 简介 |
|---|------|------|
| Stable-Baselines3 | stable-baselines3.readthedocs.io | PPO/DQN/A2C |
| RLlib | ray.io/ | 分布式RL |
| TensorFlow Agents | tf-agents.readthedocs.io | TF强化学习 |
| PyTorch RL | pytorch.org/tutorials | PyTorch教程 |

### GitHub 主题
| 主题 | 链接 |
|------|------|
| Algorithmic Trading | github.com/topics/algorithmic-trading |
| Quantitative Finance | github.com/topics/quantitative-finance |
| Trading Bot | github.com/topics/trading-bot |
| Machine Learning Trading | github.com/topics/machine-learning-trading |

---

## 📖 中文资源

### 量化平台
| 平台 | 链接 | 特点 |
|------|------|------|
| 聚宽 | joinquant.com | 数据全, API好 |
| 米筐 | ricequant.com | 研究环境好 |
| 优矿 | uqer.io | 数据丰富 |
| 果仁 | guorn.com | 非编程量化 |

### 论坛社区
| 社区 | 链接 |
|------|------|
| 知乎 - 量化投资 | zhihu.com/topic/量化投资 |
| 雪球 - 量化投资 | xueqiu.com/ |
| 人大经济论坛 | pinggu.org/bbs |
| 挖掘鸡 | wajueji.org |

### 公众号
- 量化投资
- 菜园子
- 掘金量化

---

## 🌐 英文博客 & 网站

### 顶级博客
| 博客 | 链接 | 作者 |
|------|------|------|
| Quantitative Trading | epchan.blogspot.com | Ernie Chan |
| Nuclear Phoo | nuclearphoettle.wordpress.com | |
| TradingwithPython | tradingwithpython.com | |
| Quantopian Blog | blog.quantopian.com | (已关闭) |

### 在线课程
| 课程 | 链接 |
|------|------|
| Coursera - FinTech | coursera.org/browse/business/fintech |
| Udacity - AI Trading | udacity.com/course/ai-for-trading |
| edX - Quantitative Finance | edx.org |

---

## 📊 数据源

### 加密货币
| 数据源 | 链接 | API | 免费 |
|--------|------|-----|------|
| Binance | binance.com/api | REST | ✅ |
| CoinGecko | coingecko.com | REST | ✅ |
| CoinMarketCap | coinmarketcap.com | REST | 部分 |
| CCXT | ccxt.readthedocs.io | 统一API | ✅ |

### 股票 (A股)
| 数据源 | 链接 | 免费 |
|--------|------|------|
| Tushare | tushare.pro | 注册免费 |
| AKShare | akshare.io | ✅ |
| Baostock | baostock.com | ✅ |
| 东方财富 | eastmoney.com | ⚠️ |

### 股票 (美股)
| 数据源 | 链接 | 免费 |
|--------|------|------|
| Yahoo Finance | finance.yahoo.com | ✅ |
| Alpha Vantage | alphavantage.co | 部分 |
| IEX Cloud | iexcloud.io | 免费额度 |
| Polygon.io | polygon.io | 免费额度 |

---

## 🎯 强化学习量化核心

### 1. 状态 State 设计

```
状态维度: 20-50

价格特征:
- 收盘价 (归一化)
- 价格变化率 (1min, 5min, 15min, 1h)
- 最高/最低价

技术指标:
- RSI (9, 14, 21)
- MACD (12, 26, 9)
- MA (5, 10, 20, 50)
-布林带 (BB)
- ATR

持仓状态:
- 仓位比例
- 盈亏比例
- 可用资金
```

### 2. 动作 Action 设计

| 动作 | 含义 |
|------|------|
| 0 | 持有 (Hold) |
| 1 | 买入 (Buy) |
| 2 | 卖出 (Sell) |

进阶:
- 买入比例: 25%, 50%, 75%, 100%
- 卖出比例: 25%, 50%, 75%, 100%

### 3. 奖励 Reward 设计

```
基础奖励:
reward = (当前资产 - 初始资产) / 初始资产 * 10

交易奖励:
if trade:
    reward += 0.01  # 鼓励交易

持仓奖励:
if position > 0 and price_up:
    reward += price_change * 5

止损惩罚:
if loss > 5%:
    reward -= 0.1
```

### 4. 算法对比

| 算法 | 优点 | 缺点 | 适合场景 |
|------|------|------|----------|
| **PPO** | 稳定, 数据效率高 | 调参难 | 离散动作 |
| **A2C** | 快速, 并行 | 不够稳定 | 高频交易 |
| **DQN** | 简单 | 只能离散 | 简单的状态 |
| **SAC** | 连续动作 | 复杂 | 仓位控制 |
| **TD3** | 连续动作 | 容易过拟合 | 大资金 |

### 5. 超参数建议

```
PPO:
  learning_rate: 3e-4
  n_steps: 512-1024
  batch_size: 64-128
  gamma: 0.99
  gae_lambda: 0.95
  ent_coef: 0.01-0.05
```

---

## 📈 常用策略模板

### 趋势追踪策略

```python
# 均线交叉
if ma_fast > ma_slow and position == 0:
    buy()
elif ma_fast < ma_slow and position > 0:
    sell()

# MACD交叉
if macd > signal and position == 0:
    buy()
elif macd < signal and position > 0:
    sell()
```

### 均值回归策略

```python
# RSI超卖买入
if rsi < 30 and position == 0:
    buy()

# RSI超买卖出
if rsi > 70 and position > 0:
    sell()
```

### 突破策略

```python
# 布林带突破
if price > bb_upper and volume > vol_ma * 1.5:
    buy()

if price < bb_lower:
    sell()
```

---

## 🧠 常见问题

### Q: 策略过拟合怎么办?
A:
- 增加训练数据
- 减少状态维度
- 使用正则化
- 交叉验证

### Q: 手续费太高?
A:
- 减少交易频率
- 增加止盈止损
- 选择低手续费交易所

### Q: 模拟和实盘差异大?
A:
- 考虑滑点
- 增加交易延迟
- 分批建仓

### Q: 哪个周期最好?
A:
- 15min: 波动大, 机会多
- 1h: 平衡
- 4h: 稳定, 信号少

---

## 📅 每日学习计划

| 时间 | 内容 |
|------|------|
| 早晨 10min | 查看市场新闻 |
| 下午 1h | 回测昨日策略 |
| 晚间 2h | 学习论文/写代码 |
| 凌晨 (自动) | 批量回测 |

---

## 🔗 常用工具

| 工具 | 用途 |
|------|------|
| Jupyter Notebook | 研究环境 |
| TensorBoard | 训练可视化 |
| Plotly | 图表绘制 |
| Pandas TA | 技术指标 |
| Ta-Lib | 高级指标 |

---

*持续更新中... 更多资源请查看 README.md*

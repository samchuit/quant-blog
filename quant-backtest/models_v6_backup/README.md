# v6 模型版本记录

## 保存时间
2026-03-07 14:24

## 训练配置
- 训练数据: 2年 (730天)
- 每币种步数: 200,000
- 学习率: 3e-4
- 市场模式: 全市场适配

## 性能表现

### 各币种结果
| 币种 | RL收益 | BH收益 | 超额收益 | 交易次数 |
|------|--------|--------|----------|----------|
| LINK | +212.3% | -55.3% | +267.6% | 59 |
| MATIC | +181.6% | -59.2% | +240.9% | 50 |
| DOGE | +151.6% | -44.9% | +196.5% | 42 |
| SOL | +146.9% | -41.7% | +188.6% | 49 |
| DOT | +95.8% | -85.8% | +181.5% | 56 |
| ADA | +97.3% | -64.2% | +161.4% | 8 |
| ETH | +65.5% | -49.1% | +114.6% | 134 |
| XRP | +218.3% | +120.4% | +97.8% | 47 |
| BTC | +63.9% | -0.1% | +64.0% | 30 |
| BNB | +34.3% | +29.2% | +5.1% | 23 |

### 平均成绩
- 平均RL收益: +118.7%
- 平均BH收益: -25.1%
- 平均超额收益: **+146.8%**
- 平均交易次数: 46次

## 模型文件
- ppo_v6_BTCUSDT.zip
- ppo_v6_ETHUSDT.zip (原版，效果差)
- ppo_v6_ETH_trend.zip (ETH专用趋势版，推荐)
- ppo_v6_BNBUSDT.zip
- ppo_v6_SOLUSDT.zip
- ppo_v6_XRPUSDT.zip
- ppo_v6_ADAUSDT.zip
- ppo_v6_DOGEUSDT.zip
- ppo_v6_DOTUSDT.zip
- ppo_v6_MATICUSDT.zip
- ppo_v6_LINKUSDT.zip

## 使用说明
```python
from stable_baselines3 import PPO
from optimize_v6 import TradingEnvV6, get_data

# 加载模型
model = PPO.load('ppo_v6_BTCUSDT.zip')

# 获取数据
data = get_data('BTCUSDT')

# 创建环境
env = TradingEnvV6(data, market_mode='all')

# 运行
obs, _ = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, _, done, _, _ = env.step(action)

print(f'收益: {(env.total_profit/10000-1)*100:.1f}%')
```

## 注意事项
1. ETH 请使用 ppo_v6_ETH_trend.zip 而非原版
2. 实盘需考虑手续费 (~0.1%/笔) 和滑点
3. 极端行情可能触发止损
4. 建议先模拟盘测试

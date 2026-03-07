# 量化交易策略系统

基于强化学习的加密货币量化交易系统，使用PPO算法训练交易策略。

## 项目概述

- **策略版本**: v7 (风控增强版)
- **训练数据**: 2021-2026 (~6年)
- **回测超额收益**: +94.9% (样本外验证)
- **交易品种**: BTC, ETH, BNB, SOL, XRP, ADA, DOGE, DOT, MATIC, LINK

## 策略特点

### 风控机制
- 止盈: 15%
- 止损: 8%
- 最大回撤控制: 20%
- 单日最大亏损: 5%
- 手续费: 0.1% + 滑点: 0.05%

### 技术指标
- 移动平均线 (MA5, MA20, MA60)
- RSI (相对强弱指标)
- MACD
- 布林带
- ATR (平均真实波幅)

## 文件结构

```
quant-backtest/
├── optimize_v7.py          # v7策略训练脚本
├── optimize_v8_rolling.py  # 滚动测试脚本
├── live_trading.py         # 实时交易接口
├── trading_monitor.py      # 交易监控系统
├── monitor_report.py       # 监控报告生成
├── auto_retrain.py         # 自动重训练
├── backtest_5year.py       # 5年回测脚本
├── check_position.py       # 持仓检查工具
├── 使用指南.md             # 使用文档
├── 完整使用指南.md         # 完整文档
└── models_v8_backup/       # 模型备份目录
    └── README.md           # 模型说明
```

## 快速开始

### 环境要求
- Python 3.10+
- stable-baselines3
- gymnasium
- requests

### 安装依赖
```bash
pip install stable-baselines3 gymnasium requests numpy matplotlib
```

### 运行回测
```bash
python backtest_5year.py
```

### 启动监控
```bash
# 运行一次
python trading_monitor.py --mode once

# 每日定时运行 (日线策略)
python trading_monitor.py --mode daily --daily-time 08:00
```

## 回测结果

### 6年全周期 (2021-2026)

| 币种 | RL收益 | BH收益 | 超额收益 | 最大回撤 |
|------|--------|--------|----------|----------|
| SOL | +8850% | +1593% | +7257% | 78.6% |
| XRP | +3108% | -36% | +3145% | 32.9% |
| ETH | +2318% | -14% | +2333% | 15.1% |
| BTC | +1358% | +10% | +1348% | 19.2% |
| BNB | +979% | -35% | +1014% | 41.9% |

**平均超额收益**: +3019%

### 样本外验证 (220天)

**平均超额收益**: +94.9%

## 交易接口

支持币安测试网接口:
- 获取实时价格
- 账户余额查询
- 自动下单执行

## 风险提示

1. 历史表现不代表未来收益
2. 加密货币市场波动极大
3. 实盘需考虑滑点和手续费
4. 建议先使用测试网验证

## 许可证

MIT License

## 更新日志

### 2026-03-07
- 完成v7版本优化
- 添加风控机制
- 接入币安测试网
- 实现自动监控

---

**免责声明**: 本策略仅供学习研究使用，不构成投资建议。投资有风险，入市需谨慎。

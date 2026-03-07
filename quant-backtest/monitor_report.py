"""
监控报告系统
=============
每日/每周生成策略表现报告
"""
import os
import json
import numpy as np
import requests
from datetime import datetime, timedelta
from stable_baselines3 import PPO
import warnings
warnings.filterwarnings('ignore')

WORK_DIR = r"C:\Users\Administrator\.openclaw\workspace\quant-backtest"
os.chdir(WORK_DIR)

PROXIES = {'http': 'http://192.168.0.45:7897', 'https': 'http://192.168.0.45:7897'}


def get_data(symbol, limit=365):
    """获取数据"""
    try:
        url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1d&limit={limit}'
        r = requests.get(url, proxies=PROXIES, timeout=15)
        data = json.loads(r.text)
        return np.array([[float(d[1]),float(d[2]),float(d[3]),float(d[4]),float(d[5]),float(d[7])] for d in data], dtype=np.float32)
    except:
        return None


def calculate_indicators(data):
    """计算指标"""
    close = data[:, 3]
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
        ind[i, 7] = 0  # 简化
        ind[i, 11] = 0
        ind[i, 12] = (ind[i,0] - ind[i,1]) / (ind[i,1] + 1e-8)
        ind[i, 13] = 0
    
    return ind


def get_action(model, data):
    """获取模型建议"""
    ind = calculate_indicators(data)
    i = len(data) - 1
    c = data[i, 3]
    
    obs = np.array([
        ind[i,0]/1000, ind[i,1]/1000, ind[i,2]/1000, ind[i,3]/100,
        ind[i,4], 0, 0, ind[i,7]*10, 0, 0,
        0, ind[i,11], ind[i,12]*10, ind[i,13], 0,
        0, 0, 0, 1, 0, 0, 0, 0, 0, 0
    ], dtype=np.float32)
    
    action, _ = model.predict(obs)
    actions = ['HOLD', 'BUY_SMALL', 'BUY_LARGE', 'SELL_HALF', 'SELL_ALL']
    return actions[min(action, 4)]


def generate_report():
    """生成监控报告"""
    print("="*60)
    print(f"策略监控报告 - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*60)
    
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT']
    
    report_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'coins': []
    }
    
    for sym in symbols:
        data = get_data(sym)
        if data is None:
            continue
        
        current_price = data[-1, 0]
        prev_close = data[-2, 3]
        daily_change = (current_price - prev_close) / prev_close * 100
        
        # 30天表现
        price_30d_ago = data[-30, 3] if len(data) >= 30 else data[0, 3]
        change_30d = (current_price - price_30d_ago) / price_30d_ago * 100
        
        # 技术指标
        ind = calculate_indicators(data)
        rsi = ind[-1, 3]
        bb = ind[-1, 4]
        
        # 获取模型建议
        try:
            model = PPO.load(f"ppo_v7_{sym}")
            action = get_action(model, data)
        except:
            action = "MODEL_NOT_FOUND"
        
        coin_data = {
            'symbol': sym,
            'price': float(current_price),
            'daily_change': float(daily_change),
            'change_30d': float(change_30d),
            'rsi': float(rsi),
            'bollinger': float(bb),
            'action': action
        }
        
        report_data['coins'].append(coin_data)
        
        print(f"\n{sym}:")
        print(f"  价格: ${current_price:,.2f}")
        print(f"  日涨跌: {daily_change:+.2f}%")
        print(f"  30日涨跌: {change_30d:+.2f}%")
        print(f"  RSI: {rsi:.1f}")
        print(f"  布林带位置: {bb:+.2f}σ")
        print(f"  模型建议: {action}")
    
    # 保存报告
    with open('monitor_report.json', 'w') as f:
        json.dump(report_data, f, indent=2)
    
    # 生成文本报告
    text_report = generate_text_report(report_data)
    with open('monitor_report.txt', 'w', encoding='utf-8') as f:
        f.write(text_report)
    
    print("\n" + "="*60)
    print("报告已保存: monitor_report.json, monitor_report.txt")
    
    return report_data


def generate_text_report(data):
    """生成文本报告"""
    lines = [
        "="*60,
        f"策略监控日报 - {data['timestamp']}",
        "="*60,
        ""
    ]
    
    for coin in data['coins']:
        lines.extend([
            f"{coin['symbol']}:",
            f"  价格: ${coin['price']:,.2f}",
            f"  日涨跌: {coin['daily_change']:+.2f}%",
            f"  30日涨跌: {coin['change_30d']:+.2f}%",
            f"  RSI: {coin['rsi']:.1f} (超买>70, 超卖<30)",
            f"  布林带: {coin['bollinger']:+.2f}σ",
            f"  模型建议: {coin['action']}",
            ""
        ])
    
    # 汇总
    avg_daily = np.mean([c['daily_change'] for c in data['coins']])
    actions = [c['action'] for c in data['coins']]
    
    lines.extend([
        "="*60,
        "汇总:",
        f"  平均日涨跌: {avg_daily:+.2f}%",
        f"  建议买入: {actions.count('BUY_SMALL') + actions.count('BUY_LARGE')}个",
        f"  建议卖出: {actions.count('SELL_HALF') + actions.count('SELL_ALL')}个",
        f"  建议持有: {actions.count('HOLD')}个",
        "="*60
    ])
    
    return "\n".join(lines)


def send_notification(report_data):
    """发送通知（需要配置）"""
    # 这里可以接入飞书/钉钉/Telegram等通知
    # 示例：保存到文件
    print("\n[通知] 报告已生成，请查看 monitor_report.txt")
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['once', 'daily'], default='once')
    args = parser.parse_args()
    
    if args.mode == 'once':
        report = generate_report()
    else:
        # 每日定时运行
        import schedule
        import time
        
        print("启动每日监控...")
        schedule.every().day.at("09:00").do(generate_report)
        schedule.every().day.at("21:00").do(generate_report)
        
        while True:
            schedule.run_pending()
            time.sleep(60)

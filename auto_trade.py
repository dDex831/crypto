# 文件路径：auto_trade/auto_trade.py

import os
import json
import math
import time
from datetime import datetime, timezone, timedelta

import pandas as pd
import numpy as np
from binance.client import Client
from binance.exceptions import BinanceAPIException

# ========== 1. Binance API 初始化 ==========
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")
if not API_KEY or not API_SECRET:
    raise ValueError("请先在环境变量中设置 BINANCE_API_KEY 和 BINANCE_API_SECRET")

client = Client(API_KEY, API_SECRET)

# ========== 2. 参数配置 ==========
SYMBOL = "ADAUSDT"           # 交易对
INTERVAL = Client.KLINE_INTERVAL_4HOUR  # 4h K 线
LIMIT = 100                  # 拉取最近 100 根 K 线即可
POSITION_SIDE = "LONG"       # 本示例只做多单，不做空单

# 状态文件路径（与本脚本同目录）
STATE_FILE = os.path.join(os.path.dirname(__file__), "state.json")

# ========== 3. 工具函数：加载/保存持仓状态 ==========
def load_state():
    """从 state.json 加载持仓状态，如果文件不存在，则返回初始状态。"""
    if not os.path.exists(STATE_FILE):
        return {
            "inPosition": False,     # 是否持仓
            "entryTime": None,       # 持仓时的时间戳（UTC）
            "entryPrice": None       # 持仓时的进场价格
        }
    with open(STATE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_state(state: dict):
    """将持仓状态写回 state.json。在 workflow 中会检测到变更并自动提交。"""
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

# ========== 4. 拉取 4h K 线数据，并转换为 DataFrame ==========
def fetch_klines(symbol, interval, limit=LIMIT):
    """
    返回一个 pandas.DataFrame，字段包括：
    ['open_time', 'open', 'high', 'low', 'close', 'volume', ...]
    open_time 单位为毫秒。
    """
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])
    # 转换类型
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df

# ========== 5. 计算指标：Stochastic、Bollinger Band 等 ==========
def compute_indicators(df: pd.DataFrame):
    """
    输入：包含 open, high, low, close, volume 的 DataFrame，按时间升序排列。
    输出：原 DataFrame 直接新增列：
      - 'k', 'd', 'j' (Stochastic)
      - 'lowerBB' (Bollinger 下轨)
      - 以及红棒统计 redBarCount、barDrop（当根跌幅）、volume_20m_flag（量 > 20M）
    """
    # 5.1 计算 Stochastic %K, %D, %J
    low_min = df["low"].rolling(window=14).min()
    high_max = df["high"].rolling(window=14).max()
    df["k"] = 100 * (df["close"] - low_min) / (high_max - low_min)
    df["d"] = df["k"].rolling(window=3).mean()
    df["j"] = 3 * df["k"] - 2 * df["d"]

    # 5.2 计算 Bollinger Band 下轨
    basis = df["close"].rolling(window=20).mean()
    dev = df["close"].rolling(window=20).std()
    df["lowerBB"] = basis - 2 * dev

    # 5.3 计算红棒（close < open）并统计最近 10 根红棒数量
    df["is_red"] = (df["close"] < df["open"]).astype(int)
    df["redBarCount"] = df["is_red"].rolling(window=10).sum()

    # 5.4 当根“开→收”跌幅 ≥ 7%
    df["barDrop"] = ((df["open"] - df["close"]) / df["open"] * 100) >= 7

    # 5.5 原有条件细项
    df["brokenBB"] = df["low"] < df["lowerBB"]
    df["vol_increasing"] = df["volume"] > df["volume"].shift(1) * 1.5
    df["vol_over_20m"] = df["volume"] > 20_000_000

    return df

# ========== 6. 判断进场／出场信号 ==========
def check_signals(df: pd.DataFrame, state: dict):
    """
    根据最后一根 K 线（即 DataFrame 最后一行），判断是否产生新的进场信号或需要平仓。
    返回：
      - action: "BUY", "SELL_TP", "SELL_3D", 或 None
      - price:  本 K 线的 close 价格 (执行下单时可略微调整为市价单，直接传 None 即可)
    """
    last = df.iloc[-1]
    prev = df.iloc[-2]

    # 6.1 判断是否触发“进场”条件
    cond_original = (
        (last["j"] <= 15) and
        last["brokenBB"] and
        last["vol_increasing"] and
        (last["j"] > prev["j"]) and
        last["vol_over_20m"]
    )
    cond_8_of_10_red = last["redBarCount"] >= 8
    condition = cond_original or last["barDrop"] or cond_8_of_10_red
    alert_on_next_bar = condition and (not (prev.get("condition_flag", False)))

    # 6.2 判断是否需要“平仓”
    action = None
    price = last["close"]

    # 如果当前没有持仓，且本根 K 线产生进场信号
    if (not state["inPosition"]) and alert_on_next_bar:
        action = "BUY"
        return action, price

    # 如果当前持仓
    if state["inPosition"]:
        entry_price = state["entryPrice"]
        entry_time = datetime.fromisoformat(state["entryTime"])
        now = last["open_time"]  # 本根 K 线的收盘时间（UTC）
        # 6.2.1 持仓获利 ≥ 1.5% 立即平仓
        profit_pct = (price - entry_price) / entry_price * 100
        if profit_pct >= 1.5:
            action = "SELL_TP"
            return action, price

        # 6.2.2 持仓超过 3 天 (即 18 根 4h K 线) 且当前 close ≥ entry_price （回本）才平仓
        if (now >= entry_time + timedelta(days=3)) and (price >= entry_price):
            action = "SELL_3D"
            return action, price

    return None, None

# ========== 7. 下单函数 ==========
def place_order(action: str, symbol: str, quantity: float):
    """
    简化示例：默认使用市价单
    action: "BUY" 或 "SELL"
    quantity: 买入/卖出数量（以基础币种计）
    """
    try:
        if action == "BUY":
            order = client.create_order(
                symbol=symbol,
                side=Client.SIDE_BUY,
                type=Client.ORDER_TYPE_MARKET,
                quantity=quantity
            )
        else:  # SELL
            order = client.create_order(
                symbol=symbol,
                side=Client.SIDE_SELL,
                type=Client.ORDER_TYPE_MARKET,
                quantity=quantity
            )
        print(f"{datetime.now()} 下单成功：{action}, 订单 ID: {order['orderId']}")
        return order
    except BinanceAPIException as e:
        print(f"{datetime.now()} 下单失败：{e}")
        return None

# ========== 8. 计算仓位数量（全仓 300 USDT 为例）==========
def calc_quantity(usdt_amount: float, price: float):
    """
    简化：你原策略是 300 USDT 全仓买入，默认按市价计算买多少 ADA。
    向下取整保留足够最小精度。
    """
    # 假设交易对 ADAUSDT 的最小下单数量精度 (step size) 是 0.001，你可以在 API 中查询交易规则来动态获取
    step_size = 0.001
    qty = usdt_amount / price
    qty = math.floor(qty / step_size) * step_size
    return float(format(qty, f".3f"))

# ========== 9. 主流程 ==========
def main():
    # 9.1 加载持仓状态
    state = load_state()

    # 9.2 拉取数据、计算指标
    df = fetch_klines(SYMBOL, INTERVAL, limit=LIMIT)
    df = compute_indicators(df)

    # —— 在这里插入调试打印，用来检查 K 线和指标数值 —— #
    # 打印最近 5 根 4h K 线，以及它们的 open/high/low/close/volume
    # 以及刚算出来的 k, d, j, lowerBB, redBarCount, barDrop, brokenBB, vol_increasing, vol_over_20m
    print("\n===== 最近 5 根 4h K 线与指标 =====")
    cols_to_show = [
        "open_time", "open", "high", "low", "close", "volume",
        "k", "d", "j", "lowerBB", "redBarCount", "barDrop",
        "brokenBB", "vol_increasing", "vol_over_20m"
    ]
    # 只选最后 5 根
    print(df[cols_to_show].tail(5).to_string(index=False))
    print("====================================\n")
    # —— 调试打印结束 —— #



    # 9.3 判断信号
    action, price = check_signals(df, state)

    if action == "BUY":
        # 9.3.1 计算买入数量（假设全仓 300 USDT）
        usdt_amount = 300
        qty = calc_quantity(usdt_amount, price)

        # 9.3.2 下买单
        order = place_order("BUY", SYMBOL, qty)
        if order:
            # 更新状态
            state["inPosition"] = True
            state["entryPrice"] = price
            # entryTime 记录为本根 K 线收盘时间（ISO 格式）
            last_time = df.iloc[-1]["open_time"]
            state["entryTime"] = last_time.replace(tzinfo=timezone.utc).isoformat()
            print(f"{datetime.now()} 记录持仓状态：入场价格 {price}，时间 {state['entryTime']}")

    elif action in ("SELL_TP", "SELL_3D"):
        # 9.3.3 平仓前先查询当前持仓数量
        # 这里示例直接按照 state["entryPrice"] 计算的数量来卖出，实际可调用账户信息查询精确数量
        entry_price = state["entryPrice"]
        qty = calc_quantity(300, entry_price)
        order = place_order("SELL", SYMBOL, qty)
        if order:
            print(f"{datetime.now()} 执行平仓动作：{action}")
            # 清空状态
            state["inPosition"] = False
            state["entryPrice"] = None
            state["entryTime"] = None

    else:
        print(f"{datetime.now()} 无操作信号。当前持仓状态：{state}")

    # 9.4 把 state 写回文件，后续由 workflow 检测变更并提交
    save_state(state)

if __name__ == "__main__":
    main()

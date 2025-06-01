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
            "entryPrice": None,      # 持仓时的进场价格
            "entryQty": None         # 持仓时的合约数量
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
      - 'lowerBB' (Bollinger 下轨，ddof=0，与 TradingView 一致)
      - 以及红棒统计 redBarCount、barDrop（当根跌幅）、vol_increasing、vol_over_20m
    """
    # 5.1 计算 Stochastic %K, %D, %J
    low_min = df["low"].rolling(window=14).min()
    high_max = df["high"].rolling(window=14).max()
    df["k"] = 100 * (df["close"] - low_min) / (high_max - low_min)
    df["d"] = df["k"].rolling(window=3).mean()
    df["j"] = 3 * df["k"] - 2 * df["d"]

    # 5.2 计算 Bollinger Band 下轨 (总体标准差 ddof=0)
    basis = df["close"].rolling(window=20).mean()
    dev = df["close"].rolling(window=20).std(ddof=0)
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
      - price:  本 K 线的 close 价格
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

    if (not state["inPosition"]) and alert_on_next_bar:
        action = "BUY"
        return action, price

    if state["inPosition"]:
        entry_price = state["entryPrice"]
        entry_time = datetime.fromisoformat(state["entryTime"])
        now = last["open_time"]  # 本根 K 线的收盘时间（UTC）
        # 6.2.1 持仓获利 ≥ 1.5% 立即平仓
        profit_pct = (price - entry_price) / entry_price * 100
        if profit_pct >= 1.5:
            action = "SELL_TP"
            return action, price

        # 6.2.2 持仓超过 3 天 (即 18 根 4h K 线) 且当前 close ≥ entry_price（回本）才平仓
        if (now >= entry_time + timedelta(days=3)) and (price >= entry_price):
            action = "SELL_3D"
            return action, price

    return None, None

# ========== 7. 下单函数 ==========
def place_order(action: str, symbol: str, quantity: float):
    """
    简化示例：默认使用市价单
    action: "BUY" 或 "SELL"
    quantity: 买入/卖出数量
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

# ========== 8. 计算仓位数量（按 USDT 余额） ===========
def calc_quantity(usdt_amount: float, price: float):
    """
    计算在当前价格下，用 usdt_amount（USDT）可以购买多少 ADA（现货或合约按市价）。向下取整保留精度到 0.001。
    """
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

    # —— 调试打印：最近 5 根 K 线与指标 —— #
    print("\n===== 最近 5 根 4h K 线与指标 =====")
    cols_to_show = [
        "open_time", "open", "high", "low", "close", "volume",
        "k", "d", "j", "lowerBB", "redBarCount", "barDrop",
        "brokenBB", "vol_increasing", "vol_over_20m"
    ]
    print(df[cols_to_show].tail(5).to_string(index=False))
    print("====================================\n")
    # —— 调试打印结束 —— #

    # 9.3 判断信号
    action, price = check_signals(df, state)

    if action == "BUY":
        usdt_amount = 200
        # 9.3.1 计算买入数量
        qty = calc_quantity(usdt_amount, price)

        # 9.3.2 下买单
        order = place_order("BUY", SYMBOL, qty)
        if order:
            # 更新状态：记录 inPosition、entryPrice、entryTime 以及 entryQty
            state["inPosition"] = True
            state["entryPrice"] = price
            state["entryQty"] = qty
            last_time = df.iloc[-1]["open_time"]
            state["entryTime"] = last_time.replace(tzinfo=timezone.utc).isoformat()
            print(
                f"{datetime.now()} 记录持仓状态：入场价格 {price}，入场数量 {qty}，时间 {state['entryTime']}"
            )

    elif action in ("SELL_TP", "SELL_3D"):
        # 9.3.3 平仓：直接用保存的 entryQty，而不是重新计算
        qty = state.get("entryQty")
        if qty is None:
            print(f"{datetime.now()} 错误：找不到 entryQty，无法平仓")
        else:
            order = place_order("SELL", SYMBOL, qty)
            if order:
                print(f"{datetime.now()} 执行平仓动作：{action}，卖出数量 {qty}")
                # 清空状态
                state["inPosition"] = False
                state["entryPrice"] = None
                state["entryTime"] = None
                state["entryQty"] = None

    else:
        print(f"{datetime.now()} 无操作信号。当前持仓状态：{state}")

    # 9.4 把 state 写回文件
    save_state(state)

# ========== 10. 测试函数：市价单买入 20 ADA 合约 (5 倍杠杆)，3 秒后卖出 ==========
def test_trade_futures():
    """
    在 USDT-M 永续合约账户上执行：
    1) 设置 ADAUSDT 永续合约杠杆为 5 倍
    2) 市价买入 20 张 ADAUSDT 合约
    3) 等待 3 秒
    4) 市价卖出 20 张（平仓）
    """
    try:
        # 10.1 设置杠杆为 5 倍
        leverage_resp = client.futures_change_leverage(symbol=SYMBOL, leverage=5)
        print(f"{datetime.now()} 杠杆设置响应：{leverage_resp}")

        # 10.2 市价买入 20 张 ADAUSDT 永续合约
        buy_order = client.futures_create_order(
            symbol=SYMBOL,
            side="BUY",
            type="MARKET",
            quantity=20  # 这里改为买入 20 张合约
        )
        print(f"{datetime.now()} 下单买入 20 张 ADA 合约，订单信息：{buy_order}")

        # 10.3 等待 3 秒
        time.sleep(3)

        # 10.4 市价卖出 20 张进行平仓
        sell_order = client.futures_create_order(
            symbol=SYMBOL,
            side="SELL",
            type="MARKET",
            quantity=20  # 平仓时同样卖出 20 张
        )
        print(f"{datetime.now()} 下单卖出 20 张 ADA 合约，订单信息：{sell_order}")

    except BinanceAPIException as e:
        print(f"{datetime.now()} 测试下单失败：{e}")

if __name__ == "__main__":
    # 运行主流程
    main()

    # —— 执行测试下单 —— #
    # print("\n===== 开始测试：合约账户市价买入20 ADA，3 秒后卖出 =====")
    # test_trade_futures()
    # print("===== 测试结束 =====\n")

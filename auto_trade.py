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

# ===============================
# 1. Binance API 初始化（Spot）
# ===============================
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")
if not API_KEY or not API_SECRET:
    raise ValueError("请先在环境变量中设置 BINANCE_API_KEY 和 BINANCE_API_SECRET")

client = Client(API_KEY, API_SECRET)

# ===============================
# 2. 参数配置
# ===============================
SYMBOL = "ADAUSDT"               # Spot 交易对
INTERVAL = Client.KLINE_INTERVAL_4HOUR  # 4h K 线
LIMIT = 100                      # 拉取最近 100 根 K 线即可
POSITION_SIDE = "LONG"           # 这里只做多单示例，不做空单

# 从 SYMBOL 提取 base asset（用于检查现货余额）
BASE_ASSET = SYMBOL.replace("USDT", "")  # "ADA"

# 状态文件路径（与本脚本同目录）
STATE_FILE = os.path.join(os.path.dirname(__file__), "state.json")


# ===============================
# 3. LOT_SIZE 参数获取
# ===============================
def get_lot_size_params(symbol: str):
    """
    通过交易对信息获取 minQty 和 stepSize。
    """
    info = client.get_symbol_info(symbol)
    for f in info["filters"]:
        if f["filterType"] == "LOT_SIZE":
            return float(f["minQty"]), float(f["stepSize"])
    raise ValueError(f"LOT_SIZE filter not found for {symbol}")


# ===============================
# 4. 计算现货下单数量（按 USDT 余额）
# ===============================
def calc_quantity(usdt_amount: float, price: float, symbol: str):
    """
    计算在当前价格下，用 usdt_amount（USDT）可以在现货市场买多少 base asset（如 ADA）。
    根据该交易对的 minQty 与 stepSize 向下取整，确保合法。
    返回 0.0 表示低于最小下单量，应跳过下单。
    """
    min_qty, step = get_lot_size_params(symbol)
    raw_qty = usdt_amount / price
    # 向下取整到 step 的整数倍
    qty = math.floor(raw_qty / step) * step
    if qty < min_qty:
        return 0.0
    # 格式化小数位数与 step 保持一致
    precision = int(round(-math.log10(step)))
    return float(format(qty, f".{precision}f"))


# ===============================
# 5. 状态加载/保存
# ===============================
def load_state():
    if not os.path.exists(STATE_FILE):
        return {
            "inPosition": False,
            "strategy": None,
            "entryTime": None,
            "entryPrice": None,
            "entryQty": None
        }
    with open(STATE_FILE, "r", encoding="utf-8") as f:
        state = json.load(f)
    if "strategy" not in state:
        state["strategy"] = None
    return state

def save_state(state: dict):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


# ===============================
# 6. 拉取 K 线并计算指标
# ===============================
def fetch_klines(symbol, interval, limit=LIMIT):
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df

def compute_indicators(df: pd.DataFrame):
    low_min = df["low"].rolling(window=14).min()
    high_max = df["high"].rolling(window=14).max()
    df["k"] = 100 * (df["close"] - low_min) / (high_max - low_min)
    df["d"] = df["k"].rolling(window=3).mean()
    df["j"] = 3 * df["k"] - 2 * df["d"]
    basis = df["close"].rolling(window=20).mean()
    dev = df["close"].rolling(window=20).std(ddof=0)
    df["lowerBB"] = basis - 2 * dev
    df["upperBB"] = basis + 2 * dev
    df["is_red"] = (df["close"] < df["open"]).astype(int)
    df["redBarCount"] = df["is_red"].rolling(window=10).sum()
    df["barDrop"] = ((df["open"] - df["close"]) / df["open"] * 100) >= 7
    df["brokenBB"] = df["low"] < df["lowerBB"]
    df["vol_increasing"] = df["volume"] > df["volume"].shift(1) * 1.45
    df["vol_over_20m"] = df["volume"] > 20_000_000
    return df


# ===============================
# 7. 信号判断（同前）
# ===============================
def check_signals(df: pd.DataFrame, state: dict):
    if len(df) < 5:
        return None, None
    last  = df.iloc[-2]
    prev  = df.iloc[-3]
    prev2 = df.iloc[-4]
    prev3 = df.iloc[-5]
    price = last["close"]

    # 动量策略进场/出场
    pct0 = (last["close"] - last["open"]) / last["open"] * 100
    pct1 = (prev["close"] - prev["open"]) / prev["open"] * 100
    pct2 = (prev2["close"] - prev2["open"]) / prev2["open"] * 100
    pct3 = (prev3["close"] - prev3["open"]) / prev3["open"] * 100
    sum3 = pct1 + pct2 + pct3
    upperShadowPct = (last["high"] - last["close"]) / last["close"] * 100
    mom_entry = (
        (pct0 > 7) and 
        (sum3 <= 20) and 
        (upperShadowPct <= 2) and 
        (not state.get("inPosition", False))
    )
    in_mom = state.get("inPosition", False) and state.get("strategy") == "mom"
    touchedUpperBB = last["high"] >= last["upperBB"]
    if in_mom:
        profitPctMom = (price - state["entryPrice"]) / state["entryPrice"] * 100
        if (not touchedUpperBB) and ((price < prev["close"]) or (profitPctMom >= 1.5)):
            return "SELL_MOM", price
    if mom_entry:
        return "BUY_MOM", price

    # 原有策略
    j        = last["j"]
    j_prev   = prev["j"]
    brokenBB = last["brokenBB"]
    vol_inc  = last["vol_increasing"]
    vol_20m  = last["vol_over_20m"]
    cond_A = (j <= 15) and brokenBB and vol_inc and (j > j_prev) and vol_20m
    cond_B = last["barDrop"]
    cond_C = last["redBarCount"] >= 8
    orig_entry = (cond_A or cond_B or cond_C) and (not state.get("inPosition", False))
    j2        = prev["j"]
    brokenBB2 = prev["brokenBB"]
    vol_inc2  = prev["vol_increasing"]
    vol_202m  = prev["vol_over_20m"]
    condA2 = (j2 <= 15) and brokenBB2 and vol_inc2 and (j2 > df.iloc[-4]["j"]) and vol_202m
    condB2 = prev["barDrop"]
    condC2 = prev["redBarCount"] >= 8
    prev_condition = condA2 or condB2 or condC2
    alert_orig = orig_entry and not prev_condition
    in_orig = state.get("inPosition", False) and state.get("strategy") == "orig"
    if in_orig:
        entry_price = state["entryPrice"]
        last_time   = last["open_time"]
        entry_time  = datetime.fromisoformat(state["entryTime"])
        profitPctOrig = (price - entry_price) / entry_price * 100
        if profitPctOrig >= 1.5:
            return "SELL_TP", price
        if (last_time >= entry_time + timedelta(days=3)) and (price >= entry_price):
            return "SELL_3D", price
    if alert_orig:
        return "BUY_ORIG", price

    return None, None


# ===============================
# 8. 下单函数（Spot 市价单）
# ===============================
def place_order_spot(action: str, symbol: str, quantity: float):
    try:
        if action == "BUY":
            order = client.order_market_buy(symbol=symbol, quantity=quantity)
        else:
            order = client.order_market_sell(symbol=symbol, quantity=quantity)
        print(f"{datetime.now()} 下单成功：{action}, 订单 ID: {order.get('orderId', order.get('clientOrderId', 'N/A'))}")
        return order
    except BinanceAPIException as e:
        print(f"{datetime.now()} 下单失败：{e}")
        return None


# ===============================
# 9. 同步本地状态 vs. API
# ===============================
def sync_position_with_api_spot(state: dict):
    if not state.get("inPosition", False):
        return state
    try:
        balance = client.get_asset_balance(asset=BASE_ASSET)
        free_amount = float(balance.get("free", 0.0))
        if free_amount == 0.0:
            print(f"{datetime.now()} 检测到现货持仓为0 → 重置本地状态。")
            state.update({"inPosition": False, "strategy": None, "entryPrice": None, "entryTime": None, "entryQty": None})
            save_state(state)
    except BinanceAPIException as e:
        print(f"{datetime.now()} 同步现货持仓时出错：{e}")
    return state


# ===============================
# 10. 主流程
# ===============================
def main():
    state = load_state()
    state = sync_position_with_api_spot(state)

    df = fetch_klines(SYMBOL, INTERVAL, limit=LIMIT)
    df = compute_indicators(df)
    action, price = check_signals(df, state)

    print("\n===== 最近 5 根 4h K 线与指标 =====")
    cols = ["open_time","open","high","low","close","volume","k","d","j","lowerBB","upperBB","redBarCount","barDrop","brokenBB","vol_increasing","vol_over_20m"]
    print(df[cols].tail(5).to_string(index=False))
    print("====================================\n")

    if action in ("BUY_MOM", "BUY_ORIG"):
        usdt_amount = 320
        qty = calc_quantity(usdt_amount, price, SYMBOL)
        if qty <= 0:
            print(f"{datetime.now()} 计算的下单量 {qty} 小于最小下单量，跳过下单")
        else:
            order = place_order_spot("BUY", SYMBOL, qty)
            if order:
                fills = order.get("fills", [])
                if fills:
                    total_qty = sum(float(f["qty"]) for f in fills)
                    avg_price = sum(float(f["price"]) * float(f["qty"]) for f in fills) / total_qty
                else:
                    total_qty, avg_price = qty, price

                state["inPosition"] = True
                state["strategy"]   = "mom" if action=="BUY_MOM" else "orig"
                state["entryPrice"] = avg_price
                state["entryQty"]   = total_qty
                last_time = df.iloc[-2]["open_time"]
                state["entryTime"] = last_time.replace(tzinfo=timezone.utc).isoformat()
                print(f"{datetime.now()} 记录仓位（现货）：策略 {state['strategy']}, 入场价 {avg_price:.6f}, 数量 {total_qty:.3f}, 时间 {state['entryTime']}")

    elif action in ("SELL_MOM", "SELL_TP", "SELL_3D"):
        qty = state.get("entryQty")
        if qty is None:
            print(f"{datetime.now()} 错误：找不到 entryQty，无法平仓")
        else:
            order = place_order_spot("SELL", SYMBOL, qty)
            if order:
                fills = order.get("fills", [])
                if fills:
                    total_qty = sum(float(f["qty"]) for f in fills)
                    avg_price = sum(float(f["price"]) * float(f["qty"]) for f in fills) / total_qty
                else:
                    total_qty, avg_price = qty, price

                print(f"{datetime.now()} 平仓成功：数量 {total_qty:.3f}, 价格 {avg_price:.6f}, 动作 {action}")
                state.update({"inPosition": False, "strategy": None, "entryPrice": None, "entryTime": None, "entryQty": None})

    else:
        print(f"{datetime.now()} 无操作信号。当前持仓状态：{state}")

    save_state(state)


# ===============================
# 11. 测试函数（可选）
# ===============================
def test_trade_spot():
    try:
        ticker = client.get_symbol_ticker(symbol=SYMBOL)
        price = float(ticker["price"])
        qty = calc_quantity(1 * price, price, SYMBOL)
        buy_order = client.order_market_buy(symbol=SYMBOL, quantity=qty)
        print(f"{datetime.now()} Spot 下单买入 ADA，订单信息：{buy_order}")
        time.sleep(3)
        sell_order = client.order_market_sell(symbol=SYMBOL, quantity=qty)
        print(f"{datetime.now()} Spot 下单卖出 ADA，订单信息：{sell_order}")
    except BinanceAPIException as e:
        print(f"{datetime.now()} Spot 测试下单失败：{e}")

if __name__ == "__main__":
    main()
    # 若需单独测试，下方取消注释：
    # test_trade_spot()

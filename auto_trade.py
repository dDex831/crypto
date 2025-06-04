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
# 3. 工具函数：加载/保存持仓状态
# ===============================
def load_state():
    """
    从 state.json 加载持仓状态，如果文件不存在，则返回初始状态。
    """
    if not os.path.exists(STATE_FILE):
        return {
            "inPosition": False,     # 是否持仓
            "strategy": None,        # "orig" 或 "mom"
            "entryTime": None,       # 持仓时的时间戳（UTC，ISO 格式）
            "entryPrice": None,      # 持仓时的实际成交均价（USD 计价）
            "entryQty": None         # 持仓时的实际成交数量（ADA 数量）
        }
    with open(STATE_FILE, "r", encoding="utf-8") as f:
        state = json.load(f)
        # 兼容旧版本没有 strategy 的情况
        if "strategy" not in state:
            state["strategy"] = None
        return state

def save_state(state: dict):
    """
    将持仓状态写回 state.json。
    """
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

# ===============================
# 4. 拉取 4h K 线数据，并转换为 DataFrame
# ===============================
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

# ===============================
# 5. 计算指标：Stochastic、Bollinger Band 等
# ===============================
def compute_indicators(df: pd.DataFrame):
    """
    输入：包含 open, high, low, close, volume 的 DataFrame，按时间升序排列。
    输出：在原 DataFrame 直接新增列：
      - 'k', 'd', 'j' (Stochastic)
      - 'lowerBB', 'upperBB' (Bollinger 下轨和上轨，ddof=0，与 TradingView 一致)
      - 以及红棒统计 redBarCount、barDrop、vol_increasing、vol_over_20m
    """
    # 1. 计算 Stochastic %K, %D, %J
    low_min = df["low"].rolling(window=14).min()
    high_max = df["high"].rolling(window=14).max()
    df["k"] = 100 * (df["close"] - low_min) / (high_max - low_min)
    df["d"] = df["k"].rolling(window=3).mean()
    df["j"] = 3 * df["k"] - 2 * df["d"]

    # 2. 计算 Bollinger Band 上下轨 (总体标准差 ddof=0)
    basis = df["close"].rolling(window=20).mean()
    dev = df["close"].rolling(window=20).std(ddof=0)
    df["lowerBB"] = basis - 2 * dev
    df["upperBB"] = basis + 2 * dev

    # 3. 红棒 = close < open，最近 10 根红棒数量
    df["is_red"] = (df["close"] < df["open"]).astype(int)
    df["redBarCount"] = df["is_red"].rolling(window=10).sum()

    # 4. 当根“开→收”跌幅 ≥ 7%
    df["barDrop"] = ((df["open"] - df["close"]) / df["open"] * 100) >= 7

    # 5. 其余条件
    df["brokenBB"] = df["low"] < df["lowerBB"]
    df["vol_increasing"] = df["volume"] > df["volume"].shift(1) * 1.5
    df["vol_over_20m"] = df["volume"] > 20_000_000

    return df

# ===============================
# 6. 判断进场／出场信号（和原来完全相同）
# ===============================
def check_signals(df: pd.DataFrame, state: dict):
    """
    结合“原有策略”与“动量策略”：
      - 动量策略：本根 K 线涨幅 > 7%，前三根涨幅合 ≤ 20%，上影线相对收盘价 ≤ 2% → 进场
                   平仓：持有动量仓 && (本根收盘 < 前收 OR 浮盈 ≥ 1.5%) && 本根未触及布林上轨 → 立刻平仓
      - 原有策略：与先前保持一致
      - 两者互斥，只能持有其中之一

    返回：
      - action: "BUY_MOM", "BUY_ORIG", "SELL_MOM", "SELL_TP", "SELL_3D" 或 None
      - price: 触发信号时 用的收盘价
    """
    # 行数不足则无操作
    if len(df) < 5:
        return None, None

    # last 对应上一根已收盘 4h K 线，prev 及 prev2、prev3 用于动量和原有判断
    last  = df.iloc[-2]
    prev  = df.iloc[-3]
    prev2 = df.iloc[-4]
    prev3 = df.iloc[-5]
    price = last["close"]

    # ========== 动量策略部分 ==========
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

    # ========== 原有策略部分 ==========
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

        # 1) 持仓浮盈 ≥1.5% 立即平仓
        if profitPctOrig >= 1.5:
            return "SELL_TP", price
        # 2) 持仓超过3天 且 回本时平仓
        if (last_time >= entry_time + timedelta(days=3)) and (price >= entry_price):
            return "SELL_3D", price

    if alert_orig:
        return "BUY_ORIG", price

    return None, None

# ===============================
# 7. 下单函数（Spot 市价单）
# ===============================
def place_order_spot(action: str, symbol: str, quantity: float):
    """
    action: "BUY" 或 "SELL"
    quantity: 现货下单数量（base asset 数量，例如 ADA）
    这里示例都用市价单。
    """
    try:
        if action == "BUY":
            order = client.order_market_buy(
                symbol=symbol,
                quantity=quantity
            )
        else:  # SELL
            order = client.order_market_sell(
                symbol=symbol,
                quantity=quantity
            )
        print(f"{datetime.now()} 下单成功：{action}, 订单 ID: {order.get('orderId', order.get('clientOrderId', 'N/A'))}")
        return order
    except BinanceAPIException as e:
        print(f"{datetime.now()} 下单失败：{e}")
        return None

# ===============================
# 8. 计算现货下单数量（按 USDT 余额）
# ===============================
def calc_quantity(usdt_amount: float, price: float):
    """
    计算在当前价格下，用 usdt_amount（USDT）可以在现货市场买多少 base asset（ADA）。
    先按比例算出数量，然后向下取整保留到 0.001 精度。
    """
    step_size = 0.001
    qty = usdt_amount / price
    qty = math.floor(qty / step_size) * step_size
    return float(format(qty, f".3f"))

# ===============================
# 9. 同步本地状态 vs. Spot 账户真实持仓
# ===============================
def sync_position_with_api_spot(state: dict):
    """
    如果本地 state["inPosition"] == True，就去 Binance Spot API 查询余额，
    如果该资产（BASE_ASSET）可用余额（free） == 0，则说明实际已无持仓，需把本地 state 清空。
    """
    if not state.get("inPosition", False):
        return state

    try:
        balance = client.get_asset_balance(asset=BASE_ASSET)
        free_amount = float(balance.get("free", 0.0))
        if free_amount == 0.0:
            print(f"{datetime.now()} 检测到现货持仓为0 → 重置本地状态。")
            state["inPosition"] = False
            state["strategy"]   = None
            state["entryPrice"] = None
            state["entryTime"]  = None
            state["entryQty"]   = None
            save_state(state)
    except BinanceAPIException as e:
        print(f"{datetime.now()} 同步现货持仓时出错：{e}")

    return state

# ===============================
# 10. 主流程
# ===============================
def main():
    # 10.1 加载持仓状态
    state = load_state()

    # 10.2 同步本地 state 与现货账户真实持仓状态
    state = sync_position_with_api_spot(state)

    # 10.3 拉取数据、计算指标
    df = fetch_klines(SYMBOL, INTERVAL, limit=LIMIT)
    df = compute_indicators(df)

    action, price = check_signals(df, state)

    # —— 调试打印：最近 5 根 4h K 线与指标 —— #
    print("\n===== 最近 5 根 4h K 线与指标 =====")
    cols_to_show = [
        "open_time", "open", "high", "low", "close", "volume",
        "k", "d", "j", "lowerBB", "upperBB", "redBarCount", "barDrop",
        "brokenBB", "vol_increasing", "vol_over_20m"
    ]
    print(df[cols_to_show].tail(5).to_string(index=False))
    print("====================================\n")

    # ============================
    # 10.4 根据信号下单（Spot）
    # ============================
    if action == "BUY_MOM":
        usdt_amount = 200
        qty = calc_quantity(usdt_amount, price)
        order = place_order_spot("BUY", SYMBOL, qty)
        if order:
            # Spot 买单返回的 fills 可以用来计算平均价格
            fills = order.get("fills", [])
            if fills:
                total_qty = sum(float(f["qty"]) for f in fills)
                avg_price = sum(float(f["price"]) * float(f["qty"]) for f in fills) / total_qty
            else:
                total_qty = qty
                avg_price = price

            state["inPosition"] = True
            state["strategy"]   = "mom"
            state["entryPrice"] = avg_price
            state["entryQty"]   = total_qty
            last_time = df.iloc[-2]["open_time"]
            state["entryTime"] = last_time.replace(tzinfo=timezone.utc).isoformat()
            print(f"{datetime.now()} 记录动量仓（现货）：入场价 {avg_price:.6f}, 数量 {total_qty:.3f}, 时间 {state['entryTime']}")

    elif action == "BUY_ORIG":
        usdt_amount = 200
        qty = calc_quantity(usdt_amount, price)
        order = place_order_spot("BUY", SYMBOL, qty)
        if order:
            fills = order.get("fills", [])
            if fills:
                total_qty = sum(float(f["qty"]) for f in fills)
                avg_price = sum(float(f["price"]) * float(f["qty"]) for f in fills) / total_qty
            else:
                total_qty = qty
                avg_price = price

            state["inPosition"] = True
            state["strategy"]   = "orig"
            state["entryPrice"] = avg_price
            state["entryQty"]   = total_qty
            last_time = df.iloc[-2]["open_time"]
            state["entryTime"] = last_time.replace(tzinfo=timezone.utc).isoformat()
            print(f"{datetime.now()} 记录原有仓（现货）：入场价 {avg_price:.6f}, 数量 {total_qty:.3f}, 时间 {state['entryTime']}")

    elif action == "SELL_MOM":
        qty = state.get("entryQty")
        if qty is None:
            print(f"{datetime.now()} 错误：找不到 entryQty，无法平动量仓（现货）")
        else:
            order = place_order_spot("SELL", SYMBOL, qty)
            if order:
                fills = order.get("fills", [])
                if fills:
                    total_qty = sum(float(f["qty"]) for f in fills)
                    avg_price = sum(float(f["price"]) * float(f["qty"]) for f in fills) / total_qty
                else:
                    total_qty = qty
                    avg_price = price

                print(f"{datetime.now()} 动量仓（现货）平仓价 {avg_price:.6f}, 数量 {total_qty:.3f}")
                state["inPosition"] = False
                state["strategy"]   = None
                state["entryPrice"] = None
                state["entryTime"]  = None
                state["entryQty"]   = None

    elif action == "SELL_TP" or action == "SELL_3D":
        qty = state.get("entryQty")
        if qty is None:
            print(f"{datetime.now()} 错误：找不到 entryQty，无法平原有仓（现货）")
        else:
            order = place_order_spot("SELL", SYMBOL, qty)
            if order:
                fills = order.get("fills", [])
                if fills:
                    total_qty = sum(float(f["qty"]) for f in fills)
                    avg_price = sum(float(f["price"]) * float(f["qty"]) for f in fills) / total_qty
                else:
                    total_qty = qty
                    avg_price = price

                print(f"{datetime.now()} 原有仓（现货）平仓价 {avg_price:.6f}, 数量 {total_qty:.3f}, 动作 {action}")
                state["inPosition"] = False
                state["strategy"]   = None
                state["entryPrice"] = None
                state["entryTime"]  = None
                state["entryQty"]   = None

    else:
        print(f"{datetime.now()} 无操作信号。当前持仓状态：{state}")

    # 10.5 把 state 写回文件
    save_state(state)

# ===============================
# 11. (可选) 测试函数：Spot 市价单买入/卖出示例
# ===============================
def test_trade_spot():
    """
    在 Spot 账户上执行：
    1) 市价买入 1 ADA（示例，总价近 1 * ADA 当前价格）
    2) 等待 3 秒
    3) 市价卖出 1 ADA（平仓）
    """
    try:
        # 11.1 获取市场最新价格
        ticker = client.get_symbol_ticker(symbol=SYMBOL)
        price = float(ticker["price"])
        qty = calc_quantity(1 * price, price)  # 假设用 1 USDT 买入 ADA
        # 11.2 市价买入
        buy_order = client.order_market_buy(symbol=SYMBOL, quantity=qty)
        print(f"{datetime.now()} Spot 下单买入 ADA，订单信息：{buy_order}")
        # 11.3 等待 3 秒
        time.sleep(3)
        # 11.4 市价卖出同样的 qty
        sell_order = client.order_market_sell(symbol=SYMBOL, quantity=qty)
        print(f"{datetime.now()} Spot 下单卖出 ADA，订单信息：{sell_order}")

    except BinanceAPIException as e:
        print(f"{datetime.now()} Spot 测试下单失败：{e}")

if __name__ == "__main__":
    main()

    # —— 如果想单独测试 Spot 下单，可以取消下面两行注释 —— #
    # print("\n===== 开始测试：Spot 市价买入 1 ADA，3 秒后卖出 =====")
    # test_trade_spot()

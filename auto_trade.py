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
            "entryTime": None,       # 持仓时的时间戳（UTC，ISO 格式）
            "entryPrice": None,      # 持仓时的实际成交均价
            "entryQty": None         # 持仓时的实际成交数量（合约张数或现货币数）
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
    # 1. 计算 Stochastic %K, %D, %J
    low_min = df["low"].rolling(window=14).min()
    high_max = df["high"].rolling(window=14).max()
    df["k"] = 100 * (df["close"] - low_min) / (high_max - low_min)
    df["d"] = df["k"].rolling(window=3).mean()
    df["j"] = 3 * df["k"] - 2 * df["d"]

    # 2. 计算 Bollinger Band 下轨 (总体标准差 ddof=0)
    basis = df["close"].rolling(window=20).mean()
    dev = df["close"].rolling(window=20).std(ddof=0)
    df["lowerBB"] = basis - 2 * dev

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


# ========== 6. 判断进场／出场信号 ==========
def check_signals(df: pd.DataFrame, state: dict):
    """
    完全参照你给的 Pine Script：
      - 用 “倒数第2 行” 作为 last（上一根已收盘 4h K 线）
      - 用 “倒数第3 行” 作为 prev（再前一根已收盘 4h K 线）
      - 严格沿用 Pine Script 里所有条件（j<=15、跌破布林带、量增、当根量＞20M、8/10 红棒、bar_drop）
      - 平仓：获利 ≥ 1.5% → 立即；或者持仓超过 72 根（约 3 天）且 “回本（close == entryPrice）” 才平
    返回：
      - action: "BUY" / "SELL_TP" / "SELL_3D" / None
      - price: 触发信号时 用的收盘价
    """
    # 确保行数足够：至少要 3 根 K 线才能取到 df.iloc[-3]
    if len(df) < 3:
        return None, None

    # —— 对应 Pine Script 的 last = df.iloc[-2]，prev = df.iloc[-3] —— #
    last = df.iloc[-2]
    prev = df.iloc[-3]
    price = last["close"]

    # —— 计算 Stochastic、Bollinger、红棒等 —— #
    j        = last["j"]
    j_prev   = prev["j"]
    brokenBB = last["brokenBB"]
    vol_inc  = last["vol_increasing"]
    vol_20m  = last["vol_over_20m"]

    # 条件 A： j ≤ 15 且 跌破布林带 且 量增 且 j 向上 且 当根量 > 20M
    cond_original = (j <= 15) and brokenBB and vol_inc and (j > j_prev) and vol_20m

    # 条件 B： 当根“开 → 收”跌幅 ≥ 7%
    bar_drop = last["barDrop"]  # 已在 compute_indicators 里算好

    # 条件 C： 最近 10 根中 ≥ 8 根为红棒（close < open）
    #       DataFrame 已在 compute_indicators 里有 redBarCount = 最近 10 根红棒数
    cond_8_of_10_red = last["redBarCount"] >= 8

    # 合并：A or B or C
    condition = cond_original or bar_drop or cond_8_of_10_red

    # alert_on_next_bar = condition and not condition[1]  —— 这里的 “condition[1]” 对应 prev 的 condition 标志
    # 但我们没有把 condition 存进 df，所以下面手动复刻：上一根 prev 的 condition_prev
    # 所以先算 prev_condition：
    j2        = prev["j"]
    brokenBB2 = prev["brokenBB"]
    vol_inc2  = prev["vol_increasing"]
    vol_202m  = prev["vol_over_20m"]
    cond_orig_prev = (j2 <= 15) and brokenBB2 and vol_inc2 and (j2 > df.iloc[-4]["j"]) and vol_202m \
                     if len(df) >= 4 else False
    bar_drop_prev = prev["barDrop"]
    cond_red_prev = prev["redBarCount"] >= 8
    condition_prev = cond_orig_prev or bar_drop_prev or cond_red_prev

    alert_on_next_bar = condition and (not condition_prev)

    # —— 进场逻辑 —— #
    if (not state.get("inPosition", False)) and alert_on_next_bar:
        # BUY：用上一根 K 线的收盘价作买入价
        return "BUY", price

    # —— 平仓逻辑 —— #
    if state.get("inPosition", False):
        entry_price = state["entryPrice"]
        # Pine Script 里 entryBar = bar_index，当时存的是 index，所以此处只取时间
        # 但我们 Python 版只关心“持仓条数” -> 可用 K 线数来算是否 72 根
        # 所以如果要判断“持仓超过 72 根”可以设定：假定 state 里存了 entry_bar_index 或 entry_time
        #
        # 为了跟你原来的逻辑保持一致，这里直接取“上一根 K 线的 open_time”与 “entryTime” 对比
        last_time = last["open_time"]  # 上一根 K 线的开盘时间（UTC）
        entry_time = datetime.fromisoformat(state["entryTime"])

        # 4-1. 如果（last.close - entryPrice）/ entryPrice * 100 ≥ 1.5% → SELL_TP
        profit_pct = (price - entry_price) / entry_price * 100
        if profit_pct >= 1.5:
            return "SELL_TP", price

        # 4-2. 如果持仓超过 72 根 4h K 线（也是约 3 天），且 “回本（last.close == entryPrice）” 才平仓
        #     Pine Script 用的是 bar_index 计数，但我们用时间也可等效：用 entryTime + 3 天判断
        if (last_time >= entry_time + timedelta(days=3)) and (price >= entry_price):
            return "SELL_3D", price

    return None, None

# ========== 7. 下单函数（针对永续合约） ==========
def place_order(action: str, symbol: str, quantity: float):
    """
    action: "BUY" 或 "SELL"
    quantity: 合约下单张数（USDT-M 永续合约）
    这里示例都用市价单，如果要改成限价或其他类型，请自行修改 type/price 参数。
    """
    try:
        if action == "BUY":
            order = client.futures_create_order(
                symbol=symbol,
                side="BUY",
                type="MARKET",
                quantity=quantity
            )
        else:  # SELL
            order = client.futures_create_order(
                symbol=symbol,
                side="SELL",
                type="MARKET",
                quantity=quantity
            )
        print(f"{datetime.now()} 下单成功：{action}, 订单 ID: {order['orderId']}")
        return order
    except BinanceAPIException as e:
        print(f"{datetime.now()} 下单失败：{e}")
        return None

# ========== 8. 计算合约下单张数（按 USDT 余额） ===========
def calc_quantity(usdt_amount: float, price: float):
    """
    计算在当前价格下，用 usdt_amount（USDT）可以在永续合约买多少张 ADA。
    先按比例算出数量，然后向下取整保留到 0.001 张。
    """
    step_size = 0.001
    qty = usdt_amount / price
    qty = math.floor(qty / step_size) * step_size
    return float(format(qty, f".3f"))

# ========== 9. 检查本地状态 vs. 永续合约真实持仓 ==========
def sync_position_with_api(state: dict):
    """
    如果本地 state["inPosition"] == True，就去 Binance API 查询
    永续合约仓位（futures position），如果发现该 symbol 在 币安合约
    账户中 positionAmt == 0，则说明实际已无持仓，需把本地 state 清空。
    """
    if not state.get("inPosition", False):
        # 本地没持仓，直接返回
        return state

    try:
        # 下面 API: 获取该 symbol 在永续合约账户的持仓信息
        positions = client.futures_position_information(symbol=SYMBOL)
        # positions 是一个列表，每个元素里有 "positionAmt" 表示该仓位
        # 取出 positionAmt
        actual_amt = 0.0
        for pos in positions:
            # pos["symbol"] 里就应该是 "ADAUSDT"
            if pos.get("symbol") == SYMBOL:
                actual_amt = float(pos.get("positionAmt", 0))
                break

        if actual_amt == 0.0:
            # 本地 state 标示 inPosition，但 API 读出来是 0 => 实际已爆仓或平仓
            print(f"{datetime.now()} 检测到本地 inPosition=True，但合约仓位为 0 → 重置本地状态。")
            state["inPosition"] = False
            state["entryPrice"] = None
            state["entryTime"] = None
            state["entryQty"] = None
            save_state(state)  # 直接把清空后的 state 持久化
    except BinanceAPIException as e:
        print(f"{datetime.now()} 同步仓位时出错：{e}")

    return state

# ========== 10. 主流程 ==========
def main():
    # 10.1 加载持仓状态
    state = load_state()

    # 10.2 同步本地 state 与币安永续合约真实持仓状态
    state = sync_position_with_api(state)

    # 10.3 拉取数据、计算指标
    df = fetch_klines(SYMBOL, INTERVAL, limit=LIMIT)
    df = compute_indicators(df)

    action, price = check_signals(df, state)

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

    # 10.4 判断信号
    

    if action == "BUY":
        usdt_amount = 200
        # 10.4.1 计算买入张数
        qty = calc_quantity(usdt_amount, price)

        # 10.4.2 直接在永续合约下市价单买入
        order = place_order("BUY", SYMBOL, qty)
        if order:
            # 从 order["fills"] 中提取实际成交数量与加权均价
            fills = order.get("fills", [])
            total_qty = sum(float(f["qty"]) for f in fills) if fills else qty
            if fills:
                avg_price = sum(float(f["price"]) * float(f["qty"]) for f in fills) / total_qty
            else:
                avg_price = price

            # 更新状态：记录 inPosition、entryPrice、entryQty 及 entryTime
            state["inPosition"] = True
            state["entryPrice"] = avg_price
            state["entryQty"] = total_qty
            last_time = df.iloc[-2]["open_time"]
            state["entryTime"] = last_time.replace(tzinfo=timezone.utc).isoformat()
            print(
                f"{datetime.now()} 记录持仓状态：实际入场价 {avg_price:.6f}，入场数量 {total_qty:.3f}，时间 {state['entryTime']}"
            )

    elif action in ("SELL_TP", "SELL_3D"):
        # 10.4.3 平仓：直接用保存的 entryQty
        qty = state.get("entryQty")
        if qty is None:
            print(f"{datetime.now()} 错误：找不到 entryQty，无法平仓")
        else:
            order = place_order("SELL", SYMBOL, qty)
            if order:
                fills = order.get("fills", [])
                total_qty = sum(float(f["qty"]) for f in fills) if fills else qty
                if fills:
                    avg_price = sum(float(f["price"]) * float(f["qty"]) for f in fills) / total_qty
                else:
                    avg_price = None

                print(f"{datetime.now()} 实际平仓价 {avg_price:.6f}，平仓数量 {total_qty:.3f}，动作 {action}")
                # 清空状态
                state["inPosition"] = False
                state["entryPrice"] = None
                state["entryTime"] = None
                state["entryQty"] = None

    else:
        print(f"{datetime.now()} 无操作信号。当前持仓状态：{state}")

    # 10.5 把 state 写回文件
    save_state(state)

# ========== 11. 测试函数：市价单买入 20 张 ADA 永续合约，3 秒后卖出 ==========
def test_trade_futures():
    """
    在 USDT-M 永续合约账户上执行：
    1) 设置 ADAUSDT 永续合约杠杆为 3 倍
    2) 市价买入 20 张 ADAUSDT 合约
    3) 等待 3 秒
    4) 市价卖出 20 张（平仓）
    """
    try:
        # 11.1 设置杠杆为 3 倍
        leverage_resp = client.futures_change_leverage(symbol=SYMBOL, leverage=3)
        print(f"{datetime.now()} 杠杆设置响应：{leverage_resp}")

        # 11.2 市价买入 20 张 ADAUSDT 永续合约
        buy_order = client.futures_create_order(
            symbol=SYMBOL,
            side="BUY",
            type="MARKET",
            quantity=20
        )
        print(f"{datetime.now()} 下单买入 20 张 ADA 合约，订单信息：{buy_order}")

        # 11.3 等待 3 秒
        time.sleep(3)

        # 11.4 市价卖出 20 张进行平仓
        sell_order = client.futures_create_order(
            symbol=SYMBOL,
            side="SELL",
            type="MARKET",
            quantity=20
        )
        print(f"{datetime.now()} 下单卖出 20 张 ADA 合约，订单信息：{sell_order}")

    except BinanceAPIException as e:
        print(f"{datetime.now()} 测试下单失败：{e}")

if __name__ == "__main__":
    # 运行主流程
    main()

    # —— 如果需要自己测试合约下单，可以取消下面两行注释 —— #
    # print("\n===== 开始测试：合约账户市价买入20 ADA，3 秒后卖出 =====")
    # test_trade_futures()

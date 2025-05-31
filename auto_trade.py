# auto_trade.py

import os
import json
import pandas as pd
import numpy as np

from pybit.unified_trading import HTTP
from pybit.exceptions import FailedRequestError

# ----------------------------
# 1. 从环境变量读取 Bybit API Key/Secret
# ----------------------------
BYBIT_API_KEY    = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")
if not BYBIT_API_KEY or not BYBIT_API_SECRET:
    raise ValueError("请先在环境变量设定 BYBIT_API_KEY 和 BYBIT_API_SECRET")

# ----------------------------
# 2. 用新版 Pybit v5 建立 HTTP 客户端 (USDT 永续合约)
#    如果要跑 Testnet，把 testnet=True
# ----------------------------
client = HTTP(
    testnet=False,               # 正式网：False；要用测试网就改成 True
    api_key    = BYBIT_API_KEY,
    api_secret = BYBIT_API_SECRET
)

# 2.1 尝试设置杠杆为 3 倍，如果出错（IP 限制、API 权限等），捕获异常并继续
try:
    client.set_leverage(symbol="ADAUSDT", leverage=3)
    print("杠杆设置成功：ADAUSDT = 3 倍")
except FailedRequestError as e:
    print(f"Warning: 设置杠杆时发生错误，已跳过。错误信息：{e}")

# ----------------------------
# 3. 参数设置
# ----------------------------
SYMBOL      = "ADAUSDT"   # 交易对：ADA/USDT 永续合约
TIMEFRAME   = "60"        # K 线周期：60 = 60 分钟 (1h)
FETCH_LIMIT = 200         # 拉取最近 200 根 60 分钟 K 线

# 下单时要动态计算 “可用余额 × 杠杆 ÷ 标记价” 得出合约数
LOT_SIZE    = None        

# Stochastic 参数（Bar 数量）
STOCH_K     = 14
STOCH_D     = 3

# Bollinger Bands 参数（Bar 数量、标准差倍数）
BB_LENGTH   = 20
BB_MULT     = 2.0

# ----------------------------
# 4. 持仓状态保存 (state.json)
# ----------------------------
STATE_FILENAME = "state.json"

def load_state():
    if os.path.isfile(STATE_FILENAME):
        with open(STATE_FILENAME, "r") as f:
            return json.load(f)
    else:
        return {"in_position": False, "entry_price": None, "entry_time": None}

def save_state(state):
    with open(STATE_FILENAME, "w") as f:
        json.dump(state, f)

# ----------------------------
# 5. 拉 K 线并转换成 pandas.DataFrame
# ----------------------------
def fetch_klines(symbol, interval, limit=200):
    """
    Bybit Unified Trading REST API query_kline → 返回 DataFrame:
      index: open_time (pd.Timestamp)
      columns: open, high, low, close, volume
    """
    resp = client.get_kline(
        symbol   = symbol,
        interval = interval,
        limit    = limit,
        _unified = True
    )
    if "result" not in resp or not isinstance(resp["result"], list):
        raise RuntimeError(f"query_kline 返回异常: {resp}")
    data = resp["result"]
    df = pd.DataFrame(data)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df.set_index("open_time")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df

# ----------------------------
# 6. 计算可用余额并返回可下单合约数
# ----------------------------
def calculate_lot_size(symbol, leverage):
    """
    1. 从钱包余额 API 拿到可用 USDT
    2. 用最近一根 K 线收盘价当作“标记价”近似
    3. 公式：lot_count = floor(available_usdt * leverage / mark_price)
    """
    balances = client.get_wallet_balance(coin="USDT")
    if "result" not in balances or "USDT" not in balances["result"]:
        raise RuntimeError(f"get_wallet_balance 返回异常: {balances}")
    available_usdt = float(balances["result"]["USDT"]["available_balance"])

    # 拿最近 5 根 60 分 K 线，最后一根收盘价当作标记价
    klines = fetch_klines(symbol, TIMEFRAME, limit=5)
    mark_price = klines["close"].iloc[-1]

    raw_size = (available_usdt * leverage) / mark_price
    # ADAUSDT 永续合约一般精度到小数后 3 位
    size = float(f"{raw_size:.3f}")
    return size

# ----------------------------
# 7. 用 pandas + numpy 手动计算 Stochastic & Bollinger Bands
# ----------------------------
def compute_indicators(df):
    """
    输入 df：必须包含 open, high, low, close, volume
    输出 df：新增 stoch_k, stoch_d, stoch_j, bb_lower
    """
    # 7.1 %K = (CLOSE - N 期内最低价) / (N 期内最高价 - N 期内最低价) * 100
    low_n   = df["low"].rolling(window=STOCH_K).min()
    high_n  = df["high"].rolling(window=STOCH_K).max()
    stoch_k = (df["close"] - low_n) / (high_n - low_n) * 100

    # 7.2 %D = %K 的 3 期简单移动平均
    stoch_d = stoch_k.rolling(window=STOCH_D).mean()

    # 7.3 %J = 3 * %K - 2 * %D
    stoch_j = 3 * stoch_k - 2 * stoch_d

    # 7.4 Bollinger Band 下轨 = 20 期收盘价的均线 - 2 * 20 期收盘价的标准差
    ma20    = df["close"].rolling(window=BB_LENGTH).mean()
    std20   = df["close"].rolling(window=BB_LENGTH).std()
    bb_lower= ma20 - BB_MULT * std20

    df_ = df.copy()
    df_["stoch_k"]  = stoch_k
    df_["stoch_d"]  = stoch_d
    df_["stoch_j"]  = stoch_j
    df_["bb_lower"] = bb_lower
    return df_

# ----------------------------
# 8. 核心策略逻辑：判断买入/卖出、控制持仓
# ----------------------------
def strategy_logic(df, state):
    in_position = state["in_position"]
    entry_price = state["entry_price"]
    entry_time  = state["entry_time"]

    # 如果 Bar 数量不足，就先不操作
    if len(df) < max(STOCH_K, BB_LENGTH) + 2:
        return state

    # 计算所有指标
    df_ind = compute_indicators(df)

    # 拿最后一根 K 线的数据
    last_idx   = df_ind.index[-1]
    last_open  = df_ind["open"].iloc[-1]
    last_close = df_ind["close"].iloc[-1]
    last_low   = df_ind["low"].iloc[-1]
    last_vol   = df_ind["volume"].iloc[-1]
    prev_vol   = df_ind["volume"].iloc[-2]
    prev_j     = df_ind["stoch_j"].iloc[-2]
    curr_j     = df_ind["stoch_j"].iloc[-1]
    lower_band = df_ind["bb_lower"].iloc[-1]

    # 8.1 原始买入条件：%J ≤ 15 且 跌破 BB 下轨、成交量放大、%J 向上、当根量 > 20M
    cond_j_low       = curr_j <= 15
    cond_broken_bb   = last_low < lower_band
    cond_vol_up      = last_vol > prev_vol
    cond_j_rising    = curr_j > prev_j
    cond_vol_over20m = last_vol > 20_000_000
    buy_cond         = cond_j_low and cond_broken_bb and cond_vol_up and cond_j_rising and cond_vol_over20m

    # 8.2 或当根 K 线跌幅 ≥ 7%，也算买入
    bar_drop = ((last_open - last_close) / last_open * 100) >= 7
    final_buy_cond = buy_cond or bar_drop

    size = None
    # 8.3 如果当前没持仓且满足买入条件 → 全仓 3 倍杠杆下多单
    if (not in_position) and final_buy_cond:
        size = calculate_lot_size(SYMBOL, leverage=3)
        print(f"[{last_idx}] 买入信号成立 → 杠杆×3 下单，合约数={size}, 价格≈{last_close}")
        resp = client.place_active_order(
            symbol           = SYMBOL,
            side             = "Buy",
            order_type       = "Market",
            qty              = size,
            time_in_force    = "GoodTillCancel",
            reduce_only      = False,
            close_on_trigger = False
        )
        print("多单下单响应：", resp)
        in_position = True
        entry_price = last_close
        entry_time  = last_idx.strftime("%Y-%m-%d %H:%M:%S")

    # 8.4 如果已经持仓，检查平仓条件
    elif in_position:
        # 如果上面买入时没算 size，就在这里再算一次
        if size is None:
            size = calculate_lot_size(SYMBOL, leverage=3)

        profit_pct = (last_close - entry_price) / entry_price * 100

        # (A) 如果盈利 ≥ 1.5% → 平仓
        if profit_pct >= 1.5:
            print(f"[{last_idx}] 盈利 {profit_pct:.2f}% ≥ 1.5%，市价平仓 Sell")
            resp = client.place_active_order(
                symbol           = SYMBOL,
                side             = "Sell",
                order_type       = "Market",
                qty              = size,
                time_in_force    = "GoodTillCancel",
                reduce_only      = False,
                close_on_trigger = False
            )
            print("空单平仓响应：", resp)
            in_position = False
            entry_price = None
            entry_time  = None

        # (B) 如果持仓 ≥ 3 天 (4320 分钟) 且收盘价回到进场价 → 平仓
        if entry_time:
            held_minutes = (pd.to_datetime(last_idx) - pd.to_datetime(entry_time)).total_seconds() / 60
            if held_minutes >= 4320 and abs(last_close - entry_price) < 1e-8:
                print(f"[{last_idx}] 持仓 {held_minutes/60:.1f} 小时 (>72 小时) 且回本 → 平仓")
                resp = client.place_active_order(
                    symbol           = SYMBOL,
                    side             = "Sell",
                    order_type       = "Market",
                    qty              = size,
                    time_in_force    = "GoodTillCancel",
                    reduce_only      = False,
                    close_on_trigger = False
                )
                print("3 天回本平仓响应：", resp)
                in_position = False
                entry_price = None
                entry_time  = None

    # 8.5 更新并返回最新持仓状态
    new_state = {
        "in_position": in_position,
        "entry_price": entry_price,
        "entry_time":  entry_time,
    }
    return new_state

# ----------------------------
# 9. 主程序：拉 K 线 → 执行策略 → 下单 → 更新 state.json
# ----------------------------
def main():
    state = load_state()
    try:
        df = fetch_klines(SYMBOL, TIMEFRAME, limit=FETCH_LIMIT)
        new_state = strategy_logic(df, state)
        if new_state != state:
            save_state(new_state)
    except Exception as e:
        print("执行过程中出错：", str(e))
        # 抛出异常让 GitHub Actions 返回非零退出码，以便看到 Failure
        raise

if __name__ == "__main__":
    main()

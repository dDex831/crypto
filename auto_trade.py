# auto_trade.py
import os
import json
import time
import pandas as pd
import pandas_ta as ta
from pybit import HTTP

# ----------------------------
# 1. 從環境變數讀取 Bybit API 金鑰
# ----------------------------
BYBIT_API_KEY    = os.getenv("JWvjtDVvOOMuBmKhxs")
BYBIT_API_SECRET = os.getenv("6JqBoMYOJuanbTlQqFf2gyeerK0L9qhiVgw6")
if not BYBIT_API_KEY or not BYBIT_API_SECRET:
    raise ValueError("請先在環境變數設置 BYBIT_API_KEY 和 BYBIT_API_SECRET")

# ----------------------------
# 2. 初始化 Bybit HTTP 客戶端 (正式環境)
#    若要測試請改成 endpoint='https://api-testnet.bybit.com'
# ----------------------------
client = HTTP(
    endpoint = "https://api.bybit.com",
    api_key    = BYBIT_API_KEY,
    api_secret = BYBIT_API_SECRET
)
# 2.1 设置杠杆为 3 倍（仅需执行一次或每次下单前再次确认）
client.set_leverage(symbol="ADAUSDT", leverage=3)

# ----------------------------
# 3. 參數設定
# ----------------------------
SYMBOL      = "ADAUSDT"   # 交易对：ADA/USDT 永续合约
TIMEFRAME   = "60"         # K 线周期：1 分钟
FETCH_LIMIT = 200         # 拉取最近 200 根 K 线

# LOT_SIZE 这里先占位，后面会动态计算
LOT_SIZE    = None    
# ----------------------------
# 4. 進場 / 持倉狀態儲存 (state.json)
#    用這份檔案記錄是否已持倉 & 進場價格，避免每次重啟都重新判斷
STOCH_K   = 14
STOCH_D   = 3
BB_LENGTH = 20
BB_MULT   = 2.0


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
# 5. 抓取 K 线并转成 pandas.DataFrame
# ----------------------------
def fetch_klines(symbol, interval, limit=200):
    resp = client.query_kline(
        symbol   = symbol,
        interval = interval,
        limit    = limit,
        _unified = True
    )
    data = resp["result"]
    df = pd.DataFrame(data)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df.set_index("open_time")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df

# ----------------------------
# 6. 计算可用余额并动态算出下单张数
# ----------------------------
def calculate_lot_size(symbol, leverage):
    """
    1. 先从钱包余额 API 拿到可用 USDT
    2. 最新标记价格：直接用合约本身的 '标记价格' 或 最后一根 K 线收盘价。这里用最新收盘价近似。
    3. 公式：lot = floor(available_usdt × leverage / mark_price)
    """
    # 6.1 拿到 USDT 可用余额 (wallet only)
    balances = client.get_wallet_balance(coin="USDT")
    available_usdt = float(balances["result"]["USDT"]["available_balance"])

    # 6.2 拿最新一根 K 线的收盘价当作标记价格 (也可以调用 client.latest_information_for_symbol 来拿 mark_price)
    klines = fetch_klines(symbol, TIMEFRAME, limit=5)
    mark_price = klines["close"].iloc[-1]

    # 6.3 理论最大下单合约数量
    raw_size = (available_usdt * leverage) / mark_price

    # 6.4 Bybit 永续合约对 ADAUSDT 是 0.001 合约精度 (视情况调整)
    #     可以调用 client.query_symbol 或手动查看 stepSize。这里假设量级精确到小数点后 3 位。
    size = float(f"{raw_size:.3f}")

    return size

# ----------------------------
# 7. 核心策略逻辑 (与之前相同，只是下单前先把 LOT_SIZE 动态算好)
# ----------------------------
def strategy_logic(df, state):
    in_position = state["in_position"]
    entry_price = state["entry_price"]
    entry_time  = state["entry_time"]

    # 7.1 计算指标
    stoch = ta.stoch(df["high"], df["low"], df["close"],
                     length=STOCH_K, k=STOCH_K, d=STOCH_D)
    df["stoch_k"] = stoch[f"STOCHk_{STOCH_K}_{STOCH_D}"]
    df["stoch_d"] = stoch[f"STOCHd_{STOCH_K}_{STOCH_D}"]
    df["stoch_j"] = 3 * df["stoch_k"] - 2 * df["stoch_d"]
    bb = ta.bbands(df["close"], length=BB_LENGTH, std=BB_MULT)
    df["bb_lower"] = bb[f"BBL_{BB_LENGTH}_{BB_MULT}_2.0"]

    # 7.2 取最后一根 K 线数据
    last_idx   = df.index[-1]
    last_open  = df["open"].iloc[-1]
    last_close = df["close"].iloc[-1]
    last_low   = df["low"].iloc[-1]
    last_vol   = df["volume"].iloc[-1]
    prev_vol   = df["volume"].iloc[-2]
    prev_j     = df["stoch_j"].iloc[-2]
    curr_j     = df["stoch_j"].iloc[-1]
    lower_band = df["bb_lower"].iloc[-1]

    cond_j_low       = curr_j <= 15
    cond_broken_bb   = last_low < lower_band
    cond_vol_up      = last_vol > prev_vol
    cond_j_rising    = curr_j > prev_j
    cond_vol_over20m = last_vol > 20_000_000
    buy_cond         = (cond_j_low and cond_broken_bb and
                        cond_vol_up and cond_j_rising and cond_vol_over20m)

    bar_drop = ((last_open - last_close) / last_open * 100) >= 7
    final_buy_cond = buy_cond or bar_drop

    # 7.3 如果没持仓 & 满足买入条件 → “全部可用余额”按 3×杠杆下单
    if (not in_position) and final_buy_cond:
        # 计算此刻允许的最大开仓张数
        size = calculate_lot_size(SYMBOL, leverage=3)

        print(f"[{last_idx}] 买入信号成立，杠杆=3 倍，下单数量：{size} 合约 (市价多单，近似价格={last_close})")
        resp = client.place_active_order(
            symbol           = SYMBOL,
            side             = "Buy",
            order_type       = "Market",
            qty              = size,
            time_in_force    = "GoodTillCancel",
            reduce_only      = False,
            close_on_trigger = False
        )
        print("多单响应：", resp)

        in_position = True
        entry_price = last_close
        entry_time  = last_idx.strftime("%Y-%m-%d %H:%M:%S")

    # 7.4 如果已持仓，检查平仓条件
    elif in_position:
        profit_pct = (last_close - entry_price) / entry_price * 100

        # 平仓条件 1：盈利 ≥ 1.5%
        if profit_pct >= 1.5:
            print(f"[{last_idx}] 盈利 {profit_pct:.2f}% ≥ 1.5%，市价平仓 Sell")
            resp = client.place_active_order(
                symbol           = SYMBOL,
                side             = "Sell",
                order_type       = "Market",
                qty              = size,  # 上次下入的 size
                time_in_force    = "GoodTillCancel",
                reduce_only      = False,
                close_on_trigger = False
            )
            print("空单响应：", resp)
            in_position = False
            entry_price = None
            entry_time  = None

        # 平仓条件 2：持仓 ≥ 3 天且回本
        if entry_time:
            held_minutes = (pd.to_datetime(last_idx) -
                            pd.to_datetime(entry_time)
                           ).total_seconds() / 60
            if held_minutes >= 4320 and abs(last_close - entry_price) < 1e-8:
                print(f"[{last_idx}] 持仓 {held_minutes/60:.1f} 小时 (>72 小时) 且回本 → 市价平仓 Sell")
                resp = client.place_active_order(
                    symbol           = SYMBOL,
                    side             = "Sell",
                    order_type       = "Market",
                    qty              = size,
                    time_in_force    = "GoodTillCancel",
                    reduce_only      = False,
                    close_on_trigger = False
                )
                print("3 天回本空单响应：", resp)
                in_position = False
                entry_price = None
                entry_time  = None

    # 7.5 更新持仓状态并返回
    new_state = {
        "in_position": in_position,
        "entry_price": entry_price,
        "entry_time":  entry_time
    }
    return new_state

# ----------------------------
# 8. 主程序：拉 K 线 → 运行策略 → 下单 → 更新 state.json
# ----------------------------
def main():
    state = load_state()

    try:
        df = fetch_klines(SYMBOL, TIMEFRAME, limit=FETCH_LIMIT)
        new_state = strategy_logic(df, state)
        if new_state != state:
            save_state(new_state)
    except Exception as e:
        print("执行出错：", str(e))

if __name__ == "__main__":
    main()
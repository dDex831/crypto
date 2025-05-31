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
BYBIT_API_KEY    = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")
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

# ----------------------------
# 3. 參數設定
# ----------------------------
SYMBOL        = "BTCUSDT"   # 交易對
TIMEFRAME     = "1"         # K 線週期 "1" = 1 分鐘，可改成 "5"、"15"、"60"...
FETCH_LIMIT   = 200         # 最多抓 200 根 K 線，足夠計算各種指標
LOT_SIZE      = 0.001       # 每次下單數量 (請依你的策略與資金調整)
STOCH_K       = 14
STOCH_D       = 3
BB_LENGTH     = 20
BB_MULT       = 2.0

# ----------------------------
# 4. 進場 / 持倉狀態儲存 (state.json)
#    用這份檔案記錄是否已持倉 & 進場價格，避免每次重啟都重新判斷
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
# 5. 抓取 K 線並轉成 pandas.DataFrame
# ----------------------------
def fetch_klines(symbol, interval, limit=200):
    """
    Bybit REST API query_kline → 回傳 DataFrame:
      index: open_time (pd.Timestamp)
      columns: open, high, low, close, volume
    """
    resp = client.query_kline(
        symbol = symbol,
        interval = interval,
        limit = limit,
        _unified = True
    )
    data = resp["result"]
    df = pd.DataFrame(data)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df.set_index("open_time")
    # cast to float
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df

# ----------------------------
# 6. 核心策略邏輯
# ----------------------------
def strategy_logic(df, state):
    """
    輸入:
      df: 包含最近若干根 K 線 (index=open_time, open, high, low, close, volume)
      state: {"in_position": bool, "entry_price": float, "entry_time": str}
    輸出:
      new_state: 更新後的 state dict
    """
    in_position = state["in_position"]
    entry_price = state["entry_price"]
    entry_time  = state["entry_time"]

    # 1. 計算 Stochastic %K/%D/%J
    stoch = ta.stoch(df["high"], df["low"], df["close"], length=STOCH_K, k=STOCH_K, d=STOCH_D)
    df["stoch_k"] = stoch[f"STOCHk_{STOCH_K}_{STOCH_D}"]
    df["stoch_d"] = stoch[f"STOCHd_{STOCH_K}_{STOCH_D}"]
    df["stoch_j"] = 3 * df["stoch_k"] - 2 * df["stoch_d"]

    # 2. 計算布林下緣 (Lower Band)
    bb = ta.bbands(df["close"], length=BB_LENGTH, std=BB_MULT)
    df["bb_lower"] = bb[f"BBL_{BB_LENGTH}_{BB_MULT}_2.0"]

    # 3. 判斷買入條件 (最後一根 K 線)
    last_idx     = df.index[-1]
    last_open    = df["open"].iloc[-1]
    last_close   = df["close"].iloc[-1]
    last_low     = df["low"].iloc[-1]
    last_volume  = df["volume"].iloc[-1]
    prev_volume  = df["volume"].iloc[-2]
    prev_j       = df["stoch_j"].iloc[-2]
    curr_j       = df["stoch_j"].iloc[-1]
    lower_band   = df["bb_lower"].iloc[-1]

    cond_j_low       = curr_j <= 15
    cond_broken_bb   = last_low < lower_band
    cond_vol_up      = last_volume > prev_volume
    cond_j_rising    = curr_j > prev_j
    cond_vol_over20m = last_volume > 20_000_000
    buy_cond         = cond_j_low and cond_broken_bb and cond_vol_up and cond_j_rising and cond_vol_over20m

    # 4. 當根 K 線跌幅 ≥ 7% 也視為買入
    bar_drop = ((last_open - last_close) / last_open * 100) >= 7

    final_buy_cond = buy_cond or bar_drop

    # 5. 如果目前沒持倉 & 符合買入則下多單
    if (not in_position) and final_buy_cond:
        print(f"[{last_idx}] 進場信號成立 → 用市價多單下單 (price={last_close})")
        # 下多單
        resp = client.place_active_order(
            symbol         = SYMBOL,
            side           = "Buy",
            order_type     = "Market",
            qty            = LOT_SIZE,
            time_in_force  = "GoodTillCancel",
            reduce_only    = False,
            close_on_trigger = False
        )
        print("多單回應：", resp)
        # 更新持倉狀態
        in_position = True
        entry_price = last_close
        entry_time  = last_idx.strftime("%Y-%m-%d %H:%M:%S")

    # 6. 如果已持倉，檢查平倉條件
    elif in_position:
        # 計算目前獲利百分比
        profit_pct = (last_close - entry_price) / entry_price * 100

        # 6-1. 獲利 ≥ 1.5% → 平倉 
        if profit_pct >= 1.5:
            print(f"[{last_idx}] 獲利 {profit_pct:.2f}% ≥ 1.5%，平倉 SELL")
            resp = client.place_active_order(
                symbol          = SYMBOL,
                side            = "Sell",
                order_type      = "Market",
                qty             = LOT_SIZE,
                time_in_force   = "GoodTillCancel",
                reduce_only     = False,
                close_on_trigger= False
            )
            print("空單回應：", resp)
            in_position = False
            entry_price = None
            entry_time  = None

        # 6-2. 持倉超過 3 天 (4320 根 1 分鐘 K 線, 約 4320 分鐘) 且回本才平倉
        #     注意：最後一根 K 線時間戳 - entry_time timestamp
        if entry_time:
            held_minutes = (pd.to_datetime(last_idx) - pd.to_datetime(entry_time)).total_seconds() / 60
            if held_minutes >= 4320 and abs(last_close - entry_price) < 1e-8:
                print(f"[{last_idx}] 持倉 {held_minutes/60:.1f} 小時 (超過 72 小時) 且收盤回本 → 平倉")
                resp = client.place_active_order(
                    symbol          = SYMBOL,
                    side            = "Sell",
                    order_type      = "Market",
                    qty             = LOT_SIZE,
                    time_in_force   = "GoodTillCancel",
                    reduce_only     = False,
                    close_on_trigger= False
                )
                print("3 天回本空單回應：", resp)
                in_position = False
                entry_price = None
                entry_time  = None

    # 7. 更新 state 並返回
    new_state = {
        "in_position": in_position,
        "entry_price": entry_price,
        "entry_time":  entry_time
    }
    return new_state

# ----------------------------
# 7. 主程式：拉 K 線 → 計算 → 下單 → 更新 state (並存到 local 檔)
# ----------------------------
def main():
    # 讀取本地存檔
    state = load_state()

    try:
        # 1) 取得最近 K 線
        df = fetch_klines(SYMBOL, TIMEFRAME, limit=FETCH_LIMIT)

        # 2) 交由策略邏輯判斷並下單
        new_state = strategy_logic(df, state)

        # 3) 如果狀態改變就存檔
        if new_state != state:
            save_state(new_state)

    except Exception as e:
        print("執行過程出錯：", str(e))

# 直接執行
if __name__ == "__main__":
    main()

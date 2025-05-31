# auto_trade.py

import os
import json
import pandas as pd
import numpy as np

# ❶ 把下面這行改成「unified_trading」裡的 HTTP
from pybit.unified_trading import HTTP

# ----------------------------
# 1. 從環境變數讀取 Bybit API Key/Secret
# ----------------------------
BYBIT_API_KEY    = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")
if not BYBIT_API_KEY or not BYBIT_API_SECRET:
    raise ValueError("請先在環境變數設置 BYBIT_API_KEY 和 BYBIT_API_SECRET")

# ----------------------------
# 2. 初始化 Bybit HTTP 客戶端 (USDT 永續合約，正式環境)
#    如果想跑 Testnet，把 endpoint 改成 "https://api-testnet.bybit.com"
# ----------------------------
client = HTTP(
    api_key    = BYBIT_API_KEY,
    api_secret = BYBIT_API_SECRET,
    endpoint   = "https://api.bybit.com"
)

# 2.1 設定槓桿為 3 倍（只需呼叫一次就生效）
client.set_leverage(symbol="ADAUSDT", leverage=3)

# ----------------------------
# 3. 參數設定
# ----------------------------
SYMBOL      = "ADAUSDT"   # 交易對：ADA/USDT 永續合約
TIMEFRAME   = "60"        # K 線週期：60 = 60 分鐘 (1h)
FETCH_LIMIT = 200         # 拉取最近 200 根 60 分鐘 K 線

# 下單時會動態計算 LOT_SIZE（「可用餘額 × 槓桿 ÷ 標記價」），先留 None
LOT_SIZE    = None        

# Stochastic 參數（Bar 數量）
STOCH_K     = 14
STOCH_D     = 3

# Bollinger Band 參數（Bar 數量、標準差倍數）
BB_LENGTH   = 20
BB_MULT     = 2.0

# ----------------------------
# 4. 進場／持倉狀態儲存 (state.json)
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
    Bybit Unified Trading REST API query_kline → 返回 DataFrame:
      index: open_time (pd.Timestamp)
      columns: open, high, low, close, volume
    """
    # ❷ 這裡用的是 unified_trading.HTTP.query_kline
    resp = client.query_kline(
        symbol   = symbol,
        interval = interval,
        limit    = limit,
        _unified = True
    )
    if "result" not in resp or not isinstance(resp["result"], list):
        raise RuntimeError(f"query_kline 回傳格式異常: {resp}")
    data = resp["result"]
    df = pd.DataFrame(data)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df.set_index("open_time")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df

# ----------------------------
# 6. 計算可用餘額並動態算出下單合約數
# ----------------------------
def calculate_lot_size(symbol, leverage):
    """
    1. 先從錢包餘額 API 拿到可用 USDT
    2. 用最近一根 K 線收盤價當作「標記價」近似
    3. 公式：lot = floor(available_usdt × leverage / mark_price)
    """
    balances = client.get_wallet_balance(coin="USDT")
    if "result" not in balances or "USDT" not in balances["result"]:
        raise RuntimeError(f"get_wallet_balance 返回異常: {balances}")
    available_usdt = float(balances["result"]["USDT"]["available_balance"])

    # 再用最近 5 根 K 線收盤價當作標記價
    klines = fetch_klines(symbol, TIMEFRAME, limit=5)
    mark_price = klines["close"].iloc[-1]

    raw_size = (available_usdt * leverage) / mark_price
    # 對 ADAUSDT 永續合約，一般精度到小數點後 3 位
    size = float(f"{raw_size:.3f}")
    return size

# ----------------------------
# 7. 用純 pandas+numpy 手動計算 Stochastic 和 Bollinger Band
# ----------------------------
def compute_indicators(df):
    """
    輸入 df: 包含 open, high, low, close, volume
    輸出 df: 新增 stoch_k, stoch_d, stoch_j, bb_lower
    """
    # 7.1 %K = (CLOSE - N 期內最低價) / (N 期內最高價 - N 期內最低價) × 100
    low_n   = df["low"].rolling(window=STOCH_K).min()
    high_n  = df["high"].rolling(window=STOCH_K).max()
    stoch_k = (df["close"] - low_n) / (high_n - low_n) * 100

    # 7.2 %D = %K 的 3 期 SMA
    stoch_d = stoch_k.rolling(window=STOCH_D).mean()

    # 7.3 %J = 3×%K - 2×%D
    stoch_j = 3 * stoch_k - 2 * stoch_d

    # 7.4 Bollinger Band 下軌 = MA20 - 2×StdDev20
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
# 8. 核心策略邏輯
# ----------------------------
def strategy_logic(df, state):
    in_position = state["in_position"]
    entry_price = state["entry_price"]
    entry_time  = state["entry_time"]

    # 如果 Bar 數量不足，就先不下單
    if len(df) < max(STOCH_K, BB_LENGTH) + 2:
        return state

    # 計算所有指標
    df_ind = compute_indicators(df)

    # 取最後一根 K 線資料
    last_idx   = df_ind.index[-1]
    last_open  = df_ind["open"].iloc[-1]
    last_close = df_ind["close"].iloc[-1]
    last_low   = df_ind["low"].iloc[-1]
    last_vol   = df_ind["volume"].iloc[-1]
    prev_vol   = df_ind["volume"].iloc[-2]
    prev_j     = df_ind["stoch_j"].iloc[-2]
    curr_j     = df_ind["stoch_j"].iloc[-1]
    lower_band = df_ind["bb_lower"].iloc[-1]

    # 8.1 原始進場：%J ≤ 15 且 跌破 BB 下軌、成交量放大、%J 向上、當根量 > 20M
    cond_j_low       = curr_j <= 15
    cond_broken_bb   = last_low < lower_band
    cond_vol_up      = last_vol > prev_vol
    cond_j_rising    = curr_j > prev_j
    cond_vol_over20m = last_vol > 20_000_000
    buy_cond         = cond_j_low and cond_broken_bb and cond_vol_up and cond_j_rising and cond_vol_over20m

    # 8.2 或者 Bar 跌幅 ≥ 7% ( (open-close)/open*100 ) 也算買入
    bar_drop = ((last_open - last_close) / last_open * 100) >= 7
    final_buy_cond = buy_cond or bar_drop

    size = None
    # 8.3 如果目前無持倉且滿足進場條件 → 全倉 3 倍槓桿下多單
    if (not in_position) and final_buy_cond:
        size = calculate_lot_size(SYMBOL, leverage=3)
        print(f"[{last_idx}] 進場信號成立 → 槓桿×3 下單，合約數={size}, 價格≈{last_close}")
        resp = client.place_active_order(
            symbol           = SYMBOL,
            side             = "Buy",
            order_type       = "Market",
            qty              = size,
            time_in_force    = "GoodTillCancel",
            reduce_only      = False,
            close_on_trigger = False
        )
        print("多單下單回應：", resp)
        in_position = True
        entry_price = last_close
        entry_time  = last_idx.strftime("%Y-%m-%d %H:%M:%S")

    # 8.4 如果已持倉，檢查平倉
    elif in_position:
        if size is None:
            # 如果上面買入時沒記 size，就在這裡再算一次
            size = calculate_lot_size(SYMBOL, leverage=3)

        profit_pct = (last_close - entry_price) / entry_price * 100

        # (A) 獲利 ≥ 1.5% → 平倉
        if profit_pct >= 1.5:
            print(f"[{last_idx}] 獲利 {profit_pct:.2f}% ≥ 1.5%，市價平倉 Sell")
            resp = client.place_active_order(
                symbol           = SYMBOL,
                side             = "Sell",
                order_type       = "Market",
                qty              = size,
                time_in_force    = "GoodTillCancel",
                reduce_only      = False,
                close_on_trigger = False
            )
            print("空單平倉回應：", resp)
            in_position = False
            entry_price = None
            entry_time  = None

        # (B) 持倉 ≥ 3 天 (4320 分鐘) 且收盤價 == 進場價 → 平倉
        if entry_time:
            held_minutes = (pd.to_datetime(last_idx) - pd.to_datetime(entry_time)).total_seconds() / 60
            if held_minutes >= 4320 and abs(last_close - entry_price) < 1e-8:
                print(f"[{last_idx}] 持倉 {held_minutes/60:.1f} 小時 (>72 小時) 且回本 → 平倉")
                resp = client.place_active_order(
                    symbol           = SYMBOL,
                    side             = "Sell",
                    order_type       = "Market",
                    qty              = size,
                    time_in_force    = "GoodTillCancel",
                    reduce_only      = False,
                    close_on_trigger = False
                )
                print("3 天回本平倉回應：", resp)
                in_position = False
                entry_price = None
                entry_time  = None

    # 8.5 更新並回傳最新持倉狀態
    new_state = {
        "in_position": in_position,
        "entry_price": entry_price,
        "entry_time":  entry_time
    }
    return new_state

# ----------------------------
# 9. 主程式：拉 K 線 → 執行策略 → 下單 → 更新 state.json
# ----------------------------
def main():
    state = load_state()
    try:
        df = fetch_klines(SYMBOL, TIMEFRAME, limit=FETCH_LIMIT)
        new_state = strategy_logic(df, state)
        if new_state != state:
            save_state(new_state)
    except Exception as e:
        print("執行過程出錯：", str(e))
        # 抛出异常讓 Actions 返回非零碼，方便你在 GitHub Actions log 裡看到 Failure
        raise

if __name__ == "__main__":
    main()

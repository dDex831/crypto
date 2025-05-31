# auto_trade.py

import os
import json
import pandas as pd
import numpy as np

# ▶ 注意：新版 Pybit v5 以上，要這麼導入 HTTP：
from pybit.unified_trading import HTTP

# ----------------------------
# 1. 從環境變數讀取 Bybit API Key/Secret
# ----------------------------
BYBIT_API_KEY    = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")
if not BYBIT_API_KEY or not BYBIT_API_SECRET:
    raise ValueError("請先在環境變數設置 BYBIT_API_KEY 和 BYBIT_API_SECRET")

# ----------------------------
# 2. 用新版 Pybit v5 建立 HTTP 客戶端 (USDT 永續合約)
#    如果要跑正式主網 (mainnet)，使用 testnet=False
#    如果要跑測試網 (Testnet)，使用 testnet=True
#    這裡預設連正式網：testnet=False
# ----------------------------
client = HTTP(
    testnet=False,
    api_key    = BYBIT_API_KEY,
    api_secret = BYBIT_API_SECRET
)

# 2.1 設置槓桿為 3 倍（只需要執行一次，就會套用到你之後所有下單）
client.set_leverage(symbol="ADAUSDT", leverage=3)

# ----------------------------
# 3. 參數設定
# ----------------------------
SYMBOL      = "ADAUSDT"   # 交易對：ADA/USDT 永續合約
TIMEFRAME   = "60"        # K 線週期：60 = 60 分鐘 (1h)
FETCH_LIMIT = 200         # 拉取最近 200 根 60 分鐘 K 線

# 下單時要動態計算「可用餘額 × 3 倍槓桿 ÷ 標記價」得出合約數
LOT_SIZE    = None        

# Stochastic 參數 (Bar 數量)
STOCH_K     = 14
STOCH_D     = 3

# Bollinger Bands 參數 (Bar 數量、標準差倍數)
BB_LENGTH   = 20
BB_MULT     = 2.0

# ----------------------------
# 4. 持倉狀態儲存 (state.json)
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
# 5. 拉 K 線並轉成 pandas.DataFrame
# ----------------------------
def fetch_klines(symbol, interval, limit=200):
    """
    Bybit Unified Trading REST API query_kline → 回傳 DataFrame:
      index: open_time (pd.Timestamp)
      columns: open, high, low, close, volume
    """
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
# 6. 計算可用餘額並回傳可下單合約數
# ----------------------------
def calculate_lot_size(symbol, leverage):
    """
    1. 先從 wallet balance API 拿到可用的 USDT
    2. 再用最近一根 K 線收盤價當作「標記價」近似
    3. 公式：lot_count = floor(available_usdt × leverage / mark_price)
    """
    balances = client.get_wallet_balance(coin="USDT")
    if "result" not in balances or "USDT" not in balances["result"]:
        raise RuntimeError(f"get_wallet_balance 回傳異常: {balances}")
    available_usdt = float(balances["result"]["USDT"]["available_balance"])

    # 拿最近 5 根 60 分鐘 K 線，最後一筆的收盤價當作「標記價」近似
    klines = fetch_klines(symbol, TIMEFRAME, limit=5)
    mark_price = klines["close"].iloc[-1]

    raw_size = (available_usdt * leverage) / mark_price
    # ADAUSDT 永續合約通常 precision 到小數第 3 位
    size = float(f"{raw_size:.3f}")
    return size

# ----------------------------
# 7. 用 pandas + numpy 手動計算 Stochastic & Bollinger Bands
# ----------------------------
def compute_indicators(df):
    """
    輸入 df: 必須包含 open/high/low/close/volume
    輸出 df: 新增如下欄位：
      - stoch_k: Stochastic %K
      - stoch_d: Stochastic %D = %K 的 3 期 SMA
      - stoch_j: Stochastic %J = 3×%K - 2×%D
      - bb_lower: Bollinger Band 下軌 = MA20 - 2×StdDev20
    """
    # 7.1 %K = (收盤 - N 期內最低) / (N 期內最高 - N 期內最低) × 100
    low_n   = df["low"].rolling(window=STOCH_K).min()
    high_n  = df["high"].rolling(window=STOCH_K).max()
    stoch_k = (df["close"] - low_n) / (high_n - low_n) * 100

    # 7.2 %D = %K 的 STOCH_D 期簡單移動平均 (SMA)
    stoch_d = stoch_k.rolling(window=STOCH_D).mean()

    # 7.3 %J = 3×%K - 2×%D
    stoch_j = 3 * stoch_k - 2 * stoch_d

    # 7.4 Bollinger Band 下軌 = 20 期收盤價的均線 - 2×20 期收盤價的標準差
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
# 8. 策略核心邏輯：判斷買入或賣出，並控制持倉
# ----------------------------
def strategy_logic(df, state):
    in_position = state["in_position"]
    entry_price = state["entry_price"]
    entry_time  = state["entry_time"]

    # 如果資料筆數不夠，就先不操作
    if len(df) < max(STOCH_K, BB_LENGTH) + 2:
        return state

    # 計算指標
    df_ind = compute_indicators(df)

    # 拿最後一根 60 分鐘 K 線的數值
    last_idx   = df_ind.index[-1]
    last_open  = df_ind["open"].iloc[-1]
    last_close = df_ind["close"].iloc[-1]
    last_low   = df_ind["low"].iloc[-1]
    last_vol   = df_ind["volume"].iloc[-1]
    prev_vol   = df_ind["volume"].iloc[-2]
    prev_j     = df_ind["stoch_j"].iloc[-2]
    curr_j     = df_ind["stoch_j"].iloc[-1]
    lower_band = df_ind["bb_lower"].iloc[-1]

    # 8.1 原始買入條件：%J ≤ 15 且 跌破布林下軌、成交量放大、%J 向上、當根量 > 20M
    cond_j_low       = curr_j <= 15
    cond_broken_bb   = last_low < lower_band
    cond_vol_up      = last_vol > prev_vol
    cond_j_rising    = curr_j > prev_j
    cond_vol_over20m = last_vol > 20_000_000
    buy_cond         = (
        cond_j_low
        and cond_broken_bb
        and cond_vol_up
        and cond_j_rising
        and cond_vol_over20m
    )

    # 8.2 或者當根 K 線跌幅 ≥ 7%，也算買入
    bar_drop = ((last_open - last_close) / last_open * 100) >= 7
    final_buy_cond = buy_cond or bar_drop

    size = None
    # 8.3 如果當前沒持倉且滿足買入條件 → 用「可用餘額 × 3 倍槓桿」下多單
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

    # 8.4 如果已經持倉，檢查平倉條件
    elif in_position:
        # 如果上面買入時沒算 size，就在這裡再算一次
        if size is None:
            size = calculate_lot_size(SYMBOL, leverage=3)

        profit_pct = (last_close - entry_price) / entry_price * 100

        # (A) 如果盈利 ≥ 1.5% → 平倉
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

        # (B) 如果持倉 ≥ 3 天 (4320 分鐘) 且收盤價回到進場價 → 平倉
        if entry_time:
            held_minutes = (
                pd.to_datetime(last_idx) - pd.to_datetime(entry_time)
            ).total_seconds() / 60
            if held_minutes >= 4320 and abs(last_close - entry_price) < 1e-8:
                print(
                    f"[{last_idx}] 持倉 {held_minutes/60:.1f} 小時 (>72 小時) 且回本 → 平倉"
                )
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

    # 8.5 更新持倉狀態並回傳
    new_state = {
        "in_position": in_position,
        "entry_price": entry_price,
        "entry_time":  entry_time,
    }
    return new_state

# ----------------------------
# 9. 主程式：拉 K 線→執行策略→下單→更新 state.json
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
        # 抛出例外讓 GitHub Actions 返回非零碼，以便看見 Failure
        raise

if __name__ == "__main__":
    main()

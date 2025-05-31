# auto_trade.py

import os
import json
import requests
import pandas as pd
import numpy as np

from pybit.unified_trading import HTTP as BybitV5HTTP
from pybit.exceptions import FailedRequestError

# ----------------------------
# 1. 从环境变量读取 Bybit API Key/Secret
# ----------------------------
BYBIT_API_KEY    = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")
if not BYBIT_API_KEY or not BYBIT_API_SECRET:
    raise ValueError("请先在环境变量设定 BYBIT_API_KEY 和 BYBIT_API_SECRET")

# ----------------------------
# 是否启用“测试下单模式”：直接下市价单 5 合约（3 倍杠杆）
# ----------------------------
TEST_ORDER = os.getenv("TEST_ORDER", "false").lower() == "true"

# ----------------------------
# 2. 用 pybit v5 建立 HTTP 客户端 (USDT 永续合约)
#    如果要跑 Testnet，把 testnet=True
# ----------------------------
client = BybitV5HTTP(
    testnet=False,
    api_key    = BYBIT_API_KEY,
    api_secret = BYBIT_API_SECRET
)

# ----------------------------
# 3. 先设杠杆 3 倍
# ----------------------------
def ensure_leverage():
    try:
        client.set_leverage(symbol="ADAUSDT", leverage=3)
        print(f"[{pd.Timestamp.now()}] 杠杆设置成功：ADAUSDT = 3 倍")
    except FailedRequestError as e:
        print(f"[{pd.Timestamp.now()}] Warning: 设置杠杆时发生错误，已跳过。错误信息：{e}")

print(f"[{pd.Timestamp.now()}] 初始化：即将设置杠杆 3 倍。")
ensure_leverage()

# ----------------------------
# 4. 参数设置
# ----------------------------
SYMBOL      = "ADAUSDT"   # 交易对：ADA/USDT 永续合约
FETCH_LIMIT = 200         # 拉取最近 200 根 1 小时 K 线

# Stochastic 参数（Bar 数量）
STOCH_K     = 14
STOCH_D     = 3

# Bollinger Bands 参数（Bar 数量、标准差倍数）
BB_LENGTH   = 20
BB_MULT     = 2.0

# ----------------------------
# 5. 持仓状态保存 (state.json)
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
# 6. 拉 K 线并转换成 pandas.DataFrame（使用 CryptoCompare 公共接口）
#    并转换成台湾本地时间 (Asia/Taipei)
# ----------------------------
def fetch_klines(symbol: str, limit: int = 200) -> pd.DataFrame:
    """
    通过 CryptoCompare 公共 REST API 获取 ADA/USDT 小时 K 线，不带 API Key。
    将 open_time 从 UTC 转成 Asia/Taipei 时区，让你直接看到台湾本地时间。
    返回 pandas.DataFrame:
      index: open_time (pd.Timestamp with tz='Asia/Taipei')
      columns: open, high, low, close, volume
    参数：
      symbol: "ADAUSDT"  -> 拆成 fsym="ADA" 和 tsym="USDT"
      limit: 最多返回几根历史 K 线，最大200
    """
    fsym = symbol[:-4]   # "ADAUSDT" -> "ADA"
    tsym = symbol[-4:]   # "ADAUSDT" -> "USDT"

    url = "https://min-api.cryptocompare.com/data/v2/histohour"
    params = {
        "fsym": fsym,
        "tsym": tsym,
        "limit": limit
    }

    print(f"[{pd.Timestamp.now()}] → 开始请求 CryptoCompare K 线 (symbol={symbol}, limit={limit})")
    try:
        r = requests.get(url, params=params, timeout=10)
    except Exception as e:
        raise RuntimeError(f"调用 CryptoCompare K-line 接口时网络错误: {e}")

    print(f"[{pd.Timestamp.now()}] ← CryptoCompare HTTP 状态码: {r.status_code}")
    if r.status_code != 200:
        raise RuntimeError(f"CryptoCompare K-line HTTP 错误: Status {r.status_code}, Response: {r.text}")

    try:
        data = r.json()
    except Exception as e:
        raise RuntimeError(f"CryptoCompare K-line 返回无法解析为 JSON: {e} | Raw Response: {r.text}")

    if data.get("Response") != "Success" or "Data" not in data or "Data" not in data["Data"]:
        raise RuntimeError(f"CryptoCompare K-line 数据格式异常: {data}")

    ohlc_list = data["Data"]["Data"]
    df = pd.DataFrame(ohlc_list)

    # 把 "time"（Unix 秒）转成 UTC 时区的 Timestamp
    df["open_time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    # 再把它从 UTC 转到 Asia/Taipei
    df["open_time"] = df["open_time"].dt.tz_convert("Asia/Taipei")
    # 设成索引
    df = df.set_index("open_time")

    # 转成我们需要的列，并转换为 float
    df2 = pd.DataFrame({
        "open": df["open"].astype(float),
        "high": df["high"].astype(float),
        "low":  df["low"].astype(float),
        "close": df["close"].astype(float),
        # 使用 volumefrom 做 volume
        "volume": df["volumefrom"].astype(float)
    }, index=df.index)

    print(f"[{pd.Timestamp.now()}] ← 成功拿到 {len(df2)} 根 K 线，最后一根时间（台北时区）：{df2.index[-1]}, 收盘价：{df2['close'].iloc[-1]:.6f}")
    return df2

# ----------------------------
# 7. 计算可用余额并返回可下单合约数
# ----------------------------
def calculate_lot_size(symbol: str, leverage: int) -> float:
    """
    1. 从 Bybit v5 Wallet Balance API 拿到可用 USDT
    2. 用最近一根 K 线收盘价当作“标记价”近似
    3. 公式：lot_count = floor(available_usdt * leverage / mark_price)
    """
    balances = client.get_wallet_balance(coin="USDT")
    if "result" not in balances or "USDT" not in balances["result"]:
        raise RuntimeError(f"get_wallet_balance 返回异常: {balances}")
    available_usdt = float(balances["result"]["USDT"]["available_balance"])
    print(f"[{pd.Timestamp.now()}] 可用 USDT 余额: {available_usdt:.4f}")

    # 拿最近 5 根 1h K 线，最后一根的 close 当作标记价
    klines = fetch_klines(symbol, limit=5)
    mark_price = klines["close"].iloc[-1]
    print(f"[{pd.Timestamp.now()}] 标记价 (最近 1h 收盘): {mark_price:.6f}")

    raw_size = (available_usdt * leverage) / mark_price
    # ADAUSDT 永续合约一般精度到小数后 3 位
    size = float(f"{raw_size:.3f}")
    print(f"[{pd.Timestamp.now()}] 计算出可下单合约数: {size:.3f}")
    return size

# ----------------------------
# 8. 用 pandas + numpy 手动计算 Stochastic & Bollinger Bands
# ----------------------------
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    输入 df：必须包含 open, high, low, close, volume
    输出 df：新增 stoch_k, stoch_d, stoch_j, bb_lower
    """
    low_n   = df["low"].rolling(window=STOCH_K).min()
    high_n  = df["high"].rolling(window=STOCH_K).max()
    stoch_k = (df["close"] - low_n) / (high_n - low_n) * 100
    stoch_d = stoch_k.rolling(window=STOCH_D).mean()
    stoch_j = 3 * stoch_k - 2 * stoch_d

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
# 9. 核心策略逻辑：判断买入/卖出、控制持仓
#    增加打印最后几行指标 & 信号打印
# ----------------------------
def strategy_logic(df: pd.DataFrame, state: dict) -> dict:
    in_position = state["in_position"]
    entry_price = state["entry_price"]
    entry_time  = state["entry_time"]

    # 如果 Bar 数量不足，就先不操作
    if len(df) < max(STOCH_K, BB_LENGTH) + 2:
        print(f"[{pd.Timestamp.now()}] K 线数量不足，结束当前轮询 (共 {len(df)} 根)。")
        return state

    # 计算所有指标
    df_ind = compute_indicators(df)
    # 打印最后 3 根指标示例
    print(f"[{pd.Timestamp.now()}] → 指标计算完成，示例（最后 3 行）:")
    print(df_ind.tail(3)[["stoch_k", "stoch_d", "stoch_j", "bb_lower"]])

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

    print(f"[{pd.Timestamp.now()}] 最后一根 K 线时间 (台北时区): {last_idx}, 收盘价: {last_close:.6f}, 低: {last_low:.6f}, 成交量: {last_vol:.2f}")
    print(f"[{pd.Timestamp.now()}] %J 前:{prev_j:.2f}，当前:{curr_j:.2f}，BB 下轨: {lower_band:.6f}")

    # 9.1 原始买入条件：%J ≤ 15 且 跌破 BB 下轨、成交量放大、%J 向上、当根量 > 20M
    cond_j_low       = curr_j <= 15
    cond_broken_bb   = last_low < lower_band
    cond_vol_up      = last_vol > prev_vol
    cond_j_rising    = curr_j > prev_j
    cond_vol_over20m = last_vol > 20_000_000
    buy_cond         = cond_j_low and cond_broken_bb and cond_vol_up and cond_j_rising and cond_vol_over20m

    # 9.2 或当根 K 线跌幅 ≥ 7%，也算买入
    bar_drop = ((last_open - last_close) / last_open * 100) >= 7
    final_buy_cond = buy_cond or bar_drop

    size = None
    # 9.3 如果当前没持仓且满足买入条件 → 全仓 3 倍杠杆下多单
    if (not in_position) and final_buy_cond:
        print(f"[{pd.Timestamp.now()}] 买入条件满足，准备下多单。")
        size = calculate_lot_size(SYMBOL, leverage=3)
        print(f"[{pd.Timestamp.now()}] → 下单信息：杠杆×3，合约数={size:.3f}, 价格≈{last_close:.6f}")
        resp = client.place_order(
            category  = "linear",
            symbol    = SYMBOL,
            side      = "Buy",
            orderType = "Market",
            qty       = size
        )
        print(f"[{pd.Timestamp.now()}] 多单下单响应：{resp}")
        in_position = True
        entry_price = last_close
        entry_time  = last_idx.strftime("%Y-%m-%d %H:%M:%S")

    # 9.4 如果已经持仓，则检查平仓条件
    elif in_position:
        print(f"[{pd.Timestamp.now()}] 当前已有持仓，检查平仓条件。")
        if size is None:
            size = calculate_lot_size(SYMBOL, leverage=3)

        profit_pct = (last_close - entry_price) / entry_price * 100
        print(f"[{pd.Timestamp.now()}] 盈亏比例: {profit_pct:.2f}% (当前价 {last_close:.6f} vs 进场价 {entry_price:.6f})")

        # (A) 如果盈利 ≥ 1.5% → 平仓
        if profit_pct >= 1.5:
            print(f"[{pd.Timestamp.now()}] 盈利 ≥ 1.5%，市价平仓 Sell。")
            resp = client.place_order(
                category  = "linear",
                symbol    = SYMBOL,
                side      = "Sell",
                orderType = "Market",
                qty       = size
            )
            print(f"[{pd.Timestamp.now()}] 空单平仓响应：{resp}")
            in_position = False
            entry_price = None
            entry_time  = None

        # (B) 如果持仓 ≥ 3 天 (4320 分钟) 且收盘价回到进场价 → 平仓
        if entry_time:
            held_minutes = (pd.to_datetime(last_idx) - pd.to_datetime(entry_time)).total_seconds() / 60
            if held_minutes >= 4320 and abs(last_close - entry_price) < 1e-8:
                print(f"[{pd.Timestamp.now()}] 持仓 {held_minutes/60:.1f} 小时 (>72 小时) 且回本 → 平仓。")
                resp = client.place_order(
                    category  = "linear",
                    symbol    = SYMBOL,
                    side      = "Sell",
                    orderType = "Market",
                    qty       = size
                )
                print(f"[{pd.Timestamp.now()}] 3 天回本平仓响应：{resp}")
                in_position = False
                entry_price = None
                entry_time  = None

    # 9.5 更新并返回最新持仓状态
    new_state = {
        "in_position": in_position,
        "entry_price": entry_price,
        "entry_time":  entry_time,
    }
    return new_state

# ----------------------------
# 10. 主程序：拉 K 线 → 或直接测试下单 → 执行策略 → 下单 → 更新 state.json
# ----------------------------
def main():
    state = load_state()
    print(f"[{pd.Timestamp.now()}] 当前持仓状态: {state}")

    # 如果 TEST_ORDER=True，就直接市价买入 5 合约，然后结束脚本
    if TEST_ORDER:
        print(f"[{pd.Timestamp.now()}] TEST_ORDER 模式：直接市价买入 5 合约 (USDⓈ-M ADAUSDT)，已设杠杆 3 倍。")
        try:
            resp = client.place_order(
                category  = "linear",
                symbol    = SYMBOL,
                side      = "Buy",
                orderType = "Market",
                qty       = "5"
            )
            print(f"[{pd.Timestamp.now()}] 测试下单 5 合约响应：{resp}")
        except Exception as e:
            print(f"[{pd.Timestamp.now()}] 测试下单过程中出错：{e}")
        return  # 结束脚本

    # 非测试模式 → 正常策略逻辑
    try:
        df = fetch_klines(SYMBOL, limit=FETCH_LIMIT)
        new_state = strategy_logic(df, state)
        if new_state != state:
            print(f"[{pd.Timestamp.now()}] ⚡ 持仓状态改变: {state} → {new_state}，保存状态。")
            save_state(new_state)
        else:
            print(f"[{pd.Timestamp.now()}] 持仓状态无变化: {state}")
    except Exception as e:
        print(f"[{pd.Timestamp.now()}] 执行过程中出错：{e}")
        raise

if __name__ == "__main__":
    main()

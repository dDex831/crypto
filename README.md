# Crypto Auto Trade Bot (Binance Spot)

這是一個使用 Python 撰寫的 **加密貨幣自動交易腳本**，連接 Binance API，根據技術分析邏輯進行自動下單操作。

## 📌 功能特色

- 使用 Binance Spot API 進行下單與查詢
- 支援 ADA/USDT 現貨交易
- 自動計算下單數量（符合 Binance 最小交易單位）
- 日誌記錄交易與錯誤資訊
- 可擴充進行技術策略判斷（如布林帶、動量策略等）

## ⚙️ 使用方式

### 1. 安裝依賴套件

```bash
pip install python-binance pandas numpy


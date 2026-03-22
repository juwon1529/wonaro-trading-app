import pandas as pd
from data import get_stock_data
from analysis import calculate_indicators

# 기본 관심 종목 리스트
DEFAULT_WATCHLIST = ["AAPL", "MSFT", "TSLA", "NVDA", "AMZN"]

def analyze_watchlist(watchlist=DEFAULT_WATCHLIST, period="3mo"):
    """
    여러 종목의 데이터를 한 번에 수집하고 지표를 계산하여 DataFrame으로 반환합니다.
    """
    results = []

    for ticker in watchlist:
        # 1. 데이터 수집
        hist, current_price, error = get_stock_data(ticker, period)
        
        if error or hist is None or hist.empty:
            continue

        # 2. 지표 계산
        hist_ind = calculate_indicators(hist)
        
        if hist_ind is None or hist_ind.empty:
            continue

        # 3. 최신 데이터 추출
        latest = hist_ind.iloc[-1]
        ma20 = latest["MA20"]
        rsi = latest["RSI"]

        # 4. RSI 상태 판별
        if pd.isna(rsi):
            rsi_status = "데이터 부족"
        elif rsi > 70:
            rsi_status = "🔴 과매수"
        elif rsi < 30:
            rsi_status = "🟢 과매도"
        else:
            rsi_status = "⚪ 중립"

        # 5. 결과 저장
        results.append({
            "종목": ticker,
            "현재가": current_price,
            "MA20": ma20,
            "RSI": rsi,
            "RSI 상태": rsi_status
        })

    # 결과를 표(DataFrame) 형태로 변환하여 반환
    return pd.DataFrame(results)
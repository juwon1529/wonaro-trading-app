# data.py
import yfinance as yf

def get_stock_data(ticker, period="3mo"):
    try:
        ticker = ticker.strip().upper()
        if not ticker:
            return None, None, "종목 코드가 비어 있습니다."
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        if hist.empty:
            return None, None, f"{ticker} 종목의 주가 데이터를 찾을 수 없습니다."
        current_price = round(hist["Close"].iloc[-1], 2)
        return hist, current_price, None
    except Exception as e:
        return None, None, f"데이터 수집 중 오류가 발생했습니다: {str(e)}"

def get_macro_data():
    tickers = {"환율": "KRW=X", "VIX": "^VIX", "미국10년물": "^TNX"}
    macro_results = {}
    try:
        for name, symbol in tickers.items():
            data = yf.Ticker(symbol).history(period="5d")
            if len(data) >= 2:
                current = round(data["Close"].iloc[-1], 2)
                prev = round(data["Close"].iloc[-2], 2)
                diff = round(current - prev, 2)
                macro_results[name] = {"current": current, "diff": diff}
            else:
                macro_results[name] = {"current": "-", "diff": "-"}
        return macro_results
    except Exception as e:
        print(f"거시 지표 수집 오류: {e}")
        return None

def get_recent_news(ticker):
    """
    해당 종목의 최신 뉴스 제목과 링크를 가져옵니다.
    yfinance의 반환 구조 변경에 대비하여 안전하게 추출합니다.
    """
    try:
        stock = yf.Ticker(ticker)
        news_data = stock.news
        
        if not news_data:
            return []
        
        news_list = []
        for n in news_data:
            # yfinance 버전에 따라 title 또는 content 속성에 제목이 있을 수 있음
            title = n.get('title') or n.get('content', {}).get('title') or "제목 없음"
            
            # yfinance 버전에 따라 link 또는 content 속성에 링크가 있을 수 있음
            link = n.get('link') or n.get('content', {}).get('clickThroughUrl') or n.get('url') or "#"
            
            # 둘 다 유효한 경우에만 리스트에 추가
            if title != "제목 없음" and link != "#":
                news_list.append({"title": title, "link": link})
                
            # 5개만 수집하면 중단
            if len(news_list) >= 5:
                break
                
        return news_list
    except Exception as e:
        print(f"뉴스 수집 오류: {e}")
        return []
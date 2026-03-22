# analysis.py
import pandas as pd
import requests
import json

def calculate_indicators(df):
    if df is None or df.empty: return None
    df = df.copy()
    df["MA20"] = df["Close"].rolling(window=20).mean().round(2)
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, 1e-9)
    df["RSI"] = (100 - (100 / (1 + rs))).round(2)
    
    vwap_window = 20
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP'] = (df['Typical_Price'] * df['Volume']).rolling(window=vwap_window).sum() / df['Volume'].rolling(window=vwap_window).sum()
    df['VWAP'] = df['VWAP'].round(2)
    return df

def analyze_news_sentiment(news_list):
    if not news_list: return 0, "뉴스 데이터 없음"
    positive_words = ['up', 'surge', 'jump', 'gain', 'bull', 'buy', 'beat', 'growth', 'profit']
    negative_words = ['down', 'drop', 'fall', 'plunge', 'bear', 'sell', 'miss', 'loss', 'risk']
    score = 0
    for news in news_list:
        title = news['title'].lower()
        for word in positive_words:
            if word in title: score += 10
        for word in negative_words:
            if word in title: score -= 10
    score = max(min(score, 100), -100)
    if score > 10: status = "🟢 긍정적"
    elif score < -10: status = "🔴 부정적"
    else: status = "⚪ 중립적"
    return score, status

def analyze_past_patterns(df_2y):
    if df_2y is None or len(df_2y) < 50: return None
    df = df_2y.copy()
    latest = df.iloc[-1]
    current_trend = "상승장" if latest['Close'] > latest['MA20'] else "하락장"
    current_rsi = "과매수" if latest['RSI'] >= 70 else ("과매도" if latest['RSI'] <= 30 else "중립")

    df['Trend'] = df.apply(lambda x: "상승장" if x['Close'] > x['MA20'] else "하락장", axis=1)
    df['RSI_Zone'] = df['RSI'].apply(lambda r: "과매수" if r >= 70 else ("과매도" if r <= 30 else "중립"))
    df['Return_5d'] = (df['Close'].shift(-5) / df['Close']) - 1

    condition = (df['Trend'] == current_trend) & (df['RSI_Zone'] == current_rsi)
    similar_cases = df[condition].iloc[:-5]
    if len(similar_cases) == 0: return {"match_count": 0, "win_rate": 0, "avg_return": 0}

    up_cases = similar_cases[similar_cases['Return_5d'] > 0]
    win_rate = round((len(up_cases) / len(similar_cases)) * 100, 2)
    avg_return = round(similar_cases['Return_5d'].mean() * 100, 2)
    return {"match_count": len(similar_cases), "win_rate": win_rate, "avg_return": avg_return}

# ==========================================
# 🧠 AI 프롬프트 (가독성/구간 분리 최적화)
# ==========================================
def generate_ai_analysis(provider, api_key, ticker, current_price, ma20, rsi, vwap, volume, macro_text, sentiment_status, pattern_win_rate):
    if not api_key or not api_key.strip(): return f"{provider} API 키가 없습니다."

    prompt = f"""
너는 '스마트 머니 콘셉트(SMC)'와 'ICT 기법'에 통달한 기관 트레이더다.
아래 데이터를 바탕으로 TradingView 스타일의 전문적인 트레이딩 셋업을 제안해라.

[입력 데이터]
- 종목: {ticker} (현재가: {current_price})
- 기술적 지표: MA20 = {ma20}, RSI = {rsi}
- 세력 추적(SMC): VWAP = {vwap}, 최근 거래량 = {volume}
- 거시 경제: {macro_text}
- 투심(Sentiment): {sentiment_status}
- 과거 통계적 승률: {pattern_win_rate}%

[출력 형식] (반드시 아래의 마크다운 소제목과 리스트 기호를 엄격하게 지켜서 가독성 좋게 작성할 것)

### 💡 1. 시장 구조 및 SMC 분석
- **VWAP 기준:** [분석 내용]
- **유동성(Liquidity) 관점:** [분석 내용]
- **AMD 모델 관점:** [분석 내용]

### 📊 2. 데이터 기반 팩트 체크
- **통계 승률:** [분석 내용]
- **매크로 및 투심:** [분석 내용]

### ⏳ 3. 기간별 가격 움직임 예측
#### 🔹 단기 (1~5일)
- **예측:** [상승/하락/횡보]
- **근거:** [이유]

#### 🔹 중기 (1~4주)
- **예측:** [상승/하락/횡보]
- **근거:** [이유]

#### 🔹 장기 (1~3개월)
- **예측:** [상승/하락/횡보]
- **근거:** [이유]

### 🎯 4. 최종 트레이딩 셋업
- **추천 포지션:** [Long / Short / Wait]
- **진입 타점:** [가격대]
- **익절가(TP):** [가격대]
- **손절가(SL):** [가격대]
"""
    try:
        if provider == "OpenAI (GPT-4o)":
            from openai import OpenAI
            client = OpenAI(api_key=api_key.strip())
            response = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], temperature=0.5)
            return response.choices[0].message.content
        elif provider == "Anthropic (Claude 3.5)":
            import anthropic
            client = anthropic.Anthropic(api_key=api_key.strip())
            response = client.messages.create(model="claude-3-5-sonnet-20240620", max_tokens=1000, temperature=0.5, messages=[{"role": "user", "content": prompt}])
            return response.content[0].text
        elif provider == "Google (Gemini 1.5)":
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key.strip()}"
            response = requests.post(url, headers={'Content-Type': 'application/json'}, json={"contents": [{"parts": [{"text": prompt}]}]})
            if response.status_code == 200: return response.json()['candidates'][0]['content']['parts'][0]['text']
            else: return f"❌ Gemini 통신 에러"
    except Exception as e:
        return f"파이썬 오류: {str(e)}"

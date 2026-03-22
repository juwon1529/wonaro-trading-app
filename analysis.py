# analysis.py
import pandas as pd
import requests
import json

def calculate_indicators(df):
    if df is None or df.empty:
        return None
    df = df.copy()
    
    # 기본 지표
    df["MA20"] = df["Close"].rolling(window=20).mean().round(2)
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, 1e-9)
    df["RSI"] = (100 - (100 / (1 + rs))).round(2)
    
    # 🚀 트레이딩뷰/SMC(스마트머니) 필수 지표 추가
    # 1. Anchored VWAP (단순화를 위해 최근 20일 기준 누적 거래량 가중 평균)
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

    match_count = len(similar_cases)
    if match_count == 0: return {"match_count": 0, "win_rate": 0, "avg_return": 0}

    up_cases = similar_cases[similar_cases['Return_5d'] > 0]
    win_rate = round((len(up_cases) / match_count) * 100, 2)
    avg_return = round(similar_cases['Return_5d'].mean() * 100, 2)
    return {"match_count": match_count, "win_rate": win_rate, "avg_return": avg_return}

# ==========================================
# 🧠 AI 프롬프트 (SMC 및 기간별 예측 적용)
# ==========================================
def generate_ai_analysis(provider, api_key, ticker, current_price, ma20, rsi, vwap, volume, macro_text, sentiment_status, pattern_win_rate):
    if not api_key or not api_key.strip(): return f"{provider} API 키가 없습니다."

    prompt = f"""
너는 '스마트 머니 콘셉트(SMC)'와 'ICT 기법'에 통달한 기관 트레이더다.
아래 데이터를 바탕으로 TradingView 스타일의 전문적인 트레이딩 셋업을 제안해라.

[분석 파라미터 (현재 입력된 데이터)]
- 종목: {ticker}
- 현재가: {current_price}
- 기본 기술적 지표: MA20 = {ma20}, RSI = {rsi}
- 기관/세력 추적 지표 (SMC 기반 추정치):
  1) VWAP(거래량 가중 평균가): {vwap} (현재가가 이보다 높으면 매수 우위, 낮으면 매도 우위)
  2) 최근 거래량: {volume} (Volume Profile 및 Order Flow 유입 강도 추정용)
- 거시 경제: {macro_text}
- 투심(Sentiment): {sentiment_status}
- 과거 통계적 승률: {pattern_win_rate}%

[트레이딩 수익 목적 고려 데이터 (개념적 분석)]
위 수치들을 바탕으로 아래 항목들이 현재 차트에서 어떻게 작용하고 있을지 논리적으로 추론하라:
- Liquidity Sweep (개미털기 유동성 확보 여부)
- Order Flow (매수/매도 압력 우위)
- AMD Model (Accumulation, Manipulation, Distribution 중 현재 어느 단계인지)

[출력 형식] (반드시 마크다운으로 깔끔하게 작성할 것)
💡 **[시장 구조 및 SMC 분석]** (VWAP, 유동성, AMD 관점에서의 현재 위치)
📊 **[데이터 기반 팩트 체크]** (통계 승률, 매크로, 투심 요약)
⏳ **[기간별 가격 움직임 예측 (Price Action)]**
- **단기 (1~5일):** [예측 및 근거]
- **중기 (1~4주):** [예측 및 근거]
- **장기 (1~3개월):** [예측 및 근거]
🎯 **[트레이딩 셋업 (Trading Setup)]** 진입 타점, 익절가(TP), 손절가(SL) 및 최종 포지션 추천 (Long / Short / Wait)
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
            models_to_try = ["gemini-1.5-flash-latest", "gemini-1.5-pro-latest", "gemini-pro"]
            for model_name in models_to_try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key.strip()}"
                response = requests.post(url, headers={'Content-Type': 'application/json'}, json={"contents": [{"parts": [{"text": prompt}]}]})
                if response.status_code == 200: return f"(*{model_name}*)\n\n" + response.json()['candidates'][0]['content']['parts'][0]['text']
            return f"Gemini 통신 에러 (API 키 또는 한도 문제)"
    except Exception as e:
        return f"오류 발생: {str(e)}"
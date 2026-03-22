# app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from data import get_stock_data, get_macro_data, get_recent_news
from analysis import calculate_indicators, generate_ai_analysis, analyze_news_sentiment, analyze_past_patterns

# UI 설정
st.set_page_config(page_title="Wonaro", layout="wide", initial_sidebar_state="expanded")

# --- 전체 앱 텍스트 가독성 향상을 위한 전역 CSS 주입 ---
st.markdown("""
    <style>
        /* 기본 텍스트 및 배경 가독성 최적화 */
        .stApp {
            background-color: #0E1117;
        }
        p, div, span, li {
            color: #D1D4DC !important;
        }
        h1, h2, h3, h4, h5, h6, b, strong {
            color: #FFFFFF !important;
            font-weight: 600 !important;
        }
        /* AI 리포트 내부 마크다운 텍스트 색상 강제 지정 */
        .ai-report-box p, .ai-report-box li {
            color: #F5F5F5 !important;
            line-height: 1.6;
            font-size: 15px;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300)
def load_cached_stock_data(ticker, period="3mo"):
    return get_stock_data(ticker, period)

@st.cache_data(ttl=3600)
def load_pattern_data(ticker):
    return get_stock_data(ticker, period="2y")

@st.cache_data(ttl=600)
def load_cached_macro_data():
    return get_macro_data()

st.markdown("<h1 style='text-align: center; color: #FFFFFF; margin-bottom: 0px;'>Wonaro</h1>", unsafe_allow_html=True)

# 1. 글로벌 매크로 보드 (텍스트 가독성 상향)
macro_data = load_cached_macro_data()
macro_text = "데이터 없음"
if macro_data:
    krw, vix, tnx = macro_data.get("환율", {}), macro_data.get("VIX", {}), macro_data.get("미국10년물", {})
    st.markdown(f"""
    <div style='background-color:#1E222D; padding:12px; border-radius:8px; text-align:center; font-size:15px; border: 1px solid #2A2E39; margin-bottom: 20px;'>
        <b style='color:#FFFFFF;'>[Macro]</b> &nbsp;&nbsp; 
        <span style='color:#D1D4DC;'>KRW/USD:</span> <b style='color:#FFF;'>{krw.get('current','-')}</b> <span style='color:#888;'>({krw.get('diff','-')})</span> &nbsp;&nbsp;|&nbsp;&nbsp; 
        <span style='color:#D1D4DC;'>VIX:</span> <b style='color:#FFF;'>{vix.get('current','-')}</b> <span style='color:#888;'>({vix.get('diff','-')})</span> &nbsp;&nbsp;|&nbsp;&nbsp; 
        <span style='color:#D1D4DC;'>US10Y:</span> <b style='color:#FFF;'>{tnx.get('current','-')}%</b> <span style='color:#888;'>({tnx.get('diff','-')}%)</span>
    </div>
    """, unsafe_allow_html=True)
    macro_text = f"환율 {krw.get('current')}원, VIX {vix.get('current')}, 국채 {tnx.get('current')}%"

# 사이드바 설정
with st.sidebar:
    st.markdown("<h3 style='color:#FFF;'>⚙️ Wonaro Settings</h3>", unsafe_allow_html=True)
    ai_provider = st.selectbox("AI Engine", ["OpenAI (GPT-4o)", "Anthropic (Claude 3.5)", "Google (Gemini 1.5)"])
    api_key = st.text_input("API Key", type="password")
    period_mapping = {"1개월": "1mo", "3개월": "3mo", "6개월": "6mo", "1년": "1y"}
    period_key = st.selectbox("Timeframe", list(period_mapping.keys()), index=1)
    period = period_mapping[period_key]

# 메인 검색창
with st.form(key="search_form"):
    col_search, col_btn = st.columns([5, 1])
    with col_search:
        ticker = st.text_input("Symbol", placeholder="예: AAPL, NVDA, BTC-USD (입력 후 엔터)", label_visibility="collapsed")
    with col_btn:
        submit_button = st.form_submit_button("Chart & Analyze")

if submit_button and ticker.strip():
    ticker = ticker.strip().upper()
    with st.spinner("Loading Trading Data..."):
        hist, current_price, error_message = load_cached_stock_data(ticker, period)
        hist_2y, _, _ = load_pattern_data(ticker)

    if error_message or hist is None:
        st.error(error_message or "데이터를 불러오지 못했습니다.")
    else:
        hist_ind = calculate_indicators(hist)
        hist_2y_ind = calculate_indicators(hist_2y)
        latest = hist_ind.iloc[-1]
        
        chart_col, info_col = st.columns([2.5, 1.2])

        # ==========================================
        # 좌측: TradingView 스타일 차트
        # ==========================================
        with chart_col:
            st.markdown(f"<h3 style='color:#FFF; margin-bottom:0px;'>{ticker} <span style='font-size:24px; color:#26A69A;'>${current_price}</span></h3>", unsafe_allow_html=True)
            
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.6, 0.2, 0.2])

            fig.add_trace(go.Candlestick(
                x=hist_ind.index, open=hist_ind['Open'], high=hist_ind['High'], low=hist_ind['Low'], close=hist_ind['Close'],
                name='Price', increasing_line_color='#26A69A', decreasing_line_color='#EF5350'
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(x=hist_ind.index, y=hist_ind['MA20'], line=dict(color='#FFD600', width=1.5), name='MA20'), row=1, col=1)
            
            if 'VWAP' in hist_ind.columns:
                fig.add_trace(go.Scatter(x=hist_ind.index, y=hist_ind['VWAP'], line=dict(color='#E1BEE7', width=1.5, dash='dot'), name='VWAP'), row=1, col=1)

            colors = ['#26A69A' if row['Close'] >= row['Open'] else '#EF5350' for index, row in hist_ind.iterrows()]
            fig.add_trace(go.Bar(x=hist_ind.index, y=hist_ind['Volume'], marker_color=colors, name='Volume'), row=2, col=1)

            fig.add_trace(go.Scatter(x=hist_ind.index, y=hist_ind['RSI'], line=dict(color='#29B6F6', width=1.5), name='RSI'), row=3, col=1)
            fig.add_hline(y=70, line_dash="dot", line_color="#EF5350", row=3, col=1)
            fig.add_hline(y=30, line_dash="dot", line_color="#26A69A", row=3, col=1)

            fig.update_layout(
                xaxis_rangeslider_visible=False,
                margin=dict(l=10, r=10, t=10, b=10),
                height=700,
                template="plotly_dark",
                plot_bgcolor="#131722",
                paper_bgcolor="#131722",
                showlegend=False
            )
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#2A2E39')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#2A2E39')
            
            # 최신 문법으로 경고 해결
            st.plotly_chart(fig)

        # ==========================================
        # 우측: 지표 시각화 및 AI 리포트
        # ==========================================
        with info_col:
            vwap_val = latest.get("VWAP", "N/A")
            vol_val = latest.get("Volume", 0)
            rsi_val = latest.get("RSI", 50)
            
            news_list = get_recent_news(ticker)
            sentiment_score, sentiment_status = analyze_news_sentiment(news_list)
            pattern_stats = analyze_past_patterns(hist_2y_ind)
            win_rate = pattern_stats['win_rate'] if pattern_stats else "N/A"

            if rsi_val >= 70:
                rsi_text, rsi_color = "🔴 과매수 (Overbought)", "#EF5350"
            elif rsi_val <= 30:
                rsi_text, rsi_color = "🟢 과매도 (Oversold)", "#26A69A"
            else:
                rsi_text, rsi_color = "⚪ 중립 (Neutral)", "#FFB74D"

            sent_percent = (sentiment_score + 100) / 2 
            if sentiment_score > 10: sent_color = "#26A69A"
            elif sentiment_score < -10: sent_color = "#EF5350"
            else: sent_color = "#FFB74D"

            st.markdown("<h4 style='color:#FFF; margin-top:0px;'>📊 Market Status</h4>", unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style='background-color:#1E222D; padding:15px; border-radius:8px; border: 1px solid #2A2E39; margin-bottom: 15px;'>
                <div style='margin-bottom: 15px;'>
                    <span style='color:#8B93A6; font-size:13px; font-weight:600;'>RSI (14) STATUS</span><br>
                    <span style='font-size:20px; font-weight:bold; color:{rsi_color};'>{rsi_val} - {rsi_text}</span>
                </div>
                <div>
                    <span style='color:#8B93A6; font-size:13px; font-weight:600;'>NEWS SENTIMENT</span><br>
                    <div style='display:flex; justify-content:space-between; align-items:flex-end;'>
                        <span style='font-size:20px; font-weight:bold; color:{sent_color};'>{sentiment_score} <span style='font-size:14px'>({sentiment_status})</span></span>
                    </div>
                    <div style='width:100%; background-color:#131722; height:8px; border-radius:4px; margin-top:8px; overflow:hidden;'>
                        <div style='width:{sent_percent}%; background-color:{sent_color}; height:100%; border-radius:4px; transition: width 0.5s ease-in-out;'></div>
                    </div>
                    <div style='display:flex; justify-content:space-between; font-size:11px; color:#8B93A6; margin-top:4px;'>
                        <span>Ext. Fear (-100)</span>
                        <span>Neutral (0)</span>
                        <span>Ext. Greed (100)</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div style='background-color:#1E222D; padding:15px; border-radius:8px; border: 1px solid #2A2E39; font-size:13px; margin-bottom:15px;'>
                <b style='color:#FFF;'>[고급 트레이딩 분석 파라미터]</b><br>
                <span style='color:#A3A6AF;'>• <b>VWAP & Vol:</b> 기관 평단가 및 유입 강도</span><br>
                <span style='color:#A3A6AF;'>• <b>Liq Sweep:</b> 유동성 사냥(휩소) 확인</span><br>
                <span style='color:#A3A6AF;'>• <b>AMD Model:</b> 축적/조작/분배 사이클 판별</span>
            </div>
            """, unsafe_allow_html=True)
            
            # 최신 문법으로 경고 해결
            if st.button("🧠 기간별 움직임 예측 (AI)", type="primary"):
                if not api_key:
                    st.error("좌측 설정에서 API 키를 입력하세요.")
                else:
                    with st.spinner("SMC 흐름 분석 및 기간별 예측 중..."):
                        ai_result = generate_ai_analysis(
                            provider=ai_provider, api_key=api_key, ticker=ticker, current_price=current_price,
                            ma20=latest['MA20'], rsi=latest['RSI'], vwap=vwap_val, volume=vol_val,
                            macro_text=macro_text, sentiment_status=sentiment_status, pattern_win_rate=win_rate
                        )
                        st.markdown(f"""
                        <div class="ai-report-box" style='background-color:#131722; padding:20px; border:1px solid #26A69A; border-radius:8px; margin-top:10px;'>
                            {ai_result}
                        </div>
                        """, unsafe_allow_html=True)
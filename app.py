import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_datareader.data as web
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- AUTO-REFRESH (5 MINUTOS) ---
try:
    from streamlit_autorefresh import st_autorefresh
    count = st_autorefresh(interval=300000, key="focusterminalupdate")
except:
    pass

# --- 1. CONFIGURAÇÃO VISUAL ---
st.set_page_config(page_title="PRO TERMINAL V24 CARRY", layout="wide", initial_sidebar_state="expanded")

# CSS AGRESSIVO
st.markdown("""
<style>
    .stApp {background-color: #000000;}
    h1, h2, h3, h4 {color: #e0e0e0 !important; font-family: 'Consolas', monospace;}
    p, label, div {color: #b0b0b0 !important;}
    [data-testid="stSidebar"] {background-color: #0e0e0e; border-right: 1px solid #333;}
    .matrix-box {padding: 15px; margin-bottom: 10px; text-align: center; border-radius: 4px; font-weight: bold; font-family: 'Consolas', monospace; font-size: 18px; border: 1px solid #000;}
    .matrix-green {background-color: #00ff00; color: black; box-shadow: 0 0 15px #00ff00;}
    .matrix-red {background-color: #ff0000; color: white; box-shadow: 0 0 15px #ff0000;}
    .matrix-gray {background-color: #333; color: white;}
    div[data-testid="stMetric"] {background-color: #111; border: 1px solid #333; padding: 10px; border-radius: 5px;}
    .stTabs [data-baseweb="tab-list"] {border-bottom: 1px solid #333;}
    .stTabs [data-baseweb="tab"] {background-color: #111; color: #fff;}
    .stTabs [aria-selected="true"] {background-color: #333 !important; color: #ffff00 !important; border-top: 2px solid #ffff00;}
    .signal-box {padding: 20px; border-radius: 8px; text-align: center; margin-bottom: 20px; border: 2px solid;}
    .sig-buy {background-color: #002600; border-color: #00ff00; color: #00ff00;}
    .sig-sell {background-color: #260000; border-color: #ff0000; color: #ff0000;}
    .sig-neutral {background-color: #222; border-color: #888; color: #ccc;}
    
    /* ATR BOX */
    .atr-card {border: 1px solid #444; padding: 10px; border-radius: 5px; background: #0a0a0a; margin-bottom: 5px;}
    .atr-label {font-size: 12px; color: #888;}
    .atr-val {font-size: 16px; font-weight: bold; color: #fff;}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. CONFIGURAÇÃO (29 PARIDADES)
# ==============================================================================
PAIRS_CONFIG = {
    "EURUSD": {"symbol": "EURUSD=X", "base": "EUR", "quote": "USD"},
    "GBPUSD": {"symbol": "GBPUSD=X", "base": "GBP", "quote": "USD"},
    "USDJPY": {"symbol": "USDJPY=X", "base": "USD", "quote": "JPY"},
    "AUDUSD": {"symbol": "AUDUSD=X", "base": "AUD", "quote": "USD"},
    "USDCAD": {"symbol": "USDCAD=X", "base": "USD", "quote": "CAD"},
    "USDCHF": {"symbol": "USDCHF=X", "base": "USD", "quote": "CHF"},
    "NZDUSD": {"symbol": "NZDUSD=X", "base": "NZD", "quote": "USD"},
    "EURGBP": {"symbol": "EURGBP=X", "base": "EUR", "quote": "GBP"},
    "EURJPY": {"symbol": "EURJPY=X", "base": "EUR", "quote": "JPY"},
    "EURCHF": {"symbol": "EURCHF=X", "base": "EUR", "quote": "CHF"},
    "EURAUD": {"symbol": "EURAUD=X", "base": "EUR", "quote": "AUD"},
    "EURCAD": {"symbol": "EURCAD=X", "base": "EUR", "quote": "CAD"},
    "EURNZD": {"symbol": "EURNZD=X", "base": "EUR", "quote": "NZD"},
    "GBPJPY": {"symbol": "GBPJPY=X", "base": "GBP", "quote": "JPY"},
    "GBPCHF": {"symbol": "GBPCHF=X", "base": "GBP", "quote": "CHF"},
    "GBPAUD": {"symbol": "GBPAUD=X", "base": "GBP", "quote": "AUD"},
    "GBPCAD": {"symbol": "GBPCAD=X", "base": "GBP", "quote": "CAD"},
    "GBPNZD": {"symbol": "GBPNZD=X", "base": "GBP", "quote": "NZD"},
    "AUDJPY": {"symbol": "AUDJPY=X", "base": "AUD", "quote": "JPY"},
    "AUDCAD": {"symbol": "AUDCAD=X", "base": "AUD", "quote": "CAD"},
    "AUDCHF": {"symbol": "AUDCHF=X", "base": "AUD", "quote": "CHF"},
    "AUDNZD": {"symbol": "AUDNZD=X", "base": "AUD", "quote": "NZD"},
    "CADJPY": {"symbol": "CADJPY=X", "base": "CAD", "quote": "JPY"},
    "CADCHF": {"symbol": "CADCHF=X", "base": "CAD", "quote": "CHF"},
    "CHFJPY": {"symbol": "CHFJPY=X", "base": "CHF", "quote": "JPY"},
    "NZDJPY": {"symbol": "NZDJPY=X", "base": "NZD", "quote": "JPY"},
    "NZDCAD": {"symbol": "NZDCAD=X", "base": "NZD", "quote": "CAD"},
    "NZDCHF": {"symbol": "NZDCHF=X", "base": "NZD", "quote": "CHF"},
    "XAUUSD (GOLD)": {"symbol": "GC=F", "base": "XAU", "quote": "USD"},
}

FRED_CODES = {
    "USD": {"rate": "FEDFUNDS", "cpi": "CPIAUCSL"},
    "EUR": {"rate": "IRSTCI01EZM156N", "cpi": "CP0000EZ19M086NEST"},
    "JPY": {"rate": "IRSTCI01JPM156N", "cpi": "JPNCPIALLMINMEI"},
    "GBP": {"rate": "IUDSOIA", "cpi": "GBRCPIALLMINMEI"},
    "AUD": {"rate": "IRSTCI01AUM156N", "cpi": "AUSCPIALLMINMEI"},
    "CAD": {"rate": "IRSTCI01CAM156N", "cpi": "CANCPIALLMINMEI"},
    "CHF": {"rate": "IRSTCI01CHM156N", "cpi": "CHECPIALLMINMEI"},
    "NZD": {"rate": "IRSTCI01NZM156N", "cpi": "NZLCPIALLMINMEI"},
}

# ==============================================================================
# 3. SIDEBAR
# ==============================================================================
st.sidebar.title("ATIVO")
selected_pair_key = st.sidebar.selectbox("SELECIONE O PAR:", list(PAIRS_CONFIG.keys()))
current_config = PAIRS_CONFIG[selected_pair_key]
ticker_symbol = current_config["symbol"]
base_curr = current_config["base"]
quote_curr = current_config["quote"]
st.sidebar.markdown("---")
st.sidebar.title("MATRIX 🔴")

# ==============================================================================
# 4. FUNÇÕES DE DADOS E LÓGICA DO ROBÔ
# ==============================================================================
@st.cache_data(ttl=3600*12)
def get_macro_data_dynamic(base, quote):
    def fetch(curr):
        if curr not in FRED_CODES: return None, None 
        try:
            codes = FRED_CODES[curr]
            start = datetime.datetime.now() - datetime.timedelta(days=365*3)
            r = web.DataReader(codes["rate"], 'fred', start).dropna()
            if r.empty: return None, None
            rate = r.iloc[-1,0]
            c = web.DataReader(codes["cpi"], 'fred', start).dropna()
            if len(c) < 13: return rate, None
            inf = ((c.iloc[-1,0] - c.iloc[-13,0]) / c.iloc[-13,0]) * 100
            return rate, inf
        except: return None, None
    br, bi = fetch(base); qr, qi = fetch(quote)
    return br, bi, qr, qi

def fetch_data_universal(symbol, interval, period):
    try:
        df = yf.download(symbol, interval=interval, period=period, progress=False, threads=False)
        if df.empty: return pd.DataFrame()
        df = df.reset_index()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        mapping = {'Date':'time', 'Datetime':'time', 'Open':'open', 'High':'high', 'Low':'low', 'Close':'close', 'Volume':'volume'}
        df.rename(columns=mapping, inplace=True)
        return df
    except: return pd.DataFrame()

@st.cache_data(ttl=290)
def get_market_data(tf_name, symbol):
    configs = {
        "M15": {"i": "15m", "p": "5d"}, "H1": {"i": "1h", "p": "1mo"},
        "H4": {"i": "1h", "p": "1mo"}, "D1": {"i": "1d", "p": "1y"},
        "W1": {"i": "1wk", "p": "5y"}, 
        "MN1": {"i": "1mo", "p": "5y"}
    }
    cfg = configs[tf_name]
    df = fetch_data_universal(symbol, cfg['i'], cfg['p'])
    if df.empty: return pd.DataFrame()
    if tf_name == "H4":
        df = df.set_index('time')
        df = df.resample('4h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna().reset_index()
    df['EMA6'] = df['close'].ewm(span=6, adjust=False).mean()
    vol = df['volume'].replace(0, 1) 
    df['MFC_Line'] = ((df['close'] - df['open']) * vol).ewm(span=14).mean()
    df['Delta_Abs'] = abs(df['close'] - df['open']) * vol
    df['BASE_Power'] = df.apply(lambda x: x['Delta_Abs'] if x['close'] >= x['open'] else 0, axis=1).ewm(span=14).mean()
    df['QUOTE_Power'] = df.apply(lambda x: x['Delta_Abs'] if x['close'] < x['open'] else 0, axis=1).ewm(span=14).mean()
    return df

def analyze_signal(df, spread_macro):
    score = 0
    if spread_macro is not None:
        if spread_macro > 0.5: score -= 2
        elif spread_macro < -0.5: score += 2
    if df['close'].iloc[-1] > df['EMA6'].iloc[-1]: score += 1
    else: score -= 1
    if df['BASE_Power'].iloc[-1] > df['QUOTE_Power'].iloc[-1]: score += 1
    else: score -= 1
    if df['MFC_Line'].iloc[-1] > 0: score += 1
    else: score -= 1
    return score

def analyze_signal_verbose(df, spread_macro, base_c, quote_c):
    score = 0; reasons = []
    if spread_macro is not None:
        if spread_macro > 0.5: score -= 2; reasons.append("📉 MACRO Venda")
        elif spread_macro < -0.5: score += 2; reasons.append("📈 MACRO Compra")
    if df['close'].iloc[-1] > df['EMA6'].iloc[-1]: score += 1; reasons.append("📈 Preço > EMA6")
    else: score -= 1; reasons.append("📉 Preço < EMA6")
    if df['BASE_Power'].iloc[-1] > df['QUOTE_Power'].iloc[-1]: score += 1; reasons.append(f"🛡️ Força {base_c}")
    else: score -= 1; reasons.append(f"⚔️ Força {quote_c}")
    if df['MFC_Line'].iloc[-1] > 0: score += 1
    else: score -= 1
    return score, reasons

# ATR CALC
def get_atr_targets(symbol):
    try:
        d1 = fetch_data_universal(symbol, "1d", "3mo")
        w1 = fetch_data_universal(symbol, "1wk", "1y")
        mn1 = fetch_data_universal(symbol, "1mo", "2y")
        def calc_atr_values(df):
            if len(df) < 15: return None, None, None
            df['tr'] = pd.concat([df['high']-df['low'], (df['high']-df['close'].shift()).abs(), (df['low']-df['close'].shift()).abs()], axis=1).max(axis=1)
            atr = df['tr'].ewm(span=14).mean().iloc[-1]
            return atr, df['open'].iloc[-1], df['close'].iloc[-1]
        targets = {}
        atr_d, open_d, curr_d = calc_atr_values(d1)
        if atr_d: targets['D1'] = {'top': open_d + atr_d, 'bottom': open_d - atr_d, 'atr': atr_d, 'curr': curr_d}
        atr_w, open_w, curr_w = calc_atr_values(w1)
        if atr_w: targets['W1'] = {'top': open_w + atr_w, 'bottom': open_w - atr_w, 'atr': atr_w, 'curr': curr_w}
        atr_m, open_m, curr_m = calc_atr_values(mn1)
        if atr_m: targets['MN1'] = {'top': open_m + atr_m, 'bottom': open_m - atr_m, 'atr': atr_m, 'curr': curr_m}
        return targets
    except: return {}

# ==============================================================================
# 5. EXECUÇÃO
# ==============================================================================
base_r, base_i, quote_r, quote_i = get_macro_data_dynamic(base_curr, quote_curr)
spread = None
if base_r is not None and quote_r is not None and base_i is not None and quote_i is not None:
    spread = (base_r - base_i) - (quote_r - quote_i)

tfs = ["M15", "H1", "H4", "D1", "W1", "MN1"]
cache = {}

for tf in tfs:
    df = get_market_data(tf, ticker_symbol)
    cache[tf] = df
    if not df.empty:
        if tf in ["M15", "H1", "H4"]:
            last = df['close'].iloc[-1]; ema = df['EMA6'].iloc[-1]
            status_color = "matrix-green" if last > ema else "matrix-red"
            icon = "▲" if last > ema else "▼"
            label = "TIMING"
        else:
            sc = analyze_signal(df, spread)
            if sc > 0: status_color = "matrix-green"; icon = "🛡️"
            elif sc < 0: status_color = "matrix-red"; icon = "⚔️"
            else: status_color = "matrix-gray"; icon = "⚖️"
            label = "MACRO"
        st.sidebar.markdown(f"<div class='matrix-box {status_color}'>{tf} {icon}<br><span style='font-size:10px; color:black; opacity:0.7'>{label}</span></div>", unsafe_allow_html=True)

st.title(f"{selected_pair_key} COMMAND LIVE (Update: {datetime.datetime.now().strftime('%H:%M')})")

c1, c2, c3, c4 = st.columns(4)
def fmt(v): return f"{v}%" if v is not None else "N/D"
c1.metric(f"{base_curr} RATE", fmt(base_r)); c2.metric(f"{quote_curr} RATE", fmt(quote_r))
c3.metric("SPREAD REAL", f"{spread:.2f}%" if spread else "N/D")
if spread: 
    if spread > 0.5: c4.success(f"VIÉS: COMPRA {base_curr}")
    elif spread < -0.5: c4.error(f"VIÉS: VENDA {base_curr}")
    else: c4.warning("LATERAL")
else: c4.info("MACRO OFF")

# --- VOLATILIDADE 3 ANOS ---
st.markdown("---")
st.header("📊 VOLATILIDADE SEMANAL (3 ANOS)")
df_w1_vol = cache["W1"]
if not df_w1_vol.empty and len(df_w1_vol) > 150:
    df_3y = df_w1_vol.tail(156).copy()
    df_3y['RangePct'] = ((df_3y['high'] - df_3y['low']) / df_3y['open']) * 100
    avg_move = df_3y['RangePct'].mean()
    curr_move = df_3y['RangePct'].iloc[-1]
    remaining = avg_move - curr_move
    vc1, vc2, vc3 = st.columns(3)
    vc1.metric("MÉDIA MOVIMENTO (3 ANOS)", f"{avg_move:.2f}%")
    vc2.metric("MOVIMENTO ATUAL (SEMANA)", f"{curr_move:.2f}%")
    if remaining > 0: vc3.metric("FALTA PARA ALCANÇAR", f"{remaining:.2f}%", delta="Espaço disponível")
    else: vc3.metric("FALTA PARA ALCANÇAR", f"0.00%", delta="Exaustão", delta_color="inverse"); st.warning(f"Exaustão! Já moveu {curr_move:.2f}% (Média: {avg_move:.2f}%)")
else: st.info("Calculando volatilidade...")

# --- ATR TARGETS ---
st.markdown("---")
st.header("🎯 ATR SNIPER TARGETS (LIMITE DE VOLATILIDADE)")
atr_data = get_atr_targets(ticker_symbol)
if atr_data:
    ac1, ac2, ac3 = st.columns(3)
    if 'D1' in atr_data:
        d = atr_data['D1']
        with ac1: st.markdown(f"<div class='atr-card'><div class='atr-label'>DIÁRIO (D1)</div><div style='color:#00ff00'>⬆️ {d['top']:.5f}</div><div style='color:#ff0000'>⬇️ {d['bottom']:.5f}</div><div class='atr-label'>ATR: {d['atr']:.5f}</div></div>", unsafe_allow_html=True)
    if 'W1' in atr_data:
        w = atr_data['W1']
        with ac2: st.markdown(f"<div class='atr-card'><div class='atr-label'>SEMANAL (W1)</div><div style='color:#00ff00'>⬆️ {w['top']:.5f}</div><div style='color:#ff0000'>⬇️ {w['bottom']:.5f}</div><div class='atr-label'>ATR: {w['atr']:.5f}</div></div>", unsafe_allow_html=True)
    if 'MN1' in atr_data:
        m = atr_data['MN1']
        with ac3: st.markdown(f"<div class='atr-card'><div class='atr-label'>MENSAL (MN1)</div><div style='color:#00ff00'>⬆️ {m['top']:.5f}</div><div style='color:#ff0000'>⬇️ {m['bottom']:.5f}</div><div class='atr-label'>ATR: {m['atr']:.5f}</div></div>", unsafe_allow_html=True)
else: st.info("Calculando alvos ATR...")

st.markdown("---")

tabs = st.tabs(tfs)
for i, tf in enumerate(tfs):
    with tabs[i]:
        df = cache[tf]
        if not df.empty:
            score, reasons = analyze_signal_verbose(df, spread, base_curr, quote_curr)
            if score >= 3: txt="COMPRA FORTE"; css="sig-buy"
            elif score > 0: txt="COMPRA MODERADA"; css="sig-buy"
            elif score <= -3: txt="VENDA FORTE"; css="sig-sell"
            elif score < 0: txt="VENDA MODERADA"; css="sig-sell"
            else: txt="NEUTRO"; css="sig-neutral"
            st.markdown(f"<div class='signal-box {css}'><h2>{txt} ({score})</h2></div>", unsafe_allow_html=True)
            with st.expander("Ver Detalhes do Diagnóstico"):
                for r in reasons: st.write(r)
            
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.5, 0.15, 0.15, 0.2],
                                subplot_titles=(f"PRICE {tf}", "MFC", "TREND", "BATALHA"))
            fig.add_trace(go.Candlestick(x=df['time'], open=df['open'], high=df['high'], low=df['low'], close=df['close']), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['time'], y=df['EMA6'], line=dict(color='#00FFFF', width=2)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['time'], y=df['MFC_Line'], line=dict(color='#FFFF00'), fill='tozeroy'), row=2, col=1)
            fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)
            fig.add_trace(go.Scatter(x=df['time'], y=df['MFC_Line'].ewm(span=9).mean(), line=dict(color='#ff00ff', dash='dot')), row=3, col=1)
            fig.add_trace(go.Scatter(x=df['time'], y=df['BASE_Power'], line=dict(color='#0000FF', width=2), name=base_curr), row=4, col=1)
            fig.add_trace(go.Scatter(x=df['time'], y=df['QUOTE_Power'], line=dict(color='#FF0000', width=2), name=quote_curr), row=4, col=1)
            fig.update_layout(paper_bgcolor='black', plot_bgcolor='#111', font=dict(color='#ccc'), xaxis_rangeslider_visible=False, height=1000, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

# --- SCANNER DUPLO (GENERALS + ROYAL FLUSH + CARRY) ---
st.header("🏆 PORTFOLIO GENERALS")
@st.cache_data(ttl=14400)
def run_scanner_enhanced():
    results = []
    progress_bar = st.progress(0); total = len(PAIRS_CONFIG)
    for idx, (nome_par, cfg) in enumerate(PAIRS_CONFIG.items()):
        progress_bar.progress((idx + 1) / total)
        sym = cfg['symbol']; base = cfg['base']; quote = cfg['quote']
        try:
            b_r, b_i, q_r, q_i = get_macro_data_dynamic(base, quote)
            local_spread = None
            if b_r is not None and q_r is not None: local_spread = (b_r - b_i) - (q_r - q_i)
            
            # CÁLCULO DE CARRY (NOMINAL DIFF)
            carry_diff = (b_r - q_r) if (b_r and q_r) else 0.0
            
            def process_tf_score(interval, period):
                df_raw = fetch_data_universal(sym, interval, period)
                if df_raw.empty or len(df_raw) < 14: return 0, df_raw
                df_raw['EMA6'] = df_raw['close'].ewm(span=6, adjust=False).mean()
                vol = df_raw['volume'].replace(0, 1)
                df_raw['MFC_Line'] = ((df_raw['close'] - df_raw['open']) * vol).ewm(span=14).mean()
                df_raw['Delta_Abs'] = abs(df_raw['close'] - df_raw['open']) * vol
                df_raw['BASE_Power'] = df_raw.apply(lambda x: x['Delta_Abs'] if x['close'] >= x['open'] else 0, axis=1).ewm(span=14).mean()
                df_raw['QUOTE_Power'] = df_raw.apply(lambda x: x['Delta_Abs'] if x['close'] < x['open'] else 0, axis=1).ewm(span=14).mean()
                return analyze_signal(df_raw, local_spread), df_raw
            
            s_d1, df_d1 = process_tf_score("1d", "1y"); s_w1, _ = process_tf_score("1wk", "5y"); s_mn1, _ = process_tf_score("1mo", "5y")
            final = 0
            if s_d1 > 0 and s_w1 > 0 and s_mn1 > 0: final = 3
            elif s_d1 < 0 and s_w1 < 0 and s_mn1 < 0: final = -3
            
            is_royal = False
            swap_tag = ""
            
            if final != 0:
                # Royal Logic
                def check_soldier(interval, period, direction):
                    df_s = fetch_data_universal(sym, interval, period)
                    if df_s.empty: return False
                    df_s['EMA6'] = df_s['close'].ewm(span=6, adjust=False).mean()
                    last_p = df_s['close'].iloc[-1]; last_ema = df_s['EMA6'].iloc[-1]
                    return last_p > last_ema if direction > 0 else last_p < last_ema
                if check_soldier("1h", "1mo", final) and check_soldier("15m", "5d", final): is_royal = True
                
                # Swap Logic
                if final > 0 and carry_diff > 0.1: swap_tag = " [SWAP+]"
                elif final < 0 and carry_diff < -0.1: swap_tag = " [SWAP+]"
                
                last_p = df_d1['close'].iloc[-1]; prev_p = df_d1['close'].iloc[-2]
                
                results.append({
                    "Par": f"{nome_par}{swap_tag}", 
                    "Preço": last_p, 
                    "Var(%)": ((last_p - prev_p) / prev_p) * 100, 
                    "Tendência": "BULL 🛡️" if final > 0 else "BEAR ⚔️", 
                    "RawScore": final, 
                    "Royal": is_royal,
                    "Carry(%)": f"{carry_diff:.2f}%"
                })
        except: continue
    progress_bar.empty()
    return pd.DataFrame(results)

with st.spinner("Escaneando..."):
    df_res = run_scanner_enhanced()

if not df_res.empty:
    top_buy = df_res[df_res['RawScore'] == 3].sort_values(by='Var(%)', ascending=False)
    top_sell = df_res[df_res['RawScore'] == -3].sort_values(by='Var(%)', ascending=True)
    c1, c2 = st.columns(2)
    with c1: st.success("COMPRA GENERAL"); st.dataframe(top_buy[['Par', 'Preço', 'Var(%)', 'Carry(%)', 'Tendência']], use_container_width=True)
    with c2: st.error("VENDA GENERAL"); st.dataframe(top_sell[['Par', 'Preço', 'Var(%)', 'Carry(%)', 'Tendência']], use_container_width=True)

    st.markdown("---"); st.header("💎 ROYAL FLUSH (TOTAL ALIGNMENT)")
    rb = df_res[(df_res['RawScore'] == 3) & (df_res['Royal'] == True)].sort_values(by='Var(%)', ascending=False)
    rs = df_res[(df_res['RawScore'] == -3) & (df_res['Royal'] == True)].sort_values(by='Var(%)', ascending=True)
    rc1, rc2 = st.columns(2)
    with rc1: st.success("ROYAL COMPRA"); st.dataframe(rb[['Par', 'Preço', 'Var(%)', 'Carry(%)']], use_container_width=True)
    with rc2: st.error("ROYAL VENDA"); st.dataframe(rs[['Par', 'Preço', 'Var(%)', 'Carry(%)']], use_container_width=True)
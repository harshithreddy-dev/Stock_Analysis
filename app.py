import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="FinSight", page_icon="📊", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ── Force entire app white/light ── */
html, body, .stApp, [data-testid="stAppViewContainer"] {
    background-color: #f1f5f9 !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Sidebar force white ── */
[data-testid="stSidebar"], 
[data-testid="stSidebar"] > div,
[data-testid="stSidebar"] > div:first-child {
    background-color: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
}

/* ── Sidebar text force dark ── */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] .stMarkdown {
    color: #1e293b !important;
}

/* ── Sidebar caption ── */
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] small {
    color: #64748b !important;
}

/* ── Sidebar input boxes ── */
[data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] select,
[data-testid="stSidebar"] .stSelectbox div {
    background-color: #f8fafc !important;
    color: #1e293b !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
}

/* ── Sidebar slider ── */
[data-testid="stSidebar"] .stSlider > div > div > div {
    background: #1a3a6b !important;
}

/* ── Sidebar button ── */
[data-testid="stSidebar"] .stButton > button {
    background: #1a3a6b !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    width: 100% !important;
    padding: 0.65rem !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    margin-top: 0.5rem !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #0f2444 !important;
}

/* ── Main content area ── */
.block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; }

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #0f2444 0%, #1a3a6b 60%, #1e4d8c 100%);
    border-radius: 16px;
    padding: 2.5rem 2.5rem;
    margin-bottom: 1.5rem;
    color: white;
}
.hero .tag {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-bottom: 1rem;
    letter-spacing: 1px;
}
.hero h1 { font-size: 2.4rem; font-weight: 700; margin: 0 0 0.5rem 0; }
.hero p  { font-size: 1rem; opacity: 0.85; margin: 0; line-height: 1.7; }

/* ── Section title ── */
.section-title {
    font-size: 1rem;
    font-weight: 700;
    color: #0f2444;
    border-left: 4px solid #1a3a6b;
    padding-left: 0.75rem;
    margin: 1.8rem 0 1rem 0;
}

/* ── Step cards ── */
.step-grid { display: flex; gap: 1rem; margin: 1rem 0 1.5rem 0; flex-wrap: wrap; }
.step-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.3rem 1.4rem;
    flex: 1; min-width: 180px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.step-card .num   { font-size: 1.8rem; font-weight: 700; color: #1a3a6b; line-height: 1; margin-bottom: 0.5rem; }
.step-card .label { font-size: 0.9rem; font-weight: 600; color: #1e293b; margin-bottom: 0.3rem; }
.step-card .desc  { font-size: 0.8rem; color: #64748b; line-height: 1.6; }

/* ── Tip boxes ── */
.tip-box {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-size: 0.83rem;
    color: #334155;
    margin-bottom: 0.8rem;
    line-height: 1.6;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
}
.tip-box b { color: #1a3a6b; }

/* ── Glossary ── */
.glossary-grid { display: flex; flex-wrap: wrap; gap: 0.8rem; margin-top: 0.5rem; }
.glossary-item {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 0.9rem 1rem;
    flex: 1; min-width: 200px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
}
.glossary-item .term { font-size: 0.85rem; font-weight: 700; color: #1a3a6b; margin-bottom: 0.25rem; }
.glossary-item .def  { font-size: 0.78rem; color: #475569; line-height: 1.5; }

/* ── KPI cards ── */
.kpi-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.2rem 1rem;
    text-align: center;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.kpi-label { font-size: 0.72rem; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 0.3rem; }
.kpi-value { font-size: 1.9rem; font-weight: 700; }
.kpi-hint  { font-size: 0.72rem; color: #94a3b8; margin-top: 0.2rem; }

/* ── Suggestion cards ── */
.sug-ok   { background:#f0fdf4; border:1px solid #bbf7d0; border-radius:10px; padding:1rem 1.2rem; margin-bottom:0.7rem; color:#166534; font-size:0.88rem; line-height:1.6; }
.sug-warn { background:#fffbeb; border:1px solid #fde68a; border-radius:10px; padding:1rem 1.2rem; margin-bottom:0.7rem; color:#92400e; font-size:0.88rem; line-height:1.6; }

/* ── Dataframe ── */
.stDataFrame { border-radius: 10px !important; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ──
with st.sidebar:
    st.markdown("""
    <div style="padding: 0.5rem 0 1rem 0;">
        <div style="font-size:1.3rem; font-weight:700; color:#0f2444;">📊 FinSight</div>
        <div style="font-size:0.78rem; color:#64748b; margin-top:2px;">Portfolio Analyzer</div>
    </div>
    <hr style="border:none; border-top:1px solid #e2e8f0; margin: 0 0 1rem 0;">
    <div style="font-size:0.8rem; font-weight:700; color:#1a3a6b; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:0.3rem;">
        Step 1 — Stock Tickers
    </div>
    <div style="font-size:0.75rem; color:#64748b; margin-bottom:0.5rem;">
        Indian stocks need <b>.NS</b> at the end (e.g. RELIANCE.NS)
    </div>
    """, unsafe_allow_html=True)

    ticker_input = st.text_area("Tickers", "RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS", label_visibility="collapsed")
    tickers = [t.strip() for t in ticker_input.split(",") if t.strip()]

    st.markdown("""
    <div style="font-size:0.8rem; font-weight:700; color:#1a3a6b; text-transform:uppercase; letter-spacing:0.5px; margin: 1rem 0 0.3rem 0;">
        Step 2 — Time Period
    </div>
    <div style="font-size:0.75rem; color:#64748b; margin-bottom:0.5rem;">How far back should we look?</div>
    """, unsafe_allow_html=True)

    period = st.selectbox("Period", ["6mo", "1y", "2y"], index=1, label_visibility="collapsed",
        format_func=lambda x: {"6mo": "Last 6 Months", "1y": "Last 1 Year", "2y": "Last 2 Years"}[x])

    st.markdown("""
    <div style="font-size:0.8rem; font-weight:700; color:#1a3a6b; text-transform:uppercase; letter-spacing:0.5px; margin: 1rem 0 0.3rem 0;">
        Step 3 — Your Allocation
    </div>
    <div style="font-size:0.75rem; color:#64748b; margin-bottom:0.5rem;">How much % of your money is in each stock?</div>
    """, unsafe_allow_html=True)

    weights = {}
    total = 0
    for t in tickers:
        default = 100 // len(tickers)
        weights[t] = st.slider(t, 0, 100, default) / 100
        total += weights[t]

    if abs(total - 1.0) > 0.05:
        st.markdown(f'<div style="background:#fff7ed;border:1px solid #fed7aa;border-radius:8px;padding:0.6rem 0.8rem;font-size:0.78rem;color:#9a3412;margin-top:0.5rem;">⚠️ Weights add up to {total:.0%}. Adjust to reach 100%.</div>', unsafe_allow_html=True)

    st.markdown('<hr style="border:none;border-top:1px solid #e2e8f0;margin:1rem 0;">', unsafe_allow_html=True)
    analyze = st.button("🔍 Analyze Portfolio")

# ── HERO ──
st.markdown("""
<div class="hero">
    <div class="tag">PORTFOLIO ANALYTICS</div>
    <h1>📊 FinSight</h1>
    <p>Enter your stocks on the left and click <b>Analyze Portfolio</b><br>
    to get a full performance, risk &amp; AI rebalancing report — instantly.</p>
</div>
""", unsafe_allow_html=True)

# ── LANDING PAGE ──
if not analyze:
    st.markdown('<div class="section-title">How It Works</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="step-grid">
        <div class="step-card"><div class="num">01</div><div class="label">Enter Your Stocks</div><div class="desc">Type tickers in the sidebar. Indian stocks need .NS at the end (e.g. RELIANCE.NS).</div></div>
        <div class="step-card"><div class="num">02</div><div class="label">Set Allocations</div><div class="desc">Drag sliders to set how much % of your money is in each stock. Should total 100%.</div></div>
        <div class="step-card"><div class="num">03</div><div class="label">Click Analyze</div><div class="desc">FinSight fetches live data and calculates performance, risk, and AI suggestions.</div></div>
        <div class="step-card"><div class="num">04</div><div class="label">Read Your Report</div><div class="desc">See growth charts, sector exposure, risk vs return, and AI rebalancing flags.</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">What You\'ll See</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="tip-box">📈 <b>Portfolio Growth Chart</b><br>See how ₹100 invested would have grown over time — each line is one stock.</div>', unsafe_allow_html=True)
        st.markdown('<div class="tip-box">🥧 <b>Allocation Pie</b><br>Visual breakdown of how your money is split across your stocks.</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="tip-box">🌍 <b>Sector Exposure</b><br>Too much in one industry (IT, Banking) makes you vulnerable. This chart shows the spread.</div>', unsafe_allow_html=True)
        st.markdown('<div class="tip-box">🔗 <b>Correlation Matrix</b><br>Stocks that move together offer less safety. This shows how connected your picks are.</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="tip-box">⚖️ <b>Risk vs Return</b><br>Is each stock earning its risk? Top-left = ideal. Bottom-right = avoid.</div>', unsafe_allow_html=True)
        st.markdown('<div class="tip-box">🤖 <b>AI Rebalancing</b><br>Machine learning flags if you\'re over-concentrated or holding a drag on your portfolio.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">📖 Beginner\'s Glossary</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="glossary-grid">
        <div class="glossary-item"><div class="term">Annual Return</div><div class="def">How much your investment grew in one year (%). Higher is better.</div></div>
        <div class="glossary-item"><div class="term">Sharpe Ratio</div><div class="def">Return adjusted for risk. Above 1.0 is good. Below 0 = risk not worth it.</div></div>
        <div class="glossary-item"><div class="term">Volatility</div><div class="def">How much the stock price swings up and down. High = more risky.</div></div>
        <div class="glossary-item"><div class="term">Max Drawdown</div><div class="def">The biggest % drop from a peak. -20% = fell 20% from its highest point.</div></div>
        <div class="glossary-item"><div class="term">Diversification</div><div class="def">Spreading money across sectors so one bad stock doesn't hurt everything.</div></div>
        <div class="glossary-item"><div class="term">Rebalancing</div><div class="def">Adjusting your stock weights to manage risk and improve returns over time.</div></div>
    </div>
    """, unsafe_allow_html=True)

# ── ANALYSIS ──
else:
    with st.spinner("Fetching live market data..."):
        try:
            prices = yf.download(tickers, period=period, auto_adjust=True, progress=False)["Close"]
            if isinstance(prices, pd.Series):
                prices = prices.to_frame(name=tickers[0])
            prices = prices.dropna()
            sectors = {}
            for t in tickers:
                try:
                    sectors[t] = yf.Ticker(t).info.get("sector", "Unknown")
                except:
                    sectors[t] = "Unknown"
        except Exception as e:
            st.error(f"Could not fetch data: {e}. Check your ticker symbols.")
            st.stop()

    dr       = prices.pct_change().dropna()
    cum_ret  = (1 + dr).cumprod() - 1
    ann_ret  = dr.mean() * 252
    ann_vol  = dr.std() * np.sqrt(252)
    sharpe   = (ann_ret - 0.065) / ann_vol
    max_dd   = ((prices - prices.cummax()) / prices.cummax()).min()
    w        = pd.Series(weights)

    port_ret    = float((ann_ret * w).sum())
    port_vol    = float((ann_vol * w).sum())
    port_sharpe = float((sharpe * w).sum())
    port_dd     = float((max_dd * w).sum())

    # KPIs
    st.markdown('<div class="section-title">Portfolio Overview</div>', unsafe_allow_html=True)
    def kpi(label, val, hint, color):
        return f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value" style="color:{color}">{val}</div><div class="kpi-hint">{hint}</div></div>'

    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(kpi("Annual Return",  f"{port_ret:.1%}",    "Higher = better",         "#16a34a" if port_ret>0 else "#dc2626"), unsafe_allow_html=True)
    c2.markdown(kpi("Sharpe Ratio",   f"{port_sharpe:.2f}", ">1.0 good, <0 risky",     "#16a34a" if port_sharpe>1 else ("#f59e0b" if port_sharpe>0 else "#dc2626")), unsafe_allow_html=True)
    c3.markdown(kpi("Volatility",     f"{port_vol:.1%}",    "How much prices swing",   "#0f2444"), unsafe_allow_html=True)
    c4.markdown(kpi("Max Drawdown",   f"{port_dd:.1%}",     "Worst drop from peak",    "#dc2626" if port_dd<-0.2 else "#f59e0b"), unsafe_allow_html=True)

    # Growth chart
    st.markdown('<div class="section-title">📈 Portfolio Growth Over Time</div>', unsafe_allow_html=True)
    st.caption("How ₹100 invested would have grown — each line is one stock.")
    fig = px.line(cum_ret*100, labels={"value":"Return (%)","index":"Date","variable":"Stock"}, color_discrete_sequence=px.colors.qualitative.Bold)
    fig.update_layout(paper_bgcolor="white", plot_bgcolor="#f8fafc", hovermode="x unified", margin=dict(l=10,r=10,t=10,b=10))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#e2e8f0", zeroline=True, zerolinecolor="#cbd5e1")
    st.plotly_chart(fig, use_container_width=True)

    # Pie + Sector
    c1,c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">🥧 How Your Money Is Split</div>', unsafe_allow_html=True)
        st.caption("Current allocation across your stocks.")
        fig2 = px.pie(values=list(weights.values()), names=list(weights.keys()), hole=0.45, color_discrete_sequence=px.colors.qualitative.Bold)
        fig2.update_layout(margin=dict(l=10,r=10,t=10,b=10), paper_bgcolor="white")
        st.plotly_chart(fig2, use_container_width=True)
    with c2:
        st.markdown('<div class="section-title">🌍 Sector Exposure</div>', unsafe_allow_html=True)
        st.caption("High exposure to one sector = higher risk if that industry falls.")
        sdf = pd.DataFrame({"sector":sectors,"weight":weights}).groupby("sector")["weight"].sum().reset_index()
        sdf["pct"] = sdf["weight"]*100
        fig3 = px.bar(sdf, x="sector", y="pct", text=sdf["pct"].apply(lambda x:f"{x:.0f}%"),
                      color="pct", color_continuous_scale=["#bfdbfe","#1a3a6b"],
                      labels={"pct":"Allocation (%)","sector":"Sector"})
        fig3.update_layout(paper_bgcolor="white", plot_bgcolor="#f8fafc", coloraxis_showscale=False, margin=dict(l=10,r=10,t=10,b=10))
        fig3.update_traces(textposition="outside")
        fig3.update_xaxes(showgrid=False)
        fig3.update_yaxes(gridcolor="#e2e8f0")
        st.plotly_chart(fig3, use_container_width=True)

    # Risk vs Return
    st.markdown('<div class="section-title">⚖️ Risk vs Return</div>', unsafe_allow_html=True)
    st.caption("Top-left = low risk, high return (ideal). Bottom-right = high risk, low return (avoid).")
    scat = pd.DataFrame({"Annual Return (%)":ann_ret*100,"Volatility (Risk %)":ann_vol*100,"Sharpe":sharpe,"Weight":pd.Series({t:weights[t]*100 for t in tickers})})
    fig4 = px.scatter(scat, x="Volatility (Risk %)", y="Annual Return (%)", text=scat.index,
                      size="Weight", color="Sharpe", color_continuous_scale="RdYlGn", size_max=50)
    fig4.update_traces(textposition="top center", marker=dict(opacity=0.85, line=dict(width=1,color="white")))
    fig4.update_layout(paper_bgcolor="white", plot_bgcolor="#f8fafc", margin=dict(l=10,r=10,t=10,b=10))
    fig4.update_xaxes(gridcolor="#e2e8f0")
    fig4.update_yaxes(gridcolor="#e2e8f0")
    st.plotly_chart(fig4, use_container_width=True)

    # Correlation
    st.markdown('<div class="section-title">🔗 Correlation Between Stocks</div>', unsafe_allow_html=True)
    st.caption("Close to 1.0 = move together (less safe). Close to 0 = independent (better diversification).")
    fig5 = px.imshow(dr.corr(), text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    fig5.update_layout(margin=dict(l=10,r=10,t=10,b=10), paper_bgcolor="white")
    st.plotly_chart(fig5, use_container_width=True)

    # ML Rebalancing
    st.markdown('<div class="section-title">🤖 AI Rebalancing Suggestions</div>', unsafe_allow_html=True)
    st.caption("Machine learning analyses your risk clusters and flags concentration issues.")

    feat = pd.DataFrame({"return":ann_ret,"volatility":ann_vol,"sharpe":sharpe})
    if len(tickers) >= 2:
        feat["cluster"] = KMeans(n_clusters=min(3,len(tickers)), random_state=42, n_init=10).fit_predict(StandardScaler().fit_transform(feat))
    else:
        feat["cluster"] = 0
    feat["weight"] = pd.Series(weights)
    feat["sector"] = pd.Series(sectors)

    sugg = []
    for sec, exp in feat.groupby("sector")["weight"].sum().items():
        if exp > 0.4:
            sugg.append(("warn", f"<b>Sector overload — {sec} ({exp:.0%})</b><br>More than 40% in one sector is risky. Consider spreading into other industries."))
    for tkr, row in feat.iterrows():
        if row["sharpe"] < 0 and row["weight"] > 0.15:
            sugg.append(("warn", f"<b>{tkr} — Not earning its risk</b><br>Negative Sharpe ratio but {row['weight']:.0%} of your portfolio. Consider reducing this position."))
    for cl, exp in feat.groupby("cluster")["weight"].sum().items():
        if exp > 0.6:
            sugg.append(("warn", f"<b>Risk cluster concentration ({exp:.0%})</b><br>Most stocks behave similarly. Add stocks from different risk profiles for better protection."))
    if not sugg:
        sugg.append(("ok", "<b>✅ Portfolio looks healthy!</b><br>No major issues detected. Keep monitoring and rebalance if any stock grows beyond 40% of your portfolio."))

    for stype, msg in sugg:
        st.markdown(f'<div class="{"sug-ok" if stype=="ok" else "sug-warn"}">{"✅" if stype=="ok" else "⚠️"} {msg}</div>', unsafe_allow_html=True)

    # Table
    st.markdown('<div class="section-title">📋 Stock-by-Stock Breakdown</div>', unsafe_allow_html=True)
    st.caption("Full summary of every stock in your portfolio.")
    st.dataframe(pd.DataFrame({
        "Stock": tickers,
        "Sector": [sectors.get(t,"Unknown") for t in tickers],
        "Allocation": [f"{weights[t]:.0%}" for t in tickers],
        "Annual Return": [f"{ann_ret[t]:.1%}" for t in tickers],
        "Volatility": [f"{ann_vol[t]:.1%}" for t in tickers],
        "Sharpe Ratio": [f"{sharpe[t]:.2f}" for t in tickers],
        "Max Drawdown": [f"{max_dd[t]:.1%}" for t in tickers],
    }), use_container_width=True, hide_index=True)
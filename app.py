import os
import json
import time
import requests
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─── 1. CONFIGURATION ────────────────────────────────────────
load_dotenv()

SERPER_API_KEY = st.secrets.get("SERPER_API_KEY") or os.getenv("SERPER_API_KEY")
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

# ─── 2. CORE LOGIC (UNCHANGED) ───────────────────────────────

def get_reviews_data(product_name):
    if not SERPER_API_KEY:
        st.error("Serper API Key missing!")
        return None
    url = "https://google.serper.dev/search"
    payload = {
        "q": f"{product_name} user reviews pros cons reddit amazon",
        "num": 10, "gl": "us", "autocorrect": True
    }
    headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        response.raise_for_status()
        results = response.json()
        combined_text = ""
        organic = results.get('organic', [])
        if not organic:
            return None
        for item in organic:
            snippet = item.get('snippet', '')
            combined_text += f"{snippet} "
        return combined_text.strip()
    except Exception as e:
        st.error(f"Search API Error: {e}")
        return None


def analyze_sentiment(review_text, product_name):
    if not GEMINI_API_KEY:
        st.error("Gemini API Key missing!")
        return None
    genai.configure(api_key=GEMINI_API_KEY)
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
    except:
        model = genai.GenerativeModel('gemini-flash-latest')
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
    prompt = f"""
    ROLE: You are a Senior Product Strategist and Market Analyst.
    TASK: Provide a deep-dive sentiment analysis for '{product_name}' based on the following web data:
    
    DATA:
    {review_text}
    
    INSTRUCTIONS:
    1. Summarize the 'vibe' in a sophisticated, 2-3 sentence paragraph. 
    2. Identify the top 3 'Strengths'—be specific (e.g., instead of "Good battery," say "Exceptional 20-hour battery life even with ANC active").
    3. Identify the top 3 'Weaknesses'—focus on recurring user complaints or deal-breakers.
    4. Assign a precise 'Sentiment Score' from 0-100 based on the data.

    OUTPUT FORMAT (STRICT JSON ONLY):
    {{
        "score": 85,
        "vibe": "Detailed 2-3 sentence analysis here...",
        "pros": ["Detailed strength 1", "Detailed strength 2", "Detailed strength 3"],
        "cons": ["Detailed weakness 1", "Detailed weakness 2", "Detailed weakness 3"]
    }}
    """
    for attempt in range(3):
        try:
            response = model.generate_content(
                prompt,
                safety_settings=safety_settings,
                generation_config={"response_mime_type": "application/json"}
            )
            if response and response.text:
                return json.loads(response.text)
        except Exception as e:
            if "429" in str(e):
                wait = (attempt + 1) * 15
                st.warning(f"Quota busy. Retrying in {wait}s…")
                time.sleep(wait)
                continue
            st.error(f"AI Error: {e}")
            break
    return None


# ─── 3. COMPARE HELPERS (NEW) ────────────────────────────────

def get_price(product_name):
    """Fetch estimated retail price via Serper shopping search."""
    if not SERPER_API_KEY:
        return "N/A"
    url = "https://google.serper.dev/shopping"
    payload = {"q": product_name, "num": 3, "gl": "us"}
    headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        items = response.json().get('shopping', [])
        prices = []
        for item in items:
            p = item.get('price', '')
            if p:
                # Strip currency symbol and commas, parse float
                cleaned = p.replace('$', '').replace(',', '').strip()
                try:
                    prices.append(float(cleaned))
                except ValueError:
                    pass
        if prices:
            avg = sum(prices) / len(prices)
            return f"~${avg:,.0f}"
        return "N/A"
    except Exception:
        return "N/A"


def safe_price_float(price_str):
    """Convert any price representation (str or number) to float. Returns 0.0 on failure."""
    try:
        cleaned = str(price_str).replace('~', '').replace('$', '').replace(',', '').strip()
        return float(cleaned) if cleaned and cleaned != 'N/A' else 0.0
    except (ValueError, TypeError):
        return 0.0


def analyze_compare(review_text, product_name):
    """Like analyze_sentiment but also returns a price_estimate."""
    if not GEMINI_API_KEY:
        return None
    genai.configure(api_key=GEMINI_API_KEY)
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
    except:
        model = genai.GenerativeModel('gemini-flash-latest')
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
    prompt = f"""
    ROLE: You are a Senior Product Strategist and Market Analyst.
    TASK: Analyse '{product_name}' from the following web data for a head-to-head comparison.

    DATA:
    {review_text}

    INSTRUCTIONS:
    1. Assign a Sentiment Score 0-100.
    2. Write a single punchy verdict sentence (max 20 words).
    3. List the top 3 specific strengths.
    4. List the top 3 specific weaknesses / complaints.
    5. Estimate the typical retail price in USD based on your knowledge (e.g. "$349"). If unknown, use "N/A".

    OUTPUT FORMAT (STRICT JSON ONLY):
    {{
        "score": 85,
        "vibe": "One punchy verdict sentence.",
        "pros": ["Strength 1", "Strength 2", "Strength 3"],
        "cons": ["Weakness 1", "Weakness 2", "Weakness 3"],
        "price": "$349"
    }}
    """
    for attempt in range(3):
        try:
            response = model.generate_content(
                prompt,
                safety_settings=safety_settings,
                generation_config={"response_mime_type": "application/json"}
            )
            if response and response.text:
                return json.loads(response.text)
        except Exception as e:
            if "429" in str(e):
                wait = (attempt + 1) * 15
                st.warning(f"Quota busy. Retrying in {wait}s…")
                time.sleep(wait)
                continue
            st.error(f"AI Error: {e}")
            break
    return None


# ─── 4. CHART HELPERS ────────────────────────────────────────

def score_to_label(score):
    if score >= 80: return "Highly Recommended", "#22c98a"
    if score >= 65: return "Worth Buying",        "#5bc8a8"
    if score >= 45: return "Mixed Signals",        "#f0a500"
    return "Not Recommended",                      "#e85d5d"


def build_gauge(score):
    _, accent = score_to_label(score)
    steps = [
        {"range": [0,  25],  "color": "rgba(232,93,93,0.1)"},
        {"range": [25, 50],  "color": "rgba(240,165,0,0.1)"},
        {"range": [50, 75],  "color": "rgba(91,200,168,0.1)"},
        {"range": [75, 100], "color": "rgba(34,201,138,0.1)"},
    ]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={
            "font": {"color": accent, "size": 56, "family": "DM Mono, monospace"},
            "suffix": ""
        },
        gauge={
            "axis": {
                "range": [0, 100],
                "tickvals": [0, 25, 50, 75, 100],
                "tickfont": {"color": "rgba(255,255,255,0.25)", "size": 10, "family": "DM Mono"},
                "tickcolor": "rgba(255,255,255,0.1)",
            },
            "bar":      {"color": accent, "thickness": 0.26},
            "bgcolor":  "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps":    steps,
            "threshold": {"line": {"color": accent, "width": 2}, "thickness": 0.82, "value": score},
        },
        title={"text": "SENTIMENT SCORE",
               "font": {"color": "rgba(255,255,255,0.25)", "size": 10, "family": "Plus Jakarta Sans"}},
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=40, b=4, l=24, r=24), height=230,
    )
    return fig


def build_radar(pros, cons):
    def clip(s, n=26): return s[:n] + "…" if len(s) > n else s

    pro_cats = [clip(p) for p in pros]
    con_cats = [clip(c) for c in cons]
    pro_vals = [90, 80, 70]
    con_vals = [65, 72, 58]

    polar_style = dict(
        radialaxis=dict(
            visible=True, range=[0, 100],
            gridcolor='rgba(255,255,255,0.07)',
            tickvals=[25, 50, 75],
            tickfont=dict(color='rgba(255,255,255,0.18)', size=8, family='DM Mono'),
        ),
        angularaxis=dict(
            gridcolor='rgba(255,255,255,0.07)',
            tickfont=dict(color='rgba(255,255,255,0.65)', size=11, family='Plus Jakarta Sans'),
            linecolor='rgba(255,255,255,0.06)',
        ),
        bgcolor='rgba(255,255,255,0.02)',
    )

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "polar"}, {"type": "polar"}]],
        subplot_titles=["▲  Strengths", "▼  Weaknesses"],
    )
    fig.add_trace(go.Scatterpolar(
        r=pro_vals + [pro_vals[0]], theta=pro_cats + [pro_cats[0]],
        fill='toself', name='Strengths',
        line=dict(color='#22c98a', width=2.5),
        fillcolor='rgba(34,201,138,0.14)',
        marker=dict(size=7, color='#22c98a'),
    ), row=1, col=1)
    fig.add_trace(go.Scatterpolar(
        r=con_vals + [con_vals[0]], theta=con_cats + [con_cats[0]],
        fill='toself', name='Weaknesses',
        line=dict(color='#e85d5d', width=2.5),
        fillcolor='rgba(232,93,93,0.14)',
        marker=dict(size=7, color='#e85d5d'),
    ), row=1, col=2)
    fig.update_layout(
        polar=dict(**polar_style), polar2=dict(**polar_style),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        margin=dict(t=48, b=24, l=40, r=40),
        height=400,
    )
    for ann in fig.layout.annotations:
        ann.font = dict(color='rgba(255,255,255,0.35)', size=11, family='DM Mono')
    return fig


def build_diverging_bar(pros, cons):
    def clip(s, n=38): return s[:n] + "…" if len(s) > n else s
    labels = [clip(c) for c in reversed(cons)] + [clip(p) for p in reversed(pros)]
    values = [-65, -72, -58, 90, 80, 70]
    colors = ['rgba(34,201,138,0.78)' if v > 0 else 'rgba(232,93,93,0.78)' for v in values]
    border = ['rgba(34,201,138,0.4)'  if v > 0 else 'rgba(232,93,93,0.4)'  for v in values]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation='h',
        marker=dict(color=colors, line=dict(color=border, width=1)),
        text=[f"+{v}" if v > 0 else str(v) for v in values],
        textposition='outside',
        textfont=dict(color='rgba(255,255,255,0.4)', size=11, family='DM Mono'),
        hovertemplate='%{y}<extra></extra>',
        width=0.5,
    ))
    fig.add_vline(x=0, line_color='rgba(255,255,255,0.18)', line_width=1.5)
    fig.add_hline(y=2.5, line_color='rgba(255,255,255,0.07)', line_width=1, line_dash='dot')
    fig.add_annotation(x=-135, y=5.3, text="STRENGTHS", showarrow=False,
                       font=dict(color='rgba(34,201,138,0.45)', size=9, family='DM Mono'),
                       xanchor='left')
    fig.add_annotation(x=-135, y=2.2, text="WEAKNESSES", showarrow=False,
                       font=dict(color='rgba(232,93,93,0.45)', size=9, family='DM Mono'),
                       xanchor='left')
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-140, 140]),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.04)',
            tickfont=dict(color='rgba(255,255,255,0.7)', size=12, family='Plus Jakarta Sans'),
            automargin=True,
        ),
        margin=dict(t=28, b=20, l=12, r=80),
        height=360, bargap=0.44,
    )
    return fig


def build_compare_score_bar(name_a, score_a, name_b, score_b):
    """Horizontal score comparison bar chart."""
    _, color_a = score_to_label(score_a)
    _, color_b = score_to_label(score_b)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name=name_a, x=[score_a], y=["Score"],
        orientation='h', marker_color=color_a,
        text=[f"{score_a}"], textposition='inside',
        textfont=dict(color='#080c18', size=13, family='DM Mono'),
        width=0.35,
    ))
    fig.add_trace(go.Bar(
        name=name_b, x=[score_b], y=["Score"],
        orientation='h', marker_color=color_b,
        text=[f"{score_b}"], textposition='inside',
        textfont=dict(color='#080c18', size=13, family='DM Mono'),
        width=0.35,
    ))
    fig.update_layout(
        barmode='group',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(range=[0, 100], showgrid=False, zeroline=False,
                   tickvals=[0, 25, 50, 75, 100],
                   tickfont=dict(color='rgba(255,255,255,0.25)', size=9, family='DM Mono')),
        yaxis=dict(showticklabels=False),
        legend=dict(font=dict(color='rgba(255,255,255,0.55)', size=11, family='Plus Jakarta Sans'),
                    bgcolor='rgba(0,0,0,0)', orientation='h',
                    x=0.5, xanchor='center', y=1.18),
        margin=dict(t=40, b=10, l=10, r=10),
        height=130,
    )
    return fig


# ─── 5. CSS ──────────────────────────────────────────────────

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Instrument+Serif:ital@0;1&display=swap');

html, body, [data-testid="stAppViewContainer"] { background: #080c18 !important; }
[data-testid="stHeader"], [data-testid="stDecoration"],
[data-testid="stSidebar"], #MainMenu, footer { display: none !important; }

/* Noise texture */
[data-testid="stAppViewContainer"]::before {
    content: ""; position: fixed; inset: 0; z-index: 0; pointer-events: none; opacity: 0.4;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.08'/%3E%3C/svg%3E");
    background-size: 200px 200px;
}
/* Ambient blobs */
[data-testid="stAppViewContainer"]::after {
    content: ""; position: fixed; inset: 0; z-index: 0; pointer-events: none;
    background:
        radial-gradient(ellipse 60% 50% at 8%  18%,  rgba(240,165,0,0.14)  0%, transparent 60%),
        radial-gradient(ellipse 50% 45% at 90% 78%,  rgba(34,201,138,0.12) 0%, transparent 58%),
        radial-gradient(ellipse 35% 30% at 52% 50%,  rgba(60,40,120,0.1)   0%, transparent 55%);
}

*, p, div, span, label { font-family: 'Plus Jakarta Sans', sans-serif !important; color: rgba(255,255,255,0.82); }
section.main > div { padding-top: 2rem; max-width: 920px; margin: auto; }

/* ── Wordmark ── */
.wm-wrap { display:flex; align-items:baseline; gap:1.1rem; padding-bottom:1.2rem; border-bottom:1px solid rgba(240,165,0,0.2); margin-bottom:0.4rem; }
.wm-logo { font-family:'Instrument Serif',Georgia,serif !important; font-size:2.8rem; font-weight:400; letter-spacing:-0.02em; line-height:1; color:#fff !important; }
.wm-logo em { font-style:italic; color:#f0a500 !important; }
.wm-sub { font-family:'DM Mono',monospace !important; font-size:0.68rem; letter-spacing:0.2em; text-transform:uppercase; color:rgba(255,255,255,0.25) !important; padding-bottom:0.2rem; }

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid rgba(255,255,255,0.07) !important;
    gap: 0.5rem;
    margin-bottom: 1rem;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    color: rgba(255,255,255,0.32) !important;
    padding: 0.6rem 1rem !important;
    border-radius: 8px 8px 0 0 !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: #f0a500 !important;
    border-bottom: 2px solid #f0a500 !important;
    background: rgba(240,165,0,0.06) !important;
}

/* ── Section markers ── */
.sec-row { display:flex; align-items:center; gap:0.9rem; margin:2.2rem 0 1rem; }
.sec-num { font-family:'DM Mono',monospace !important; font-size:0.7rem; font-weight:500; letter-spacing:0.08em; color:#f0a500 !important; border:1px solid rgba(240,165,0,0.35); border-radius:5px; padding:0.18rem 0.48rem; white-space:nowrap; }
.sec-label { font-family:'Plus Jakarta Sans',sans-serif !important; font-size:0.72rem; font-weight:600; letter-spacing:0.2em; text-transform:uppercase; color:rgba(255,255,255,0.3) !important; }
.sec-rule { flex:1; height:1px; background:rgba(255,255,255,0.07); }

/* ── Glass card ── */
.gc {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(24px); -webkit-backdrop-filter: blur(24px);
    border: 1px solid rgba(255,255,255,0.08); border-radius: 18px;
    padding: 1.6rem 1.8rem; margin-bottom: 1rem;
    box-shadow: 0 6px 28px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.06);
    position: relative; overflow: hidden;
}
.gc-amber { border-left: 3px solid rgba(240,165,0,0.6); }
.gc-green { border-left: 3px solid rgba(34,201,138,0.6); }
.gc-red   { border-left: 3px solid rgba(232,93,93,0.6); }
.gc-blue  { border-left: 3px solid rgba(96,165,250,0.6); }
.gc-warn  { border-left: 3px solid rgba(240,165,0,0.6); background: rgba(240,165,0,0.06); }
.gc-err   { border-left: 3px solid rgba(232,93,93,0.6);  background: rgba(232,93,93,0.06); }

/* ── Score ── */
.score-num { font-family:'DM Mono',monospace !important; font-size:6rem; font-weight:400; line-height:1; letter-spacing:-0.04em; }
.score-denom { font-family:'DM Mono',monospace !important; font-size:1.1rem; color:rgba(255,255,255,0.2) !important; vertical-align:super; margin-left:4px; }
.score-badge {
    display: inline-block;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.6rem; font-weight: 500;
    letter-spacing: 0.1em; text-transform: uppercase;
    border-radius: 20px; padding: 0.28rem 0.75rem;
    margin-top: 0.6rem; border: 1px solid;
    white-space: nowrap;
}

/* ── Data strip ── */
.data-strip {
    display: flex; gap: 0; margin-top: 1rem; padding-top: 0.9rem;
    border-top: 1px solid rgba(255,255,255,0.07);
}
.data-cell { flex: 1; padding-right: 0.5rem; min-width: 0; }
.data-val { font-family:'DM Mono',monospace !important; font-size:1.5rem; font-weight:400; color:#fff !important; line-height:1; white-space: nowrap; }
.data-key {
    font-family:'DM Mono',monospace !important;
    font-size: 0.58rem; letter-spacing: 0.12em;
    text-transform: uppercase; color: rgba(255,255,255,0.28) !important;
    margin-top: 0.25rem; display: block;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}

/* ── Verdict quote ── */
.verdict-quote { font-family:'Instrument Serif',Georgia,serif !important; font-size:1.12rem; font-style:italic; font-weight:400; line-height:1.7; color:rgba(255,255,255,0.78) !important; border-left:2px solid rgba(240,165,0,0.45); padding-left:1rem; margin:1rem 0 0.8rem; }

/* ── Product pill ── */
.prod-pill { display:inline-flex; align-items:center; gap:0.4rem; background:rgba(240,165,0,0.1); border:1px solid rgba(240,165,0,0.25); border-radius:20px; padding:0.26rem 0.85rem; font-family:'DM Mono',monospace !important; font-size:0.75rem; color:rgba(255,215,120,0.85) !important; }

/* ── Micro label ── */
.micro { font-family:'DM Mono',monospace !important; font-size:0.62rem; letter-spacing:0.18em; text-transform:uppercase; color:rgba(255,255,255,0.24) !important; margin-bottom:0.45rem; display:block; }

/* ── Pro/Con ── */
.pci { display:flex; align-items:flex-start; gap:0.7rem; padding:0.75rem 0.9rem; border-radius:11px; margin-bottom:0.5rem; font-size:0.875rem; line-height:1.48; }
.pci-pro { background:rgba(34,201,138,0.07); border:1px solid rgba(34,201,138,0.16); }
.pci-con { background:rgba(232,93,93,0.07);  border:1px solid rgba(232,93,93,0.16); }
.pci-pro .dot { color:#22c98a !important; font-size:0.8rem; margin-top:0.15rem; flex-shrink:0; }
.pci-con .dot { color:#e85d5d !important; font-size:0.8rem; margin-top:0.15rem; flex-shrink:0; }

/* ── Compare specific ── */
.compare-score-card {
    text-align: center; padding: 1.8rem 1rem;
}
.compare-score-big {
    font-family: 'DM Mono', monospace !important;
    font-size: 4.5rem; font-weight: 400; line-height: 1; letter-spacing: -0.04em;
}
.compare-product-name {
    font-family: 'Instrument Serif', serif !important;
    font-size: 1.1rem; font-style: italic;
    color: rgba(255,255,255,0.55) !important;
    margin-bottom: 0.8rem; display: block;
}
.compare-price {
    font-family: 'DM Mono', monospace !important;
    font-size: 1.8rem; color: #fff !important;
    line-height: 1;
}
.compare-price-label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.6rem; letter-spacing: 0.15em;
    text-transform: uppercase; color: rgba(255,255,255,0.28) !important;
    margin-top: 0.2rem; display: block;
}
.winner-badge {
    display: inline-block;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.65rem; letter-spacing: 0.1em;
    text-transform: uppercase; padding: 0.3rem 0.9rem;
    background: rgba(240,165,0,0.15); border: 1px solid rgba(240,165,0,0.4);
    border-radius: 20px; color: #f0a500 !important; margin-top: 0.6rem;
}
.verdict-strip {
    font-family: 'Instrument Serif', serif !important;
    font-size: 0.92rem; font-style: italic;
    color: rgba(255,255,255,0.5) !important;
    line-height: 1.5; margin-top: 0.7rem;
}

/* ── Chart caption ── */
.ch-cap { font-family:'DM Mono',monospace !important; font-size:0.7rem; color:rgba(255,255,255,0.25) !important; font-style:italic; letter-spacing:0.03em; margin-bottom:0.2rem; display:block; }

/* ── Plotly chart as glass card ── */
[data-testid="stPlotlyChart"] {
    background: rgba(255,255,255,0.04) !important;
    backdrop-filter: blur(24px); -webkit-backdrop-filter: blur(24px);
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 18px !important;
    padding: 1rem 0.5rem 0.3rem !important;
    box-shadow: 0 6px 28px rgba(0,0,0,0.28), inset 0 1px 0 rgba(255,255,255,0.06);
}

/* ── Input ── */
[data-testid="stTextInput"] input { background:rgba(255,255,255,0.05) !important; border:1px solid rgba(255,255,255,0.1) !important; border-radius:12px !important; color:white !important; padding:0.85rem 1.1rem !important; font-family:'Plus Jakarta Sans',sans-serif !important; font-size:0.96rem !important; transition:border-color 0.2s,box-shadow 0.2s; }
[data-testid="stTextInput"] input::placeholder { color:rgba(255,255,255,0.2) !important; }
[data-testid="stTextInput"] input:focus { border-color:rgba(240,165,0,0.5) !important; box-shadow:0 0 0 3px rgba(240,165,0,0.1) !important; outline:none !important; }
[data-testid="stTextInput"] label { font-family:'DM Mono',monospace !important; font-size:0.66rem !important; letter-spacing:0.16em; text-transform:uppercase; color:rgba(255,255,255,0.32) !important; }

/* ── Button ── */
[data-testid="stButton"] > button { background:linear-gradient(135deg,#f0a500 0%,#d97b2a 100%) !important; color:#080c18 !important; border:none !important; border-radius:12px !important; padding:0.82rem 2.2rem !important; font-family:'Plus Jakarta Sans',sans-serif !important; font-weight:700 !important; font-size:0.88rem !important; letter-spacing:0.03em !important; transition:all 0.2s ease !important; box-shadow:0 4px 20px rgba(240,165,0,0.3) !important; }
[data-testid="stButton"] > button:hover { transform:translateY(-1px) !important; box-shadow:0 7px 28px rgba(240,165,0,0.45) !important; }

/* ── Spinner ── */
[data-testid="stSpinner"] p { font-family:'DM Mono',monospace !important; color:rgba(255,255,255,0.32) !important; font-size:0.82rem !important; }

[data-testid="stAlert"] { background:rgba(240,165,0,0.06) !important; border:1px solid rgba(240,165,0,0.2) !important; border-radius:12px !important; }
.js-plotly-plot .plotly, .js-plotly-plot .plotly .plot-container { background: transparent !important; }
</style>
"""

# ─── 6. PAGE SETUP ───────────────────────────────────────────

st.set_page_config(page_title="VibeCheck", page_icon="🛡️", layout="centered")
st.markdown(CSS, unsafe_allow_html=True)

st.markdown("""
<div class="wm-wrap">
    <div class="wm-logo">Vibe<em>Check</em></div>
    <div class="wm-sub">Product Intelligence Engine</div>
</div>
""", unsafe_allow_html=True)

# ─── 7. TABS ─────────────────────────────────────────────────

tab_single, tab_compare = st.tabs(["Single Analysis", "Head-to-Head Compare"])

# ══════════════════════════════════════════════════════════════
# TAB 1 — SINGLE ANALYSIS
# ══════════════════════════════════════════════════════════════
with tab_single:
    st.markdown('<div style="height:0.3rem"></div>', unsafe_allow_html=True)
    product_query = st.text_input(
        "Product", placeholder="e.g. iPhone 17 Pro, Sony WH-1000XM5…",
        label_visibility="visible", key="single_input"
    )
    analyze_btn = st.button("Run Intelligence Check →", key="single_btn")

    if analyze_btn and product_query:
        with st.spinner("Scouring the web for signals…"):
            data = get_reviews_data(product_query)

        if data:
            char_count = len(data)
            with st.spinner("Quantifying sentiment with Gemini…"):
                analysis = analyze_sentiment(data, product_query)

            if analysis:
                score  = analysis.get('score', 0)
                vibe   = analysis.get('vibe', '')
                pros   = analysis.get('pros', [])
                cons   = analysis.get('cons', [])
                label, color = score_to_label(score)

                # ── 01 Overview ──────────────────────────
                st.markdown("""<div class="sec-row"><span class="sec-num">01</span><span class="sec-label">Overview</span><div class="sec-rule"></div></div>""", unsafe_allow_html=True)
                left, right = st.columns([1, 1.7], gap="large")
                with left:
                    st.markdown(f"""
                    <div class="gc gc-amber">
                        <span class="micro">Sentiment Score</span>
                        <div><span class="score-num" style="color:{color};">{score}</span><span class="score-denom">/100</span></div>
                        <div class="score-badge" style="color:{color}; border-color:{color}40; background:{color}1a;">{label}</div>
                        <div class="data-strip">
                            <div class="data-cell">
                                <div class="data-val">{len(pros)}</div>
                                <span class="data-key">Strengths</span>
                            </div>
                            <div class="data-cell">
                                <div class="data-val">{len(cons)}</div>
                                <span class="data-key">Weaknesses</span>
                            </div>
                            <div class="data-cell">
                                <div class="data-val">{char_count // 100}k</div>
                                <span class="data-key">Signals</span>
                            </div>
                        </div>
                    </div>""", unsafe_allow_html=True)
                    st.plotly_chart(build_gauge(score), use_container_width=True,
                                    config={"displayModeBar": False})
                with right:
                    st.markdown(f"""
                    <div class="gc" style="height:100%;">
                        <span class="micro">Product</span>
                        <div class="prod-pill">📦 {product_query}</div>
                        <div class="verdict-quote">{vibe}</div>
                        <div style="font-size:0.7rem;color:rgba(255,255,255,0.2);font-style:italic;margin-top:0.6rem;">Synthesised from live web search snippets &amp; community discussions</div>
                    </div>""", unsafe_allow_html=True)

                # ── 02 Strengths & Weaknesses ─────────────
                st.markdown("""<div class="sec-row"><span class="sec-num">02</span><span class="sec-label">Strengths &amp; Weaknesses</span><div class="sec-rule"></div></div>""", unsafe_allow_html=True)
                pc, cc = st.columns(2, gap="medium")
                with pc:
                    pros_html = "".join([f'<div class="pci pci-pro"><span class="dot">▲</span><span>{p}</span></div>' for p in pros])
                    st.markdown(f'<div class="gc gc-green"><span class="micro" style="color:#22c98a !important;">What people love</span>{pros_html}</div>', unsafe_allow_html=True)
                with cc:
                    cons_html = "".join([f'<div class="pci pci-con"><span class="dot">▼</span><span>{c}</span></div>' for c in cons])
                    st.markdown(f'<div class="gc gc-red"><span class="micro" style="color:#e85d5d !important;">Common complaints</span>{cons_html}</div>', unsafe_allow_html=True)

                # ── 03 Attribute Radar ────────────────────
                st.markdown("""<div class="sec-row"><span class="sec-num">03</span><span class="sec-label">Attribute Radar</span><div class="sec-rule"></div></div>""", unsafe_allow_html=True)
                st.markdown('<span class="ch-cap">Relative attribute mapping across reported strengths and weaknesses</span>', unsafe_allow_html=True)
                st.plotly_chart(build_radar(pros, cons), use_container_width=True, config={"displayModeBar": False})

                # ── 04 Signal Strength ────────────────────
                st.markdown("""<div class="sec-row"><span class="sec-num">04</span><span class="sec-label">Signal Strength</span><div class="sec-rule"></div></div>""", unsafe_allow_html=True)
                st.markdown('<span class="ch-cap">Diverging view — positive signals vs. negative friction points</span>', unsafe_allow_html=True)
                st.plotly_chart(build_diverging_bar(pros, cons), use_container_width=True, config={"displayModeBar": False})

                # ── Footer ────────────────────────────────
                st.markdown("""<div style="text-align:center;padding:2rem 0 1rem;font-family:'DM Mono',monospace;font-size:0.65rem;color:rgba(255,255,255,0.14);letter-spacing:0.1em;">VIBECHECK · POWERED BY GEMINI AI · LIVE WEB DATA</div>""", unsafe_allow_html=True)
            else:
                st.markdown("""<div class="gc gc-warn" style="text-align:center;padding:2rem;"><div style="font-size:1.8rem;margin-bottom:0.5rem;">⚠</div><div style="font-weight:600;">AI analysis failed.</div><div style="color:rgba(255,255,255,0.35);font-size:0.85rem;margin-top:0.3rem;">Usually a quota issue. Wait 60 seconds and try again.</div></div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="gc gc-err" style="text-align:center;padding:2rem;"><div style="font-size:1.8rem;margin-bottom:0.5rem;">✕</div><div style="font-weight:600;">No search data found.</div><div style="color:rgba(255,255,255,0.35);font-size:0.85rem;margin-top:0.3rem;">Check your Serper API key and try a different product name.</div></div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 2 — HEAD-TO-HEAD COMPARE
# ══════════════════════════════════════════════════════════════
with tab_compare:
    st.markdown('<div style="height:0.3rem"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.78rem; color:rgba(255,255,255,0.32); margin-bottom:1rem; font-style:italic;">
        Enter two products to compare sentiment, pros, cons, and estimated price side by side.
    </div>""", unsafe_allow_html=True)

    col_a, col_b = st.columns(2, gap="medium")
    with col_a:
        product_a = st.text_input("Product A", placeholder="e.g. Sony WH-1000XM5", key="cmp_a")
    with col_b:
        product_b = st.text_input("Product B", placeholder="e.g. Bose QC Ultra", key="cmp_b")

    compare_btn = st.button("Run Comparison →", key="compare_btn")

    if compare_btn and product_a and product_b:

        # Fetch data for both concurrently using st.spinner
        with st.spinner(f"Researching {product_a}…"):
            data_a = get_reviews_data(product_a)
        with st.spinner(f"Researching {product_b}…"):
            data_b = get_reviews_data(product_b)

        if data_a and data_b:
            with st.spinner("Fetching live prices…"):
                scraped_price_a = get_price(product_a)
                scraped_price_b = get_price(product_b)
            with st.spinner("Running AI analysis on both products…"):
                result_a = analyze_compare(data_a, product_a)
                time.sleep(2)  # Avoid quota collision
                result_b = analyze_compare(data_b, product_b)

            if result_a and result_b:
                score_a = result_a.get('score', 0)
                score_b = result_b.get('score', 0)
                pros_a  = result_a.get('pros', [])
                cons_a  = result_a.get('cons', [])
                pros_b  = result_b.get('pros', [])
                cons_b  = result_b.get('cons', [])
                vibe_a  = result_a.get('vibe', '')
                vibe_b  = result_b.get('vibe', '')
                # Use live scraped price; fall back to AI estimate if unavailable
                ai_price_a = result_a.get('price', 'N/A')
                ai_price_b = result_b.get('price', 'N/A')
                price_a = scraped_price_a if scraped_price_a != 'N/A' else ai_price_a
                price_b = scraped_price_b if scraped_price_b != 'N/A' else ai_price_b
                label_a, color_a = score_to_label(score_a)
                label_b, color_b = score_to_label(score_b)
                winner  = product_a if score_a >= score_b else product_b

                # ── C1  Score Overview ─────────────────────
                st.markdown("""<div class="sec-row"><span class="sec-num">C1</span><span class="sec-label">Score Overview</span><div class="sec-rule"></div></div>""", unsafe_allow_html=True)

                ca, cb = st.columns(2, gap="medium")
                with ca:
                    if score_a > score_b:
                        winner_html = '<div class="winner-badge">🏆 Winner</div>'
                    elif score_a == score_b:
                        winner_html = '<div class="winner-badge" style="background:rgba(255,255,255,0.06);border-color:rgba(255,255,255,0.18);color:rgba(255,255,255,0.45)!important;">🤝 Tie</div>'
                    else:
                        winner_html = ''
                    st.markdown(f"""
                    <div class="gc gc-amber compare-score-card">
                        <span class="compare-product-name">{product_a}</span>
                        <div class="compare-score-big" style="color:{color_a};">{score_a}</div>
                        <div style="font-family:'DM Mono',monospace;font-size:0.9rem;color:rgba(255,255,255,0.2);">/100</div>
                        <div class="score-badge" style="color:{color_a};border-color:{color_a}40;background:{color_a}1a;margin-top:0.6rem;">{label_a}</div>
                        {winner_html}
                        <div class="verdict-strip">{vibe_a}</div>
                    </div>""", unsafe_allow_html=True)
                with cb:
                    if score_b > score_a:
                        winner_html_b = '<div class="winner-badge">🏆 Winner</div>'
                    elif score_a == score_b:
                        winner_html_b = '<div class="winner-badge" style="background:rgba(255,255,255,0.06);border-color:rgba(255,255,255,0.18);color:rgba(255,255,255,0.45)!important;">🤝 Tie</div>'
                    else:
                        winner_html_b = ''
                    st.markdown(f"""
                    <div class="gc gc-blue compare-score-card">
                        <span class="compare-product-name">{product_b}</span>
                        <div class="compare-score-big" style="color:{color_b};">{score_b}</div>
                        <div style="font-family:'DM Mono',monospace;font-size:0.9rem;color:rgba(255,255,255,0.2);">/100</div>
                        <div class="score-badge" style="color:{color_b};border-color:{color_b}40;background:{color_b}1a;margin-top:0.6rem;">{label_b}</div>
                        {winner_html_b}
                        <div class="verdict-strip">{vibe_b}</div>
                    </div>""", unsafe_allow_html=True)

                # Score bar comparison chart
                st.markdown('<span class="ch-cap" style="margin-top:0.5rem;">Sentiment score comparison</span>', unsafe_allow_html=True)
                st.plotly_chart(
                    build_compare_score_bar(product_a, score_a, product_b, score_b),
                    use_container_width=True, config={"displayModeBar": False}
                )

                # ── C2  Price ─────────────────────────────
                st.markdown("""<div class="sec-row"><span class="sec-num">C2</span><span class="sec-label">Estimated Retail Price</span><div class="sec-rule"></div></div>""", unsafe_allow_html=True)

                pa_col, pb_col = st.columns(2, gap="medium")
                with pa_col:
                    try:
                        pf_a = safe_price_float(price_a)
                        pf_b = safe_price_float(price_b)
                        if pf_a > 0 and pf_b > 0:
                            better_val = "Better value" if (score_a / pf_a) > (score_b / pf_b) else ""
                        else:
                            better_val = ""
                    except Exception:
                        better_val = ""
                    val_badge  = f'<div class="winner-badge" style="font-size:0.58rem;">💰 {better_val}</div>' if better_val else ''
                    st.markdown(f"""
                    <div class="gc" style="text-align:center;padding:1.4rem 1rem;">
                        <span class="compare-product-name">{product_a}</span>
                        <div class="compare-price">{price_a}</div>
                        <span class="compare-price-label">Est. retail price</span>
                        {val_badge}
                    </div>""", unsafe_allow_html=True)
                with pb_col:
                    try:
                        pf_a = safe_price_float(price_a)
                        pf_b = safe_price_float(price_b)
                        if pf_a > 0 and pf_b > 0:
                            better_val_b = "Better value" if (score_b / pf_b) > (score_a / pf_a) else ""
                        else:
                            better_val_b = ""
                    except Exception:
                        better_val_b = ""
                    val_badge_b = f'<div class="winner-badge" style="font-size:0.58rem;">💰 {better_val_b}</div>' if better_val_b else ''
                    st.markdown(f"""
                    <div class="gc" style="text-align:center;padding:1.4rem 1rem;">
                        <span class="compare-product-name">{product_b}</span>
                        <div class="compare-price">{price_b}</div>
                        <span class="compare-price-label">Est. retail price</span>
                        {val_badge_b}
                    </div>""", unsafe_allow_html=True)

                # ── C3  Pros & Cons Side by Side ──────────
                st.markdown("""<div class="sec-row"><span class="sec-num">C3</span><span class="sec-label">Strengths</span><div class="sec-rule"></div></div>""", unsafe_allow_html=True)
                sa, sb = st.columns(2, gap="medium")
                with sa:
                    pros_a_html = "".join([f'<div class="pci pci-pro"><span class="dot">▲</span><span>{p}</span></div>' for p in pros_a])
                    st.markdown(f'<div class="gc gc-green"><span class="micro" style="color:#22c98a !important;">{product_a}</span>{pros_a_html}</div>', unsafe_allow_html=True)
                with sb:
                    pros_b_html = "".join([f'<div class="pci pci-pro"><span class="dot">▲</span><span>{p}</span></div>' for p in pros_b])
                    st.markdown(f'<div class="gc gc-green"><span class="micro" style="color:#22c98a !important;">{product_b}</span>{pros_b_html}</div>', unsafe_allow_html=True)

                st.markdown("""<div class="sec-row"><span class="sec-num">C4</span><span class="sec-label">Weaknesses</span><div class="sec-rule"></div></div>""", unsafe_allow_html=True)
                wa, wb = st.columns(2, gap="medium")
                with wa:
                    cons_a_html = "".join([f'<div class="pci pci-con"><span class="dot">▼</span><span>{c}</span></div>' for c in cons_a])
                    st.markdown(f'<div class="gc gc-red"><span class="micro" style="color:#e85d5d !important;">{product_a}</span>{cons_a_html}</div>', unsafe_allow_html=True)
                with wb:
                    cons_b_html = "".join([f'<div class="pci pci-con"><span class="dot">▼</span><span>{c}</span></div>' for c in cons_b])
                    st.markdown(f'<div class="gc gc-red"><span class="micro" style="color:#e85d5d !important;">{product_b}</span>{cons_b_html}</div>', unsafe_allow_html=True)

                st.markdown("""<div style="text-align:center;padding:2rem 0 1rem;font-family:'DM Mono',monospace;font-size:0.65rem;color:rgba(255,255,255,0.14);letter-spacing:0.1em;">VIBECHECK · HEAD-TO-HEAD · POWERED BY GEMINI AI</div>""", unsafe_allow_html=True)

            else:
                st.markdown("""<div class="gc gc-warn" style="text-align:center;padding:2rem;"><div style="font-size:1.8rem;">⚠</div><div style="font-weight:600;margin-top:0.5rem;">AI analysis failed for one or both products.</div><div style="color:rgba(255,255,255,0.35);font-size:0.85rem;margin-top:0.3rem;">Quota limit likely hit. Wait 60s and retry.</div></div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="gc gc-err" style="text-align:center;padding:2rem;"><div style="font-size:1.8rem;">✕</div><div style="font-weight:600;margin-top:0.5rem;">Could not fetch data for one or both products.</div><div style="color:rgba(255,255,255,0.35);font-size:0.85rem;margin-top:0.3rem;">Check your Serper API key.</div></div>""", unsafe_allow_html=True)
    elif compare_btn:
        st.markdown("""<div class="gc" style="text-align:center;padding:1.5rem;"><div style="color:rgba(255,255,255,0.35);font-size:0.88rem;">Please enter both product names to compare.</div></div>""", unsafe_allow_html=True)
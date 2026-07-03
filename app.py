import os
import html as _html
import json
import time
import re
import requests
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
import google.generativeai as genai
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def h(text):
    """Strip any HTML tags (including escaped entities like &lt;div&gt;), then HTML-escape user/AI text so it never breaks the surrounding HTML structure."""
    if not text:
        return ""
    # Unescape HTML entities (e.g. &lt;div&gt; -> <div>) so regex can match and strip them
    unescaped = _html.unescape(str(text))
    cleaned = re.sub(r'<[^>]*>', '', unescaped)
    return _html.escape(cleaned)

# CONFIGURATION 
load_dotenv()

SERPER_API_KEY = st.secrets.get("SERPER_API_KEY") or os.getenv("SERPER_API_KEY")
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

# CORE LOGIC

def get_reviews_data(product_name):
    if not SERPER_API_KEY:
        st.error("Serper API Key missing!")
        return None
    url = "https://google.serper.dev/search"
    payload = {
        "q": f"{product_name} user reviews pros cons India Flipkart Amazon India price",
        "num": 10, "gl": "in", "hl": "en", "autocorrect": True
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
    5. Do NOT include any HTML, XML, or Markdown tags in your response. Ensure all text in vibe, pros, and cons is plain text only.

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


# COMPARE HELPERS (NEW)

def get_price(product_name):
    """Fetch estimated retail price via Serper shopping search (India, INR)."""
    if not SERPER_API_KEY:
        return None
    url = "https://google.serper.dev/shopping"
    # Try multiple queries to maximise the chance of finding a price
    queries = [
        f"{product_name} price India",
        f"{product_name} buy online India",
        product_name,
    ]
    headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}
    for query in queries:
        try:
            payload = {"q": query, "num": 5, "gl": "in", "hl": "en"}
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            items = response.json().get('shopping', [])
            prices = []
            for item in items:
                p = item.get('price', '')
                if p:
                    # Strip ₹, Rs., $, commas etc.
                    cleaned = p.replace('₹', '').replace('Rs.', '').replace('Rs', '')\
                               .replace('$', '').replace(',', '').strip()
                    try:
                        prices.append(float(cleaned))
                    except ValueError:
                        pass
            if prices:
                avg = sum(prices) / len(prices)
                return f"~₹{avg:,.0f}"
        except Exception:
            continue
    return None


def safe_price_float(price_str):
    """Convert any price representation (str or number) to float. Returns 0.0 on failure."""
    try:
        cleaned = (
            str(price_str)
            .replace('~', '').replace('₹', '').replace('Rs.', '').replace('Rs', '')
            .replace('$', '').replace(',', '').strip()
        )
        return float(cleaned) if cleaned and cleaned.upper() != 'N/A' else 0.0
    except (ValueError, TypeError):
        return 0.0


def format_inr(price_str):
    """Normalize and format price to Indian Rupees (INR), converting USD if needed."""
    if not price_str:
        return "₹Price on request"
    price_str = str(price_str).strip()
    if price_str.upper() in ['N/A', 'NONE', '—', '-', 'NULL', '']:
        return "₹Price on request"
    
    # Check if the price is in USD/Dollars
    if '$' in price_str or 'USD' in price_str.upper():
        try:
            digits = ''.join(c for c in price_str if c.isdigit() or c == '.')
            val = float(digits)
            inr_val = int(val * 85)
            return f"₹{inr_val:,}"
        except:
            pass

    # Extract all digits to normalize and format
    digits = ''.join(c for c in price_str if c.isdigit())
    if digits:
        try:
            return f"₹{int(digits):,}"
        except ValueError:
            pass
            
    # Fallback to appending ₹ if not present
    if not (price_str.startswith('₹') or price_str.startswith('Rs') or price_str.startswith('INR')):
        return f"₹{price_str}"
        
    return price_str


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
    5. Estimate the typical retail price in INR (Indian Rupees) based on your knowledge
       (e.g. "₹29999"). You MUST provide a realistic estimate — never return "N/A".
       If the product is not officially sold in India, convert from USD at ~₹85/USD.
       Do NOT include any HTML, XML, or Markdown tags in your response. Ensure all text in vibe, pros, and cons is plain text only.

    OUTPUT FORMAT (STRICT JSON ONLY):
    {{
        "score": 85,
        "vibe": "One punchy verdict sentence.",
        "pros": ["Strength 1", "Strength 2", "Strength 3"],
        "cons": ["Weakness 1", "Weakness 2", "Weakness 3"],
        "price": "₹29999"
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


# CHART HELPERS

def score_to_label(score):
    if score >= 80: return "Highly Recommended", "#10b981"
    if score >= 65: return "Worth Buying",        "#34d399"
    if score >= 45: return "Mixed Signals",        "#f59e0b"
    return "Not Recommended",                      "#ef4444"


def build_gauge(score):
    _, accent = score_to_label(score)
    steps = [
        {"range": [0,  25],  "color": "rgba(239,68,68,0.06)"},
        {"range": [25, 50],  "color": "rgba(245,158,11,0.06)"},
        {"range": [50, 75],  "color": "rgba(52,211,153,0.06)"},
        {"range": [75, 100], "color": "rgba(16,185,129,0.06)"},
    ]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={
            "font": {"color": accent, "size": 56, "family": "JetBrains Mono, monospace"},
            "suffix": ""
        },
        gauge={
            "axis": {
                "range": [0, 100],
                "tickvals": [0, 25, 50, 75, 100],
                "tickfont": {"color": "rgba(255,255,255,0.3)", "size": 10, "family": "JetBrains Mono"},
                "tickcolor": "rgba(255,255,255,0.1)",
            },
            "bar":      {"color": accent, "thickness": 0.26},
            "bgcolor":  "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps":    steps,
            "threshold": {"line": {"color": accent, "width": 2}, "thickness": 0.82, "value": score},
        },
        title={"text": "SENTIMENT SCORE",
               "font": {"color": "rgba(255,255,255,0.3)", "size": 10, "family": "Plus Jakarta Sans"}},
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
            gridcolor='rgba(255,255,255,0.06)',
            tickvals=[25, 50, 75],
            tickfont=dict(color='rgba(255,255,255,0.25)', size=8, family='JetBrains Mono'),
        ),
        angularaxis=dict(
            gridcolor='rgba(255,255,255,0.06)',
            tickfont=dict(color='rgba(255,255,255,0.7)', size=11, family='Plus Jakarta Sans'),
            linecolor='rgba(255,255,255,0.05)',
        ),
        bgcolor='rgba(255,255,255,0.01)',
    )

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "polar"}, {"type": "polar"}]],
        subplot_titles=["▲  Strengths", "▼  Weaknesses"],
    )
    fig.add_trace(go.Scatterpolar(
        r=pro_vals + [pro_vals[0]], theta=pro_cats + [pro_cats[0]],
        fill='toself', name='Strengths',
        line=dict(color='#10b981', width=2.5),
        fillcolor='rgba(16,185,129,0.08)',
        marker=dict(size=7, color='#10b981'),
    ), row=1, col=1)
    fig.add_trace(go.Scatterpolar(
        r=con_vals + [con_vals[0]], theta=con_cats + [con_cats[0]],
        fill='toself', name='Weaknesses',
        line=dict(color='#ef4444', width=2.5),
        fillcolor='rgba(239,68,68,0.08)',
        marker=dict(size=7, color='#ef4444'),
    ), row=1, col=2)
    fig.update_layout(
        polar=dict(**polar_style), polar2=dict(**polar_style),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        margin=dict(t=48, b=24, l=40, r=40),
        height=400,
    )
    for ann in fig.layout.annotations:
        ann.font = dict(color='rgba(255,255,255,0.4)', size=11, family='JetBrains Mono')
    return fig


def build_diverging_bar(pros, cons):
    def clip(s, n=38): return s[:n] + "…" if len(s) > n else s
    labels = [clip(c) for c in reversed(cons)] + [clip(p) for p in reversed(pros)]
    values = [-65, -72, -58, 90, 80, 70]
    colors = ['rgba(16,185,129,0.75)' if v > 0 else 'rgba(239,68,68,0.75)' for v in values]
    border = ['rgba(16,185,129,0.35)'  if v > 0 else 'rgba(239,68,68,0.35)'  for v in values]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation='h',
        marker=dict(color=colors, line=dict(color=border, width=1)),
        text=[f"+{v}" if v > 0 else str(v) for v in values],
        textposition='outside',
        textfont=dict(color='rgba(255,255,255,0.5)', size=11, family='JetBrains Mono'),
        hovertemplate='%{y}<extra></extra>',
        width=0.5,
    ))
    fig.add_vline(x=0, line_color='rgba(255,255,255,0.15)', line_width=1.5)
    fig.add_hline(y=2.5, line_color='rgba(255,255,255,0.06)', line_width=1, line_dash='dot')
    fig.add_annotation(x=-135, y=5.3, text="STRENGTHS", showarrow=False,
                       font=dict(color='rgba(16,185,129,0.5)', size=9, family='JetBrains Mono'),
                       xanchor='left')
    fig.add_annotation(x=-135, y=2.2, text="WEAKNESSES", showarrow=False,
                       font=dict(color='rgba(239,68,68,0.5)', size=9, family='JetBrains Mono'),
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
        textfont=dict(color='#000000', size=13, family='JetBrains Mono'),
        width=0.35,
    ))
    fig.add_trace(go.Bar(
        name=name_b, x=[score_b], y=["Score"],
        orientation='h', marker_color=color_b,
        text=[f"{score_b}"], textposition='inside',
        textfont=dict(color='#000000', size=13, family='JetBrains Mono'),
        width=0.35,
    ))
    fig.update_layout(
        barmode='group',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(range=[0, 100], showgrid=False, zeroline=False,
                   tickvals=[0, 25, 50, 75, 100],
                   tickfont=dict(color='rgba(255,255,255,0.3)', size=9, family='JetBrains Mono')),
        yaxis=dict(showticklabels=False),
        legend=dict(font=dict(color='rgba(255,255,255,0.6)', size=11, family='Plus Jakarta Sans'),
                    bgcolor='rgba(0,0,0,0)', orientation='h',
                    x=0.5, xanchor='center', y=1.18),
        margin=dict(t=40, b=10, l=10, r=10),
        height=130,
    )
    return fig


# CSS 
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"] { background: #000000 !important; }
[data-testid="stHeader"], [data-testid="stDecoration"],
[data-testid="stSidebar"], #MainMenu, footer { display: none !important; }

/* Disable ambient blue gradients and noise */
[data-testid="stAppViewContainer"]::before { display: none !important; }
[data-testid="stAppViewContainer"]::after { display: none !important; }

*, p, div, span, label { font-family: 'Plus Jakarta Sans', sans-serif !important; color: rgba(255,255,255,0.85); }
section.main > div { padding-top: 0; max-width: 960px; margin: auto; }
/* Allow hero to bleed edge-to-edge */
section.main > div:first-child { max-width: 100% !important; padding: 0 !important; overflow: visible !important; }
[data-testid="block-container"] { padding-top: 0 !important; padding-left: 0 !important; padding-right: 0 !important; }

/* ── Hero wrap: full-viewport ── */
.hero-outer {
    position: relative;
    width: 100vw;
    left: 50%;
    transform: translateX(-50%);
    margin-bottom: 0;
    overflow: hidden;
    border-bottom: 1px solid rgba(255,255,255,0.08);
}

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid rgba(255,255,255,0.08) !important;
    gap: 0.5rem;
    margin-bottom: 1rem;
    margin-top: 1.2rem;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: rgba(255,255,255,0.4) !important;
    padding: 0.6rem 1rem !important;
    border-radius: 6px 6px 0 0 !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: #ffffff !important;
    border-bottom: 2px solid #ffffff !important;
    background: rgba(255,255,255,0.05) !important;
}

/* ── Section markers ── */
.sec-row { display:flex; align-items:center; gap:0.9rem; margin:2.2rem 0 1rem; }
.sec-num { font-family:'JetBrains Mono',monospace !important; font-size:0.7rem; font-weight:700; color:#ffffff !important; border:1px solid rgba(255,255,255,0.15); border-radius:4px; padding:0.18rem 0.48rem; white-space:nowrap; background: rgba(255,255,255,0.04); }
.sec-label { font-family:'Plus Jakarta Sans',sans-serif !important; font-size:0.7rem; font-weight:600; letter-spacing:0.15em; text-transform:uppercase; color:rgba(255,255,255,0.35) !important; }
.sec-rule { flex:1; height:1px; background:rgba(255,255,255,0.08); }

/* ── Matte Cards ── */
.gc {
    background: #09090b;
    border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 12px;
    padding: 1.6rem 1.8rem; margin-bottom: 1rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.6);
    position: relative; overflow: hidden;
}
.gc-amber { border-left: 3px solid #ffffff; }
.gc-green { border-left: 3px solid #10b981; }
.gc-red   { border-left: 3px solid #ef4444; }
.gc-blue  { border-left: 3px solid #71717a; }
.gc-warn  { border-left: 3px solid #f59e0b;  background: rgba(245,158,11,0.03); }
.gc-err   { border-left: 3px solid #ef4444;  background: rgba(239,68,68,0.03); }

/* ── Score ── */
.score-num { font-family:'Plus Jakarta Sans',sans-serif !important; font-size:6rem; font-weight:800; line-height:1; letter-spacing:-0.04em; }
.score-denom { font-family:'JetBrains Mono',monospace !important; font-size:1.1rem; color:rgba(255,255,255,0.2) !important; vertical-align:super; margin-left:4px; }
.score-badge {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.58rem; font-weight: 600;
    letter-spacing: 0.05em; text-transform: uppercase;
    border-radius: 4px; padding: 0.28rem 0.75rem;
    margin-top: 0.6rem; border: 1px solid;
    white-space: nowrap;
}

/* ── Data strip ── */
.data-strip {
    display: flex; gap: 0; margin-top: 1rem; padding-top: 0.9rem;
    border-top: 1px solid rgba(255,255,255,0.08);
}
.data-cell { flex: 1; padding-right: 0.5rem; min-width: 0; }
.data-val { font-family:'JetBrains Mono',monospace !important; font-size:1.5rem; font-weight:600; color:#fff !important; line-height:1; white-space: nowrap; }
.data-key {
    font-family:'JetBrains Mono',monospace !important;
    font-size: 0.56rem; letter-spacing: 0.08em;
    text-transform: uppercase; color: rgba(255,255,255,0.4) !important;
    margin-top: 0.25rem; display: block;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}

/* ── Verdict quote ── */
.verdict-quote {
    font-family:'Plus Jakarta Sans',sans-serif !important;
    font-size:1rem; font-weight:400; line-height:1.75;
    color:rgba(255,255,255,0.8) !important;
    border-left:2px solid rgba(255,255,255,0.25);
    padding-left:1rem; margin:1rem 0 0.8rem;
}

/* ── Product pill ── */
.prod-pill {
    display:inline-flex; align-items:center; gap:0.4rem;
    background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.15);
    border-radius:4px; padding:0.26rem 0.85rem;
    font-family:'JetBrains Mono',monospace !important;
    font-size:0.73rem; color:rgba(255,255,255,0.85) !important;
}

/* ── Micro label ── */
.micro { font-family:'JetBrains Mono',monospace !important; font-size:0.6rem; letter-spacing:0.1em; text-transform:uppercase; color:rgba(255,255,255,0.4) !important; margin-bottom:0.45rem; display:block; }

/* ── Pro/Con ── */
.pci { display:flex; align-items:flex-start; gap:0.7rem; padding:0.75rem 0.9rem; border-radius:8px; margin-bottom:0.5rem; font-size:0.875rem; line-height:1.5; }
.pci-pro { background:rgba(16,185,129,0.05); border:1px solid rgba(16,185,129,0.12); }
.pci-con { background:rgba(239,68,68,0.05);  border:1px solid rgba(239,68,68,0.12); }
.pci-pro .dot { color:#10b981 !important; font-size:0.8rem; margin-top:0.15rem; flex-shrink:0; }
.pci-con .dot { color:#ef4444 !important; font-size:0.8rem; margin-top:0.15rem; flex-shrink:0; }

/* ── Compare specific ── */
.compare-score-card {
    text-align: center; padding: 1.8rem 1rem;
}
.compare-score-big {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 4.5rem; font-weight: 800; line-height: 1; letter-spacing: -0.03em;
}
.compare-product-name {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 1rem; font-weight: 600; letter-spacing: 0.01em;
    color: rgba(255,255,255,0.5) !important;
    margin-bottom: 0.8rem; display: block;
}
.compare-price {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.8rem; color: #fff !important;
    line-height: 1;
}
.compare-price-label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.6rem; letter-spacing: 0.08em;
    text-transform: uppercase; color: rgba(255,255,255,0.4) !important;
    margin-top: 0.2rem; display: block;
}
.winner-badge {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.62rem; letter-spacing: 0.05em;
    text-transform: uppercase; padding: 0.3rem 0.9rem;
    background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.18);
    border-radius: 4px; color: #ffffff !important; margin-top: 0.6rem;
}
.verdict-strip {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.9rem; font-style: italic;
    color: rgba(255,255,255,0.5) !important;
    line-height: 1.6; margin-top: 0.7rem;
}

/* ── Chart caption ── */
.ch-cap { font-family:'JetBrains Mono',monospace !important; font-size:0.68rem; color:rgba(255,255,255,0.3) !important; font-style:italic; letter-spacing:0.03em; margin-bottom:0.2rem; display:block; }

/* ── Plotly chart as glass card ── */
[data-testid="stPlotlyChart"] {
    background: #09090b !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 12px !important;
    padding: 1rem 0.5rem 0.3rem !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.6);
}

/* ── Input ── */
[data-testid="stTextInput"] input { background:#09090b !important; border:1px solid rgba(255,255,255,0.08) !important; border-radius:8px !important; color:white !important; padding:0.85rem 1.1rem !important; font-family:'Plus Jakarta Sans',sans-serif !important; font-size:0.96rem !important; transition:border-color 0.2s,box-shadow 0.2s; }
[data-testid="stTextInput"] input::placeholder { color:rgba(255,255,255,0.35) !important; }
[data-testid="stTextInput"] input:focus { border-color:rgba(255,255,255,0.25) !important; box-shadow:0 0 0 3px rgba(255,255,255,0.03) !important; outline:none !important; }
[data-testid="stTextInput"] label { font-family:'JetBrains Mono',monospace !important; font-size:0.64rem !important; letter-spacing:0.08em; text-transform:uppercase; color:rgba(255,255,255,0.5) !important; }

/* ── Button ── */
[data-testid="stButton"] > button { background:#ffffff !important; border:1px solid #ffffff !important; border-radius:8px !important; padding:0.82rem 2.2rem !important; font-family:'Plus Jakarta Sans',sans-serif !important; font-weight:600 !important; font-size:0.88rem !important; transition:all 0.2s ease !important; box-shadow:none !important; }
[data-testid="stButton"] > button, [data-testid="stButton"] > button * { color:#000000 !important; }
[data-testid="stButton"] > button:hover { background:#e4e4e7 !important; border-color:#e4e4e7 !important; transform:translateY(-1px) !important; }
[data-testid="stButton"] > button:hover, [data-testid="stButton"] > button:hover * { color:#000000 !important; }

/* ── Spinner ── */
[data-testid="stSpinner"] p { font-family:'JetBrains Mono',monospace !important; color:rgba(255,255,255,0.45) !important; font-size:0.8rem !important; }

[data-testid="stAlert"] { background:#09090b !important; border:1px solid rgba(255,255,255,0.08) !important; border-radius:8px !important; }
/* Spline iframe: fullscreen — pointer-events auto so cursor tracking works */
[data-testid="stIFrame"] {
    border-radius: 0 !important;
    width: 100vw !important;
    max-width: 100vw !important;
    left: 50% !important;
    transform: translateX(-50%) !important;
    position: relative !important;
}
.js-plotly-plot .plotly, .js-plotly-plot .plotly .plot-container { background: transparent !important; }
</style>
"""

# ─── 5b. MINI LOADING OVERLAY HTML ──────────────────────────

LOADING_HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<script type="module" src="https://unpkg.com/@splinetool/viewer@1.9.79/build/spline-viewer.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;700;900&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  html, body { width:100%; height:100%; background:#000; overflow:hidden; }

  .stage {
    position: relative;
    width: 100%;
    height: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
  }

  /* Miniature Spline — desaturated, dim */
  .spline-mini {
    position: absolute;
    inset: 0;
    opacity: 0.35;
    filter: grayscale(0.4) brightness(0.7);
  }
  spline-viewer { width:100%; height:100%; display:block; }

  /* Dark vignette so text pops */
  .vignette {
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at center, rgba(0,0,0,0.2) 0%, rgba(0,0,0,0.85) 100%);
    pointer-events: none;
  }

  /* Watermark blocker */
  .wm-block {
    position: absolute;
    bottom: 0; right: 0;
    width: 200px; height: 70px;
    background: #000;
    z-index: 99999;
    pointer-events: none;
  }

  /* Content */
  .content {
    position: relative;
    z-index: 10;
    text-align: center;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 1.4rem;
  }

  /* Wordmark */
  .wordmark {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 2.6rem;
    font-weight: 900;
    letter-spacing: -0.04em;
    color: #fff;
    opacity: 0.92;
  }
  .wordmark span {
    background: linear-gradient(135deg, #fff 30%, #6b6b7b 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }

  /* Animated thinking dots line */
  .status-line {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.5);
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  .dots span {
    display: inline-block;
    animation: blink 1.4s ease-in-out infinite;
    opacity: 0;
  }
  .dots span:nth-child(2) { animation-delay: 0.22s; }
  .dots span:nth-child(3) { animation-delay: 0.44s; }
  @keyframes blink {
    0%, 80%, 100% { opacity: 0; }
    40%           { opacity: 1; }
  }

  /* Shimmer progress bar */
  .progress-wrap {
    width: 220px;
    height: 2px;
    background: rgba(255,255,255,0.08);
    border-radius: 2px;
    overflow: hidden;
  }
  .progress-bar {
    height: 100%;
    width: 40%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.7), transparent);
    border-radius: 2px;
    animation: shimmer 1.6s ease-in-out infinite;
  }
  @keyframes shimmer {
    0%   { transform: translateX(-120%); }
    100% { transform: translateX(320%); }
  }

  /* Cycling status messages */
  #msg { min-height: 1em; }
</style>
</head>
<body>
<div class="stage">
  <div class="spline-mini">
    <spline-viewer url="https://prod.spline.design/bBLr7TJSLnKwHY5A/scene.splinecode"
                   loading-anim-type="none"></spline-viewer>
  </div>
  <div class="vignette"></div>
  <div class="wm-block"></div>

  <div class="content">
    <div class="wordmark">VIBE<span>CHECK</span></div>

    <div class="status-line">
      <span id="msg">Scouring the web</span>
      <span class="dots"><span>.</span><span>.</span><span>.</span></span>
    </div>

    <div class="progress-wrap"><div class="progress-bar"></div></div>
  </div>
</div>

<script>
  /* Suppress Spline watermark in shadow DOM */
  const sl = setInterval(() => {
    const v = document.querySelector('spline-viewer');
    if (v && v.shadowRoot) {
      const s = document.createElement('style');
      s.textContent = '#logo,#spline-logo,a[href*="spline"],.spline-watermark,[class*="watermark"],[id*="logo"]{display:none!important}';
      v.shadowRoot.appendChild(s);
      clearInterval(sl);
    }
  }, 30);
  setTimeout(() => clearInterval(sl), 12000);

  /* Cycle through thinking messages */
  const msgs = [
    'Scouring the web',
    'Reading signals',
    'Weighing sentiment',
    'Crunching numbers',
    'Consulting Gemini',
    'Building your report',
  ];
  let i = 0;
  const el = document.getElementById('msg');
  setInterval(() => { i = (i + 1) % msgs.length; el.textContent = msgs[i]; }, 1800);
</script>
</body>
</html>
"""

# ─── 6. PAGE SETUP ───────────────────────────────────────────

st.set_page_config(page_title="VibeCheck", page_icon="🤖", layout="centered")
st.markdown(CSS, unsafe_allow_html=True)

# ─── HERO: 3D Robot ──────────────────────────────────────────
st.markdown('<div class="hero-outer">', unsafe_allow_html=True)

components.html("""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<script type="module" src="https://unpkg.com/@splinetool/viewer@1.9.79/build/spline-viewer.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@800;900&display=swap" rel="stylesheet">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  html, body { width: 100%; height: 100vh; overflow: hidden; background: #000; }

  .scene { position: relative; width: 100%; height: 100vh; }

  spline-viewer { position: absolute; inset: 0; width: 100%; height: 100%; display: block; }

  /* Suppress every possible Spline watermark element */
  .scene::after {
    content: '';
    position: absolute;
    bottom: 0; right: 0;
    width: 200px; height: 70px;
    background: #000;
    z-index: 99999;
    pointer-events: none;
  }

  /* ── Huge brand logo overlay ── */
  .brand {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    z-index: 100;
    text-align: center;
    pointer-events: none;
    user-select: none;
  }
  .brand-wordmark {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: clamp(4rem, 12vw, 9rem);
    font-weight: 900;
    letter-spacing: -0.04em;
    line-height: 0.9;
    color: #ffffff;
    text-shadow:
      0 0 80px rgba(255,255,255,0.12),
      0 4px 32px rgba(0,0,0,0.9);
  }
  .brand-wordmark span {
    background: linear-gradient(135deg, #ffffff 30%, #6b6b7b 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  .brand-sub {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: clamp(0.6rem, 1.5vw, 0.85rem);
    font-weight: 500;
    letter-spacing: 0.35em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.38);
    margin-top: 1rem;
  }

  /* ── Scroll arrow ── */
  .scroll-hint {
    position: absolute;
    bottom: 2rem;
    left: 50%;
    transform: translateX(-50%);
    z-index: 100;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.4rem;
    pointer-events: none;
  }
  .scroll-hint span {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 0.58rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.28);
  }
  .scroll-hint svg {
    animation: bounce 1.8s ease-in-out infinite;
  }
  @keyframes bounce {
    0%, 100% { transform: translateY(0); opacity: 0.3; }
    50%       { transform: translateY(6px); opacity: 0.7; }
  }
</style>
</head>
<body>
<div class="scene">
  <spline-viewer
    url="https://prod.spline.design/bBLr7TJSLnKwHY5A/scene.splinecode"
    loading-anim-type="spinner-small-dark">
  </spline-viewer>

  <!-- Brand logo on top -->
  <div class="brand">
    <div class="brand-wordmark">VIBE<span>CHECK</span></div>
    <div class="brand-sub">Product Intelligence Engine</div>
  </div>

  <!-- Scroll hint -->
  <div class="scroll-hint">
    <span>Scroll to explore</span>
    <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
      <path d="M3 6l6 6 6-6" stroke="rgba(255,255,255,0.5)" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
  </div>
</div>

<script>
  /* Suppress Spline shadow-DOM logo */
  const suppressLogo = setInterval(() => {
    const viewer = document.querySelector('spline-viewer');
    if (viewer && viewer.shadowRoot) {
      const shadow = viewer.shadowRoot;
      const s = document.createElement('style');
      s.textContent = `
        #logo, #spline-logo, a[href*="spline"], .spline-watermark,
        [class*="watermark"], [id*="logo"] { display: none !important; }
      `;
      shadow.appendChild(s);
      clearInterval(suppressLogo);
    }
  }, 30);
  setTimeout(() => clearInterval(suppressLogo), 12000);
</script>
</body>
</html>
""", height=700, scrolling=False)

st.markdown('</div>', unsafe_allow_html=True)

# TABS


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
        _loader = st.empty()
        with _loader:
            components.html(LOADING_HTML, height=280, scrolling=False)
        with st.spinner("Scouring the web for signals…"):
            data = get_reviews_data(product_query)

        if data:
            char_count = len(data)
            with st.spinner("Quantifying sentiment with Gemini…"):
                analysis = analyze_sentiment(data, product_query)

            _loader.empty()

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
                    st.markdown(f"""<div class="gc gc-amber">
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
                    st.markdown(f"""<div class="gc" style="height:100%;">
<span class="micro">Product</span>
<div class="prod-pill">📦 {h(product_query)}</div>
<div class="verdict-quote">{h(vibe)}</div>
<div style="font-size:0.7rem;color:rgba(255,255,255,0.2);font-style:italic;margin-top:0.6rem;">Synthesised from live web search snippets &amp; community discussions</div>
</div>""", unsafe_allow_html=True)

                # ── 02 Strengths & Weaknesses ─────────────
                st.markdown("""<div class="sec-row"><span class="sec-num">02</span><span class="sec-label">Strengths &amp; Weaknesses</span><div class="sec-rule"></div></div>""", unsafe_allow_html=True)
                pc, cc = st.columns(2, gap="medium")
                with pc:
                    pros_html = "".join([f'<div class="pci pci-pro"><span class="dot">▲</span><span>{h(p)}</span></div>' for p in pros])
                    st.markdown(f'<div class="gc gc-green"><span class="micro" style="color:#22c98a !important;">What people love</span>{pros_html}</div>', unsafe_allow_html=True)
                with cc:
                    cons_html = "".join([f'<div class="pci pci-con"><span class="dot">▼</span><span>{h(c)}</span></div>' for c in cons])
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
                st.markdown("""<div style="text-align:center;padding:2rem 0 1rem;font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:rgba(255,255,255,0.14);letter-spacing:0.1em;">VIBECHECK · LIVE WEB DATA</div>""", unsafe_allow_html=True)
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
        _cmp_loader = st.empty()
        with _cmp_loader:
            components.html(LOADING_HTML, height=280, scrolling=False)

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

            _cmp_loader.empty()

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
                ai_price_a = result_a.get('price', '')
                ai_price_b = result_b.get('price', '')
                price_a = format_inr(scraped_price_a if scraped_price_a else ai_price_a)
                price_b = format_inr(scraped_price_b if scraped_price_b else ai_price_b)
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
                    st.markdown(f"""<div class="gc gc-amber compare-score-card">
<span class="compare-product-name">{h(product_a)}</span>
<div class="compare-score-big" style="color:{color_a};">{score_a}</div>
<div style="font-family:'JetBrains Mono',monospace;font-size:0.9rem;color:rgba(255,255,255,0.2);">/100</div>
<div class="score-badge" style="color:{color_a};border-color:{color_a}40;background:{color_a}1a;margin-top:0.6rem;">{label_a}</div>
{winner_html}
<div class="verdict-strip">{h(vibe_a)}</div>
</div>""", unsafe_allow_html=True)
                with cb:
                    if score_b > score_a:
                        winner_html_b = '<div class="winner-badge">🏆 Winner</div>'
                    elif score_a == score_b:
                        winner_html_b = '<div class="winner-badge" style="background:rgba(255,255,255,0.06);border-color:rgba(255,255,255,0.18);color:rgba(255,255,255,0.45)!important;">🤝 Tie</div>'
                    else:
                        winner_html_b = ''
                    st.markdown(f"""<div class="gc gc-blue compare-score-card">
<span class="compare-product-name">{h(product_b)}</span>
<div class="compare-score-big" style="color:{color_b};">{score_b}</div>
<div style="font-family:'JetBrains Mono',monospace;font-size:0.9rem;color:rgba(255,255,255,0.2);">/100</div>
<div class="score-badge" style="color:{color_b};border-color:{color_b}40;background:{color_b}1a;margin-top:0.6rem;">{label_b}</div>
{winner_html_b}
<div class="verdict-strip">{h(vibe_b)}</div>
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
                    st.markdown(f"""<div class="gc" style="text-align:center;padding:1.4rem 1rem;">
<span class="compare-product-name">{h(product_a)}</span>
<div class="compare-price">{h(price_a)}</div>
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
                    st.markdown(f"""<div class="gc" style="text-align:center;padding:1.4rem 1rem;">
<span class="compare-product-name">{h(product_b)}</span>
<div class="compare-price">{h(price_b)}</div>
<span class="compare-price-label">Est. retail price</span>
{val_badge_b}
</div>""", unsafe_allow_html=True)

                # ── C3  Pros & Cons Side by Side ──────────
                st.markdown("""<div class="sec-row"><span class="sec-num">C3</span><span class="sec-label">Strengths</span><div class="sec-rule"></div></div>""", unsafe_allow_html=True)
                sa, sb = st.columns(2, gap="medium")
                with sa:
                    pros_a_html = "".join([f'<div class="pci pci-pro"><span class="dot">▲</span><span>{h(p)}</span></div>' for p in pros_a])
                    st.markdown(f'<div class="gc gc-green"><span class="micro" style="color:#22c98a !important;">{h(product_a)}</span>{pros_a_html}</div>', unsafe_allow_html=True)
                with sb:
                    pros_b_html = "".join([f'<div class="pci pci-pro"><span class="dot">▲</span><span>{h(p)}</span></div>' for p in pros_b])
                    st.markdown(f'<div class="gc gc-green"><span class="micro" style="color:#22c98a !important;">{h(product_b)}</span>{pros_b_html}</div>', unsafe_allow_html=True)

                st.markdown("""<div class="sec-row"><span class="sec-num">C4</span><span class="sec-label">Weaknesses</span><div class="sec-rule"></div></div>""", unsafe_allow_html=True)
                wa, wb = st.columns(2, gap="medium")
                with wa:
                    cons_a_html = "".join([f'<div class="pci pci-con"><span class="dot">▼</span><span>{h(c)}</span></div>' for c in cons_a])
                    st.markdown(f'<div class="gc gc-red"><span class="micro" style="color:#e85d5d !important;">{h(product_a)}</span>{cons_a_html}</div>', unsafe_allow_html=True)
                with wb:
                    cons_b_html = "".join([f'<div class="pci pci-con"><span class="dot">▼</span><span>{h(c)}</span></div>' for c in cons_b])
                    st.markdown(f'<div class="gc gc-red"><span class="micro" style="color:#e85d5d !important;">{h(product_b)}</span>{cons_b_html}</div>', unsafe_allow_html=True)

                st.markdown("""<div style="text-align:center;padding:2rem 0 1rem;font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:rgba(255,255,255,0.14);letter-spacing:0.1em;">VIBECHECK · HEAD-TO-HEAD COMPARE</div>""", unsafe_allow_html=True)

            else:
                st.markdown("""<div class="gc gc-warn" style="text-align:center;padding:2rem;"><div style="font-size:1.8rem;">⚠</div><div style="font-weight:600;margin-top:0.5rem;">AI analysis failed for one or both products.</div><div style="color:rgba(255,255,255,0.35);font-size:0.85rem;margin-top:0.3rem;">Quota limit likely hit. Wait 60s and retry.</div></div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="gc gc-err" style="text-align:center;padding:2rem;"><div style="font-size:1.8rem;">✕</div><div style="font-weight:600;margin-top:0.5rem;">Could not fetch data for one or both products.</div><div style="color:rgba(255,255,255,0.35);font-size:0.85rem;margin-top:0.3rem;">Check your Serper API key.</div></div>""", unsafe_allow_html=True)
    elif compare_btn:
        st.markdown("""<div class="gc" style="text-align:center;padding:1.5rem;"><div style="color:rgba(255,255,255,0.35);font-size:0.88rem;">Please enter both product names to compare.</div></div>""", unsafe_allow_html=True)

"""Global CSS injected once per page via ``st.markdown(_CSS, unsafe_allow_html=True)``."""

from __future__ import annotations

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Plus+Jakarta+Sans:ital,wght@0,400;0,500;0,600;0,700;1,400&family=Sora:wght@500;600;700;800&display=swap');

:root {
    --bg: #0a0c12;
    --bg-elev: #0e111a;
    --surface: #131826;
    --surface-2: #171d2e;
    --border: rgba(255,255,255,0.08);
    --border-strong: rgba(255,255,255,0.16);

    --text-primary: #edf0f7;
    --text-secondary: #99a2b8;
    --text-muted: #5b6378;

    --primary: #818cf8;
    --primary-strong: #6366f1;
    --primary-soft: rgba(129,140,248,0.12);
    --primary-border: rgba(129,140,248,0.30);

    --cyan: #38d9e8;
    --cyan-soft: rgba(56,217,232,0.12);
    --cyan-border: rgba(56,217,232,0.30);

    --fuchsia: #e879f9;
    --fuchsia-soft: rgba(232,121,249,0.12);
    --fuchsia-border: rgba(232,121,249,0.30);

    --success: #34d399;
    --success-soft: rgba(52,211,153,0.12);
    --success-border: rgba(52,211,153,0.30);

    --warning: #fbbf24;
    --warning-soft: rgba(251,191,36,0.12);
    --warning-border: rgba(251,191,36,0.30);

    --danger: #fb7185;
    --danger-soft: rgba(251,113,133,0.12);
    --danger-border: rgba(251,113,133,0.30);

    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 18px;
    --shadow-card: 0 1px 2px rgba(0,0,0,0.45), 0 12px 28px -12px rgba(0,0,0,0.55);
}

/* ── Base app shell ──────────────────────────────────────────────────── */
.stApp { background: var(--bg) !important; }
[data-testid="stHeader"] { background: var(--bg) !important; }
[data-testid="stDecoration"] {
    background: linear-gradient(90deg, var(--primary-strong), var(--cyan), var(--fuchsia)) !important;
    height: 3px !important;
}
/* Removed .stApp span and .stApp div to protect native icon fonts */
[data-testid="stAppViewContainer"], .stApp, .stApp p, .stApp label {
    font-family: 'Plus Jakarta Sans', sans-serif;
}
[data-testid="stSidebar"] {
    background: var(--bg-elev) !important;
    border-right: 1px solid var(--border);
}
/* Collapse the empty native header and force zero top padding */
[data-testid="stSidebarHeader"] {
    padding: 0 !important;
    min-height: 0 !important;
    height: 0 !important;
}
[data-testid="stSidebarContent"], 
[data-testid="stSidebarUserContent"] { 
    padding-top: 1rem !important; /* Adjust this value to nudge it exactly where you want */
}
/* Reduce the flexbox gap between Streamlit widgets in the sidebar */
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: 0.75rem !important; /* Streamlit's default is usually 1rem or 1.5rem */
}

/* Shrink the massive default margins on st.divider() */
[data-testid="stSidebar"] hr {
    margin: 1rem 0 !important; /* Default is often over 2em */
}
[data-testid="stCaptionContainer"], .stCaption { color: var(--text-muted) !important; }
.ir-icon { display: inline-block; vertical-align: -3px; flex-shrink: 0; }

/* ── Native widget skinning ──────────────────────────────────────────── */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: var(--radius-sm) !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px var(--primary-soft) !important;
}
[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
[data-testid="stMultiSelect"] div[data-baseweb="select"] > div {
    background: var(--surface) !important;
    border-color: var(--border) !important;
    border-radius: var(--radius-sm) !important;
}
[data-testid="stSlider"] [role="slider"] {
    background-color: var(--primary) !important;
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 4px var(--primary-soft) !important;
}
[data-testid="stTickBarMin"], [data-testid="stTickBarMax"] { color: var(--text-muted) !important; }
[data-testid="stWidgetLabel"] p {
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    font-size: 13px !important;
}
[data-testid="baseButton-secondary"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: var(--radius-sm) !important;
    transition: border-color .15s ease, background .15s ease;
}
[data-testid="baseButton-secondary"]:hover {
    border-color: var(--primary-border) !important;
    background: var(--primary-soft) !important;
    color: var(--primary) !important;
}
[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, var(--primary-strong), #5457e0) !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    box-shadow: 0 4px 16px -4px rgba(99,102,241,0.55) !important;
}
[data-testid="baseButton-primary"]:hover { filter: brightness(1.08); }
[data-testid="stForm"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 14px 16px;
}
[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
}
[data-testid="stAlert"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
}
[data-testid="stStatusWidget"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 100px !important;
    box-shadow: var(--shadow-card) !important;
}
[data-testid="stStatusWidget"] svg { display: none !important; }
[data-testid="stStatusWidget"]::before {
    content: "";
    width: 13px; height: 13px;
    border-radius: 50%;
    border: 2px solid var(--primary-border);
    border-top-color: var(--primary);
    display: inline-block;
    margin-right: 6px;
    animation: ir-spin .7s linear infinite;
}
[data-testid="stSpinner"] > div { color: var(--text-secondary) !important; }
@keyframes ir-spin { to { transform: rotate(360deg); } }

/* ── Brand header ─────────────────────────────────────────────────────── */
.ir-brand { display: flex; align-items: center; gap: 10px; }
.ir-brand-badge {
    width: 36px; height: 36px;
    border-radius: 9px;
    background: linear-gradient(135deg, var(--primary-strong), var(--cyan));
    display: flex; align-items: center; justify-content: center;
    color: #fff;
    box-shadow: 0 4px 14px -4px rgba(99,102,241,0.6);
    flex-shrink: 0;
}
.ir-brand-title {
    font-family: 'Sora', sans-serif;
    font-weight: 700;
    font-size: 16.5px;
    color: var(--text-primary);
    letter-spacing: -0.2px;
}

.ir-page-header { display: flex; align-items: center; gap: 14px; margin-bottom: 4px; }
.ir-page-header .ir-brand-badge { width: 42px; height: 42px; border-radius: 12px; }
.ir-page-title {
    font-family: 'Sora', sans-serif;
    font-weight: 800;
    font-size: 26px;
    color: var(--text-primary);
    letter-spacing: -0.4px;
    line-height: 1.15;
}
.ir-page-subtitle {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 13px;
    color: var(--text-muted);
    margin-top: 1px;
}

/* ── Chips (meta pills + stats row) ──────────────────────────────────── */
.ir-chip-row { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0 18px; }
.ir-chip {
    display: inline-flex; align-items: center; gap: 6px;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 12.5px; font-weight: 600;
    padding: 6px 13px;
    border-radius: 100px;
    border: 1px solid var(--border);
    background: var(--surface);
    color: var(--text-secondary);
    line-height: 1;
}
.ir-chip b { color: var(--text-primary); font-weight: 700; }
.ir-chip-primary { border-color: var(--primary-border); background: var(--primary-soft); color: var(--primary); }
.ir-chip-cyan { border-color: var(--cyan-border); background: var(--cyan-soft); color: var(--cyan); }
.ir-chip-fuchsia { border-color: var(--fuchsia-border); background: var(--fuchsia-soft); color: var(--fuchsia); }

/* ── Section / eyebrow labels (sidebar) ──────────────────────────────── */
.ir-sec {
    display: flex; align-items: center; gap: 7px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10.5px; font-weight: 700;
    letter-spacing: 1.3px; text-transform: uppercase;
    color: var(--text-muted);
    margin: 18px 0 10px;
}
.ir-sec .ir-icon { color: var(--primary); }
.ir-eyebrow {
    display: flex; align-items: center; gap: 6px;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 12px; font-weight: 600;
    color: var(--text-secondary);
    margin: 10px 0 8px;
    padding-left: 10px;
    border-left: 2px solid var(--primary-border);
}
.ir-eyebrow .ir-icon { color: var(--primary); }

/* ── Connection status badge ─────────────────────────────────────────── */
.ir-status {
    display: flex; align-items: center; gap: 7px;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 13px; font-weight: 600;
    padding: 7px 12px;
    border-radius: var(--radius-sm);
    border: 1px solid var(--border);
}
.ir-status-ok { background: var(--success-soft); border-color: var(--success-border); color: var(--success); }
.ir-status-bad { background: var(--danger-soft); border-color: var(--danger-border); color: var(--danger); }
.ir-status-sub { color: var(--text-secondary); font-weight: 500; }

/* ── Query insights panel ────────────────────────────────────────────── */
.ir-qp {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--primary);
    border-radius: 0 var(--radius-md) var(--radius-md) 0;
    padding: 16px 20px;
    margin-bottom: 18px;
}
.ir-qp-head {
    display: flex; align-items: center; gap: 7px;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 12.5px; font-weight: 700;
    color: var(--text-secondary);
    text-transform: uppercase; letter-spacing: 0.6px;
    margin-bottom: 10px;
}
.ir-qp-head .ir-icon { color: var(--primary); }
.ir-qp-row { display: flex; align-items: baseline; gap: 8px; margin-bottom: 7px; flex-wrap: wrap; }
.ir-qp-label { font-size: 12px; color: var(--text-muted); font-weight: 600; min-width: 72px; }
.ir-qp-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    color: var(--text-primary);
}
.ir-spell {
    display: inline-flex; align-items: center; gap: 4px;
    font-size: 10.5px; font-weight: 700;
    letter-spacing: 0.5px; text-transform: uppercase;
    color: var(--warning);
    background: var(--warning-soft);
    border: 1px solid var(--warning-border);
    padding: 2px 9px;
    border-radius: 100px;
    margin-left: 8px;
    vertical-align: middle;
}
.ir-token {
    display: inline-flex; align-items: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--text-secondary);
    background: var(--surface-2);
    border: 1px solid var(--border);
    padding: 2px 9px;
    border-radius: 100px;
    margin: 2px 3px 2px 0;
}
.ir-expand-term {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: var(--cyan);
    background: var(--cyan-soft);
    border: 1px solid var(--cyan-border);
    padding: 2px 9px;
    border-radius: 100px;
    margin: 2px 3px 2px 0;
    display: inline-flex;
}

/* ── Result card ──────────────────────────────────────────────────────── */
.ir-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 18px 22px 16px;
    margin-bottom: 14px;
    box-shadow: var(--shadow-card);
    transition: border-color .18s ease, transform .18s ease, box-shadow .18s ease;
}
.ir-card:hover {
    border-color: var(--primary-border);
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.4), 0 20px 36px -16px rgba(99,102,241,0.28);
}
.ir-card-header {
    display: flex; align-items: center; justify-content: space-between;
    flex-wrap: wrap; gap: 10px;
    margin-bottom: 12px;
}
.ir-card-left { display: flex; align-items: center; gap: 9px; }

.ir-rank {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; font-weight: 700;
    color: var(--text-secondary);
    background: var(--surface-2);
    border: 1px solid var(--border);
    width: 26px; height: 26px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
}
.ir-rank-1 { color: #f5c451; background: rgba(245,196,81,0.12); border-color: rgba(245,196,81,0.4); }
.ir-rank-2 { color: #c7cedb; background: rgba(199,206,219,0.10); border-color: rgba(199,206,219,0.32); }
.ir-rank-3 { color: #e0a37a; background: rgba(224,163,122,0.12); border-color: rgba(224,163,122,0.36); }

.ir-doc-id {
    display: inline-flex; align-items: center; gap: 5px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12.5px; font-weight: 600;
    color: var(--primary);
    background: var(--primary-soft);
    border: 1px solid var(--primary-border);
    padding: 4px 11px;
    border-radius: var(--radius-sm);
    letter-spacing: 0.2px;
    user-select: all;
    cursor: text;
}
.ir-doc-id .ir-icon { opacity: 0.75; }

.ir-score-block { display: flex; flex-direction: column; align-items: flex-end; gap: 5px; min-width: 110px; }
.ir-score {
    display: inline-flex; align-items: center; gap: 5px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12.5px; font-weight: 600;
    color: var(--success);
    background: var(--success-soft);
    border: 1px solid var(--success-border);
    padding: 4px 11px;
    border-radius: var(--radius-sm);
}
.ir-score-bar {
    width: 100%; height: 4px;
    background: var(--surface-2);
    border-radius: 100px;
    overflow: hidden;
}
.ir-score-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--success), #6ee7b7);
    border-radius: 100px;
}

.ir-title {
    font-family: 'Sora', sans-serif;
    font-size: 15.5px; font-weight: 600;
    color: var(--text-primary);
    line-height: 1.5;
    margin-bottom: 7px;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
.ir-snippet {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 13.5px;
    color: var(--text-secondary);
    line-height: 1.75;
    display: -webkit-box;
    -webkit-line-clamp: 4;
    -webkit-box-orient: vertical;
    overflow: hidden;
    -webkit-mask-image: linear-gradient(to bottom, black 75%, transparent 100%);
    mask-image: linear-gradient(to bottom, black 75%, transparent 100%);
}
.ir-snippet-muted { color: var(--text-muted) !important; font-style: italic; -webkit-line-clamp: 2; }

/* Hide the transient "Missing Submit Button" warning inside forms */
[data-testid="stForm"] [data-testid="stAlert"] {
    display: none !important;
}
/* Target the main block container using both class and modern test-ids */
.block-container, 
[data-testid="stMainBlockContainer"] {
    padding-top: 2.5rem !important;
}

/* Optional: If the empty top header bar is also pushing things down, shrink it */
[data-testid="stHeader"] {
    height: 2.5rem !important; 
    min-height: 2.5rem !important;
}

/* ── Tab navigation ───────────────────────────────────────────────────── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: transparent !important;
    gap: 4px;
    border-bottom: 1px solid var(--border) !important;
    padding-bottom: 0 !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 13.5px !important;
    color: var(--text-muted) !important;
    background: transparent !important;
    border: none !important;
    padding: 10px 18px 12px !important;
    border-radius: var(--radius-sm) var(--radius-sm) 0 0 !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: var(--primary) !important;
    background: var(--primary-soft) !important;
    border-bottom: 2px solid var(--primary) !important;
}
[data-testid="stTabsContent"] { padding-top: 20px !important; }

/* ── Metric stat card ─────────────────────────────────────────────────── */
.ir-stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 16px 18px;
    display: flex; flex-direction: column; gap: 4px;
}
.ir-stat-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10.5px; font-weight: 700;
    letter-spacing: 1px; text-transform: uppercase;
    color: var(--text-muted);
}
.ir-stat-value {
    font-family: 'Sora', sans-serif;
    font-size: 26px; font-weight: 800;
    color: var(--text-primary);
    line-height: 1.1;
}
.ir-stat-sub {
    font-size: 12px;
    color: var(--text-secondary);
    margin-top: 2px;
}

/* ── Plotly chart wrapper ─────────────────────────────────────────────── */
.ir-chart-wrap {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 14px;
    margin-bottom: 14px;
}
.ir-chart-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 13px; font-weight: 700;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-bottom: 10px;
}

/* ── Cluster card ─────────────────────────────────────────────────────── */
.ir-cluster-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 14px 16px;
    margin-bottom: 10px;
    border-top: 3px solid transparent;
    transition: transform .15s ease, box-shadow .15s ease;
}
.ir-cluster-card:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-card);
}
.ir-cluster-header {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 10px;
}
.ir-cluster-id {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px; font-weight: 700;
    padding: 3px 10px;
    border-radius: 100px;
    border: 1px solid;
}
.ir-cluster-pct {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 12px; font-weight: 600;
    color: var(--text-secondary);
}
.ir-cluster-terms {
    display: flex; flex-wrap: wrap; gap: 5px;
    margin-bottom: 8px;
}
.ir-cluster-term {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 100px;
    background: var(--surface-2);
    border: 1px solid var(--border);
    color: var(--text-secondary);
}

/* ── Cluster badge on result cards ───────────────────────────────────── */
.ir-cluster-badge {
    display: inline-flex; align-items: center; gap: 5px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; font-weight: 600;
    padding: 3px 9px;
    border-radius: 100px;
    border: 1px solid;
    margin-right: 6px;
}

/* ── Eval delta table ─────────────────────────────────────────────────── */
.ir-delta-pos { color: var(--success); font-weight: 700; }
.ir-delta-neg { color: var(--danger); font-weight: 700; }
.ir-delta-neu { color: var(--text-muted); }
</style>
"""

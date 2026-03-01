# ============================================
# TASK 18: STREAMLIT CHATBOT INTERFACE
# SGX Stock Advisor Bot
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="SGX Stock Advisor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS
# ============================================
st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

    /* Root variables */
    :root {
        --navy: #0d1b2a;
        --green: #00b894;
        --green-light: #00cec9;
        --amber: #fdcb6e;
        --red: #e17055;
        --slate: #1e2d3d;
        --text: #dfe6e9;
        --text-muted: #8fa3b1;
        --card-bg: #162435;
        --border: #2d4a63;
    }

    /* Global */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        color: var(--text);
    }

    .stApp {
        background-color: var(--navy);
    }

    /* Main header */
    .app-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
        border-bottom: 1px solid var(--border);
        margin-bottom: 1.5rem;
    }

    .app-header h1 {
        font-family: 'DM Serif Display', serif;
        font-size: 2rem;
        color: var(--green);
        margin: 0;
        letter-spacing: -0.5px;
    }

    .app-header p {
        color: var(--text-muted);
        font-size: 0.9rem;
        margin: 0.25rem 0 0 0;
    }

    /* Chat messages */
    .stChatMessage {
        background: var(--card-bg) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        margin-bottom: 0.75rem !important;
    }

    /* Stock card */
    .stock-card {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
        border-left: 4px solid var(--green);
    }

    .stock-card.medium {
        border-left-color: var(--amber);
    }

    .stock-card.low {
        border-left-color: var(--red);
    }

    .stock-card h4 {
        margin: 0 0 0.25rem 0;
        font-size: 1rem;
        font-weight: 600;
        color: #ffffff;
    }

    .stock-card .ticker {
        font-size: 0.8rem;
        color: var(--text-muted);
        font-weight: 500;
        letter-spacing: 0.5px;
    }

    .stock-card .metrics {
        display: flex;
        gap: 1.5rem;
        margin-top: 0.5rem;
        flex-wrap: wrap;
    }

    .metric-item {
        font-size: 0.82rem;
        color: var(--text-muted);
    }

    .metric-item span {
        color: var(--text);
        font-weight: 500;
    }

    .confidence-badge {
        display: inline-block;
        padding: 0.15rem 0.6rem;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-left: 0.5rem;
    }

    .badge-high {
        background: rgba(0, 184, 148, 0.15);
        color: var(--green);
        border: 1px solid var(--green);
    }

    .badge-medium {
        background: rgba(253, 203, 110, 0.15);
        color: var(--amber);
        border: 1px solid var(--amber);
    }

    .badge-low {
        background: rgba(225, 112, 85, 0.15);
        color: var(--red);
        border: 1px solid var(--red);
    }

    /* Choice buttons */
    .stButton > button {
        background: var(--slate) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        padding: 0.5rem 1.25rem !important;
        transition: all 0.2s ease !important;
    }

    .stButton > button:hover {
        background: var(--green) !important;
        color: var(--navy) !important;
        border-color: var(--green) !important;
        transform: translateY(-1px) !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: var(--slate) !important;
        border-right: 1px solid var(--border) !important;
    }

    [data-testid="stSidebar"] .stMarkdown {
        color: var(--text);
    }

    .sidebar-section {
        background: rgba(13, 27, 42, 0.5);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 0.75rem 1rem;
        margin-bottom: 1rem;
    }

    .sidebar-section h3 {
        color: var(--green);
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 0 0 0.5rem 0;
    }

    .disclaimer-box {
        background: rgba(225, 112, 85, 0.08);
        border: 1px solid rgba(225, 112, 85, 0.3);
        border-radius: 10px;
        padding: 0.75rem 1rem;
        font-size: 0.78rem;
        color: #b2bec3;
        line-height: 1.5;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: var(--slate) !important;
        border-radius: 8px !important;
        color: var(--text) !important;
        font-size: 0.875rem !important;
    }

    /* Divider */
    hr {
        border-color: var(--border) !important;
    }

    /* Chat input */
    .stChatInput {
        border-top: 1px solid var(--border) !important;
        padding-top: 0.75rem !important;
    }

    [data-testid="stChatInput"] textarea {
        background-color: var(--slate) !important;
        border-color: var(--border) !important;
        color: var(--text) !important;
        border-radius: 10px !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# DATA LOADING (CACHED)
# ============================================
@st.cache_resource
def load_rf_model():
    """Load the trained Random Forest model."""
    model_path = 'data/ml/rf_model.joblib'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

@st.cache_data
def load_app_data():
    """Load all required data files."""
    data = {}

    try:
        data['feature_cols'] = pd.read_csv(
            'data/ml/feature_columns.csv')['Feature'].tolist()
    except Exception:
        data['feature_cols'] = []

    try:
        data['master_df'] = pd.read_csv('data/master_features_latest.csv')
    except Exception:
        data['master_df'] = pd.DataFrame()

    try:
        data['raw_df'] = pd.read_csv('data/master_features_raw.csv')
    except Exception:
        data['raw_df'] = pd.DataFrame()

    try:
        data['universe'] = pd.read_csv('data/stock_universe_clean.csv')
    except Exception:
        data['universe'] = pd.DataFrame()

    try:
        data['feature_importance'] = pd.read_csv(
            'data/ml/feature_importance.csv')
    except Exception:
        data['feature_importance'] = pd.DataFrame()

    return data

# ============================================
# FACTOR SCORING MODEL
# ============================================
def compute_factor_score(row):
    """
    Factor-based scoring model.
    Sᵢ = 0.20·rev_growth + 0.10·eps + 0.10·momentum
       + 0.15·roe + 0.10·margin + 0.10·low_debt + 0.05·mcap
       + 0.10·low_vol + 0.10·low_drawdown
    Growth: 40%, Quality: 40%, Risk: 20%
    """
    score = 0

    # GROWTH FACTORS (40%)
    if pd.notna(row.get('Revenue_Growth')):
        rev = np.clip(row['Revenue_Growth'], -0.5, 1.0)
        score += ((rev + 0.5) / 1.5 * 100) * 0.20

    if pd.notna(row.get('EPS')):
        eps = np.clip(row['EPS'], -1, 5)
        score += ((eps + 1) / 6 * 100) * 0.10

    if pd.notna(row.get('Return_6M')):
        mom = np.clip(row['Return_6M'], -0.5, 1.0)
        score += ((mom + 0.5) / 1.5 * 100) * 0.10

    # QUALITY FACTORS (40%)
    if pd.notna(row.get('ROE')):
        roe = np.clip(row['ROE'], -0.2, 0.4)
        score += ((roe + 0.2) / 0.6 * 100) * 0.15

    if pd.notna(row.get('Profit_Margin')):
        margin = np.clip(row['Profit_Margin'], -0.2, 0.5)
        score += ((margin + 0.2) / 0.7 * 100) * 0.10

    if pd.notna(row.get('Debt_to_Equity')):
        debt = np.clip(row['Debt_to_Equity'], 0, 300)
        score += ((300 - debt) / 300 * 100) * 0.10

    if pd.notna(row.get('Market_Cap')):
        mcap_log = np.log10(max(row['Market_Cap'], 1e6))
        score += np.clip((mcap_log - 6) / 5.2 * 100, 0, 100) * 0.05

    # RISK FACTORS (20%)
    if pd.notna(row.get('Volatility')) and row['Volatility'] > 0:
        vol = np.clip(row['Volatility'], 0.1, 0.8)
        score += ((0.8 - vol) / 0.7 * 100) * 0.10

    if pd.notna(row.get('Max_Drawdown')):
        dd = np.clip(row['Max_Drawdown'], -0.6, 0)
        score += ((dd + 0.6) / 0.6 * 100) * 0.10

    return score

# ============================================
# PREPARE ML FEATURES (with sector encoding)
# ============================================
def prepare_ml_features(df, feature_cols):
    """Add sector one-hot encoding and align columns to feature_cols."""
    df = df.copy()

    # Add sector dummies
    if 'Sector' in df.columns:
        sector_dummies = pd.get_dummies(df['Sector'], prefix='Sector')
        df = pd.concat([df, sector_dummies], axis=1)

    # Ensure all expected sector columns exist
    for col in feature_cols:
        if col.startswith('Sector_') and col not in df.columns:
            df[col] = 0

    # Select only the feature columns, fill missing
    X = df[feature_cols].copy()
    X = X.fillna(X.median())
    return X

# ============================================
# INVESTOR SUITABILITY
# ============================================
def determine_suitability(stock_data):
    """Determine investor suitability based on stock characteristics."""
    vol = stock_data.get('Volatility', 0.3)
    rev_growth = stock_data.get('Revenue_Growth', 0)
    div_yield = stock_data.get('Dividend_Yield', 0)
    ret_6m = stock_data.get('Return_6M', 0)
    dte = stock_data.get('Debt_to_Equity', 50)

    growth_score = 0
    conservative_score = 0

    if vol > 0.4:
        growth_score += 2
    elif vol < 0.25:
        conservative_score += 2
    else:
        growth_score += 1
        conservative_score += 1

    if rev_growth > 0.15:
        growth_score += 2
    elif rev_growth > 0.05:
        growth_score += 1
    elif rev_growth < 0:
        conservative_score += 1

    if pd.notna(div_yield):
        if div_yield > 0.04:
            conservative_score += 2
        elif div_yield > 0.02:
            conservative_score += 1

    if ret_6m > 0.2:
        growth_score += 1
    elif ret_6m < 0:
        conservative_score += 1

    if pd.notna(dte):
        if dte < 30:
            conservative_score += 1
        elif dte > 100:
            growth_score += 1

    if growth_score >= conservative_score + 2:
        return 'Growth-focused investors'
    elif conservative_score >= growth_score + 2:
        return 'Conservative investors'
    else:
        return 'Balanced investors'

# ============================================
# RECOMMENDATION ENGINE
# ============================================
def get_recommendations(risk_level, preference, top_n=5):
    """
    Generate stock recommendations based on user's risk level and preference.

    Args:
        risk_level: 'Low', 'Medium', or 'High'
        preference: 'Growth', 'Blue-Chip', or 'Both'
        top_n: Number of recommendations to return (5 or 10)

    Returns:
        DataFrame of recommended stocks with scores and confidence levels
    """
    rf_model = load_rf_model()
    app_data = load_app_data()

    if app_data['master_df'].empty:
        return pd.DataFrame()

    df = app_data['master_df'].copy()

    # Merge company names
    if not app_data['universe'].empty and 'Company_Name' not in df.columns:
        df = df.merge(
            app_data['universe'][['Ticker', 'Company_Name']],
            on='Ticker', how='left'
        )

    # Compute factor scores
    df['Factor_Score'] = df.apply(compute_factor_score, axis=1)

    # Apply sector/preference filter
    growth_sectors = ['Technology', 'Healthcare', 'Consumer Cyclical',
                      'Consumer Defensive', 'Communication Services']
    bluechip_sectors = ['Financial Services', 'Industrials', 'Real Estate',
                        'Energy', 'Utilities', 'Basic Materials']

    if preference == 'Growth':
        df_filtered = df[df['Sector'].isin(growth_sectors)].copy()
        if len(df_filtered) < 5:
            df_filtered = df.copy()
    elif preference == 'Blue-Chip':
        df_filtered = df[df['Sector'].isin(bluechip_sectors)].copy()
        if len(df_filtered) < 5:
            df_filtered = df.copy()
    else:
        df_filtered = df.copy()

    # Apply risk-level filter
    if risk_level == 'Low':
        vol_threshold = 0.35
        df_filtered = df_filtered[
            df_filtered['Volatility'].fillna(1.0) <= vol_threshold
        ].copy()
        if len(df_filtered) < 5:
            df_filtered = df.copy()
    elif risk_level == 'High':
        # Include all — high-risk users can see everything
        pass

    # Get ML predictions
    feature_cols = app_data['feature_cols']
    if rf_model is not None and feature_cols:
        X = prepare_ml_features(df_filtered, feature_cols)
        df_filtered['ML_Probability'] = rf_model.predict_proba(X)[:, 1]
    else:
        df_filtered['ML_Probability'] = 0.5

    # Rank by each model
    df_filtered['Factor_Rank'] = df_filtered['Factor_Score'].rank(
        ascending=False, method='min').astype(int)
    df_filtered['ML_Rank'] = df_filtered['ML_Probability'].rank(
        ascending=False, method='min').astype(int)

    # Top N sets
    top5_factor = set(df_filtered.nsmallest(5, 'Factor_Rank')['Ticker'])
    top5_ml = set(df_filtered.nsmallest(5, 'ML_Rank')['Ticker'])
    top10_factor = set(df_filtered.nsmallest(10, 'Factor_Rank')['Ticker'])
    top10_ml = set(df_filtered.nsmallest(10, 'ML_Rank')['Ticker'])

    # Assign confidence
    def assign_confidence(row):
        ticker = row['Ticker']
        if ticker in (top5_factor & top5_ml):
            return 'HIGH'
        elif ticker in (top10_factor & top10_ml):
            return 'MEDIUM'
        elif ticker in (top10_factor | top10_ml):
            return 'LOW'
        return 'NOT_RECOMMENDED'

    df_filtered['Confidence'] = df_filtered.apply(assign_confidence, axis=1)

    # Combined normalised score
    factor_min = df_filtered['Factor_Score'].min()
    factor_max = df_filtered['Factor_Score'].max()
    if factor_max > factor_min:
        df_filtered['Factor_Norm'] = (
            (df_filtered['Factor_Score'] - factor_min) /
            (factor_max - factor_min)
        )
    else:
        df_filtered['Factor_Norm'] = 0.5

    df_filtered['Combined_Score'] = (
        df_filtered['Factor_Norm'] + df_filtered['ML_Probability']
    ) / 2

    # Get recommended stocks
    recommended = df_filtered[
        df_filtered['Confidence'] != 'NOT_RECOMMENDED'
    ].copy()

    # Sort: confidence order then combined score
    conf_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
    recommended['_conf_order'] = recommended['Confidence'].map(conf_order)
    recommended = recommended.sort_values(
        ['_conf_order', 'Combined_Score'], ascending=[True, False]
    ).head(top_n).drop(columns=['_conf_order'])

    return recommended

# ============================================
# EXPLANATION GENERATOR (Markdown, for Streamlit)
# ============================================
SECTOR_RISKS = {
    'Financial Services': "Exposed to credit cycles, interest rate changes, and MAS regulatory requirements.",
    'Technology': "Faces valuation risks, rapid obsolescence, and increasing regulatory scrutiny.",
    'Real Estate': "Sensitive to interest rate changes and economic cycles affecting occupancy.",
    'Industrials': "Cyclical and sensitive to economic slowdowns and supply chain disruptions.",
    'Consumer Cyclical': "Vulnerable to economic downturns and changing consumer preferences.",
    'Consumer Defensive': "Offers stability but may face margin pressure from inflation.",
    'Healthcare': "Subject to regulatory approval risks and patent expirations.",
    'Energy': "Highly sensitive to commodity price fluctuations.",
    'Communication Services': "Faces intense competition and high capital expenditure requirements.",
    'Utilities': "Stable but limited growth potential, sensitive to regulatory changes.",
    'Basic Materials': "Cyclical, exposed to commodity price volatility.",
}
DEFAULT_SECTOR_RISK = "May be affected by sector-specific and macroeconomic factors."


def generate_stock_explanation(stock_data, top_features=None):
    """
    Generate a Streamlit-ready markdown explanation for a stock.
    """
    ticker = stock_data.get('Ticker', 'N/A')
    company = stock_data.get('Company_Name', ticker)
    sector = stock_data.get('Sector', 'Unknown')
    confidence = stock_data.get('Confidence', 'LOW')
    factor_score = stock_data.get('Factor_Score', 0)
    factor_rank = stock_data.get('Factor_Rank', 'N/A')
    ml_prob = stock_data.get('ML_Probability', 0)
    ml_rank = stock_data.get('ML_Rank', 'N/A')
    price = stock_data.get('Price', None)

    md = []

    # Price line
    if price and not pd.isna(price):
        md.append(f"**Current Price:** S${price:.2f}")

    md.append("")

    # --- WHY RECOMMENDED ---
    md.append("**📈 Why Recommended**")
    md.append("")

    growth_items = []
    rev_growth = stock_data.get('Revenue_Growth')
    if pd.notna(rev_growth) and rev_growth > 0:
        strength = "Strong" if rev_growth > 0.10 else "Positive"
        growth_items.append(
            f"**{strength} Revenue Growth:** {rev_growth:.1%} YoY")

    eps = stock_data.get('EPS')
    if pd.notna(eps) and eps > 0:
        growth_items.append(f"**Positive Earnings:** EPS of S${eps:.2f}")

    ret_6m = stock_data.get('Return_6M')
    if pd.notna(ret_6m) and ret_6m > 0:
        strength = "Strong" if ret_6m > 0.15 else "Positive"
        growth_items.append(
            f"**{strength} Momentum:** Up {ret_6m:.1%} in 6 months")

    if growth_items:
        md.append("*Growth:*")
        for item in growth_items:
            md.append(f"- {item}")
        md.append("")

    quality_items = []
    roe = stock_data.get('ROE')
    if pd.notna(roe) and roe > 0.10:
        label = "Excellent" if roe > 0.15 else "Good"
        quality_items.append(f"**{label} ROE:** {roe:.1%}")

    margin = stock_data.get('Profit_Margin')
    if pd.notna(margin) and margin > 0.10:
        quality_items.append(f"**Healthy Profit Margin:** {margin:.1%}")

    dte = stock_data.get('Debt_to_Equity')
    if pd.notna(dte) and dte < 80:
        level = "Low" if dte < 30 else "Moderate"
        quality_items.append(f"**{level} Debt:** D/E of {dte:.1f}%")

    mcap = stock_data.get('Market_Cap')
    if pd.notna(mcap):
        if mcap > 10e9:
            quality_items.append(
                f"**Large Cap:** S${mcap/1e9:.1f}B (established company)")
        elif mcap > 1e9:
            quality_items.append(f"**Mid Cap:** S${mcap/1e9:.1f}B")

    if quality_items:
        md.append("*Quality:*")
        for item in quality_items:
            md.append(f"- {item}")
        md.append("")

    # Risk profile
    md.append("*Risk Profile:*")
    vol = stock_data.get('Volatility')
    if pd.notna(vol):
        if vol < 0.25:
            md.append(f"- 🟢 **Low Volatility:** {vol:.1%} annualised")
        elif vol < 0.40:
            md.append(f"- 🟡 **Moderate Volatility:** {vol:.1%} annualised")
        else:
            md.append(f"- 🔴 **High Volatility:** {vol:.1%} annualised")

    max_dd = stock_data.get('Max_Drawdown')
    if pd.notna(max_dd):
        md.append(f"- **Max Drawdown:** {max_dd:.1%}")

    md.append("")

    # --- RISK CONSIDERATIONS ---
    md.append("**⚠️ Risk Considerations**")
    md.append("")
    if pd.notna(dte):
        if dte > 150:
            md.append(f"- 🔴 High debt level (D/E = {dte:.1f}%)")
        elif dte > 80:
            md.append(f"- 🟡 Moderate debt level (D/E = {dte:.1f}%)")
        else:
            md.append(f"- 🟢 Low debt level (D/E = {dte:.1f}%)")

    sector_risk = SECTOR_RISKS.get(sector, DEFAULT_SECTOR_RISK)
    md.append(f"- **Sector:** {sector_risk}")
    md.append("")

    # --- MODEL INSIGHTS ---
    md.append("**🤖 Model Insights**")
    md.append("")
    md.append("| Model | Rank | Score |")
    md.append("|-------|------|-------|")
    md.append(f"| Factor Model | #{factor_rank} | {factor_score:.1f}/100 |")
    md.append(
        f"| ML Model | #{ml_rank} | {ml_prob:.1%} outperform probability |")
    md.append("")

    in_factor_top10 = (
        isinstance(factor_rank, (int, float)) and factor_rank <= 10)
    in_ml_top10 = (
        isinstance(ml_rank, (int, float)) and ml_rank <= 10)

    if in_factor_top10 and in_ml_top10:
        md.append("> ✅ **Both models agree** on this recommendation.")
    elif in_factor_top10:
        md.append("> 📊 **Factor Model pick** — strong fundamental metrics.")
    elif in_ml_top10:
        md.append("> 🧠 **ML Model pick** — pattern-based selection.")

    if top_features:
        md.append("")
        md.append("*Key ML features:*")
        for f in top_features[:3]:
            val = stock_data.get(f)
            if pd.notna(val):
                if isinstance(val, float):
                    display = f"{val:.2%}" if abs(val) < 1 else f"{val:.2f}"
                    md.append(f"- `{f}`: {display}")

    md.append("")

    # --- INVESTOR SUITABILITY ---
    suitability = determine_suitability(stock_data)
    md.append("**👤 Suitable For**")
    md.append("")
    md.append(f"*{suitability}*")
    md.append("")

    # --- DISCLAIMER ---
    md.append("---")
    md.append(
        "*⚠️ For educational purposes only. Not financial advice. "
        "Consult a licensed financial advisor before investing.*"
    )

    return '\n'.join(md)

# ============================================
# INTENT RECOGNITION
# ============================================
def detect_intent(user_message):
    """Simple keyword-based intent detection (FSM routing)."""
    msg = user_message.lower().strip()

    # Risk level
    if any(w in msg for w in ['low', 'safe', 'conservative', 'stable', 'minimal']):
        return 'risk_low'
    if any(w in msg for w in ['medium', 'moderate', 'balanced', 'mid']):
        return 'risk_medium'
    if any(w in msg for w in ['high', 'aggressive', 'risky', 'growth-oriented']):
        return 'risk_high'

    # Preference
    if any(w in msg for w in ['growth', 'aggressive', 'tech', 'high return', 'startup']):
        return 'pref_growth'
    if any(w in msg for w in ['blue chip', 'blue-chip', 'bluechip', 'stable',
                               'dividend', 'large cap', 'safe', 'conservative']):
        return 'pref_bluechip'
    if any(w in msg for w in ['both', 'all', 'either', 'no preference', "don't mind"]):
        return 'pref_both'

    # Follow-up actions
    if any(w in msg for w in ['more', 'detail', 'why', 'explain', 'tell me']):
        return 'explain_more'
    if any(w in msg for w in ['alternative', 'other', 'different', 'more stock',
                               'show more', 'top 10']):
        return 'show_alternatives'
    if any(w in msg for w in ['restart', 'start over', 'start again', 'reset',
                               'new', 'begin']):
        return 'restart'

    return 'unclear'

# ============================================
# SESSION STATE INITIALISATION
# ============================================
def init_session_state():
    """Initialise all session state variables."""
    defaults = {
        'state': 'welcome',          # FSM state
        'messages': [],              # Chat history
        'risk_level': None,          # 'Low', 'Medium', 'High'
        'preference': None,          # 'Growth', 'Blue-Chip', 'Both'
        'recommendations': None,     # DataFrame
        'top_n': 5,                  # 5 or 10
        'show_buttons': True,        # Whether to show choice buttons
        'selected_stock': None,      # Ticker for detail view
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

# ============================================
# CHAT HELPERS
# ============================================
def add_bot_message(content, role='assistant'):
    st.session_state.messages.append({'role': role, 'content': content})

def add_user_message(content):
    st.session_state.messages.append({'role': 'user', 'content': content})

def reset_session():
    """Full reset back to welcome state."""
    for key in ['state', 'messages', 'risk_level', 'preference',
                'recommendations', 'top_n', 'show_buttons', 'selected_stock']:
        if key in st.session_state:
            del st.session_state[key]
    init_session_state()

# ============================================
# RENDER FUNCTIONS
# ============================================
def render_stock_card(row, show_detail_button=True):
    """Render a compact stock card in the chat."""
    conf = row.get('Confidence', 'LOW')
    card_class = {'HIGH': '', 'MEDIUM': 'medium', 'LOW': 'low'}.get(conf, 'low')
    badge_class = {
        'HIGH': 'badge-high', 'MEDIUM': 'badge-medium', 'LOW': 'badge-low'
    }.get(conf, 'badge-low')

    conf_emoji = {'HIGH': '🟢', 'MEDIUM': '🟡', 'LOW': '🟠'}.get(conf, '🟠')
    company = row.get('Company_Name', row.get('Ticker', 'N/A'))
    ticker = row.get('Ticker', 'N/A')
    sector = row.get('Sector', 'Unknown')
    factor_rank = row.get('Factor_Rank', 'N/A')
    ml_rank = row.get('ML_Rank', 'N/A')
    factor_score = row.get('Factor_Score', 0)
    ml_prob = row.get('ML_Probability', 0)

    # Format optional metrics
    price_str = ""
    price = row.get('Price')
    if pd.notna(price):
        price_str = f"S${price:.2f}"

    ret_str = ""
    ret = row.get('Return_6M')
    if pd.notna(ret):
        arrow = "▲" if ret > 0 else "▼"
        ret_str = f"{arrow} {abs(ret):.1%} (6M)"

    vol_str = ""
    vol = row.get('Volatility')
    if pd.notna(vol):
        vol_str = f"{vol:.1%} vol."

    st.markdown(f"""
    <div class="stock-card {card_class}">
        <h4>{company}
            <span class="confidence-badge {badge_class}">{conf_emoji} {conf}</span>
        </h4>
        <div class="ticker">{ticker} · {sector}</div>
        <div class="metrics">
            {"<div class='metric-item'>Price: <span>" + price_str + "</span></div>" if price_str else ""}
            {"<div class='metric-item'>Return: <span>" + ret_str + "</span></div>" if ret_str else ""}
            {"<div class='metric-item'>Volatility: <span>" + vol_str + "</span></div>" if vol_str else ""}
            <div class='metric-item'>Factor: <span>#{factor_rank} ({factor_score:.1f})</span></div>
            <div class='metric-item'>ML: <span>#{ml_rank} ({ml_prob:.0%})</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if show_detail_button:
        if st.button(f"📋 More about {ticker}", key=f"detail_{ticker}"):
            st.session_state.selected_stock = ticker
            st.session_state.state = 'showing_detail'
            add_user_message(f"Tell me more about {ticker}")
            add_bot_message(f"Here's the full breakdown for **{ticker}**:")
            st.rerun()


def render_recommendations(recs, top_n=5):
    """Render the recommendation list in the chat."""
    if recs is None or len(recs) == 0:
        st.markdown("*No recommendations available for your criteria. "
                    "Try a different preference or risk level.*")
        return

    display = recs.head(top_n)

    high = display[display['Confidence'] == 'HIGH']
    medium = display[display['Confidence'] == 'MEDIUM']
    low = display[display['Confidence'] == 'LOW']

    if len(high) > 0:
        st.markdown("**🟢 High Confidence** — both models agree")
        for _, row in high.iterrows():
            render_stock_card(row.to_dict())

    if len(medium) > 0:
        st.markdown("**🟡 Medium Confidence** — both models in Top 10")
        for _, row in medium.iterrows():
            render_stock_card(row.to_dict())

    if len(low) > 0:
        st.markdown("**🟠 Lower Confidence** — single model pick")
        for _, row in low.iterrows():
            render_stock_card(row.to_dict())


def render_detail_view(ticker):
    """Render full explanation for a single stock."""
    app_data = load_app_data()
    recs = st.session_state.recommendations

    if recs is None or len(recs) == 0:
        st.markdown("*Stock details not available.*")
        return

    stock_row = recs[recs['Ticker'] == ticker]
    if len(stock_row) == 0:
        st.markdown(f"*Details for {ticker} not found in recommendations.*")
        return

    stock_data = stock_row.iloc[0].to_dict()

    # Merge company name if needed
    if 'Company_Name' not in stock_data and not app_data['universe'].empty:
        match = app_data['universe'][
            app_data['universe']['Ticker'] == ticker
        ]
        if len(match) > 0:
            stock_data['Company_Name'] = match.iloc[0]['Company_Name']

    # Get top ML features
    top_features = []
    if not app_data['feature_importance'].empty:
        top_features = app_data['feature_importance'].head(5)[
            'Feature'].tolist()

    explanation = generate_stock_explanation(stock_data, top_features)
    st.markdown(explanation)

# ============================================
# CONVERSATION MANAGER (FSM)
# ============================================
def handle_user_input(user_input):
    """
    Process user input and advance the FSM state.
    Called when the user types in the chat input.
    """
    intent = detect_intent(user_input)
    state = st.session_state.state

    # --- RESTART intent works from any state ---
    if intent == 'restart':
        reset_session()
        return

    # --- STATE: welcome ---
    # Expecting risk level
    if state == 'welcome':
        if intent == 'risk_low':
            st.session_state.risk_level = 'Low'
        elif intent == 'risk_medium':
            st.session_state.risk_level = 'Medium'
        elif intent == 'risk_high':
            st.session_state.risk_level = 'High'
        else:
            add_bot_message(
                "I didn't quite catch that. Please choose your risk tolerance: "
                "**Low**, **Medium**, or **High**.")
            st.session_state.show_buttons = True
            return

        add_bot_message(
            f"Got it — **{st.session_state.risk_level} risk**. 👍\n\n"
            "Now, what kind of stocks are you interested in?\n\n"
            "- **Growth** — technology, healthcare, consumer companies with "
            "higher growth potential\n"
            "- **Blue-Chip** — large, established companies with stable "
            "dividends (banks, REITs, industrials)\n"
            "- **Both** — show me a mix of both"
        )
        st.session_state.state = 'got_risk'
        st.session_state.show_buttons = True
        return

    # --- STATE: got_risk ---
    # Expecting preference
    if state == 'got_risk':
        if intent == 'pref_growth':
            st.session_state.preference = 'Growth'
        elif intent == 'pref_bluechip':
            st.session_state.preference = 'Blue-Chip'
        elif intent == 'pref_both':
            st.session_state.preference = 'Both'
        else:
            add_bot_message(
                "Please choose one: **Growth**, **Blue-Chip**, or **Both**.")
            st.session_state.show_buttons = True
            return

        add_bot_message(
            f"Perfect — looking for **{st.session_state.preference}** stocks "
            f"with **{st.session_state.risk_level}** risk tolerance. "
            "Analysing the SGX universe now...")

        recs = get_recommendations(
            st.session_state.risk_level,
            st.session_state.preference,
            top_n=10
        )
        st.session_state.recommendations = recs
        st.session_state.top_n = 5
        st.session_state.state = 'showing_recommendations'

        n_found = len(recs) if recs is not None else 0
        if n_found == 0:
            add_bot_message(
                "I couldn't find stocks matching those exact criteria. "
                "Try a different preference or risk level.")
        else:
            add_bot_message(
                f"Here are the **Top 5** SGX stocks for you, ranked by "
                f"combined model confidence. Stocks with 🟢 **HIGH confidence** "
                f"are recommended by both my Factor Model and ML Model.\n\n"
                f"_Click **📋 More about [ticker]** on any stock for a full explanation._"
            )
        st.session_state.show_buttons = True
        return

    # --- STATE: showing_recommendations ---
    if state == 'showing_recommendations':
        if intent == 'explain_more':
            add_bot_message(
                "Click **📋 More about [ticker]** on any stock card above "
                "to see the full explanation — including growth factors, "
                "quality metrics, risk profile, and model insights.")
            st.session_state.show_buttons = True
            return

        if intent == 'show_alternatives':
            st.session_state.top_n = 10
            add_bot_message(
                "Expanding to the **Top 10** recommendations:"
            )
            st.session_state.show_buttons = True
            return

        add_bot_message(
            "You can ask me to:\n"
            "- **\"Tell me more about [ticker]\"** — full stock explanation\n"
            "- **\"Show more\"** — expand to Top 10\n"
            "- **\"Start over\"** — change your risk or preference"
        )
        st.session_state.show_buttons = True
        return

    # --- STATE: showing_detail ---
    if state == 'showing_detail':
        if intent == 'show_alternatives':
            st.session_state.state = 'showing_recommendations'
            st.session_state.show_buttons = True
            add_bot_message("Here are your recommendations again:")
            return

        add_bot_message(
            "You can:\n"
            "- **\"Show alternatives\"** — back to the full list\n"
            "- **\"Start over\"** — change your preferences"
        )
        st.session_state.show_buttons = True
        return

    # Fallback
    add_bot_message(
        "I'm not sure what you mean. You can say things like:\n"
        "- *\"Low risk\"* / *\"High risk\"*\n"
        "- *\"Growth stocks\"* / *\"Blue-chip\"* / *\"Both\"*\n"
        "- *\"Tell me more\"* / *\"Show more\"* / *\"Start over\"*"
    )


# ============================================
# BUTTON HANDLER CALLBACKS
# ============================================
def handle_button_choice(choice_text):
    """Handle button click by simulating user input."""
    add_user_message(choice_text)
    handle_user_input(choice_text)
    st.session_state.show_buttons = False

# ============================================
# SIDEBAR
# ============================================
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 0.5rem 0 1rem 0;">
            <span style="font-size:2rem;">📈</span>
            <h2 style="font-family:'DM Serif Display',serif; color:#00b894;
                       font-size:1.2rem; margin:0.25rem 0 0 0;">
                SGX Stock Advisor
            </h2>
            <p style="color:#8fa3b1; font-size:0.75rem; margin:0;">
                Powered by Factor + ML Models
            </p>
        </div>
        """, unsafe_allow_html=True)

        # How it works
        with st.expander("ℹ️ How It Works", expanded=False):
            st.markdown("""
**Three simple steps:**

1. 🎯 Tell me your **risk tolerance**
   (Low / Medium / High)

2. 📊 Choose your **preference**
   (Growth / Blue-Chip / Both)

3. 🏆 Get **personalised recommendations**
   with full explanations

---
**Two models work together:**
- **Factor Model** — scores stocks on fundamentals (ROE, revenue growth, debt, volatility)
- **ML Model** — Random Forest trained on 2020–2024 SGX data

Stocks in **both** Top 5 lists get 🟢 **HIGH confidence**.
            """)

        # Session status
        st.markdown("**📋 Current Session**")
        col1, col2 = st.columns(2)
        with col1:
            risk_display = st.session_state.get('risk_level') or '—'
            st.metric("Risk", risk_display)
        with col2:
            pref_display = st.session_state.get('preference') or '—'
            st.metric("Preference", pref_display)

        st.markdown("")

        # Restart button
        if st.button("🔄 Start Over", use_container_width=True):
            reset_session()
            st.rerun()

        st.markdown("---")

        # Disclaimer
        st.markdown("""
        <div class="disclaimer-box">
            <strong style="color:#e17055;">⚠️ Important Disclaimer</strong><br><br>
            This tool is for <strong>educational purposes only</strong> and does
            not constitute financial advice. All recommendations are generated
            by experimental models trained on historical SGX data.<br><br>
            Past performance does not guarantee future results. Always consult
            a <strong>licensed financial advisor</strong> before making any
            investment decisions. You are solely responsible for your
            investment choices.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")
        st.markdown(
            "<p style='color:#8fa3b1; font-size:0.72rem; text-align:center;'>"
            "Data: SGX · yfinance · 2020–2024<br>"
            "Models: Factor Scoring + Random Forest"
            "</p>",
            unsafe_allow_html=True
        )

# ============================================
# CHOICE BUTTONS RENDERER
# ============================================
def render_choice_buttons():
    """Render contextual clickable buttons based on current FSM state."""
    state = st.session_state.state

    if state == 'welcome':
        st.markdown(
            "<p style='color:#8fa3b1; font-size:0.82rem;"
            " margin-bottom:0.5rem;'>Choose your risk tolerance:</p>",
            unsafe_allow_html=True
        )
        cols = st.columns(3)
        choices = [
            ('🟢 Low Risk', 'Low'),
            ('🟡 Medium Risk', 'Medium'),
            ('🔴 High Risk', 'High'),
        ]
        for i, (label, value) in enumerate(choices):
            with cols[i]:
                if st.button(label, key=f"risk_{value}",
                             use_container_width=True):
                    handle_button_choice(value)
                    st.rerun()

    elif state == 'got_risk':
        st.markdown(
            "<p style='color:#8fa3b1; font-size:0.82rem;"
            " margin-bottom:0.5rem;'>Choose your stock preference:</p>",
            unsafe_allow_html=True
        )
        cols = st.columns(3)
        choices = [
            ('📈 Growth', 'Growth'),
            ('🏦 Blue-Chip', 'Blue-Chip'),
            ('⚖️ Both', 'Both'),
        ]
        for i, (label, value) in enumerate(choices):
            with cols[i]:
                if st.button(label, key=f"pref_{value}",
                             use_container_width=True):
                    handle_button_choice(value)
                    st.rerun()

    elif state == 'showing_recommendations':
        cols = st.columns(2)
        with cols[0]:
            if st.button("📊 Show Top 10", key="btn_top10",
                         use_container_width=True):
                handle_button_choice("show more")
                st.rerun()
        with cols[1]:
            if st.button("🔄 Start Over", key="btn_restart_recs",
                         use_container_width=True):
                reset_session()
                st.rerun()

    elif state == 'showing_detail':
        cols = st.columns(2)
        with cols[0]:
            if st.button("⬅️ Back to List", key="btn_back",
                         use_container_width=True):
                handle_button_choice("show alternatives")
                st.rerun()
        with cols[1]:
            if st.button("🔄 Start Over", key="btn_restart_detail",
                         use_container_width=True):
                reset_session()
                st.rerun()

# ============================================
# MAIN APP
# ============================================
def main():
    init_session_state()
    render_sidebar()

    # Header
    st.markdown("""
    <div class="app-header">
        <h1>📈 SGX Stock Advisor</h1>
        <p>Personalised SGX stock recommendations for first-time investors</p>
    </div>
    """, unsafe_allow_html=True)

    # Welcome message (only on first load)
    if not st.session_state.messages:
        add_bot_message(
            "👋 Hello! I'm your SGX Stock Advisor.\n\n"
            "I'll help you find Singapore Exchange (SGX) stocks suited to your "
            "investment goals using a combination of fundamental factor analysis "
            "and machine learning.\n\n"
            "**To get started, what is your risk tolerance?**\n\n"
            "- 🟢 **Low** — I prefer stable, lower-risk stocks\n"
            "- 🟡 **Medium** — I'm comfortable with some fluctuation\n"
            "- 🔴 **High** — I'm willing to take on more risk for higher returns"
        )
        st.session_state.show_buttons = True

    # Render all chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])

            # After the last bot message, render inline content
            if (msg == st.session_state.messages[-1]
                    and msg['role'] == 'assistant'):

                # Render recommendations if in that state
                if st.session_state.state == 'showing_recommendations':
                    if st.session_state.recommendations is not None:
                        render_recommendations(
                            st.session_state.recommendations,
                            st.session_state.top_n
                        )

                # Render stock detail if in detail state
                elif st.session_state.state == 'showing_detail':
                    if st.session_state.selected_stock:
                        render_detail_view(st.session_state.selected_stock)

    # Render choice buttons below chat
    if st.session_state.show_buttons:
        render_choice_buttons()

    # Chat input
    user_input = st.chat_input(
        "Type your message or use the buttons above..."
    )
    if user_input:
        add_user_message(user_input)
        with st.chat_message('user'):
            st.markdown(user_input)

        handle_user_input(user_input)
        st.session_state.show_buttons = True
        st.rerun()


if __name__ == '__main__':
    main()
import streamlit as st
import joblib
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from io import BytesIO
import sqlite3
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Silent download of NLTK dependencies
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Page configuration
st.set_page_config(
    page_title="Fake News Detection System",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- DATABASE SETUP (PERSISTENT STORAGE) ---
def init_db():
    conn = sqlite3.connect('veritas_audit.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS audit_log
                 (timestamp TEXT, snippet TEXT, verdict TEXT, confidence REAL)''')
    conn.commit()
    conn.close()

def log_to_db(snippet, verdict, confidence):
    conn = sqlite3.connect('veritas_audit.db')
    c = conn.cursor()
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO audit_log VALUES (?, ?, ?, ?)", (timestamp, snippet, verdict, confidence))
    conn.commit()
    conn.close()

def fetch_logs():
    conn = sqlite3.connect('veritas_audit.db')
    df = pd.read_sql_query("SELECT * FROM audit_log", conn)
    conn.close()
    return df

init_db()

# --- URL SCRAPING SETUP ---
def scrape_url(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        article_text = ' '.join([p.get_text() for p in paragraphs])
        return article_text.strip()
    except Exception as e:
        return None

# --- SESSION STATE INITIALIZATION ---
if 'admin_auth' not in st.session_state:
    st.session_state.admin_auth = False

# --- TOP CONTROLS (ADMIN ONLY) ---
col_spacer, col_admin = st.columns([8.5, 1.5])

with col_admin:
    with st.popover("Admin Login"):
        if st.session_state.admin_auth:
            st.success("Authenticated")
            if st.button("Logout"):
                st.session_state.admin_auth = False
                st.rerun()
        else:
            pwd = st.text_input("Enter Password", type="password")
            if pwd == "admin123":
                st.session_state.admin_auth = True
                st.rerun()
            elif pwd != "":
                st.error("Incorrect Password")

# --- DARK THEME CSS ---
theme_css = """
:root {
    --bg-color: #050505;
    --text-color: #FFFFFF;
    --sub-text: #94A3B8;
    --card-bg: rgba(15, 15, 18, 0.85);
    --border-color: rgba(255, 255, 255, 0.2);
    --input-bg: rgba(0, 0, 0, 0.75);
    --input-text: #FFFFFF;
    --orb-color: rgba(255, 107, 26, 0.55);
    --stream-color: rgba(255, 107, 26, 0.85);
    --card-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.5);
}
"""
chart_font_color = "#FFFFFF"
chart_sub_color = "#94A3B8"

# --- GLOBAL STYLING INJECTION ---
st.markdown(f"""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700;800&family=Space+Grotesk:wght@700;800&display=swap" rel="stylesheet">
    
    <style>
    {theme_css}
    
    /* FIX: Targeted font replacement so it doesn't break Streamlit Icons */
    html, body, .stMarkdown, p, h1, h2, h3, h4, h5, h6, label {{
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }}

    .stApp {{
        background-color: var(--bg-color) !important;
        color: var(--text-color) !important;
        overflow-x: hidden;
    }}

    header {{ visibility: hidden; }}

    /* ADMIN POPOVER BUTTON */
    div[data-testid="stPopover"] > button {{
        background-color: var(--card-bg) !important;
        color: var(--text-color) !important;
        border: 1px solid var(--border-color) !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
    }}

    /* TABS HEADERS */
    button[data-baseweb="tab"] p {{
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 800 !important;
        font-size: 1.15rem !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
        color: var(--text-color) !important;
        opacity: 0.6 !important;
    }}

    button[data-baseweb="tab"][aria-selected="true"] p {{
        color: #FF6B1A !important;
        opacity: 1 !important;
    }}
    div[data-baseweb="tab-highlight"] {{
        background-color: #FF6B1A !important;
        height: 3px !important;
    }}

    /* HERO BANNER */
    .hero-banner-container {{
        position: relative;
        width: 100%;
        height: 230px;
        background: linear-gradient(180deg, #FF6B1A 0%, #D94E10 45%, var(--bg-color) 100%);
        border-radius: 20px;
        margin-top: 5px;
        margin-bottom: 25px;
        overflow: hidden;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        border: 1px solid rgba(255, 255, 255, 0.15);
        box-shadow: var(--card-shadow);
    }}

    .hero-banner-container::before {{
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: repeating-linear-gradient(90deg, transparent, transparent 19.8%, rgba(0, 0, 0, 0.15) 20%);
        pointer-events: none;
        z-index: 1;
    }}

    .hero-banner-text {{
        position: relative;
        z-index: 2;
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 4.5rem;
        font-weight: 800;
        color: #FFFFFF !important;
        text-transform: uppercase;
        letter-spacing: -0.02em;
        text-align: center;
        line-height: 1;
        text-shadow: 0 4px 10px rgba(0,0,0,0.3);
    }}

    .hero-banner-subtitle {{
        position: relative;
        z-index: 2;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 1.05rem;
        font-weight: 700;
        color: #FFFFFF !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 8px;
        text-align: center;
        text-shadow: 0 2px 6px rgba(0,0,0,0.25);
    }}

    /* UI LAYERING */
    .main .block-container {{
        max-width: 950px;
        padding-top: 1rem !important;
        position: relative;
        z-index: 10 !important;
        margin: 0 auto;
    }}

    div[data-testid="stVerticalBlock"], 
    div[data-testid="stDataFrame"], 
    .stDataFrame,
    .stTextArea, 
    .stButton {{
        position: relative !important;
        z-index: 999 !important;
    }}

    /* INPUT TEXT AREA & BUTTON */
    .stTextArea label {{ display: none; }}
    .stTextArea textarea {{
        background-color: var(--input-bg) !important;
        color: var(--input-text) !important;
        border: 2px solid var(--border-color) !important;
        border-radius: 12px !important;
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        padding: 1.5rem !important;
        text-align: center;
        box-shadow: var(--card-shadow);
    }}
    .stTextArea textarea::placeholder {{
        color: var(--sub-text) !important;
        opacity: 0.75;
        font-weight: 600;
    }}
    .stTextArea textarea:focus {{
        border-color: #FF6B1A !important;
        outline: none;
    }}

    .stButton > button {{
        background: var(--text-color) !important;
        color: var(--bg-color) !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 800 !important;
        font-size: 1.05rem !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.7rem 2.5rem !important;
        margin: 0 auto !important;
        display: block !important;
        transition: all 0.2s ease !important;
        box-shadow: var(--card-shadow);
    }}
    .stButton > button:hover {{
        transform: translateY(-2px);
        opacity: 0.9;
    }}

    /* METRIC CARDS & CHART CONTAINERS */
    .glass-card {{
        background: var(--card-bg);
        border: 2px solid var(--border-color);
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        box-shadow: var(--card-shadow);
    }}
    .glass-label {{
        font-size: 0.85rem;
        color: var(--sub-text) !important;
        text-transform: uppercase;
        margin-bottom: 8px;
        font-weight: 800;
        letter-spacing: 0.05em;
    }}
    .glass-val {{
        font-size: 2.5rem;
        font-weight: 800;
        color: var(--text-color) !important;
    }}

    div[data-testid="stPlotlyChart"] {{
        background-color: var(--card-bg);
        border: 2px solid var(--border-color);
        border-radius: 12px;
        padding: 15px;
        box-shadow: var(--card-shadow);
    }}

    /* BACKGROUND STREAM EFFECT */
    .bg-streams {{
        position: fixed;
        bottom: 0; left: 0; right: 0;
        height: 75vh;
        background: linear-gradient(to top, var(--stream-color) 0%, rgba(255, 107, 26, 0.0) 80%);
        -webkit-mask-image: repeating-linear-gradient(90deg, black 0%, black 9.8%, transparent 9.8%, transparent 10%);
        pointer-events: none;
        z-index: 0 !important; 
    }}
    
    .bg-orb {{
        position: fixed;
        bottom: -15%; left: 50%;
        transform: translateX(-50%);
        width: 800px; height: 800px;
        background: radial-gradient(circle, var(--orb-color) 0%, transparent 65%);
        pointer-events: none;
        z-index: 0 !important;
    }}

    /* SIDE-DRIFTING ANIMATIONS LOCKED TO BACKGROUND Z-INDEX 1 */
    .space-container {{
        position: fixed;
        top: 0; left: 0;
        width: 100vw; height: 100vh;
        pointer-events: none;
        z-index: 1 !important;
        overflow: hidden;
    }}

    .floating-card {{
        position: absolute;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.72rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        background: var(--card-bg);
        border: 2px solid var(--border-color);
        color: var(--text-color) !important;
        padding: 5px 10px;
        border-radius: 6px;
        box-shadow: var(--card-shadow);
        animation: zoomForward linear infinite;
        opacity: 0;
    }}

    @keyframes zoomForward {{
        0% {{ transform: scale(0.4) translateY(30px); opacity: 0; }}
        20% {{ opacity: 0.85; }}
        80% {{ opacity: 0.85; }}
        100% {{ transform: scale(1.1) translateY(-30px); opacity: 0; }}
    }}

    .fc-1 {{ top: 22%; left: 2%; animation-duration: 16s; animation-delay: 0s; }}
    .fc-2 {{ top: 48%; left: 2.5%; animation-duration: 18s; animation-delay: 5s; }}
    .fc-3 {{ top: 76%; left: 3%; animation-duration: 20s; animation-delay: 9s; }}
    .fc-4 {{ top: 28%; right: 2%; animation-duration: 17s; animation-delay: 2s; }}
    .fc-5 {{ top: 56%; right: 2.5%; animation-duration: 19s; animation-delay: 7s; }}
    .fc-6 {{ top: 82%; right: 3%; animation-duration: 21s; animation-delay: 11s; }}

    .tag-real {{
        color: #10B981; border: 1px solid rgba(16, 185, 129, 0.6);
        background: rgba(16, 185, 129, 0.12); padding: 1px 4px;
        border-radius: 3px; margin-left: 6px; font-size: 0.65rem;
    }}
    
    .tag-fake {{
        color: #EF4444; border: 1px solid rgba(239, 68, 68, 0.6);
        background: rgba(239, 68, 68, 0.12); padding: 1px 4px;
        border-radius: 3px; margin-left: 6px; font-size: 0.65rem;
    }}
    </style>

    <div class="bg-streams"></div>
    <div class="bg-orb"></div>
    
    <!-- Floating Animation Elements locked to background -->
    <div class="space-container">
        <div class="floating-card fc-1">FEDERAL RESERVE SIGNALS RATE CUTS <span class="tag-real">REAL</span></div>
        <div class="floating-card fc-2">G20 DRAFTS JOINT SUPPLY CHAIN RESOLUTION <span class="tag-real">REAL</span></div>
        <div class="floating-card fc-3">TECH EARNINGS EXCEED Q2 EXPECTATIONS <span class="tag-real">REAL</span></div>
        <div class="floating-card fc-4">NEW TAX LOOPHOLE EXEMPTS BILLIONAIRES <span class="tag-fake">FAKE</span></div>
        <div class="floating-card fc-5">GOVERNMENT TO SEIZE ALL RETAIL GOLD <span class="tag-fake">FAKE</span></div>
        <div class="floating-card fc-6">ELECTION COMMISSION THROWS OUT 500K BALLOTS <span class="tag-fake">FAKE</span></div>
    </div>

    <!-- HERO BANNER -->
    <div class="hero-banner-container">
        <div class="hero-banner-text">FAKE NEWS DETECTOR</div>
        <div class="hero-banner-subtitle">Automated Fact-Checking & Misinformation Detection Engine</div>
    </div>
""", unsafe_allow_html=True)

# --- NLP PREPROCESSING ---
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(cleaned_words)

@st.cache_resource
def load_model():
    try:
        return joblib.load('fake_news_pipeline.pkl')
    except FileNotFoundError:
        return None

model = load_model()

# --- MULTI-TAB ARCHITECTURE ---
tab_verify, tab_export, tab_admin = st.tabs(["VERIFICATION", "EXPORT OPTION", "ADMIN RETRAINING"])

# --- TAB 1: VERIFICATION DASHBOARD ---
with tab_verify:
    if model is None:
        st.error("Trained pipeline file `fake_news_pipeline.pkl` was not found. Please ensure the model file is present in your repository.")
    else:
        user_input = st.text_area("Hidden", height=140, placeholder="Paste a URL or news article text here to verify...")
        st.markdown("<br>", unsafe_allow_html=True)

        _, col_btn, _ = st.columns([1, 1, 1])
        with col_btn:
            analyze_btn = st.button("Verify Authenticity", use_container_width=True)

        if analyze_btn:
            input_text = user_input.strip()
            if not input_text:
                st.warning("Please enter text or a URL before running analysis.")
            else:
                is_url = input_text.startswith("http://") or input_text.startswith("https://")
                text_to_analyze = input_text
                
                if is_url:
                    with st.spinner("Scraping article text from URL..."):
                        scraped = scrape_url(input_text)
                        if not scraped or len(scraped) < 50:
                            st.error("Failed to extract sufficient text from the URL. Please paste the article text manually.")
                            st.stop()
                        text_to_analyze = scraped
                        st.success("Article successfully extracted from URL!")

                with st.spinner("Processing token distributions..."):
                    cleaned_input = clean_text(text_to_analyze)
                    prediction = model.predict([cleaned_input])[0]
                    probabilities = model.predict_proba([cleaned_input])[0]
                    
                    fake_prob = probabilities[0] * 100
                    real_prob = probabilities[1] * 100
                    word_count = len(text_to_analyze.split())
                    is_real = (prediction == 1)
                    confidence = real_prob if is_real else fake_prob
                    
                    snippet = text_to_analyze[:150] + "..."
                    verdict_str = "Real News" if is_real else "Fake News"
                    log_to_db(snippet, verdict_str, confidence)
                    
                    st.markdown("<br><br>", unsafe_allow_html=True)
                    
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        v_text = "AUTHENTIC" if is_real else "FAKE NEWS"
                        v_color = "#10B981" if is_real else "#EF4444"
                        st.markdown(f"""
                        <div class="glass-card">
                            <div class="glass-label">System Verdict</div>
                            <div style="color: {v_color}; font-weight: 800; font-size: 1.5rem;">{v_text}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with c2:
                        st.markdown(f"""
                        <div class="glass-card">
                            <div class="glass-label">Confidence Score</div>
                            <div class="glass-val" style="color: {chart_font_color} !important;">{confidence:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with c3:
                        st.markdown(f"""
                        <div class="glass-card">
                            <div class="glass-label">Tokens Analyzed</div>
                            <div class="glass-val" style="color: {chart_font_color} !important;">{word_count}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)
                    col_gauge, col_feat = st.columns([1, 1])
                    
                    with col_gauge:
                        fig_gauge = go.Figure(go.Indicator(
                            mode="gauge+number", value=real_prob, domain={'x': [0, 1], 'y': [0, 1]},
                            number={'suffix': "%", 'font': {'size': 40, 'color': chart_font_color, 'family': 'Plus Jakarta Sans, sans-serif'}},
                            gauge={
                                'axis': {'range': [0, 100], 'tickwidth': 3, 'tickcolor': chart_font_color, 'tickfont': {'family': 'Plus Jakarta Sans, sans-serif', 'color': chart_font_color}},
                                'bar': {'color': "#10B981" if is_real else "#EF4444"},
                                'bgcolor': "rgba(255,255,255,0.05)", 'borderwidth': 0,
                                'steps': [{'range': [0, 50], 'color': "rgba(239, 68, 68, 0.18)"}, {'range': [50, 100], 'color': "rgba(16, 185, 129, 0.18)"}],
                            }
                        ))
                        fig_gauge.update_layout(
                            title=dict(text="<b>PROBABILITY GAUGE</b>", font=dict(family='Space Grotesk, sans-serif', size=14, color=chart_sub_color), x=0.5, y=0.85),
                            height=320, margin=dict(l=30, r=30, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)", 
                            font=dict(family='Plus Jakarta Sans, sans-serif', color=chart_font_color, size=15)
                        )
                        st.plotly_chart(fig_gauge, use_container_width=True)

                    with col_feat:
                        try:
                            vectorizer = model.named_steps['tfidf']
                            classifier = model.named_steps['classifier']
                            feature_names = np.array(vectorizer.get_feature_names_out())
                            coefs = classifier.coef_[0]
                            input_vec = vectorizer.transform([cleaned_input])
                            present_indices = input_vec.nonzero()[1]
                            
                            if len(present_indices) > 0:
                                present_words = feature_names[present_indices]
                                present_weights = coefs[present_indices]
                                feat_df = pd.DataFrame({'Keyword': present_words, 'Impact': present_weights}).sort_values(by='Impact', key=abs, ascending=False).head(5)
                                feat_df['Type'] = feat_df['Impact'].apply(lambda x: 'Real' if x > 0 else 'Fake')
                                
                                fig_bar = px.bar(feat_df, x='Impact', y='Keyword', orientation='h', color='Type', color_discrete_map={'Real': '#10B981', 'Fake': '#EF4444'})
                                fig_bar.update_layout(
                                    title=dict(text="<b>LINGUISTIC FEATURE WEIGHTS</b>", font=dict(family='Space Grotesk, sans-serif', size=14, color=chart_sub_color), x=0.5, y=0.95),
                                    height=320, margin=dict(l=20, r=20, t=60, b=20), 
                                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", 
                                    yaxis=dict(autorange="reversed", title="", color=chart_font_color, tickfont=dict(family='Plus Jakarta Sans, sans-serif', color=chart_font_color, size=14)), 
                                    xaxis=dict(title="", color=chart_font_color, gridcolor="rgba(148, 163, 184, 0.3)", tickfont=dict(family='Plus Jakarta Sans, sans-serif', color=chart_font_color, size=13)), 
                                    font=dict(family='Plus Jakarta Sans, sans-serif', color=chart_font_color, size=15, weight="bold"), 
                                    showlegend=False
                                )
                                st.plotly_chart(fig_bar, use_container_width=True)
                            else:
                                st.info("No distinct vocabulary triggers matched.")
                        except Exception:
                            st.caption("Feature weights unavailable.")

# --- TAB 2: EXPORT REPORTS ---
with tab_export:
    st.markdown("### Verified News Audit Log")
    st.write("Maintain a running archive of flagged news content. Export directly to an Excel sheet for further review.")
    
    log_df = fetch_logs()
    
    if log_df.empty:
        st.info("No articles have been verified in the database yet.")
    else:
        log_df = log_df.sort_values(by="timestamp", ascending=False)
        st.dataframe(log_df, use_container_width=True)
        
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            log_df.to_excel(writer, index=False, sheet_name='Flagged_News')
        excel_data = output.getvalue()
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            label="Download Excel Report (.xlsx)", data=excel_data,
            file_name="Audit_Log.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# --- TAB 3: ADMIN RETRAINING (LOCKED) ---
with tab_admin:
    st.markdown("### Continuously Improvable Model")
    
    if not st.session_state.admin_auth:
        st.warning("Access Denied: Please use the Admin Login button in the top right corner of the screen to access pipeline retraining tools.")
    else:
        st.write("Upload a new labeled CSV dataset to update the machine learning pipeline and adapt to evolving misinformation tactics.")
        uploaded_file = st.file_uploader("Upload labeled training data (Must contain 'text' and 'label' columns)", type="csv")
        
        if uploaded_file is not None:
            st.success("File uploaded successfully.")
            if st.button("Initialize Pipeline Retraining", use_container_width=True):
                with st.spinner("Extracting TF-IDF features and retraining Logistic Regression model..."):
                    try:
                        new_data = pd.read_csv(uploaded_file)
                        if 'text' not in new_data.columns or 'label' not in new_data.columns:
                            st.error("Error: Dataset must contain 'text' and 'label' columns.")
                        else:
                            new_data['cleaned_text'] = new_data['text'].apply(clean_text)
                            new_pipeline = Pipeline([
                                ('tfidf', TfidfVectorizer(max_features=50000, stop_words='english')),
                                ('classifier', LogisticRegression(max_iter=1000))
                            ])
                            new_pipeline.fit(new_data['cleaned_text'], new_data['label'])
                            joblib.dump(new_pipeline, 'fake_news_pipeline.pkl')
                            st.success("Model successfully retrained and saved! The dashboard will now use the updated parameters.")
                    except Exception as e:
                        st.error(f"Retraining failed: {e}")

# import streamlit as st
# import joblib
# import re
# import string
# import nltk
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer

# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('stopwords')
# nltk.download('wordnet')

# # --- Load saved artifacts ---
# model = joblib.load('sentiment_model.pkl')
# tfidf = joblib.load('tfidf_vectorizer.pkl')
# le = joblib.load('label_encoder.pkl')

# stop_words = set(stopwords.words('english'))
# negations = {'not', 'no', 'nor', "don't", "isn't", "wasn't", "didn't"}
# stop_words = stop_words - negations
# lemmatizer = WordNetLemmatizer()

# # --- Preprocessing functions (same as training) ---
# def clean_text(text):
#     text = str(text).lower()
#     text = re.sub(r'http\S+|www\.\S+', '', text)
#     text = re.sub(r'<.*?>', '', text)
#     text = re.sub(r'\d+', '', text)
#     text = text.translate(str.maketrans('', '', string.punctuation))
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text

# def preprocess(text):
#     cleaned = clean_text(text)
#     tokens = word_tokenize(cleaned)
#     tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
#     tokens = [lemmatizer.lemmatize(t) for t in tokens]
#     return ' '.join(tokens)

# # --- Streamlit UI ---
# st.set_page_config(page_title="Sentiment Analysis", page_icon="💬")
# st.title("📊 Product Review Sentiment Analyzer")
# st.write("Enter a product review below to predict its sentiment.")

# user_input = st.text_area("Review text", height=150,
#     placeholder="e.g. The battery life is amazing and camera quality is great!")

# if st.button("Predict sentiment"):
#     if user_input.strip() == "":
#         st.warning("Please enter some text.")
#     else:
#         processed = preprocess(user_input)
#         vec = tfidf.transform([processed])
#         pred = model.predict(vec)[0]
#         sentiment = le.inverse_transform([pred])[0]

#         if sentiment == 'Positive':
#             st.success(f"Sentiment: {sentiment} 😊")
#         elif sentiment == 'Negative':
#             st.error(f"Sentiment: {sentiment} 😞")
#         else:
#             st.info(f"Sentiment: {sentiment} 😐")





# import streamlit as st
# import joblib
# import re
# import string
# import numpy as np
# import nltk
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer

# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('stopwords')
# nltk.download('wordnet')

# # ================= Page config =================
# st.set_page_config(
#     page_title="Sentiment Analyzer",
#     page_icon="💬",
#     layout="centered"
# )

# # ================= Custom CSS =================
# st.markdown("""
#     <style>
#     .main-title {
#         font-size: 2.4rem;
#         font-weight: 700;
#         text-align: center;
#         margin-bottom: 0;
#     }
#     .subtitle {
#         text-align: center;
#         color: #888;
#         margin-bottom: 2rem;
#     }
#     .result-box {
#         padding: 1.2rem;
#         border-radius: 12px;
#         text-align: center;
#         font-size: 1.4rem;
#         font-weight: 600;
#         margin-top: 1rem;
#     }
#     .positive { background-color: #E6F4EA; color: #1E7E34; }
#     .negative { background-color: #FDECEA; color: #C0392B; }
#     .neutral  { background-color: #FEF6E0; color: #B7791F; }
#     </style>
# """, unsafe_allow_html=True)

# # ================= Load artifacts (cached) =================
# @st.cache_resource
# def load_artifacts():
#     model = joblib.load('sentiment_model.pkl')
#     tfidf = joblib.load('tfidf_vectorizer.pkl')
#     le = joblib.load('label_encoder.pkl')
#     return model, tfidf, le

# model, tfidf, le = load_artifacts()

# stop_words = set(stopwords.words('english'))
# negations = {'not', 'no', 'nor', "don't", "isn't", "wasn't", "didn't"}
# stop_words = stop_words - negations
# lemmatizer = WordNetLemmatizer()

# # ================= Preprocessing (same as training) =================
# def clean_text(text):
#     text = str(text).lower()
#     text = re.sub(r'http\S+|www\.\S+', '', text)
#     text = re.sub(r'<.*?>', '', text)
#     text = re.sub(r'\d+', '', text)
#     text = text.translate(str.maketrans('', '', string.punctuation))
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text

# def preprocess(text):
#     cleaned = clean_text(text)
#     tokens = word_tokenize(cleaned)
#     tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
#     tokens = [lemmatizer.lemmatize(t) for t in tokens]
#     return ' '.join(tokens)

# # ================= Sidebar =================
# with st.sidebar:
#     st.header("ℹ️ About")
#     st.write("This app predicts the sentiment of a product review as **Positive**, **Neutral**, or **Negative** using a trained ML model (TF-IDF + classifier).")
#     st.divider()
#     st.subheader("Try an example")
#     examples = {
#         "👍 Positive": "This phone is amazing, battery lasts all day and the camera is superb!",
#         "👎 Negative": "Worst purchase ever, it stopped working within two days.",
#         "😐 Neutral": "It's an okay phone, nothing special but does the job.",
#     }
#     for label, text in examples.items():
#         if st.button(label, use_container_width=True):
#             st.session_state['input_text'] = text

# # ================= Main UI =================
# st.markdown('<div class="main-title">💬 Sentiment Analyzer</div>', unsafe_allow_html=True)
# st.markdown('<div class="subtitle">Analyze the sentiment of product reviews instantly</div>', unsafe_allow_html=True)

# if 'input_text' not in st.session_state:
#     st.session_state['input_text'] = ""

# user_input = st.text_area(
#     "Enter a review:",
#     value=st.session_state['input_text'],
#     height=150,
#     placeholder="e.g. The battery life is amazing and camera quality is great!"
# )

# col1, col2 = st.columns([1, 1])
# with col1:
#     predict_clicked = st.button("🔍 Predict Sentiment", use_container_width=True, type="primary")
# with col2:
#     clear_clicked = st.button("🗑️ Clear", use_container_width=True)

# if clear_clicked:
#     st.session_state['input_text'] = ""
#     st.rerun()

# if predict_clicked:
#     if user_input.strip() == "":
#         st.warning("⚠️ Please enter some text first.")
#     else:
#         processed = preprocess(user_input)
#         vec = tfidf.transform([processed])
#         pred = model.predict(vec)[0]
#         sentiment = le.inverse_transform([pred])[0]

#         # confidence scores (if model supports it)
#         if hasattr(model, "predict_proba"):
#             probs = model.predict_proba(vec)[0]
#         elif hasattr(model, "decision_function"):
#             scores = model.decision_function(vec)[0]
#             exp_scores = np.exp(scores - np.max(scores))
#             probs = exp_scores / exp_scores.sum()
#         else:
#             probs = None

#         css_class = sentiment.lower()
#         emoji = {'positive': '😊', 'negative': '😞', 'neutral': '😐'}.get(css_class, '')

#         st.markdown(
#             f'<div class="result-box {css_class}">Sentiment: {sentiment} {emoji}</div>',
#             unsafe_allow_html=True
#         )

#         if probs is not None:
#             st.write("")
#             st.subheader("Confidence breakdown")
#             for cls, p in zip(le.classes_, probs):
#                 st.write(f"{cls}: {p*100:.1f}%")
#                 st.progress(float(p))

#         with st.expander("🔧 See processed text"):
#             st.code(processed)

# # ================= Footer =================
# st.divider()
# st.caption("Built with Streamlit · TF-IDF + Machine Learning model")








# import streamlit as st
# import joblib
# import re
# import string
# import numpy as np
# import pandas as pd
# import plotly.express as px
# import nltk
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer

# for pkg in ['punkt', 'punkt_tab', 'stopwords', 'wordnet']:
#     nltk.download(pkg)

# # ================= Page config =================
# st.set_page_config(
#     page_title="Sentiment Analyzer",
#     page_icon="💬",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # ================= Custom CSS =================
# st.markdown("""
# <style>
# .hero {
#     background: linear-gradient(135deg, #534AB7 0%, #378ADD 100%);
#     padding: 2rem; border-radius: 16px; color: white; margin-bottom: 1.5rem;
# }
# .hero h1 { margin: 0; font-size: 2rem; }
# .hero p { margin: 0.4rem 0 0; opacity: 0.9; }
# .metric-card {
#     background: var(--background-color, #fff); border: 1px solid rgba(128,128,128,0.2);
#     border-radius: 12px; padding: 1rem; text-align: center;
# }
# .result-box {
#     padding: 1.4rem; border-radius: 14px; text-align: center;
#     font-size: 1.5rem; font-weight: 700; margin: 1rem 0;
# }
# .positive { background-color: #E6F4EA; color: #1E7E34; }
# .negative { background-color: #FDECEA; color: #C0392B; }
# .neutral  { background-color: #FEF6E0; color: #B7791F; }
# </style>
# """, unsafe_allow_html=True)

# # ================= Load artifacts =================
# @st.cache_resource
# def load_artifacts():
#     model = joblib.load('sentiment_model.pkl')
#     tfidf = joblib.load('tfidf_vectorizer.pkl')
#     le = joblib.load('label_encoder.pkl')
#     return model, tfidf, le

# model, tfidf, le = load_artifacts()

# stop_words = set(stopwords.words('english'))
# negations = {'not', 'no', 'nor', "don't", "isn't", "wasn't", "didn't"}
# stop_words = stop_words - negations
# lemmatizer = WordNetLemmatizer()

# def clean_text(text):
#     text = str(text).lower()
#     text = re.sub(r'http\S+|www\.\S+', '', text)
#     text = re.sub(r'<.*?>', '', text)
#     text = re.sub(r'\d+', '', text)
#     text = text.translate(str.maketrans('', '', string.punctuation))
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text

# def preprocess(text):
#     cleaned = clean_text(text)
#     tokens = word_tokenize(cleaned)
#     tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
#     tokens = [lemmatizer.lemmatize(t) for t in tokens]
#     return ' '.join(tokens)

# def get_prediction(text):
#     processed = preprocess(text)
#     vec = tfidf.transform([processed])
#     pred = model.predict(vec)[0]
#     sentiment = le.inverse_transform([pred])[0]

#     if hasattr(model, "predict_proba"):
#         probs = model.predict_proba(vec)[0]
#     elif hasattr(model, "decision_function"):
#         scores = model.decision_function(vec)[0]
#         exp_scores = np.exp(scores - np.max(scores))
#         probs = exp_scores / exp_scores.sum()
#     else:
#         probs = np.zeros(len(le.classes_))

#     return sentiment, probs, processed

# # ================= Session state =================
# if 'history' not in st.session_state:
#     st.session_state['history'] = []
# if 'input_text' not in st.session_state:
#     st.session_state['input_text'] = ""

# # ================= Sidebar =================
# with st.sidebar:
#     st.header("💬 Sentiment Analyzer")
#     st.write("Predicts review sentiment as **Positive**, **Neutral**, or **Negative** using TF-IDF + ML.")
#     st.divider()
#     st.subheader("Try an example")
#     examples = {
#         "👍 Positive": "This phone is amazing, battery lasts all day and the camera is superb!",
#         "👎 Negative": "Worst purchase ever, it stopped working within two days.",
#         "😐 Neutral": "It's an okay phone, nothing special but does the job.",
#     }
#     for label, text in examples.items():
#         if st.button(label, use_container_width=True):
#             st.session_state['input_text'] = text
#     st.divider()
#     if st.button("🗑️ Clear history", use_container_width=True):
#         st.session_state['history'] = []
#         st.rerun()

# # ================= Hero header =================
# st.markdown("""
# <div class="hero">
#     <h1>💬 Product Review Sentiment Analyzer</h1>
#     <p>TF-IDF + Machine Learning · Real-time sentiment prediction</p>
# </div>
# """, unsafe_allow_html=True)

# # ================= KPI row =================
# hist_df = pd.DataFrame(st.session_state['history'])
# col1, col2, col3, col4 = st.columns(4)
# with col1:
#     st.metric("Total analyzed", len(hist_df) if not hist_df.empty else 0)
# with col2:
#     pos = int((hist_df['sentiment'] == 'Positive').sum()) if not hist_df.empty else 0
#     st.metric("😊 Positive", pos)
# with col3:
#     neu = int((hist_df['sentiment'] == 'Neutral').sum()) if not hist_df.empty else 0
#     st.metric("😐 Neutral", neu)
# with col4:
#     neg = int((hist_df['sentiment'] == 'Negative').sum()) if not hist_df.empty else 0
#     st.metric("😞 Negative", neg)

# st.write("")

# # ================= Tabs =================
# tab1, tab2, tab3 = st.tabs(["🔍 Single review", "📂 Batch (CSV)", "📈 History"])

# # ---- Tab 1: Single review ----
# with tab1:
#     left, right = st.columns([3, 2])

#     with left:
#         user_input = st.text_area(
#             "Enter a review:",
#             value=st.session_state['input_text'],
#             height=160,
#             placeholder="e.g. The battery life is amazing and camera quality is great!"
#         )
#         c1, c2 = st.columns(2)
#         with c1:
#             predict_clicked = st.button("🔍 Predict sentiment", use_container_width=True, type="primary")
#         with c2:
#             clear_clicked = st.button("🗑️ Clear text", use_container_width=True)

#         if clear_clicked:
#             st.session_state['input_text'] = ""
#             st.rerun()

#         if predict_clicked:
#             if user_input.strip() == "":
#                 st.warning("⚠️ Please enter some text first.")
#             else:
#                 sentiment, probs, processed = get_prediction(user_input)
#                 css_class = sentiment.lower()
#                 emoji = {'positive': '😊', 'negative': '😞', 'neutral': '😐'}.get(css_class, '')

#                 st.markdown(
#                     f'<div class="result-box {css_class}">Sentiment: {sentiment} {emoji}</div>',
#                     unsafe_allow_html=True
#                 )

#                 # save to history
#                 st.session_state['history'].append({
#                     'text': user_input[:80],
#                     'sentiment': sentiment,
#                     'positive_pct': round(float(probs[list(le.classes_).index('Positive')]) * 100, 1) if 'Positive' in le.classes_ else 0,
#                 })

#                 with right:
#                     st.subheader("Confidence breakdown")
#                     prob_df = pd.DataFrame({
#                         'Sentiment': le.classes_,
#                         'Confidence': (probs * 100).round(1)
#                     })
#                     fig = px.bar(prob_df, x='Confidence', y='Sentiment', orientation='h',
#                                  color='Sentiment', range_x=[0, 100],
#                                  color_discrete_map={'Positive': '#1E7E34', 'Negative': '#C0392B', 'Neutral': '#B7791F'})
#                     fig.update_layout(showlegend=False, height=220, margin=dict(l=0,r=0,t=10,b=0))
#                     st.plotly_chart(fig, use_container_width=True)

#                 with st.expander("🔧 Processed text (after cleaning)"):
#                     st.code(processed)

# # ---- Tab 2: Batch CSV ----
# with tab2:
#     st.write("Upload a CSV with a text column to analyze multiple reviews at once.")
#     uploaded = st.file_uploader("Upload CSV", type=["csv"])

#     if uploaded:
#         batch_df = pd.read_csv(uploaded)
#         text_col = st.selectbox("Select the text column", batch_df.columns)

#         if st.button("Run batch prediction", type="primary"):
#             with st.spinner("Analyzing..."):
#                 results = [get_prediction(str(t))[0] for t in batch_df[text_col]]
#                 batch_df['sentiment'] = results

#             st.success(f"Analyzed {len(batch_df)} reviews")
#             st.dataframe(batch_df, use_container_width=True)

#             # distribution chart
#             dist = batch_df['sentiment'].value_counts().reset_index()
#             dist.columns = ['Sentiment', 'Count']
#             fig2 = px.pie(dist, names='Sentiment', values='Count', hole=0.4,
#                           color='Sentiment',
#                           color_discrete_map={'Positive': '#1E7E34', 'Negative': '#C0392B', 'Neutral': '#B7791F'})
#             st.plotly_chart(fig2, use_container_width=True)

#             csv = batch_df.to_csv(index=False).encode('utf-8')
#             st.download_button("📥 Download results", csv, "sentiment_results.csv", "text/csv")

# # ---- Tab 3: History ----
# with tab3:
#     if hist_df.empty:
#         st.info("No predictions yet. Analyze a review in the 'Single review' tab.")
#     else:
#         st.dataframe(hist_df, use_container_width=True)
#         dist = hist_df['sentiment'].value_counts().reset_index()
#         dist.columns = ['Sentiment', 'Count']
#         fig3 = px.bar(dist, x='Sentiment', y='Count', color='Sentiment',
#                       color_discrete_map={'Positive': '#1E7E34', 'Negative': '#C0392B', 'Neutral': '#B7791F'})
#         fig3.update_layout(showlegend=False, height=300)
#         st.plotly_chart(fig3, use_container_width=True)

# # ================= Footer =================
# st.divider()
# st.caption("Built with Streamlit · TF-IDF + Machine Learning model")











from pathlib import Path
 
import pandas as pd
import plotly.express as px
import streamlit as st
 
from assets.preprocessing import get_prediction, load_artifacts
 
# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# ============================================================
# Load custom CSS from assets/style.css
# ============================================================
def load_css(path: Path) -> None:
    with open(path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
 
 
load_css(Path(__file__).parent / "assets" / "style.css")
 
# ============================================================
# Load model artifacts (cached so they only load once)
# ============================================================
@st.cache_resource
def get_artifacts():
    return load_artifacts(
        model_path="sentiment_model.pkl",
        vectorizer_path="tfidf_vectorizer.pkl",
        encoder_path="label_encoder.pkl",
    )
 
 
model, tfidf, le = get_artifacts()
 
COLOR_MAP = {"Positive": "#1E7E34", "Negative": "#C0392B", "Neutral": "#B7791F"}
 
# ============================================================
# Session state
# ============================================================
if "history" not in st.session_state:
    st.session_state["history"] = []
if "input_text" not in st.session_state:
    st.session_state["input_text"] = ""
 
# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.header("💬 Sentiment Analyzer")
    st.write(
        "Predicts review sentiment as **Positive**, **Neutral**, or "
        "**Negative** using TF-IDF + a trained ML model."
    )
    st.divider()
    st.subheader("Try an example")
 
    examples = {
        "👍 Positive": "This phone is amazing, battery lasts all day and the camera is superb!",
        "👎 Negative": "Worst purchase ever, it stopped working within two days.",
        "😐 Neutral": "It's an okay phone, nothing special but does the job.",
    }
    for label, text in examples.items():
        if st.button(label, use_container_width=True):
            st.session_state["input_text"] = text
 
    st.divider()
    if st.button("🗑️ Clear history", use_container_width=True):
        st.session_state["history"] = []
        st.rerun()
 
# ============================================================
# Hero header
# ============================================================
st.markdown(
    """
    <div class="hero">
        <h1>💬 Product Review Sentiment Analyzer</h1>
        <p>TF-IDF + Machine Learning · Real-time sentiment prediction</p>
    </div>
    """,
    unsafe_allow_html=True,
)
 
# ============================================================
# KPI row
# ============================================================
hist_df = pd.DataFrame(st.session_state["history"])
 
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total analyzed", len(hist_df) if not hist_df.empty else 0)
with col2:
    pos = int((hist_df["sentiment"] == "Positive").sum()) if not hist_df.empty else 0
    st.metric("😊 Positive", pos)
with col3:
    neu = int((hist_df["sentiment"] == "Neutral").sum()) if not hist_df.empty else 0
    st.metric("😐 Neutral", neu)
with col4:
    neg = int((hist_df["sentiment"] == "Negative").sum()) if not hist_df.empty else 0
    st.metric("😞 Negative", neg)
 
st.write("")
 
# ============================================================
# Tabs
# ============================================================
tab1, tab2, tab3 = st.tabs(["🔍 Single review", "📂 Batch (CSV)", "📈 History"])
 
# ---- Tab 1: Single review ----
with tab1:
    left, right = st.columns([3, 2])
 
    with left:
        user_input = st.text_area(
            "Enter a review:",
            value=st.session_state["input_text"],
            height=160,
            placeholder="e.g. The battery life is amazing and camera quality is great!",
        )
        c1, c2 = st.columns(2)
        with c1:
            predict_clicked = st.button(
                "🔍 Predict sentiment", use_container_width=True, type="primary"
            )
        with c2:
            clear_clicked = st.button("🗑️ Clear text", use_container_width=True)
 
        if clear_clicked:
            st.session_state["input_text"] = ""
            st.rerun()
 
        if predict_clicked:
            if user_input.strip() == "":
                st.warning("⚠️ Please enter some text first.")
            else:
                sentiment, probs, processed = get_prediction(user_input, model, tfidf, le)
                css_class = sentiment.lower()
                emoji = {"positive": "😊", "negative": "😞", "neutral": "😐"}.get(css_class, "")
 
                st.markdown(
                    f'<div class="result-box {css_class}">Sentiment: {sentiment} {emoji}</div>',
                    unsafe_allow_html=True,
                )
 
                st.session_state["history"].append(
                    {"text": user_input[:80], "sentiment": sentiment}
                )
 
                with right:
                    st.subheader("Confidence breakdown")
                    prob_df = pd.DataFrame(
                        {"Sentiment": le.classes_, "Confidence": (probs * 100).round(1)}
                    )
                    fig = px.bar(
                        prob_df,
                        x="Confidence",
                        y="Sentiment",
                        orientation="h",
                        color="Sentiment",
                        range_x=[0, 100],
                        color_discrete_map=COLOR_MAP,
                    )
                    fig.update_layout(
                        showlegend=False, height=220, margin=dict(l=0, r=0, t=10, b=0)
                    )
                    st.plotly_chart(fig, use_container_width=True)
 
                with st.expander("🔧 Processed text (after cleaning)"):
                    st.code(processed)
 
# ---- Tab 2: Batch CSV ----
with tab2:
    st.write("Upload a CSV with a text column to analyze multiple reviews at once.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
 
    if uploaded:
        batch_df = pd.read_csv(uploaded)
        text_col = st.selectbox("Select the text column", batch_df.columns)
 
        if st.button("Run batch prediction", type="primary"):
            with st.spinner("Analyzing..."):
                results = [
                    get_prediction(str(t), model, tfidf, le)[0] for t in batch_df[text_col]
                ]
                batch_df["sentiment"] = results
 
            st.success(f"Analyzed {len(batch_df)} reviews")
            st.dataframe(batch_df, use_container_width=True)
 
            dist = batch_df["sentiment"].value_counts().reset_index()
            dist.columns = ["Sentiment", "Count"]
            fig2 = px.pie(
                dist,
                names="Sentiment",
                values="Count",
                hole=0.4,
                color="Sentiment",
                color_discrete_map=COLOR_MAP,
            )
            st.plotly_chart(fig2, use_container_width=True)
 
            csv = batch_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download results", csv, "sentiment_results.csv", "text/csv")
 
# ---- Tab 3: History ----
with tab3:
    if hist_df.empty:
        st.info("No predictions yet. Analyze a review in the 'Single review' tab.")
    else:
        st.dataframe(hist_df, use_container_width=True)
        dist = hist_df["sentiment"].value_counts().reset_index()
        dist.columns = ["Sentiment", "Count"]
        fig3 = px.bar(dist, x="Sentiment", y="Count", color="Sentiment", color_discrete_map=COLOR_MAP)
        fig3.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig3, use_container_width=True)
 
# ============================================================
# Footer
# ============================================================
st.divider()
st.caption("Built with Streamlit · TF-IDF + Machine Learning model")
 
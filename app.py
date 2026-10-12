"""
app.py
======
Streamlit web application for SMS Spam Detection.
Loads the pre-trained TF-IDF + Multinomial Naive Bayes pipeline and
lets users classify any SMS message as Spam or Ham.

Run:
    streamlit run app.py
"""

import json
import pathlib
import textwrap

import joblib
# noqa: F401 — must be importable so joblib can unpickle the pipeline's preprocessor
from text_utils import clean_text  # noqa: F401
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = pathlib.Path(__file__).parent
MODEL_PATH  = BASE_DIR / "model" / "spam_pipeline.joblib"
METRICS_PATH = BASE_DIR / "model" / "metrics.json"
CM_IMG      = BASE_DIR / "assets" / "confusion_matrix.png"
MB_IMG      = BASE_DIR / "assets" / "metrics_bar.png"
DATA_PATH   = BASE_DIR / "data"  / "SMSSpamCollection"

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #f8f9fa; }

    /* Result boxes */
    .spam-box {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white; border-radius: 12px; padding: 20px 24px;
        text-align: center; font-size: 1.6rem; font-weight: 700;
        box-shadow: 0 4px 14px rgba(238,90,36,0.35);
        margin: 10px 0;
    }
    .ham-box {
        background: linear-gradient(135deg, #00b894, #00cec9);
        color: white; border-radius: 12px; padding: 20px 24px;
        text-align: center; font-size: 1.6rem; font-weight: 700;
        box-shadow: 0 4px 14px rgba(0,184,148,0.35);
        margin: 10px 0;
    }
    .confidence-box {
        background: #ffffff; border: 1px solid #dee2e6;
        border-radius: 8px; padding: 14px 18px;
        text-align: center; margin: 6px 0;
    }
    .metric-card {
        background: #ffffff; border: 1px solid #dee2e6;
        border-radius: 10px; padding: 16px;
        text-align: center; margin: 4px;
    }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #2d3436; }
    .metric-label { font-size: 0.85rem; color: #636e72; margin-top: 4px; }
    .sample-btn-area { margin-top: 8px; }
    h1 { color: #2d3436; }
    .stTextArea textarea { font-size: 1rem; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model …")
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_metrics():
    if not METRICS_PATH.exists():
        return None
    with open(METRICS_PATH) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_dataset_sample():
    if not DATA_PATH.exists():
        return None
    df = pd.read_csv(
        DATA_PATH, sep="\t", header=None,
        names=["label", "message"], encoding="latin-1",
    )
    return df


def predict(pipeline, text: str):
    """Return (label_str, spam_probability)."""
    proba = pipeline.predict_proba([text])[0]   # [ham_prob, spam_prob]
    label_num = int(np.argmax(proba))
    return ("Spam" if label_num == 1 else "Ham"), float(proba[1])


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/spam.png", width=72)
    st.title("SMS Spam Detector")
    st.markdown("---")
    st.markdown(
        "This app uses a **TF-IDF + Multinomial Naive Bayes** model "
        "trained on the [UCI SMS Spam Collection]"
        "(https://archive.ics.uci.edu/dataset/228/sms+spam+collection) "
        "dataset (5,574 messages)."
    )
    st.markdown("---")
    st.subheader("📋 How it works")
    st.markdown("""
1. Enter or paste any SMS text.
2. Click **Predict**.
3. The model returns a **Spam / Ham** verdict with confidence.
    """)
    st.markdown("---")
    st.subheader("⚠️ Disclaimer")
    st.caption(
        "Confidence scores are model probability estimates, not guaranteed "
        "certainties. Edge cases may be misclassified."
    )
    if not MODEL_PATH.exists():
        st.error(
            "Model not found. Run `python train_model.py` first.",
            icon="🚫",
        )

# ── Load model ────────────────────────────────────────────────────────────────
pipeline = load_model()
metrics  = load_metrics()

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("📱 SMS Spam Detection System")
st.markdown(
    "Classify any SMS message as **Spam** or **Not Spam** instantly "
    "using a machine learning model trained on real SMS data."
)

tab_predict, tab_evaluate, tab_explore = st.tabs(
    ["🔍 Predict", "📊 Model Evaluation", "📂 Dataset Explorer"]
)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Predict
# ─────────────────────────────────────────────────────────────────────────────
with tab_predict:
    st.subheader("Enter an SMS message")

    SAMPLE_MESSAGES = {
        "🏆 Prize winner (Spam)":
            "Congratulations! You've won a £1000 prize. "
            "Call 08700621170 to claim. T&C apply.",
        "💰 Free cash (Spam)":
            "FREE entry in 2 a weekly comp to win FA Cup final tkts 21st May 2005. "
            "Text FA to 87121 to receive entry question (std txt rate). T&C's apply.",
        "📞 Bank fraud alert (Spam)":
            "URGENT! Your mobile number has been awarded a £2,000 bonus. "
            "To claim call 09061743810. Valid 12 hours only.",
        "💬 Casual text (Ham)":
            "Hey, are you free tonight? Thought we could grab dinner.",
        "📅 Meeting reminder (Ham)":
            "Don't forget we have a team standup at 10 am tomorrow.",
        "🛒 Shopping list (Ham)":
            "Can you pick up milk and bread on your way home?",
    }

    # Persist the textarea content across reruns so that clicking Load sample
    # on one run and Predict on the next both see the same text.
    if "sms_text" not in st.session_state:
        st.session_state["sms_text"] = ""

    col_input, col_samples = st.columns([3, 1])

    with col_samples:
        st.markdown("**Try a sample:**")
        selected_sample = st.radio(
            "Sample messages", list(SAMPLE_MESSAGES.keys()),
            label_visibility="collapsed",
        )
        if st.button("Load sample →", use_container_width=True):
            st.session_state["sms_text"] = SAMPLE_MESSAGES[selected_sample]

    with col_input:
        user_input = st.text_area(
            "SMS Message",
            value=st.session_state["sms_text"],
            height=160,
            placeholder="Type or paste your SMS message here …",
            label_visibility="collapsed",
            key="sms_text",
        )

        predict_btn = st.button("🔍 Predict", type="primary", use_container_width=True)

    # ── Prediction result ────────────────────────────────────────────────────
    if predict_btn:
        if not pipeline:
            st.error("Model not loaded. Run `python train_model.py` first.", icon="🚫")
        elif not user_input.strip():
            st.warning("Please enter a message before predicting.", icon="⚠️")
        elif len(user_input.strip()) < 3:
            st.warning("Message too short. Please enter at least 3 characters.", icon="⚠️")
        else:
            label, spam_prob = predict(pipeline, user_input)
            ham_prob = 1.0 - spam_prob

            st.markdown("---")
            res_col, conf_col = st.columns(2)

            with res_col:
                st.markdown("#### 🎯 Classification Result")
                if label == "Spam":
                    st.markdown(
                        '<div class="spam-box">🚨 SPAM</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div class="ham-box">✅ NOT SPAM</div>',
                        unsafe_allow_html=True,
                    )

            with conf_col:
                st.markdown("#### 📈 Model Confidence")
                st.markdown(
                    f'<div class="confidence-box">'
                    f'<span style="font-size:1.3rem;font-weight:700;color:#e17055;">Spam: {spam_prob:.1%}</span><br>'
                    f'<span style="font-size:1.3rem;font-weight:700;color:#00b894;">Ham&nbsp;: {ham_prob:.1%}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.caption(
                    "ℹ️ Confidence is a probability estimate from the trained model. "
                    "Values close to 50 % indicate uncertainty."
                )

            # Probability bar
            st.markdown("#### Confidence Breakdown")
            prob_df = pd.DataFrame({
                "Category": ["Ham (Not Spam)", "Spam"],
                "Probability": [ham_prob, spam_prob],
            })
            fig, ax = plt.subplots(figsize=(6, 1.6))
            colors = ["#00b894", "#ff6b6b"]
            ax.barh(prob_df["Category"], prob_df["Probability"], color=colors, height=0.5)
            ax.set_xlim(0, 1)
            ax.set_xlabel("Probability")
            ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
            for i, (val, cat) in enumerate(zip(prob_df["Probability"], prob_df["Category"])):
                ax.text(val + 0.01, i, f"{val:.2%}", va="center", fontsize=9)
            ax.set_title("Prediction Probability", fontsize=11, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # Input summary
            with st.expander("📝 Input summary"):
                words = len(user_input.split())
                chars = len(user_input)
                st.write(f"**Word count:** {words}  |  **Character count:** {chars}")
                st.write(f"**Classified as:** {label}  |  **Spam probability:** {spam_prob:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Model Evaluation
# ─────────────────────────────────────────────────────────────────────────────
with tab_evaluate:
    st.subheader("📊 Model Performance on Held-Out Test Set")
    st.markdown(
        "The model is evaluated on **20 %** of the dataset (held out during training, "
        "stratified by class)."
    )

    if metrics:
        m1, m2, m3, m4 = st.columns(4)
        for col, name, key, color in [
            (m1, "Accuracy",  "accuracy",  "#4C72B0"),
            (m2, "Precision", "precision", "#55A868"),
            (m3, "Recall",    "recall",    "#C44E52"),
            (m4, "F1 Score",  "f1_score",  "#8172B2"),
        ]:
            with col:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-value" style="color:{color}">{metrics[key]:.2%}</div>'
                    f'<div class="metric-label">{name}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
    else:
        st.info("Metrics not found. Run `python train_model.py` to generate them.")

    st.markdown("---")
    img_col1, img_col2 = st.columns(2)

    with img_col1:
        st.markdown("#### Confusion Matrix")
        if CM_IMG.exists():
            st.image(str(CM_IMG), width="stretch")
        else:
            st.info("Run `python train_model.py` to generate the confusion matrix.")

    with img_col2:
        st.markdown("#### Performance Metrics Bar Chart")
        if MB_IMG.exists():
            st.image(str(MB_IMG), width="stretch")
        else:
            st.info("Run `python train_model.py` to generate the bar chart.")

    st.markdown("---")
    st.subheader("🧠 Model & Pipeline Details")
    detail_col1, detail_col2 = st.columns(2)
    with detail_col1:
        st.markdown("""
**Preprocessing**
- Lowercasing
- URL replacement → `url`
- Number replacement → `num`
- Punctuation removal
- Whitespace normalisation

**Vectorisation: TF-IDF**
- Unigrams + bigrams (1–2)
- Max features: 10,000
- Sublinear TF scaling: ✅
        """)
    with detail_col2:
        st.markdown("""
**Classifier: Multinomial Naive Bayes**
- Smoothing parameter α = 0.1
- Works directly on TF-IDF feature vectors
- Fast, interpretable, strong on text

**Training / Test split**
- 80 % training  |  20 % test
- Stratified by label (preserves class ratio)
- Random state = 42 (reproducible)
        """)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Dataset Explorer
# ─────────────────────────────────────────────────────────────────────────────
with tab_explore:
    st.subheader("📂 Dataset Overview")
    df = load_dataset_sample()

    if df is None:
        st.warning(
            "Dataset not found. Run `python train_model.py` first to download it.",
            icon="⚠️",
        )
    else:
        raw_total  = len(df)
        df_clean = df.drop_duplicates().dropna()
        spam_count = df_clean["label"].eq("spam").sum()
        ham_count  = df_clean["label"].eq("ham").sum()
        total      = len(df_clean)

        st.info(
            f"**Original dataset:** {raw_total:,} messages loaded from file "
            f"(UCI SMS Spam Collection — 5,574 messages as published, "
            f"actual file contains {raw_total:,} rows).  \n"
            f"**After deduplication:** {total:,} unique messages used for training & evaluation "
            f"({raw_total - total:,} exact duplicate rows removed).",
            icon="ℹ️",
        )

        stat1, stat2, stat3 = st.columns(3)
        with stat1:
            st.metric("Unique Messages (model dataset)", f"{total:,}")
        with stat2:
            st.metric("Ham (Not Spam)", f"{ham_count:,}", f"{ham_count/total:.1%}")
        with stat3:
            st.metric("Spam", f"{spam_count:,}", f"{spam_count/total:.1%}")

        st.markdown("---")
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown("#### Class Distribution")
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.pie(
                [ham_count, spam_count],
                labels=["Ham", "Spam"],
                colors=["#00b894", "#ff6b6b"],
                autopct="%1.1f%%",
                startangle=140,
                wedgeprops={"edgecolor": "white", "linewidth": 1.5},
            )
            ax.set_title("Ham vs Spam", fontsize=13, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with chart_col2:
            st.markdown("#### Message Length Distribution")
            df_clean = df_clean.copy()
            df_clean["msg_len"] = df_clean["message"].str.len()
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            for label_val, color, name in [("ham", "#00b894", "Ham"), ("spam", "#ff6b6b", "Spam")]:
                subset = df_clean[df_clean["label"] == label_val]["msg_len"]
                ax2.hist(subset, bins=40, alpha=0.6, color=color, label=name, density=True)
            ax2.set_xlabel("Message Length (characters)")
            ax2.set_ylabel("Density")
            ax2.set_title("Message Length by Class", fontsize=12, fontweight="bold")
            ax2.legend()
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

        st.markdown("---")
        st.subheader("Sample Messages")
        filter_label = st.selectbox("Filter by label", ["All", "Ham", "Spam"])
        n_show = st.slider("Number of messages to display", 5, 50, 10)

        display_df = df_clean.copy()
        if filter_label == "Ham":
            display_df = display_df[display_df["label"] == "ham"]
        elif filter_label == "Spam":
            display_df = display_df[display_df["label"] == "spam"]

        st.dataframe(
            display_df[["label", "message"]].sample(min(n_show, len(display_df)), random_state=42),
            use_container_width=True,
            hide_index=True,
        )

        with st.expander("📖 Dataset source & citation"):
            st.markdown(f"""
**SMS Spam Collection Dataset**
- **Source:** UCI Machine Learning Repository
- **URL:** https://archive.ics.uci.edu/dataset/228/sms+spam+collection
- **Creators:** Tiago A. Almeida and José María Gómez Hidalgo
- **Published size:** 5,574 SMS messages (4,827 ham, 747 spam)
- **Rows in file:** {raw_total:,} (the distributed file contains {raw_total:,} rows; \
{5574 - raw_total:+,} vs. the published count reflects minor version differences in the archive)
- **After deduplication:** {total:,} unique messages ({raw_total - total:,} exact duplicates removed)
- **Model trained & evaluated on:** {total:,} messages (80/20 stratified split)
- **License:** Creative Commons Attribution 4.0 International
            """)

# app.py
import os
import streamlit as st
import pandas as pd
import joblib

# ─── CACHING ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts(models_dir='saved_models'):
    """Load transformers, selector, and models into memory once."""
    # transformers
    pt       = joblib.load(os.path.join(models_dir, 'pt.pkl'))
    rs       = joblib.load(os.path.join(models_dir, 'rs.pkl'))
    ss       = joblib.load(os.path.join(models_dir, 'ss.pkl'))
    selector = joblib.load(os.path.join(models_dir, 'selector.pkl'))

    # classification models
    models = {}
    for fn in os.listdir(models_dir):
        if fn.endswith('_model.pkl'):
            name = fn.replace('_model.pkl','')
            models[name] = joblib.load(os.path.join(models_dir, fn))

    return pt, rs, ss, selector, models

def preprocess_and_select(df, pt, rs, ss, selector):
    """Drop unused cols, apply transforms, then RFE select."""
    df = df.drop(['ID', 'ZIP Code', 'Personal Loan'], axis=1, errors='ignore')
    rb_cols  = ['CCAvg', 'Mortgage']
    std_cols = ['Income', 'Experience', 'Age']

    # Transform
    df[rb_cols]  = pt.transform(df[rb_cols])
    df[rb_cols]  = rs.transform(df[rb_cols])
    df[std_cols] = ss.transform(df[std_cols])

    # RFE select
    X_sel = selector.transform(df)
    sel_cols = [f'F{i}' for i in range(X_sel.shape[1])]
    return pd.DataFrame(X_sel, columns=sel_cols)

# ─── STREAMLIT LAYOUT ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bank Loan Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏦 Bank Loan Predictor")

st.markdown(
    """
    Upload a **CSV** file (without the `Personal Loan` target column) and get  
    predictions from six classifiers.
    """
)

# Load artifacts
pt, rs, ss, selector, models = load_artifacts()

# File uploader
uploaded = st.file_uploader(
    "1️⃣ Choose a CSV file",
    type=['csv'],
    help="Your CSV must contain the features: Income, Age, Experience, CCAvg, Mortgage."
)

if uploaded:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"❌ Could not read CSV: {e}")
    else:
        st.write("### Preview of uploaded data")
        st.dataframe(df.head(), use_container_width=True)

        with st.spinner("🔄 Preprocessing & predicting..."):
            X_sel = preprocess_and_select(df.copy(), pt, rs, ss, selector)

            for name, model in models.items():
                df[f'Pred_{name}'] = model.predict(X_sel)

        st.success("✅ Predictions complete!")

        st.write("### Results")
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download results as CSV",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )
else:
    st.info("🔽 Please upload a CSV file to get started.")

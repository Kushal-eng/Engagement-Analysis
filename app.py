import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ───────────────────────────────────────────────
# Page config
# ───────────────────────────────────────────────
st.set_page_config(
    page_title="Campaign Engagement Prediction",
    page_icon="📈",
    layout="wide"
)

# ───────────────────────────────────────────────
# Load & prepare data
# ───────────────────────────────────────────────
@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv("campaign_engagement_synthetic_dataset.csv")
    
    # Label encode target
    le_label = LabelEncoder()
    df["engagement_label_encoded"] = le_label.fit_transform(df["engagement_label"])
    
    # Features to encode
    categorical_cols = ["content_type", "channel", "media_type", "day_of_week"]
    
    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[f"{col}_encoded"] = le.fit_transform(df[col])
        le_dict[col] = le
    
    # Features for modeling
    feature_cols = [
        "content_type_encoded", "channel_encoded", "media_type_encoded",
        "content_length_words", "posting_hour", "day_of_week_encoded",
        "past_engagement_rate"
    ]
    
    X = df[feature_cols]
    y = df["engagement_label_encoded"]
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Accuracy
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    return {
        "df": df,
        "model": model,
        "le_label": le_label,
        "le_dict": le_dict,
        "feature_cols": feature_cols,
        "accuracy": acc,
        "class_names": ["Low", "Medium", "High"]
    }


data = load_and_prepare_data()
df = data["df"]
model = data["model"]
le_label = data["le_label"]
le_dict = data["le_dict"]
accuracy = data["accuracy"]
class_names = data["class_names"]

# ───────────────────────────────────────────────
# HEADER
# ───────────────────────────────────────────────
st.title("📊 Marketing Campaign Engagement Prediction Dashboard")
st.markdown(
    "This dashboard predicts how well a marketing activity is likely to perform **before** it is launched."
)

st.divider()

# ───────────────────────────────────────────────
# DATASET OVERVIEW ─ two columns
# ───────────────────────────────────────────────
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("Dataset Snapshot")
    total = len(df)
    st.metric("Total campaign activities", f"{total:,}")

with col2:
    st.subheader("Engagement Distribution")
    cnt = df["engagement_label"].value_counts().reindex(["Low", "Medium", "High"])
    fig_pie = px.pie(
        values=cnt.values,
        names=cnt.index,
        color=cnt.index,
        color_discrete_map={"Low": "#ef4444", "Medium": "#f59e0b", "High": "#10b981"}
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    fig_pie.update_layout(showlegend=False, margin=dict(t=10, b=10))
    st.plotly_chart(fig_pie, use_container_width=True)

st.divider()

# ───────────────────────────────────────────────
# KEY INSIGHTS ─ four small charts
# ───────────────────────────────────────────────
st.subheader("Key Insights from Past Campaigns")

insight_cols = st.columns(4)

with insight_cols[0]:
    st.markdown("**Best performing content type**")
    order = df.groupby("content_type")["past_engagement_rate"].mean().sort_values(ascending=False).index
    fig1 = px.bar(
        df.groupby("content_type")["past_engagement_rate"].mean().reindex(order),
        text_auto=".2f",
        color=df.groupby("content_type")["past_engagement_rate"].mean().reindex(order).index
    )
    fig1.update_layout(xaxis_title=None, yaxis_title="Avg Engagement Rate", showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)

with insight_cols[1]:
    st.markdown("**Best performing channel**")
    order = df.groupby("channel")["past_engagement_rate"].mean().sort_values(ascending=False).index
    fig2 = px.bar(
        df.groupby("channel")["past_engagement_rate"].mean().reindex(order),
        text_auto=".2f",
        color=df.groupby("channel")["past_engagement_rate"].mean().reindex(order).index
    )
    fig2.update_layout(xaxis_title=None, yaxis_title=None, showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

with insight_cols[2]:
    st.markdown("**Media type performance**")
    order = df.groupby("media_type")["past_engagement_rate"].mean().sort_values(ascending=False).index
    fig3 = px.bar(
        df.groupby("media_type")["past_engagement_rate"].mean().reindex(order),
        text_auto=".2f",
        color=df.groupby("media_type")["past_engagement_rate"].mean().reindex(order).index
    )
    fig3.update_layout(xaxis_title=None, yaxis_title=None, showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)

with insight_cols[3]:
    st.markdown("**Best posting time windows**")
    df["hour_bin"] = pd.cut(df["posting_hour"], bins=[0,8,12,17,21,24], 
                            labels=["Night","Morning","Afternoon","Evening","Late Night"], include_lowest=True)
    order = df.groupby("hour_bin")["past_engagement_rate"].mean().sort_values(ascending=False).index
    fig4 = px.bar(
        df.groupby("hour_bin")["past_engagement_rate"].mean().reindex(order),
        text_auto=".2f",
        color=df.groupby("hour_bin")["past_engagement_rate"].mean().reindex(order).index
    )
    fig4.update_layout(xaxis_title=None, yaxis_title=None, showlegend=False)
    st.plotly_chart(fig4, use_container_width=True)

st.caption("Tip: **Video** content on **Instagram** posted in the **afternoon** tends to perform strongly in this dataset.")

st.divider()

# ───────────────────────────────────────────────
# MODEL INFO
# ───────────────────────────────────────────────
st.subheader("Predictive Model")
st.markdown(
    f"We trained a model that learns patterns from **{len(df)} past campaigns** "
    f"and can now predict engagement with **{accuracy:.1%} accuracy** on unseen data."
)
st.info("The model predicts whether a planned activity will likely get **Low**, **Medium**, or **High** engagement.")

st.divider()

# ───────────────────────────────────────────────
# PREDICTION SIMULATOR
# ───────────────────────────────────────────────
st.subheader("Try It Yourself – Predict Engagement")

pred_cols = st.columns([2, 3])

with pred_cols[0]:
    with st.form("prediction_form", clear_on_submit=False):
        content_type = st.selectbox("Content Type", sorted(df["content_type"].unique()))
        channel       = st.selectbox("Channel", sorted(df["channel"].unique()))
        media_type    = st.selectbox("Media Type", sorted(df["media_type"].unique()))
        length_words  = st.slider("Content Length (words)", 50, 500, 250, step=10)
        hour          = st.slider("Posting Hour (0–23)", 0, 23, 14)
        day           = st.selectbox("Day of Week", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
        past_rate     = st.slider("Past Engagement Rate (%)", 0.0, 7.0, 3.0, step=0.1)

        predict_button = st.form_submit_button("Predict Engagement", type="primary", use_container_width=True)

if predict_button:
    # Encode inputs
    try:
        ct_enc   = le_dict["content_type"].transform([content_type])[0]
        ch_enc   = le_dict["channel"].transform([channel])[0]
        mt_enc   = le_dict["media_type"].transform([media_type])[0]
        dw_enc   = le_dict["day_of_week"].transform([day])[0]

        input_vector = np.array([[
            ct_enc, ch_enc, mt_enc,
            length_words, hour, dw_enc,
            past_rate
        ]])

        proba = model.predict_proba(input_vector)[0]
        pred_class_idx = np.argmax(proba)
        pred_label = class_names[pred_class_idx]
        confidence = proba[pred_class_idx]

        color_map = {"Low": "🔴", "Medium": "🟡", "High": "🟢"}

        st.success(f"**Predicted Engagement:  {pred_label}**  {color_map[pred_label]}")
        st.metric("Confidence", f"{confidence:.0%}")

        # Simple advice
        if pred_label == "High":
            st.markdown("**Recommendation:** This activity looks very promising — **launch as planned**.")
        elif pred_label == "Medium":
            st.markdown("**Recommendation:** Acceptable performance expected. You may still **launch**, but consider small improvements (e.g. shorter text, better visuals, different time).")
        else:
            st.warning("**Recommendation:** Low expected engagement. Consider changing **media type**, **channel**, **posting time**, or **content length** before launching.")

    except Exception as e:
        st.error("Sorry — could not make prediction with current inputs. Please check selections.")

st.divider()

# ───────────────────────────────────────────────
# MANAGERIAL VALUE
# ───────────────────────────────────────────────
with st.expander("How this dashboard helps your marketing team", expanded=True):
    st.markdown("""
    - **Plan smarter** — predict performance before spending time & budget  
    - **Prioritize** high-potential activities in Zoho Projects  
    - **Reduce trial-and-error** — focus on what historically works  
    - **Optimize timing & format** — choose better channels, media types & posting hours  
    - **Build confidence** — make data-informed decisions instead of guessing
    """)

st.caption("Marketing Campaign Engagement Prediction • Synthetic dataset • Random Forest model • Streamlit dashboard")

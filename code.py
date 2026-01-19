import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Marketing Campaign Engagement Prediction Dashboard",
    layout="wide"
)

# ================= LOAD DATA =================
df = pd.read_csv(df = pd.read_csv("C:\\Users\\kotra\\Downloads\\campaign_engagement_synthetic_dataset.csv")
                )

st.title("Marketing Campaign Engagement Prediction Dashboard")
st.write(
    "This dashboard predicts how well a marketing activity is likely to perform before it is launched."
)

# ================= DATASET OVERVIEW =================
st.header("Dataset Overview")

col1, col2 = st.columns(2)

with col1:
    st.metric("Total Campaign Activities", len(df))

with col2:
    engagement_counts = df["engagement_label"].value_counts()
    st.bar_chart(engagement_counts)
    st.write(
        "This chart shows how past campaigns are distributed across engagement levels."
    )

# ================= KEY INSIGHTS =================
st.header("Key Insights")

col3, col4 = st.columns(2)

with col3:
    content_perf = df.groupby("content_type")["past_engagement_rate"].mean()
    st.bar_chart(content_perf)
    st.write(
        "Some content types consistently generate higher engagement than others."
    )

with col4:
    channel_perf = df.groupby("channel")["past_engagement_rate"].mean()
    st.bar_chart(channel_perf)
    st.write(
        "Channels differ in how effectively they engage audiences."
    )

col5, col6 = st.columns(2)

with col5:
    media_perf = df.groupby("media_type")["past_engagement_rate"].mean()
    st.bar_chart(media_perf)
    st.write(
        "Media format has a clear impact on engagement levels."
    )

with col6:
    hour_perf = df.groupby("posting_hour")["past_engagement_rate"].mean()
    st.line_chart(hour_perf)
    st.write(
        "Engagement varies depending on the time content is posted."
    )

# ================= RULE-BASED PREDICTION ENGINE =================
st.header("Predictive Model")

st.write(
    "Instead of complex algorithms, this model uses patterns from past campaign performance "
    "to estimate future engagement in a clear and explainable way."
)

# Pre-calculate averages
avg_engagement = df["past_engagement_rate"].mean()
content_avg = df.groupby("content_type")["past_engagement_rate"].mean()
channel_avg = df.groupby("channel")["past_engagement_rate"].mean()
media_avg = df.groupby("media_type")["past_engagement_rate"].mean()

# ================= INTERACTIVE PREDICTION =================
st.header("Interactive Prediction Simulator")

col7, col8, col9 = st.columns(3)

with col7:
    input_content_type = st.selectbox("Content Type", df["content_type"].unique())
    input_channel = st.selectbox("Channel", df["channel"].unique())
    input_media_type = st.selectbox("Media Type", df["media_type"].unique())

with col8:
    input_content_length = st.slider("Content Length (Words)", 50, 2000, 300)
    input_posting_hour = st.slider("Posting Hour", 0, 23, 10)

with col9:
    input_day = st.selectbox("Day of Week", df["day_of_week"].unique())
    input_past_rate = st.slider("Past Engagement Rate", 0.0, 1.0, 0.3)

if st.button("Predict Engagement"):
    score = 0

    if input_past_rate > avg_engagement:
        score += 2
    else:
        score += 1

    if content_avg[input_content_type] > avg_engagement:
        score += 2

    if channel_avg[input_channel] > avg_engagement:
        score += 2

    if media_avg[input_media_type] > avg_engagement:
        score += 2

    if 9 <= input_posting_hour <= 12 or 18 <= input_posting_hour <= 21:
        score += 2
    else:
        score += 1

    # Final decision
    if score >= 9:
        prediction = "High"
        confidence = 0.85
    elif score >= 6:
        prediction = "Medium"
        confidence = 0.65
    else:
        prediction = "Low"
        confidence = 0.45

    st.subheader(f"Predicted Engagement Level: {prediction}")
    st.write(f"Prediction Confidence: {confidence * 100:.0f}%")

    st.write(
        f"Based on patterns from similar past campaigns, this activity is expected to receive "
        f"{prediction.upper()} engagement."
    )

    # ================= DECISION SUPPORT =================
    st.header("Decision Support")

    if prediction == "High":
        st.success("This content is suitable for immediate launch and high priority.")
    elif prediction == "Medium":
        st.warning("This content is acceptable, but performance could improve with timing or media changes.")
    else:
        st.error("This content is likely to underperform. Rework content or channel strategy.")

# ================= MANAGERIAL VALUE =================
st.header("Managerial Value")

st.write(
    "This dashboard enables marketing managers to evaluate campaign ideas before execution."
)
st.write(
    "It supports smarter planning and task prioritization inside Zoho Projects."
)
st.write(
    "By reducing trial-and-error, it saves time, budget, and managerial effort."
)

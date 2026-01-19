```python
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Marketing Campaign Engagement Prediction Dashboard", layout="wide")

df = pd.read_csv("campaign_engagement_synthetic_dataset.csv")

st.title("Marketing Campaign Engagement Prediction Dashboard")
st.write("This dashboard predicts how well a marketing activity is likely to perform before it is launched.")

st.header("Dataset Overview")

col1, col2 = st.columns(2)

with col1:
    st.metric("Total Campaign Activities", len(df))

with col2:
    engagement_counts = df["engagement_label"].value_counts()
    fig, ax = plt.subplots()
    engagement_counts.plot(kind="bar", ax=ax)
    ax.set_xlabel("Engagement Level")
    ax.set_ylabel("Number of Activities")
    st.pyplot(fig)
    st.write("This chart shows how past campaigns are distributed across engagement levels.")

st.header("Key Insights")

col1, col2 = st.columns(2)

with col1:
    ct = df.groupby("content_type")["past_engagement_rate"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots()
    ct.plot(kind="bar", ax=ax)
    ax.set_xlabel("Content Type")
    ax.set_ylabel("Average Engagement Rate")
    st.pyplot(fig)
    st.write("Some content types consistently generate higher engagement than others.")

with col2:
    ch = df.groupby("channel")["past_engagement_rate"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots()
    ch.plot(kind="bar", ax=ax)
    ax.set_xlabel("Channel")
    ax.set_ylabel("Average Engagement Rate")
    st.pyplot(fig)
    st.write("Channels differ in how effectively they engage audiences.")

col3, col4 = st.columns(2)

with col3:
    mt = df.groupby("media_type")["past_engagement_rate"].mean()
    fig, ax = plt.subplots()
    mt.plot(kind="bar", ax=ax)
    ax.set_xlabel("Media Type")
    ax.set_ylabel("Average Engagement Rate")
    st.pyplot(fig)
    st.write("Media format has a clear impact on engagement levels.")

with col4:
    hour_group = df.groupby("posting_hour")["past_engagement_rate"].mean()
    fig, ax = plt.subplots()
    hour_group.plot(ax=ax)
    ax.set_xlabel("Posting Hour")
    ax.set_ylabel("Average Engagement Rate")
    st.pyplot(fig)
    st.write("Engagement varies depending on the time content is posted.")

st.header("Predictive Model")

features = [
    "content_type",
    "channel",
    "media_type",
    "content_length_words",
    "posting_hour",
    "day_of_week",
    "past_engagement_rate",
]

X = df[features]
y = df["engagement_label"]

categorical_features = ["content_type", "channel", "media_type", "day_of_week"]
numerical_features = [
    "content_length_words",
    "posting_hour",
    "past_engagement_rate",
]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numerical_features),
    ]
)

model = RandomForestClassifier(n_estimators=200, random_state=42)

pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.write(f"Model Accuracy: {accuracy * 100:.2f}%")
st.write("The model learns from past campaign performance to predict future engagement.")

st.header("Interactive Prediction Simulator")

col1, col2, col3 = st.columns(3)

with col1:
    input_content_type = st.selectbox("Content Type", df["content_type"].unique())
    input_channel = st.selectbox("Channel", df["channel"].unique())
    input_media_type = st.selectbox("Media Type", df["media_type"].unique())

with col2:
    input_content_length = st.slider("Content Length (Words)", 50, 2000, 300)
    input_posting_hour = st.slider("Posting Hour", 0, 23, 10)

with col3:
    input_day = st.selectbox("Day of Week", df["day_of_week"].unique())
    input_past_rate = st.slider("Past Engagement Rate", 0.0, 1.0, 0.3)

if st.button("Predict Engagement"):
    input_df = pd.DataFrame(
        [
            {
                "content_type": input_content_type,
                "channel": input_channel,
                "media_type": input_media_type,
                "content_length_words": input_content_length,
                "posting_hour": input_posting_hour,
                "day_of_week": input_day,
                "past_engagement_rate": input_past_rate,
            }
        ]
    )

    prediction = pipeline.predict(input_df)[0]
    probabilities = pipeline.predict_proba(input_df)[0]
    confidence = np.max(probabilities)

    st.subheader(f"Predicted Engagement Level: {prediction}")
    st.write(f"Prediction Confidence: {confidence * 100:.1f}%")
    st.write(
        f"Based on similar past campaigns, this activity is expected to receive {prediction.upper()} engagement."
    )

    st.header("Decision Support")

    if prediction == "High":
        st.success("This content is suitable for immediate launch and prioritization.")
    elif prediction == "Medium":
        st.warning(
            "This content may perform reasonably well. Consider optimizing media type or posting time."
        )
    else:
        st.error(
            "This content is likely to underperform. Rework the content strategy before launching."
        )

st.header("Managerial Value")

st.write(
    "This dashboard helps marketing managers plan campaigns more effectively by predicting engagement before execution."
)
st.write(
    "It supports task prioritization in Zoho Projects by highlighting which activities deserve immediate focus."
)
st.write(
    "By using data-driven predictions, teams can reduce trial-and-error and allocate time and budgets more efficiently."
)
```

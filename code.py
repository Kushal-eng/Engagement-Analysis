import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(
page_title="Marketing Campaign Engagement Prediction Dashboard",
layout="wide"
)

df = pd.read_csv("campaign_engagement_synthetic_dataset.csv")

st.title("Marketing Campaign Engagement Prediction Dashboard")
st.write(
"This dashboard predicts how well a marketing activity is likely to perform before it is launched."
)

st.header("Dataset Overview")

col1, col2 = st.columns(2)

with col1:
st.metric("Total Campaign Activities", len(df))

with col2:
engagement_counts = df["engagement_label"].value_counts().reset_index()
engagement_counts.columns = ["Engagement Level", "Count"]
fig = px.bar(
engagement_counts,
x="Engagement Level",
y="Count",
title="Engagement Distribution"
)
st.plotly_chart(fig, use_container_width=True)
st.write(
"This chart shows how past campaigns are distributed across engagement levels."
)

st.header("Key Insights")

col1, col2 = st.columns(2)

with col1:
ct = (
df.groupby("content_type")["past_engagement_rate"]
.mean()
.reset_index()
.sort_values(by="past_engagement_rate", ascending=False)
)
fig = px.bar(
ct,
x="content_type",
y="past_engagement_rate",
title="Average Engagement by Content Type"
)
st.plotly_chart(fig, use_container_width=True)
st.write(
"Some content types consistently generate higher engagement than others."
)

with col2:
ch = (
df.groupby("channel")["past_engagement_rate"]
.mean()
.reset_index()
.sort_values(by="past_engagement_rate", ascending=False)
)
fig = px.bar(
ch,
x="channel",
y="past_engagement_rate",
title="Average Engagement by Channel"
)
st.plotly_chart(fig, use_container_width=True)
st.write(
"Channels differ in how effectively they engage audiences."
)

col3, col4 = st.columns(2)

with col3:
mt = (
df.groupby("media_type")["past_engagement_rate"]
.mean()
.reset_index()
)
fig = px.bar(
mt,
x="media_type",
y="past_engagement_rate",
title="Impact of Media Type on Engagement"
)
st.plotly_chart(fig, use_container_width=True)
st.write(
"Media format has a clear impact on engagement levels."
)

with col4:
hour_group = (
df.groupby("posting_hour")["past_engagement_rate"]
.mean()
.reset_index()
)
fig = px.line(
hour_group,
x="posting_hour",
y="past_engagement_rate",
title="Engagement by Posting Hour"
)
st.plotly_chart(fig, use_container_width=True)
st.write(
"Engagement varies depending on the time content is posted."
)

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

categorical_features = [
"content_type",
"channel",
"media_type",
"day_of_week",
]

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

model = RandomForestClassifier(
n_estimators=200,
random_state=42
)

pipeline = Pipeline(
steps=[
("preprocessor", preprocessor),
("model", model),
]
)

X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.2,
random_state=42,
)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.write(f"Model Accuracy: {accuracy * 100:.2f}%")
st.write(
"The model learns from past campaign performance to predict future engagement."
)

st.header("Interactive Prediction Simulator")

col1, col2, col3 = st.columns(3)

with col1:
input_content_type = st.selectbox(
"Content Type",
df["content_type"].unique()
)
input_channel = st.selectbox(
"Channel",
df["channel"].unique()
)
input_media_type = st.selectbox(
"Media Type",
df["media_type"].unique()
)

with col2:
input_content_length = st.slider(
"Content Length (Words)",
50,
2000,
300
)
input_posting_hour = st.slider(
"Posting Hour",
0,
23,
10
)

with col3:
input_day = st.selectbox(
"Day of Week",
df["day_of_week"].unique()
)
input_past_rate = st.slider(
"Past Engagement Rate",
0.0,
1.0,
0.3
)

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

import streamlit as st
import pandas as pd

from utils.sentiment_utils import compute_sentiment
from utils.data_utils import load_restaurant_data, get_restaurant_options
from utils.analysis_utils import (
    compute_all_metrics,
    get_overview,
    get_competitor_view,
    get_sentiment_view,
    get_delivery_view,
    get_price_view,
    get_menu_popularity_view,
)
from agent_code import build_context, run_agent


# --------------------------------------------------
# Streamlit config
# --------------------------------------------------
st.set_page_config(
    page_title="AI Agent — Restaurant Market Analysis",
    layout="wide",
)

st.title("AI Agent — Restaurant Market Analysis")
st.caption(
    "Upload a restaurant reviews CSV to analyse performance, competition, delivery, pricing, "
    "and customer satisfaction — then ask an AI agent for insights."
)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.header("1. Upload data")
    uploaded_file = st.file_uploader(
        "Restaurant reviews CSV",
        type=["csv"],
        help="Required: name, city, cuisine, rating, review_text, review_date, lat, lon",
    )
    st.markdown("---")
    st.header("2. Filters")

if uploaded_file is None:
    st.info("⬆️ Upload a CSV to begin.")
    st.stop()


# --------------------------------------------------
# LOAD + GUARANTEE SENTIMENT (SINGLE SOURCE)
# --------------------------------------------------
@st.cache_data(show_spinner=True)
def load_and_prepare(upload):
    df = load_restaurant_data(upload)

    if "sentiment_score" not in df.columns:
        df = df.copy()
        df["sentiment_score"] = df["review_text"].astype(str).apply(compute_sentiment)

    return df


df = load_and_prepare(uploaded_file)

if df.empty:
    st.error("Loaded CSV is empty.")
    st.stop()


# --------------------------------------------------
# COMPUTE METRICS (AFTER SENTIMENT EXISTS)
# --------------------------------------------------
@st.cache_data(show_spinner=True)
def compute_metrics(df):
    return compute_all_metrics(df)


metrics = compute_metrics(df)


# --------------------------------------------------
# Sidebar filters
# --------------------------------------------------
with st.sidebar:
    name_options, city_options, cuisine_options = get_restaurant_options(df)

    selected_city = st.selectbox("City", ["All"] + city_options)
    selected_cuisine = st.selectbox("Cuisine", ["All"] + cuisine_options)

    filtered_names = []
    for n in name_options:
        row = metrics["restaurants"].loc[n]
        if (selected_city in ("All", row["city"])) and (
            selected_cuisine in ("All", row["cuisine"])
        ):
            filtered_names.append(n)

    if not filtered_names:
        st.error("No restaurants match filters.")
        st.stop()

    selected_name = st.selectbox("Restaurant", filtered_names)


# --------------------------------------------------
# Tabs
# --------------------------------------------------
tabs = st.tabs([
    "Overview",
    "Competitors",
    "Sentiment",
    "Delivery",
    "Price",
    "Menu",
    "AI Agent",
    "Raw Data"
])


# ===================== OVERVIEW =====================
with tabs[0]:
    overview = get_overview(selected_name, metrics)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Avg rating", f"{overview['avg_rating']:.2f}")
    c2.metric("Reviews", overview["review_count"])
    c3.metric("Positive %", f"{overview['positive_pct']:.1f}%")
    c4.metric("Satisfaction", f"{overview['satisfaction_score']:.2f}")
    c5.metric("Zone", overview["zone"])

    st.plotly_chart(overview["rating_chart"], use_container_width=True)


# ===================== COMPETITORS =====================
with tabs[1]:
    comp = get_competitor_view(selected_name, metrics)
    st.plotly_chart(comp["bar_chart"], use_container_width=True)


# ===================== SENTIMENT =====================
with tabs[2]:
    sent = get_sentiment_view(selected_name, metrics)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Positive", sent["counts"]["positive"])
    c2.metric("Neutral", sent["counts"]["neutral"])
    c3.metric("Negative", sent["counts"]["negative"])
    c4.metric("Avg score", f"{sent['avg_compound']:.3f}")

    st.plotly_chart(sent["chart"], use_container_width=True)

    # ---------- MOST NEGATIVE REVIEW ----------
    neg_df = df.loc[
        (df["name"] == selected_name) &
        df["sentiment_score"].notna() &
        (df["sentiment_score"] < 0)
    ]

    if not neg_df.empty:
        worst = neg_df.loc[neg_df["sentiment_score"].idxmin()]
        st.markdown("### 🚨 Most Negative Review")
        st.metric("Sentiment score", f"{worst['sentiment_score']:.3f}")
        st.write(worst["review_text"])
    else:
        st.info("No negative reviews found.")

    # ---------- DOWNLOAD ALL NEGATIVE ----------
    all_neg = df.loc[
        df["sentiment_score"].notna() &
        (df["sentiment_score"] < 0)
    ].sort_values("sentiment_score")

    if not all_neg.empty:
        st.download_button(
            "⬇️ Download all negative reviews",
            all_neg.to_csv(index=False),
            "negative_reviews.csv",
            "text/csv"
        )


# ===================== DELIVERY =====================
with tabs[3]:
    st.plotly_chart(
        get_delivery_view(selected_name, metrics),
        use_container_width=True
    )


# ===================== PRICE =====================
with tabs[4]:
    st.plotly_chart(
        get_price_view(selected_name, metrics),
        use_container_width=True
    )


# ===================== MENU =====================
with tabs[5]:
    st.plotly_chart(
        get_menu_popularity_view(selected_name, metrics),
        use_container_width=True
    )


# ===================== AI AGENT =====================
with tabs[6]:
    question = st.text_area(
        "Ask the AI agent",
        placeholder="What are the biggest customer complaints?"
    )

    if st.button("Run AI"):
        ctx = build_context(selected_name, question, metrics)
        answer = run_agent(question, ctx)
        st.write(answer)


# ===================== RAW DATA =====================
with tabs[7]:
    st.dataframe(df.head(500), use_container_width=True)

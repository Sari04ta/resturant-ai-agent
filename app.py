
import streamlit as st
import pandas as pd

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

st.set_page_config(
    page_title="AI Agent — Restaurant Market Analysis",
    layout="wide",
)

st.title("AI Agent — Restaurant Market Analysis")
st.caption(
    "Upload a restaurant reviews CSV to analyse performance, competition, delivery and pricing, "
    "and customer satisfaction — then ask an AI agent for insights."
)

with st.sidebar:
    st.header("1. Upload data")
    uploaded_file = st.file_uploader(
        "Restaurant reviews CSV",
        type=["csv"],
        help=(
            "CSV must contain at least: name, city, cuisine, rating, review_text, review_date, "
            "lat, lon. Optional but recommended: zone, num_reviews, price_range, delivery_time, "
            "menu_item_popularity."
        ),
    )
    st.markdown("---")
    st.header("2. Filters (after upload)")

if uploaded_file is None:
    st.info("⬆️ Upload the CSV in the sidebar to begin.")
    st.stop()

@st.cache_data(show_spinner=True)
def _load(upload):
    return load_restaurant_data(upload)

df = _load(uploaded_file)

if df.empty:
    st.error("Loaded dataframe is empty. Please check your CSV.")
    st.stop()

@st.cache_data(show_spinner=True)
def _compute(_df: pd.DataFrame):
    return compute_all_metrics(_df)

metrics = _compute(df)

with st.sidebar:
    name_options, city_options, cuisine_options = get_restaurant_options(df)

    selected_city = st.selectbox("City", ["All"] + city_options)
    selected_cuisine = st.selectbox("Cuisine", ["All"] + cuisine_options)

    # Filter restaurants list based on city & cuisine
    filtered_names = []
    for n in name_options:
        row = metrics["restaurants"].loc[n]
        city_ok = selected_city in ("All", str(row["city"]))
        cuisine_ok = selected_cuisine in ("All", str(row["cuisine"]))
        if city_ok and cuisine_ok:
            filtered_names.append(n)

    if not filtered_names:
        st.error("No restaurants match the current filters.")
        st.stop()

    selected_name = st.selectbox("Restaurant", filtered_names)



tabs = st.tabs([
    "Overview",
    "Competitors",
    "Sentiment",
    "Delivery Insights",
    "Price Insights",
    "Menu Popularity",
    "Nearby Restaurants (10 km)",
    "AI Agent",
    "Raw Data"
])


# -------- Overview --------
with tabs[0]:
    st.subheader(f"Overview — {selected_name}")
    overview = get_overview(selected_name, metrics)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Avg rating", f"{overview['avg_rating']:.2f}")
    c2.metric("Review count", f"{overview['review_count']}")
    c3.metric("Positive reviews", f"{overview['positive_pct']:.1f}%")
    c4.metric("Satisfaction score", f"{overview['satisfaction_score']:.2f}")
    c5.metric("Zone", overview["zone"])

    st.markdown("### Rating vs competitors")
    st.plotly_chart(overview["rating_chart"], use_container_width=True)

    if overview.get("trend_chart") is not None:
        st.markdown("### Rating & sentiment trend over time")
        st.plotly_chart(overview["trend_chart"], use_container_width=True)

# -------- Competitors --------
with tabs[1]:
    st.subheader(f"Competitor landscape — {selected_name}")
    comp = get_competitor_view(selected_name, metrics)
    st.plotly_chart(comp["bar_chart"], use_container_width=True)
    if comp.get("map") is not None:
        st.markdown("### Heatmap of sentiment around competitors")
        st.pydeck_chart(comp["map"])

# -------- Sentiment --------
with tabs[2]:
    st.subheader(f"Customer sentiment — {selected_name}")
    sent_view = get_sentiment_view(selected_name, metrics)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Positive", sent_view["counts"]["positive"])
    c2.metric("Neutral", sent_view["counts"]["neutral"])
    c3.metric("Negative", sent_view["counts"]["negative"])
    c4.metric("Avg compound", f"{sent_view['avg_compound']:.3f}")

    st.plotly_chart(sent_view["chart"], use_container_width=True)
    if not sent_view["samples"].empty:
        st.markdown("### Sample reviews")
        st.dataframe(sent_view["samples"], use_container_width=True, hide_index=True)

# -------- Delivery --------
with tabs[3]:
    st.subheader(f"Delivery performance — {selected_name}")
    delv_chart = get_delivery_view(selected_name, metrics)
    st.plotly_chart(delv_chart, use_container_width=True)

# -------- Price --------
with tabs[4]:
    st.subheader(f"Price positioning — {selected_name}")
    price_chart = get_price_view(selected_name, metrics)
    st.plotly_chart(price_chart, use_container_width=True)

# -------- Menu popularity --------
with tabs[5]:
    st.subheader(f"Menu popularity — {selected_name}")
    menu_chart = get_menu_popularity_view(selected_name, metrics)
    st.plotly_chart(menu_chart, use_container_width=True)

# -------- AI Agent --------
with tabs[6]:
    st.subheader("Ask the AI agent")
    st.markdown(
        "The agent uses all computed metrics (ratings, sentiment, delivery, pricing, zone, "
        "competitor scores, and menu popularity) to answer business questions."
    )

    user_question = st.text_area(
        "Your question",
        placeholder=(
            "Example: How is this restaurant performing compared to nearby competitors? "
            "What should they improve to increase satisfaction and repeat visits?"
        ),
    )

    provider = st.radio("LLM provider", ["OpenAI", "HuggingFace"], horizontal=True)
    temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.3, 0.05)

    if st.button("Run analysis", type="primary"):
        if not user_question.strip():
            st.warning("Please enter a question first.")
        else:
            with st.spinner("Thinking..."):
                ctx = build_context(selected_name, user_question, metrics)
                answer = run_agent(
                    user_question,
                    ctx,
                    provider="openai" if provider == "OpenAI" else "huggingface",
                    temperature=temperature,
                )
            st.markdown("### Agent answer")
            st.write(answer)

# -------- Raw data --------
with tabs[7]:
    st.subheader("Raw dataframe (first 500 rows)")
    st.dataframe(df.head(500), use_container_width=True)

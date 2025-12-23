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
    "Raw Data",
    "Debug",
    "Seating Satisfaction"
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

    st.markdown("### Rating vs competitors")
    st.plotly_chart(overview["rating_chart"], use_container_width=True)

    if overview.get("trend_chart") is not None:
        st.markdown("### Rating & sentiment trend over time")
        st.plotly_chart(overview["trend_chart"], use_container_width=True)




# ===================== COMPETITORS =====================

with tabs[1]:
    st.subheader(f"Competitor landscape — {selected_name}")
    comp = get_competitor_view(selected_name, metrics)
    st.plotly_chart(comp["bar_chart"], use_container_width=True)
    if comp.get("map") is not None:
        st.markdown("### Heatmap of sentiment around competitors")
        st.pydeck_chart(comp["map"])


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



# -------------------------------
# Nearby Restaurants (Enhanced + Auto Column Detection)
# -------------------------------


with tabs[6]:
    st.subheader("Nearby Restaurants — Enhanced Radius Search")

    st.markdown("Select a restaurant to retrieve nearby competitors within a radius.")

    # ------------------------------------------
    # AUTO-DETECT IMPORTANT COLUMNS
    # ------------------------------------------
    COLUMN_MAP = {
        "restaurant": ["restaurant_name", "restaurant", "resturant", "name", "rname", "hotel_name"],
        "city": ["city", "location", "place", "area", "city_name"],
        "lat": ["lat", "latitude"],
        "lon": ["lon", "longitude", "lng"],
        "rating": ["rating", "ratings", "stars"]
    }

    def find_column(df, possible_names):
        """Find correct column name regardless of variations."""
        df_cols = [c.lower().strip() for c in df.columns]
        for name in possible_names:
            name = name.lower().strip()
            if name in df_cols:
                return df.columns[df_cols.index(name)]
        return None

    col_restaurant = find_column(df, COLUMN_MAP["restaurant"])
    col_city = find_column(df, COLUMN_MAP["city"])
    col_lat = find_column(df, COLUMN_MAP["lat"])
    col_lon = find_column(df, COLUMN_MAP["lon"])
    col_rating = find_column(df, COLUMN_MAP["rating"])

    missing = []
    if col_restaurant is None: missing.append("restaurant")
    if col_city is None: missing.append("city")
    if col_lat is None: missing.append("latitude")
    if col_lon is None: missing.append("longitude")

    if missing:
        st.error(f"❌ Missing required columns in CSV: {', '.join(missing)}")
        st.stop()

    # ------------------------------------------
    # Restaurant selection
    # ------------------------------------------
    restaurant_list = sorted(df[col_restaurant].dropna().unique().tolist())
    r_name = st.selectbox("Select Restaurant", restaurant_list)

    # ------------------------------------------
    # Auto-fill city
    # ------------------------------------------
    possible_cities = df[df[col_restaurant] == r_name][col_city].dropna().unique().tolist()
    r_city = st.selectbox("City", possible_cities)

    # ------------------------------------------
    # Radius selection
    # ------------------------------------------
    radius = st.slider("Search Radius (km)", 1, 20, 10)

    # ------------------------------------------
    # Perform search
    # ------------------------------------------
    if st.button("Find Nearby Restaurants"):
        from utils.analysis_utils import get_nearby_restaurants

        base, nearby = get_nearby_restaurants(
            df.rename(columns={
                col_restaurant: "restaurant_name",
                col_city: "city",
                col_lat: "lat",
                col_lon: "lon"
            }),
            r_name,
            r_city,
            radius_km=radius
        )

        if base is None:
            st.error("❌ Restaurant not found in this city.")
        else:
            st.success("✅ Restaurant found! Showing results...")

            st.markdown("### 📌 Selected Restaurant (Base Location)")
            st.dataframe(base, use_container_width=True)

            # ------------------------------------------
            # Compute Danger Score
            # ------------------------------------------
            if col_rating not in df.columns:
                nearby["rating"] = 0   # fallback if no ratings

            nearby["danger_score"] = (
                (radius - nearby["distance_km"].clip(0, radius)) *
                (nearby["rating"].fillna(0) / 5)
            ).round(2)

            # ------------------------------------------
            # Show Nearby List
            # ------------------------------------------
            st.markdown("### 📍 Restaurants within Radius")
            st.dataframe(nearby, use_container_width=True)

            # ------------------------------------------
            # Top 5 Nearest
            # ------------------------------------------
            st.markdown("### ⭐ Top 5 Nearest Restaurants")
            nearest = nearby[nearby["restaurant_name"] != r_name] \
                        .sort_values("distance_km") \
                        .head(5)

            st.dataframe(
                nearest[["restaurant_name", "city", "distance_km", "rating"]],
                use_container_width=True
            )

            # ------------------------------------------
            # Danger Ranking
            # ------------------------------------------
            st.markdown("### 🔥 Competitor Danger Ranking")
            st.dataframe(
                nearby.sort_values("danger_score", ascending=False)[
                    ["restaurant_name", "rating", "distance_km", "danger_score"]
                ],
                use_container_width=True
            )

            # ------------------------------------------
            # Downloads
            # ------------------------------------------
            st.download_button(
                "Download Base Restaurant CSV",
                base.to_csv(index=False),
                "selected_restaurant.csv",
                "text/csv"
            )

            st.download_button(
                "Download Nearby Competitors CSV",
                nearby.to_csv(index=False),
                "nearby_restaurants_enhanced.csv",
                "text/csv"
            )

            # ------------------------------------------
            # Map Visualization
            # ------------------------------------------
            st.markdown("### 🗺 Nearby Restaurants Map")

            import pydeck as pdk

            layer = pdk.Layer(
                "ScatterplotLayer",
                nearby,
                get_position='[lon, lat]',
                get_color='[200, 30, 0, 160]',
                get_radius=250,
                pickable=True,
            )

            view_state = pdk.ViewState(
                latitude=nearby["lat"].mean(),
                longitude=nearby["lon"].mean(),
                zoom=12,
                pitch=45,
            )

            deck = pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip={"text": "{restaurant_name}\nDistance: {distance_km} km"}
            )

            st.pydeck_chart(deck)


# ===================== AI AGENT =====================
with tabs[7]:
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



# ===================== RAW DATA =====================
with tabs[8]:
    st.subheader("Raw dataframe (first 500 rows)")
    st.dataframe(df.head(500), use_container_width=True)

# ===================== SEATING SATISFACTION =====================
with tabs[9]:
    st.subheader(f"Seating Experience — {selected_name}")

    overview = get_overview(selected_name, metrics)

    c1, c2 = st.columns(2)

    with c1:
        st.metric(
            "Seating Satisfaction",
            f"{overview['seating_score']}/100"
        )

    with c2:
        st.metric(
            "Estimated Wait Time",
            f"{overview['estimated_wait_time']} mins"
        )

    st.markdown("### 💡 Recommendation")
    st.info(overview["seating_recommendation"])



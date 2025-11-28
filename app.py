
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


# -------------------------------
# Nearby Restaurants (Enhanced)
# -------------------------------
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




# -------- AI Agent --------
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

# -------- Raw Data --------
with tabs[8]:
    st.subheader("Raw dataframe (first 500 rows)")
    st.dataframe(df.head(500), use_container_width=True)

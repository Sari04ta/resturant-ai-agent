
from __future__ import annotations

from typing import Dict, Any
import pandas as pd
import plotly.express as px
import pydeck as pdk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def _compute_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    ana = SentimentIntensityAnalyzer()
    scores = df["review_text"].astype(str).apply(ana.polarity_scores)
    scores_df = pd.DataFrame(list(scores))
    df = df.copy()
    df["compound"] = scores_df["compound"]

    def bucket(x: float) -> str:
        if x >= 0.05:
            return "positive"
        if x <= -0.05:
            return "negative"
        return "neutral"

    df["sentiment_bin"] = df["compound"].apply(bucket)
    return df


def compute_all_metrics(df_raw: pd.DataFrame) -> Dict[str, Any]:
    df = _compute_sentiment(df_raw)

    # Sentiment per restaurant
    grp = df.groupby("name")
    sentiment_stats = grp["sentiment_bin"].value_counts().unstack(fill_value=0)
    sentiment_stats["total"] = grp.size()
    sentiment_stats["positive_pct"] = (
        sentiment_stats.get("positive", 0) / sentiment_stats["total"] * 100
    )
    sentiment_stats["avg_compound"] = grp["compound"].mean()

    # Restaurant-level stats
    rest = grp.agg(
        city=("city", "first"),
        cuisine=("cuisine", "first"),
        zone=("zone", "first"),
        avg_rating=("rating", "mean"),
        review_count=("rating", "size"),
        avg_delivery=("delivery_time", "mean"),
        price_range=("price_range", "first"),
        avg_popularity=("menu_item_popularity", "mean"),
        avg_lat=("lat", "mean"),
        avg_lon=("lon", "mean"),
    )

    rest["positive_pct"] = sentiment_stats["positive_pct"]

    # Satisfaction score: rating (50%) + positive% (30%) + popularity (20%)
    rest["satisfaction_score"] = (
        0.5 * (rest["avg_rating"] / 5.0)
        + 0.3 * (rest["positive_pct"] / 100.0)
        + 0.2 * (rest["avg_popularity"].fillna(0) / 100.0)
    )

    rest["weighted_reviews"] = rest["review_count"] * (rest["avg_rating"] / 5.0)

    # Competitor map per restaurant (same city & cuisine)
    comp_map = {}
    for name, row in rest.iterrows():
        mask = (
            (rest["city"] == row["city"])
            & (rest["cuisine"] == row["cuisine"])
            & (rest.index != name)
        )
        comp_df = rest[mask].sort_values("satisfaction_score", ascending=False)
        comp_map[name] = comp_df

    metrics: Dict[str, Any] = {
        "df": df,
        "restaurants": rest,
        "sentiment": sentiment_stats,
        "competitors": comp_map,
    }
    return metrics


def get_overview(name: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    rest = metrics["restaurants"].loc[name]
    sent = metrics["sentiment"].loc[name]
    df = metrics["df"]
    sub = df[df["name"] == name]

    # Rating vs competitors
    comp = metrics["competitors"].get(name, pd.DataFrame())
    chart_df = pd.concat(
        [
            pd.DataFrame(
                {
                    "name": [name],
                    "avg_rating": [rest["avg_rating"]],
                    "satisfaction_score": [rest["satisfaction_score"]],
                    "type": ["selected"],
                }
            ),
            pd.DataFrame(
                {
                    "name": comp.index,
                    "avg_rating": comp["avg_rating"],
                    "satisfaction_score": comp["satisfaction_score"],
                    "type": "competitor",
                }
            ),
        ],
        ignore_index=True,
    )

    rating_chart = px.bar(
        chart_df,
        x="name",
        y="avg_rating",
        color="type",
        title="Average rating vs competitors",
    )

    trend_chart = None
    if sub["review_date"].notna().any():
        tmp = sub.dropna(subset=["review_date"]).copy()
        tmp["month"] = tmp["review_date"].dt.to_period("M").dt.to_timestamp()
        by_month = tmp.groupby("month").agg(
            avg_rating=("rating", "mean"),
            avg_compound=("compound", "mean"),
        )
        by_month = by_month.reset_index()
        trend_chart = px.line(
            by_month,
            x="month",
            y=["avg_rating", "avg_compound"],
            title="Rating & sentiment over time",
        )

    return {
        "avg_rating": float(rest["avg_rating"]),
        "review_count": int(rest["review_count"]),
        "positive_pct": float(sent["positive_pct"]),
        "satisfaction_score": float(rest["satisfaction_score"]),
        "zone": rest["zone"],
        "rating_chart": rating_chart,
        "trend_chart": trend_chart,
    }


def get_competitor_view(name: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    rest = metrics["restaurants"].loc[name]
    comp = metrics["competitors"].get(name, pd.DataFrame())

    if comp.empty:
        bar_chart = px.bar(
            x=[name],
            y=[rest["satisfaction_score"]],
            labels={"x": "Restaurant", "y": "Satisfaction score"},
            title="No direct competitors found; showing only selected restaurant.",
        )
        return {"bar_chart": bar_chart, "map": None}

    bar_df = comp.copy()
    bar_df = bar_df.reset_index()
    bar_df["type"] = "competitor"

    sel_row = pd.DataFrame(
        {
            "name": [name],
            "satisfaction_score": [rest["satisfaction_score"]],
            "type": ["selected"],
        }
    )

    bar_plot_df = pd.concat(
        [sel_row, bar_df[["name", "satisfaction_score", "type"]]], ignore_index=True
    )

    bar_chart = px.bar(
        bar_plot_df,
        x="name",
        y="satisfaction_score",
        color="type",
        title="Satisfaction score vs competitors",
    )

    # Optional heatmap if lat/lon available
    df = metrics["df"]
    cols = set(df.columns)
    deck_obj = None
    if {"lat", "lon", "compound"}.issubset(cols):
        subset_names = bar_plot_df["name"].tolist()
        coord_df = df[df["name"].isin(subset_names)].dropna(subset=["lat", "lon"])
        if not coord_df.empty:
            layer = pdk.Layer(
                "HeatmapLayer",
                data=coord_df,
                get_position="[lon, lat]",
                aggregation="MEAN",
                get_weight="compound",
                radiusPixels=40,
            )
            view_state = pdk.ViewState(
                latitude=float(coord_df["lat"].mean()),
                longitude=float(coord_df["lon"].mean()),
                zoom=11,
                pitch=40,
            )
            deck_obj = pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip={"text": "{name}\nSentiment: {compound}"},
            )

    return {"bar_chart": bar_chart, "map": deck_obj}


def get_sentiment_view(name: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    sent = metrics["sentiment"].loc[name]
    df = metrics["df"]
    sub = df[df["name"] == name]

    counts = {
        "positive": int(sent.get("positive", 0)),
        "neutral": int(sent.get("neutral", 0)),
        "negative": int(sent.get("negative", 0)),
    }
    avg_compound = float(sent["avg_compound"])

    stack_df = pd.DataFrame(
        {
            "sentiment": ["positive", "neutral", "negative"],
            "count": [counts["positive"], counts["neutral"], counts["negative"]],
        }
    )
    chart = px.bar(
        stack_df,
        x="sentiment",
        y="count",
        title="Sentiment distribution",
        text_auto=True,
    )

    samples = sub[
        ["review_date", "rating", "compound", "sentiment_bin", "review_text"]
    ].copy()
    samples = samples.sort_values("compound", ascending=False).head(50)

    return {
        "counts": counts,
        "avg_compound": avg_compound,
        "chart": chart,
        "samples": samples,
    }


def get_delivery_view(name: str, metrics: Dict[str, Any]):
    df = metrics["df"]
    rest = metrics["restaurants"].loc[name]
    sub = df[df["name"] == name].copy()

    if "delivery_time" not in df.columns or sub["delivery_time"].isna().all():
        return px.bar(
            x=[name],
            y=[0],
            labels={"x": "Restaurant", "y": "Delivery time (min)"},
            title="No delivery time data available.",
        )

    # Compare with competitors
    comp = metrics["competitors"].get(name, pd.DataFrame())
    comp_names = comp.index.tolist()

    plot_df = pd.concat(
        [
            sub.assign(type="selected"),
            df[df["name"].isin(comp_names)].assign(type="competitor"),
        ],
        ignore_index=True,
    )

    chart = px.box(
        plot_df,
        x="name",
        y="delivery_time",
        color="type",
        title="Delivery time distribution (selected vs competitors)",
    )
    return chart


def get_price_view(name: str, metrics: Dict[str, Any]):
    rest = metrics["restaurants"].loc[name]
    city = rest["city"]
    cuisine = rest["cuisine"]
    rest_price = rest["price_range"]

    all_rest = metrics["restaurants"]
    mask = (all_rest["city"] == city) & (all_rest["cuisine"] == cuisine)
    subset = all_rest[mask].copy()
    subset["is_selected"] = subset.index == name

    chart = px.histogram(
        subset,
        x="price_range",
        color="is_selected",
        barmode="group",
        title=f"Price range distribution in {city} ({cuisine})",
    )
    return chart


def get_menu_popularity_view(name: str, metrics: Dict[str, Any]):
    rest = metrics["restaurants"].loc[name]
    comp = metrics["competitors"].get(name, pd.DataFrame())

    plot_df = pd.DataFrame(
        {
            "name": [name],
            "avg_popularity": [rest["avg_popularity"]],
            "type": ["selected"],
        }
    )

    if not comp.empty:
        tmp = comp.copy().reset_index()
        tmp = tmp.rename(columns={"index": "name"})
        tmp["type"] = "competitor"
        tmp = tmp[["name", "avg_popularity", "type"]]
        plot_df = pd.concat([plot_df, tmp], ignore_index=True)

    chart = px.bar(
        plot_df,
        x="name",
        y="avg_popularity",
        color="type",
        title="Average menu item popularity (0–100)",
    )
    return chart

import math

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat/2)**2 
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon/2)**2
    )
    return 2 * R * math.asin(math.sqrt(a))


def get_nearby_restaurants(df, restaurant_name, city_name, radius_km=10):
    df = df.copy()

    df["restaurant_name"] = df["restaurant_name"].str.lower()
    df["city"] = df["city"].str.lower()

    restaurant_name = restaurant_name.lower()
    city_name = city_name.lower()

    base = df[
        (df["restaurant_name"] == restaurant_name) &
        (df["city"] == city_name)
    ]

    if base.empty:
        return None, None

    base_lat = base.iloc[0]["lat"]
    base_lon = base.iloc[0]["lon"]

    df["distance_km"] = df.apply(
        lambda row: haversine(base_lat, base_lon, row["lat"], row["lon"]),
        axis=1
    )

    nearby = df[df["distance_km"] <= radius_km]

    return base, nearby


from __future__ import annotations

from typing import Dict, Any
import math
import pandas as pd
import plotly.express as px
import pydeck as pdk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ==================================================
# Seating NLP keywords
# ==================================================
SEATING_KEYWORDS = {
    "positive": [
        "spacious", "comfortable seating", "ample seating",
        "nice ambience", "family seating", "comfortable chairs",
        "airy", "peaceful", "good seating"
    ],
    "negative": [
        "crowded", "no seating", "small space", "cramped",
        "long wait", "waiting time", "queue", "standing",
        "packed", "no place to sit"
    ]
}


# ==================================================
# Seating NLP extraction
# ==================================================
def extract_seating_signal(review: str) -> int:
    review = review.lower()
    score = 0

    for kw in SEATING_KEYWORDS["positive"]:
        if kw in review:
            score += 1

    for kw in SEATING_KEYWORDS["negative"]:
        if kw in review:
            score -= 1

    return score


def estimate_wait_time(df: pd.DataFrame, restaurant_name: str) -> int:
    sub = df[df["name"] == restaurant_name]

    wait_mentions = sub["review_text"].str.lower().str.contains(
        "wait|queue|standing|crowded", regex=True
    ).sum()

    return min(60, 5 + wait_mentions * 3)


def seating_recommendation(seating_score: float, wait_time: int) -> str:
    if seating_score >= 80 and wait_time <= 10:
        return "Ideal for families and groups"
    elif seating_score >= 60:
        return "Good seating, moderate waiting expected"
    elif wait_time > 30:
        return "Better for takeaway or off-peak hours"
    else:
        return "Suitable for quick visits"


# ==================================================
# Distance calculation (Haversine)
# ==================================================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return 2 * R * math.asin(math.sqrt(a))


# ==================================================
# Sentiment computation
# ==================================================
def _compute_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    analyzer = SentimentIntensityAnalyzer()
    scores = df["review_text"].astype(str).apply(analyzer.polarity_scores)
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


# ==================================================
# CORE METRICS PIPELINE
# ==================================================
def compute_all_metrics(df_raw: pd.DataFrame) -> Dict[str, Any]:
    df = _compute_sentiment(df_raw)

    # Seating NLP signal
    df["seating_signal"] = df["review_text"].astype(str).apply(extract_seating_signal)

    grp = df.groupby("name")

    # Sentiment stats
    sentiment_stats = grp["sentiment_bin"].value_counts().unstack(fill_value=0)
    sentiment_stats["total"] = grp.size()
    sentiment_stats["positive_pct"] = (
        sentiment_stats.get("positive", 0) / sentiment_stats["total"] * 100
    )
    sentiment_stats["avg_compound"] = grp["compound"].mean()

    # Restaurant-level aggregation
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
        seating_signal=("seating_signal", "mean"),
    )

    rest["positive_pct"] = sentiment_stats["positive_pct"]

    # Satisfaction score
    rest["satisfaction_score"] = (
        0.5 * (rest["avg_rating"] / 5.0)
        + 0.3 * (rest["positive_pct"] / 100.0)
        + 0.2 * (rest["avg_popularity"].fillna(0) / 100.0)
    )

    # Seating score (0–100)
    rest["seating_score"] = (
        rest["seating_signal"].clip(-3, 3) + 3
    ) / 6 * 100

    # Wait time + recommendation
    rest["estimated_wait_time"] = rest.index.map(
        lambda n: estimate_wait_time(df, n)
    )

    rest["seating_recommendation"] = rest.apply(
        lambda r: seating_recommendation(
            r["seating_score"],
            r["estimated_wait_time"]
        ),
        axis=1
    )

    # Competitors (same city + cuisine)
    competitors = {}
    for name, row in rest.iterrows():
        mask = (
            (rest["city"] == row["city"])
            & (rest["cuisine"] == row["cuisine"])
            & (rest.index != name)
        )
        competitors[name] = rest[mask]

    return {
        "df": df,
        "restaurants": rest,
        "sentiment": sentiment_stats,
        "competitors": competitors,
    }


# ==================================================
# VIEWS (USED BY app.py)
# ==================================================
def get_overview(name: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    rest = metrics["restaurants"].loc[name]
    sent = metrics["sentiment"].loc[name]
    df = metrics["df"]
    sub = df[df["name"] == name]

    comp = metrics["competitors"].get(name, pd.DataFrame())

    chart_df = pd.concat(
        [
            pd.DataFrame({
                "name": [name],
                "avg_rating": [rest["avg_rating"]],
                "type": ["selected"]
            }),
            pd.DataFrame({
                "name": comp.index,
                "avg_rating": comp["avg_rating"],
                "type": "competitor"
            })
        ],
        ignore_index=True
    )

    rating_chart = px.bar(
        chart_df,
        x="name",
        y="avg_rating",
        color="type",
        title="Average Rating vs Competitors"
    )

    trend_chart = None
    if sub["review_date"].notna().any():
        tmp = sub.dropna(subset=["review_date"]).copy()
        tmp["month"] = tmp["review_date"].dt.to_period("M").dt.to_timestamp()
        trend = tmp.groupby("month").agg(
            avg_rating=("rating", "mean"),
            avg_compound=("compound", "mean")
        ).reset_index()

        trend_chart = px.line(
            trend,
            x="month",
            y=["avg_rating", "avg_compound"],
            title="Rating & Sentiment Trend"
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
        bar = px.bar(
            x=[name],
            y=[rest["satisfaction_score"]],
            title="No competitors found"
        )
        return {"bar_chart": bar, "map": None}

    df_bar = comp.reset_index()[["name", "satisfaction_score"]]
    df_bar["type"] = "competitor"

    selected = pd.DataFrame({
        "name": [name],
        "satisfaction_score": [rest["satisfaction_score"]],
        "type": ["selected"]
    })

    plot_df = pd.concat([selected, df_bar], ignore_index=True)

    bar_chart = px.bar(
        plot_df,
        x="name",
        y="satisfaction_score",
        color="type",
        title="Satisfaction Score vs Competitors"
    )

    return {"bar_chart": bar_chart, "map": None}


def get_sentiment_view(name: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    sent = metrics["sentiment"].loc[name]
    df = metrics["df"]
    sub = df[df["name"] == name]

    counts = {
        "positive": int(sent.get("positive", 0)),
        "neutral": int(sent.get("neutral", 0)),
        "negative": int(sent.get("negative", 0)),
    }

    chart_df = pd.DataFrame({
        "sentiment": list(counts.keys()),
        "count": list(counts.values())
    })

    chart = px.bar(
        chart_df,
        x="sentiment",
        y="count",
        title="Sentiment Distribution"
    )

    return {
        "counts": counts,
        "avg_compound": float(sent["avg_compound"]),
        "chart": chart,
        "samples": sub.sort_values("compound").head(20),
    }


def get_delivery_view(name: str, metrics: Dict[str, Any]):
    df = metrics["df"]
    sub = df[df["name"] == name]

    if "delivery_time" not in df.columns or sub["delivery_time"].isna().all():
        return px.bar(x=[name], y=[0], title="No delivery data")

    return px.box(sub, y="delivery_time", title="Delivery Time Distribution")


def get_price_view(name: str, metrics: Dict[str, Any]):
    rest = metrics["restaurants"].loc[name]
    df = metrics["restaurants"]

    subset = df[
        (df["city"] == rest["city"]) &
        (df["cuisine"] == rest["cuisine"])
    ]

    return px.histogram(
        subset,
        x="price_range",
        title="Price Range Distribution"
    )


def get_menu_popularity_view(name: str, metrics: Dict[str, Any]):
    rest = metrics["restaurants"].loc[name]
    comp = metrics["competitors"].get(name, pd.DataFrame())

    plot_df = pd.concat([
        pd.DataFrame({
            "name": [name],
            "popularity": [rest["avg_popularity"]],
            "type": ["selected"]
        }),
        pd.DataFrame({
            "name": comp.index,
            "popularity": comp["avg_popularity"],
            "type": "competitor"
        })
    ], ignore_index=True)

    return px.bar(
        plot_df,
        x="name",
        y="popularity",
        color="type",
        title="Menu Item Popularity"
    )

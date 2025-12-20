from __future__ import annotations

import pandas as pd
from typing import List, Tuple
from utils.sentiment_utils import compute_sentiment

REQUIRED_COLS = [
    "name",
    "city",
    "cuisine",
    "rating",
    "review_text",
    "review_date",
    "lat",
    "lon",
]


def _normalise_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def load_restaurant_data(file) -> pd.DataFrame:
    """
    Load restaurant CSV, normalize schema, enforce types,
    and GUARANTEE sentiment_score exists.
    """
    df = pd.read_csv(file)
    df = _normalise_cols(df)

    # Validate required columns
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    # Normalize core fields
    df["name"] = df["name"].astype(str).str.strip()
    df["city"] = df["city"].astype(str).str.strip()
    df["cuisine"] = df["cuisine"].astype(str).str.strip()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["review_text"] = df["review_text"].astype(str)
    df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    # Optional fields
    df.setdefault("zone", "Unknown")
    df.setdefault("num_reviews", 0)
    df.setdefault("price_range", "Unknown")
    df.setdefault("delivery_time", None)
    df.setdefault("menu_item_popularity", None)

    # Drop invalid rows
    df.dropna(subset=["name", "city", "cuisine", "rating", "review_text"], inplace=True)

    # 🔒 HARD GUARANTEE sentiment_score
    if "sentiment_score" not in df.columns:
        df["sentiment_score"] = df["review_text"].apply(compute_sentiment)

    return df


def get_restaurant_options(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    names = sorted(df["name"].dropna().unique().tolist())
    cities = sorted(df["city"].dropna().unique().tolist())
    cuisines = sorted(df["cuisine"].dropna().unique().tolist())
    return names, cities, cuisines

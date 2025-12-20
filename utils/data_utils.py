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

def load_restaurant_data(file):
    df = pd.read_csv(file)
    df = _normalise_cols(df)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    df["name"] = df["name"].astype(str).str.strip()
    df["city"] = df["city"].astype(str).str.strip()
    df["cuisine"] = df["cuisine"].astype(str).str.strip()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["review_text"] = df["review_text"].astype(str)
    df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    # ✅ Correct defaults
    if "zone" not in df.columns:
        df["zone"] = "Unknown"
    if "num_reviews" not in df.columns:
        df["num_reviews"] = 0
    if "price_range" not in df.columns:
        df["price_range"] = "Unknown"
    if "delivery_time" not in df.columns:
        df["delivery_time"] = None
    if "menu_item_popularity" not in df.columns:
        df["menu_item_popularity"] = None

    df.dropna(subset=["name", "city", "cuisine", "rating", "review_text"], inplace=True)
    return df






def get_restaurant_options(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    names = sorted(df["name"].dropna().unique().tolist())
    cities = sorted(df["city"].dropna().unique().tolist())
    cuisines = sorted(df["cuisine"].dropna().unique().tolist())
    return names, cities, cuisines

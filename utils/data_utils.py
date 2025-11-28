
from __future__ import annotations

import pandas as pd
from typing import List, Tuple

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
    """Load a user CSV, normalise columns and basic types.

    Required columns (case-insensitive, spaces allowed):

    - name
    - city
    - cuisine
    - rating
    - review_text
    - review_date
    - lat
    - lon

    Optional (if missing, will be created with default values):

    - zone
    - num_reviews
    - price_range
    - delivery_time
    - menu_item_popularity
    """
    df = pd.read_csv(file)
    df = _normalise_cols(df)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
        )

    df["name"] = df["name"].astype(str).str.strip()
    df["city"] = df["city"].astype(str).str.strip()
    df["cuisine"] = df["cuisine"].astype(str).str.strip()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["review_text"] = df["review_text"].astype(str)
    df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    # Optional fields
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
    name_options = sorted(df["name"].unique().tolist())
    city_options = sorted([c for c in df["city"].dropna().unique().tolist() if c])
    cuisine_options = sorted([c for c in df["cuisine"].dropna().unique().tolist() if c])
    return name_options, city_options, cuisine_options

import os
import re
import numpy as np
import pandas as pd
import argparse

# --- Configuration ---
FILE_PATH = "../data/raw/US_Accidents_March23.csv"
OUT_DIR = "../data/processed"

# --- Regular Expressions for Road Classification ---
RE_INTERSTATE = re.compile(r"\bi-\s?\d+\b", re.IGNORECASE)
RE_US_ROUTE = re.compile(r"\bus\s?-?\s?\d+\b|\bus route\b", re.IGNORECASE)
RE_STATE_ROUTE_GENERIC = re.compile(
    r"\b(state route|route)\s?\d+\b", re.IGNORECASE
)
RE_STATE_CODE_ROUTE = re.compile(
    r"\b[a-z]{1,3}\s?-?\s?\d+\b", re.IGNORECASE
)

# --- Keywords for Road Classification ---
KEYWORD_MAP = {
    "High_Speed": [
        " freeway", " fwy", " expressway", " expy", " turnpike", " tpke",
        " thruway", " trwy", " beltway", " outerbelt", " innerbelt",
        " tollway", " motorway", " autobahn", " parkway connector",
        " connector", " overcrossing"
    ],
    "Medium_Speed": [
        " parkway", " pkwy", " boulevard", " blvd", " artery", " arterial",
        " pike", " hwy", " highway", " route", " rte", " cswy",
        " causeway", " skwy", " skyway"
    ],
    "Low_Speed": [
        "street", " st", "avenue", " ave", " road", " rd", " drive", " dr",
        "lane", " ln", " court", " ct", " way", " circle", " cir", " trail",
        " trl", " alley", " aly", " loop", " pass", " path", " walk", " run",
        " cv", " cove", " cres", " crescent", " bnd", " bend", " pt", " point"
    ],
    "Structure": [
        " bridge", " brg", " tunnel", " tunl", " crossing", " viaduct"
    ]
}


def map_speed_class(street: str) -> str:
    """Map street name to a coarse road speed class."""
    if pd.isna(street) or not str(street).strip():
        return "Other/Unknown"

    s = str(street).lower().strip()

    # Priority 1: Interstates (High Speed)
    if RE_INTERSTATE.search(s):
        return "High_Speed"

    # Priority 2: High Speed Keywords
    if any(k in s for k in KEYWORD_MAP["High_Speed"]):
        return "High_Speed"

    # Priority 3: Structures (Bridges/Tunnels)
    if any(k in s for k in KEYWORD_MAP["Structure"]):
        # Special logic: structures with high-speed keywords are High_Speed
        high_speed_structures = [
            "tpke", "turnpike", "freeway", "fwy"
        ]
        if any(k in s for k in high_speed_structures):
            return "High_Speed"
        return "Medium_Speed"

    # Priority 4: Defined Routes (Medium Speed)
    if RE_US_ROUTE.search(s) or RE_STATE_ROUTE_GENERIC.search(s):
        return "Medium_Speed"

    # State-coded routes with guard against street directions (e.g., "W 11th St")
    if RE_STATE_CODE_ROUTE.search(s):
        if not re.search(r"\b([nesw]\s?\d+)(st|nd|rd|th)\b", s):
            return "Medium_Speed"

    # Priority 5: Medium Speed Keywords
    if any(k in s for k in KEYWORD_MAP["Medium_Speed"]):
        return "Medium_Speed"

    # Priority 6: Low Speed Keywords
    if any(k in s for k in KEYWORD_MAP["Low_Speed"]):
        return "Low_Speed"

    return "Other/Unknown"


def collapse_severity(sev: int) -> str:
    """Collapse 4-level severity into Low/High."""
    return "Low" if sev in (1, 2) else "High"


def bucket_precipitation(s: pd.Series) -> pd.Categorical:
    """Categorize precipitation amounts."""
    bins = [-0.01, 0.0001, 0.1, 0.3, np.inf]
    labels = ["No Rain", "Low", "Moderate", "Heavy"]
    return pd.cut(s, bins=bins, labels=labels, include_lowest=True)


def bucket_visibility(s: pd.Series) -> pd.Categorical:
    """Categorize visibility distance."""
    bins = [-0.01, 1, 3, 6, np.inf]
    labels = ["Very Low", "Low", "Moderate", "Clear"]
    return pd.cut(s, bins=bins, labels=labels, include_lowest=True)


def group_weather(cond: str) -> str:
    """Group weather conditions into broader categories."""
    if pd.isna(cond):
        return "Unknown"
    cond = str(cond).lower()

    if "rain" in cond or "drizzle" in cond:
        return "Rain"
    if "snow" in cond or "sleet" in cond:
        return "Snow"
    if "fog" in cond or "mist" in cond:
        return "Fog"
    if "storm" in cond or "thunder" in cond:
        return "Storm"
    if "clear" in cond:
        return "Clear"
    if "cloud" in cond:
        return "Cloudy"
    return "Other"


def preprocess(post: bool=False, boston: bool=False):
    """Load, clean, and feature-engineer the dataset."""
    if post:
        year_range = (2019, 2023)
    else:
        year_range = (2016, 2018)
    
    OUT_FILE = os.path.join(OUT_DIR, f"modeling_dataset_{year_range[0]}_{year_range[1]}{'_Boston' if boston else ''}.csv")
    
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loading data from {FILE_PATH}...")
    df = pd.read_csv(FILE_PATH)

    # --- Deduplication ---
    # Drop duplicates, ignoring the unique identifier "ID"
    dup_cols = [c for c in df.columns if c != "ID"]
    df = df.drop_duplicates(subset=dup_cols)

    # --- Time Processing ---
    df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors="coerce")
    df = df.dropna(subset=["Start_Time"])

    df["Year"] = df["Start_Time"].dt.year
    df["Hour"] = df["Start_Time"].dt.hour
    df["Is_Weekend"] = df["Start_Time"].dt.dayofweek >= 5

    # Restrict to stable data period
    df = df[df["Year"].between(year_range[0], year_range[1])]

    # --- Data Cleaning ---
    # Physical plausibility filters
    numeric_cols_ranges = {
        "Temperature(F)": (-40, 120),
        "Visibility(mi)": (0, 10),
        "Precipitation(in)": (0, 25),
    }

    for col, (lo, hi) in numeric_cols_ranges.items():
        if col in df.columns:
            # Keep rows within range OR rows that are already NaN
            is_valid = df[col].isna() | df[col].between(lo, hi)
            df = df[is_valid]

    # Impute missing precipitation: no record -> 0.0 in
    df["Precipitation(in)"] = df["Precipitation(in)"].fillna(0.0)

    # --- Feature Engineering ---

    # Create CityState identifier for grouping/splitting
    df["CityState"] = (
        df["City"].fillna("Unknown").astype(str) +
        ", " +
        df["State"].fillna("NA").astype(str)
    )

    if boston:
        df = df[df["CityState"] == "Boston, MA"]

    # Derived speed class from Street name
    if "Street" in df.columns:
        df["Speed_Class"] = df["Street"].apply(map_speed_class).astype("object")
    else:
        df["Speed_Class"] = "Unknown"

    # Cyclical encoding for Hour
    df["Hour_Sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["Hour_Cos"] = np.cos(2 * np.pi * df["Hour"] / 24)

    # --- Target Preparation ---
    df["Severity2"] = df["Severity"].apply(collapse_severity)

    # --- Final Formatting and Export ---
    features = [
        "Severity2", "CityState", "Hour_Sin", "Hour_Cos", "Is_Weekend",
        "Speed_Class", "Precipitation(in)", "Visibility(mi)",
        "Temperature(F)", "Traffic_Signal"
    ]

    # Keep only columns that exist
    features = [c for c in features if c in df.columns]
    df = df[features]

    # Drop rows missing crucial information
    must_have = ["Severity2", "CityState", "Hour_Sin", "Hour_Cos"]
    df = df.dropna(subset=[c for c in must_have if c in df.columns])

    print(f"Saving processed data to {OUT_FILE}...")
    df.to_csv(OUT_FILE, index=False)
    print(f"Done. Rows: {len(df):,}, Columns: {df.shape[1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess traffic accident data.")

    # Add the --post flag
    parser.add_argument(
        '--post', 
        action='store_true', 
        help="Get data from 2019-2023"
    )

    # Add the --boston flag
    parser.add_argument(
        '--boston', 
        action='store_true', 
        help="Only include Boston data"
    )

    # Parse arguments
    args = parser.parse_args()
    
    preprocess(post=args.post, boston=args.boston)
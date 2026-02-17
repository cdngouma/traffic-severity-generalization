import os
import numpy as np
import pandas as pd


FILE_PATH = "../data/raw/US_Accidents_March23.csv"
OUT_DIR = "../data/processed"
OUT_FILE = os.path.join(OUT_DIR, "modeling_dataset_2016_2018.csv")


def collapse_severity(sev: int) -> str:
    return "Low" if sev in (1, 2) else "High"


def bucket_precipitation(s: pd.Series) -> pd.Categorical:
    bins = [-0.01, 0.0001, 0.1, 0.3, np.inf]
    labels = ["No Rain", "Low", "Moderate", "Heavy"]
    return pd.cut(s, bins=bins, labels=labels, include_lowest=True)


def bucket_visibility(s: pd.Series) -> pd.Categorical:
    # Keep NaN as NaN; we'll label it as "Unknown" later
    bins = [-0.01, 1, 3, 6, np.inf]
    labels = ["Very Low", "Low", "Moderate", "Clear"]
    return pd.cut(s, bins=bins, labels=labels, include_lowest=True)


def group_weather(cond) -> str:
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


def preprocess():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(FILE_PATH)

    df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors="coerce")
    df["Year"] = df["Start_Time"].dt.year
    df = df[df["Year"].between(2016, 2016)]

    # Drop duplicates (ignoring ID)
    dup_cols = [c for c in df.columns if c != "ID"]
    df = df.drop_duplicates(subset=dup_cols)

    # Parse time
    df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors="coerce")
    df = df.dropna(subset=["Start_Time"])

    # Time features
    df["Year"] = df["Start_Time"].dt.year
    df["Hour"] = df["Start_Time"].dt.hour
    df["Is_Weekend"] = df["Start_Time"].dt.dayofweek >= 5

    # Restrict stable period
    df = df[df["Year"].between(2016, 2018)]

    # Physical plausibility filters (keep NaNs; filter only observed values)
    numeric_cols_ranges = {
        "Temperature(F)": (-40, 120),
        "Wind_Chill(F)": (-60, 80),
        "Humidity(%)": (0, 100),
        "Pressure(in)": (28, 32),
        "Visibility(mi)": (0, 10),
        "Wind_Speed(mph)": (0, 60),
        "Precipitation(in)": (0, 25),
        "Distance(mi)": (0, 100),
    }

    for col, (lo, hi) in numeric_cols_ranges.items():
        if col in df.columns:
            ok = df[col].isna() | df[col].between(lo, hi)
            df = df[ok]

    # Missing handling
    df["Precipitation(in)"] = df["Precipitation(in)"].fillna(0.0)  # assumption: missing -> no rain

    # Buckets / groupings
    df["Rain_Bucket"] = bucket_precipitation(df["Precipitation(in)"])
    df["Visibility_Bucket"] = bucket_visibility(df["Visibility(mi)"]).astype("object")
    df["Visibility_Bucket"] = df["Visibility_Bucket"].fillna("Unknown")
    df["Weather_Group"] = df["Weather_Condition"].apply(group_weather)

    # Differentiate city name with State
    df["CityState"] = df["City"].astype(str) + ", " + df["State"].astype(str)

    # Target (keep original if you want)
    df["Severity2"] = df["Severity"].apply(collapse_severity)

    # Final feature set
    features = [
        "Severity2",
        "CityState",
        "Hour",
        "Is_Weekend",
        "Rain_Bucket",
        "Weather_Group",
        "Visibility_Bucket",
        "Traffic_Signal",
    ]
    df = df[features].dropna()

    df.to_csv(OUT_FILE, index=False)
    print(f"Saved: {OUT_FILE}  |  rows={len(df):,}  cols={df.shape[1]}")


if __name__ == "__main__":
    preprocess()

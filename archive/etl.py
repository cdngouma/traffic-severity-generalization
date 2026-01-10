import pandas as pd
import numpy as np
import re


def import_traffic_data(files):
    """
    Import and merge CSV files
    
    Input:
    - files (list): List of file paths.
    """
    similar_cities = ["Philadelphia, PA", "Minneapolis, MN", "Chicago, IL", "	Pittsburgh, PA", "Buffalo, NY", "Cleveland, OH", 
                      "Seattle, WA", "Detroit, MI", "Milwaukee, WI", "Rochester, NY", "Denver, CO", "Albany, NY", "Portland, ME"]
    df = None
    for file in files:
        df_ = pd.read_csv(file)
        # filter the data by only considering certain cities
        df_ = df_[(df_["City"] + ", " + df_["State"]).isin(similar_cities)]
        # Merge tables
        df = df_ if df is None else pd.concat(df, df_)
    return df


def is_highway(street_name):
    # Check if the street name contains common highway prefixes or suffixes
    highway_prefixes = ['I-', 'US-', 'SR-', 'HWY', 'INTERSTATE', 'US HIGHWAY', 'STATE ROUTE']
    highway_suffixes = ['INTERSTATE', 'HIGHWAY', 'EXPRESSWAY', 'TURNPIKE', 'PARKWAY', 'ROUTE']
    for prefix in highway_prefixes:
        if street_name.upper().startswith(prefix):
            return True
    for suffix in highway_suffixes:
        if street_name.upper().endswith(suffix):
            return True
    return False


def calc_heat_index(t, h):
    """
    The Heat Index is a measure of how hot it feels when relative humidity is factored in with the actual air temperature.
    
    Input:
    t: temperature
    h: relative humidity
    """
    c1 = 42.379
    c2 = 2.04901523
    c3 = 10.14333127
    c4 = 0.22475541
    c5 = 0.00683783
    c6 = 0.05481717
    c7 = 0.00122874
    c8 = 0.00085282
    c9 = 0.00000199
    heat_index = -c1 + c2*t + c3*h -c4*t*h - c5*t*t - c6*h*h + c7*t*t*h + c8*t*h*h - c9*t*t*h*h
    return heat_index


def calc_wind_chill_index(t, v):
    """
    The Wind Chill Index measures the perceived decrease in temperature felt by the body on exposed skin due to wind.
    
    Input:
    t: temperature
    v: wind speed
    """
    wci = 35.74 + 0.6215*t - 35.75*np.pow(v, 0.16) + 0.4275*t*np.pow(v, 0.16)
    return wci


def categorize_weather(condition):
    # Define the categories and their regex patterns
    weather_categories = {
        'Clear': r'(clear|fair|sunny)',
        'Cloudy': r'(overcast|mostly cloudy|partly cloudy|scattered clouds|cloudy)',
        'Reduced Visibility': r'(fog|mist|haze|shallow fog|patches of fog|partial fog|smoke)',
        'Rain': r'(rain|light rain|drizzle|light drizzle|heavy rain|showers)',
        'Extreme/Stormy Weather': r'(squalls|funnel cloud|thunderstorms and snow|tornado|thunder|thunderstorm|t-storm|storm)',
        'Winter Weather': r'(snow|blowing snow|snow shower|snow and sleet|freezing rain|sleet|ice pellets)',
    }
    
    # Match the condition against the regex patterns
    for category, pattern in weather_categories.items():
        if re.search(pattern, condition, re.IGNORECASE):
            return category
    return 'Other/Unknown'


def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    elif month in [9, 10, 11]:
        return "Fall"
    else:
        return "Invalid month"


def preprocess(df):
    """
    Preprocess data
    
    Input:
    - df (DataFrame): dataset
    """
    # Transformation steps:
    
    # 1-Drop redundant and irrelevant columns
    discarded_columns = ["Country", "Source", "End_Lat", "End_Lng", "Description", "Airport_Code", "Civil_Twilight", "Nautical_Twilight", 
                         "Astronomical_Twilight", "Wind_Direction", "Wind_Chill(F)", "Timezone", "Weather_Timestamp", "Zipcode"]
    df = df.drop(columns=discarded_columns)
    
    # 2-Drop duplicates
    df = df.drop(columns=["ID"]).drop_duplicates()
    
    # 3.1-Drop rows with missing values
    df = df.dropna(subset=["Street", "Temperature(F)", "Humidity(%)", "Pressure(in)", "Visibility(mi)", "Weather_Condition"])
    # 3.2-Fill missing values
    df["Wind_Speed(mph)"] = df["Wind_Speed(mph)"].fillna(0.0)
    df["Precipitation(in)"] = df["Precipitation(in)"].fillna(0.0)
    
    # 4-Create new columns
    # Fix datatype
    df["Start_Time"] = pd.to_datetime(df["Start_Time"])
    df["End_Time"] = pd.to_datetime(df["End_Time"])
    
    df["Is_Highway"] = df["Street"].apply(is_highway)
    df["Duration(min)"] = (df["End_Time"] - df["Start_Time"]).dt.total_seconds() / 60.0
    df["Date"] = pd.to_datetime(df["Start_Time"].dt.date)
    df["Hour"] = df["Start_Time"].dt.hour
    df["Day_of_Week"] = df["Start_Time"].dt.dayofweek + 1  # index for day of the week start at 0
    df["Is_Weekend"] = df["Day_of_Week"] >= 6
    df["Month"] = df["Start_Time"].dt.month
    df["Year"] = df["Start_Time"].dt.year
    df["Is_Night"] = df["Sunrise_Sunset"] == "Night"
    df["Weather_Condition"] = df["Weather_Condition"].apply(lambda condition: categorize_weather(condition))
    df["Is_Dry_Weather"] = df["Weather_Condition"] == "Dry Weather"
    df["Heat_Index"] = df.apply(lambda x: calc_heat_index(x["Temperature(F)"], x["Humidity(%)"]), axis=1)
    df["Season"] = df["Month"].apply(get_season)
    
    # 5-Fix inconsistencies
    adjusted_ranges = ((df["Distance(mi)"] <= 30) \
        & (df["Temperature(F)"] <= 115.35) \
        & (df["Pressure(in)"] >= 25.69) \
        & (df["Visibility(mi)"] <= 20) \
        & (df["Wind_Speed(mph)"] <= 40) \
        & (df["Duration(min)"] <= 1440))
    df = df[adjusted_ranges]
    
    # Select columns
    selected_columns = ['Distance(mi)', 'Temperature(F)', 'Wind_Speed(mph)', 'Precipitation(in)', 'Duration(min)', 'Hour', 'Month', 
                        'Crossing', 'Traffic_Signal', 'Is_Highway', 'Is_Weekend', 'Is_Night', 'Weather_Condition', 'Severity']
    # Only select column that are indeed in the dataset
    for col in selected_columns:
        if col not in df.columns.tolist():
            selected_columns.remove(col)
    
    df = df[selected_columns]
    
    return df
    

def load(files):
    """
    Perform ETL (extract, transform, load) steps
    
    Input:
    - files (list): List of file paths.
    """
    # Import and merge CSV files
    df = import_traffic_data(files)
    df = preprocess(df)
    
    return df
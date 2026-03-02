import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from datetime import datetime

df = pd.read_csv("AI_project_data.csv")

# Drop unwanted columns
df.drop(columns=["Mission_name", "fuel"], inplace=True)
if "Unnamed: 7" in df.columns:
    df.drop(columns=["Unnamed: 7"], inplace=True)

# Clean whitespace
df["landed_planet"] = df["landed_planet"].str.strip()
df["nation"] = df["nation"].str.strip()

# Convert dates and calculate duration
df["launch_date"] = pd.to_datetime(df["launch_date"])
df["landed_date"] = pd.to_datetime(df["landed_date"])
df["travel_time_days"] = (df["landed_date"] - df["launch_date"]).dt.days

# RECTIFICATION: Filter out negative travel times
df = df[df["travel_time_days"] > 0]

# Convert distance to numeric
df["distance"] = pd.to_numeric(df["distance"], errors="coerce")
df.dropna(subset=["distance"], inplace=True)


le_nation = LabelEncoder()
le_planet = LabelEncoder()

df["nation_encoded"] = le_nation.fit_transform(df["nation"])
df["planet_encoded"] = le_planet.fit_transform(df["landed_planet"])

X = df[["distance", "nation_encoded", "planet_encoded"]]
y = df["travel_time_days"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

closest_approaches = [
    {"planet": "Mars", "dates": ["2027-02-19", "2029-03-25", "2031-05-04", "2033-06-27", "2035-09-15"]},
    {"planet": "Venus", "dates": ["2026-10-24", "2028-06-01", "2030-01-06", "2031-08-14", "2033-03-20"]},
    {"planet": "Jupiter", "dates": ["2026-01-10", "2027-02-11", "2028-03-13", "2029-04-12", "2030-05-13"]},
    {"planet": "Mercury", "dates": ["2026-03-07", "2026-07-13", "2026-11-04", "2027-02-20", "2027-06-21"]}
]

def get_best_launch_date(planet, user_date_str):
    user_date = datetime.strptime(user_date_str, "%Y-%m-%d")
    for item in closest_approaches:
        if item["planet"].lower() == planet.lower():
            dates = [datetime.strptime(d, "%Y-%m-%d") for d in item["dates"]]
            future_dates = [d for d in dates if d >= user_date]
            return min(future_dates) if future_dates else None
    return None

def space_travel_ai(destination_planet, nation, user_launch_date):
    destination_planet = destination_planet.strip().lower()
    nation = nation.strip()

    planet_db = {
        "mercury": {"name": "Mercury", "distance": 5000},
        "venus": {"name": "Venus", "distance": 1500},
        "mars": {"name": "Mars", "distance": 3500},
        "jupiter": {"name": "Jupiter", "distance": 9000}
    }

    if destination_planet not in planet_db:
        return "Destination planet not supported."

    planet_name = planet_db[destination_planet]["name"]
    distance = planet_db[destination_planet]["distance"]

    best_launch = get_best_launch_date(planet_name, user_launch_date)
    if best_launch is None:
        return "No suitable launch window found."

    if nation not in le_nation.classes_:
        return f"Nation '{nation}' not in training data."

    nation_code = le_nation.transform([nation])[0]
    planet_code = le_planet.transform([planet_name])[0]

    # FIX: Use DataFrame for prediction to avoid the UserWarning
    input_df = pd.DataFrame([[distance, nation_code, planet_code]],
                            columns=["distance", "nation_encoded", "planet_encoded"])

    travel_time = model.predict(input_df)[0]

    return {
        "Destination": planet_name,
        "Origin Nation": nation,
        "Preferred Launch": user_launch_date,
        "AI Optimized Launch": best_launch.strftime("%Y-%m-%d"),
        "Predicted Travel Duration": f"{int(travel_time)} days"
    }




result = space_travel_ai(destination_planet="mars", nation="China", user_launch_date="2026-05-01")
print("AI MISSION PLAN:")
print(result)
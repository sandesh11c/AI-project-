# AI-project-

# 🚀 Interstellar Travel Duration Predictor

An AI-powered system that predicts space travel duration and identifies optimal launch windows based on celestial mechanics and historical mission data.

## 🛰️ Project Overview
This project uses a **Random Forest Regressor** to estimate the time required for spacecraft to reach various planets. It uniquely combines predictive modeling with a "Best Launch Window" algorithm that calculates the most efficient departure dates based on planetary alignment (Closest Approach).

### Key Features:
* **ML Engine:** Predicts travel time using distance, origin nation, and destination planet.
* **Launch Optimizer:** A logic-based system that suggests the next best launch date based on real-world celestial windows (2026–2035).
* **Data Cleaning:** Robust preprocessing including date-time conversion, whitespace stripping, and anomaly filtering (rectifying negative travel times).

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Libraries:** Pandas, NumPy, Scikit-Learn
* **Model:** Random Forest Regressor

## 📊 How It Works
1. **Preprocessing:** The script cleans mission data, calculates `travel_time_days`, and encodes categorical variables (Nation and Planet).
2. **Training:** A Random Forest model learns the relationship between planetary distance and historical travel durations.
3. **Inference:** When a user inputs a destination and a preferred date, the AI finds the nearest "closest approach" window and predicts the arrival time.



## 🚀 Usage
1. Clone the repo: `git clone https://github.com/sandesh11c/AI-project-.git`
2. Install requirements: `pip install -r requirements.txt`
3. Run the script: `python main.py`

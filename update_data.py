import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import json
import subprocess
import sys
from datetime import datetime
from dateutil.relativedelta import relativedelta

# === Configuration ===
CSV_FILE = 'Price Dataset.csv'   # Historic data only
JSON_FILE = 'data_combined.json' # Output JSON file
GIT_COMMIT_MSG = f"Monthly data update: {datetime.now().strftime('%Y-%m-%d')}"

def read_and_forecast(csv_path, json_path):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    # Ensure Date column exists and has no nulls
    df = df.dropna(subset=['Date'])
    try:
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    except Exception as e:
        print(f"Error parsing dates: {e}")
        sys.exit(1)

    # Prepare date lists
    historical_dates = df['Date'].dt.strftime('%Y-%m-%d').tolist()
    df_prices = df.drop(columns=['Date'])

    results = {}
    for city in df_prices.columns:
        series = df_prices[city]

        # Fit damped Holt-Winters model & forecast next 60 months
        model = ExponentialSmoothing(
            series,
            trend='add',
            damped_trend=True,
            seasonal=None
        )
        fit = model.fit(optimized=True)  # allow it to pick the best damping
        forecast = fit.forecast(60)

        full_vals = pd.concat([series, forecast]).tolist()
        results[city] = full_vals

    # Generate future dates at monthly intervals
    last_date = df['Date'].iloc[-1]
    future_dates = [
        (last_date + relativedelta(months=i)).strftime('%Y-%m-%d')
        for i in range(1, 61)
    ]

    # Combine dates + values
    combined = {'dates': historical_dates + future_dates, **results}

    # Write JSON
    try:
        with open(json_path, 'w') as f:
            json.dump(combined, f, indent=2)
        print(f"Successfully wrote JSON to {json_path}")
    except Exception as e:
        print(f"Error writing JSON: {e}")
        sys.exit(1)

def git_commit_and_push(file_path, commit_msg):
    try:
        subprocess.run(['git', 'add', file_path], check=True)
        subprocess.run(['git', 'commit', '-m', commit_msg], check=True)
        subprocess.run(['git', 'push'], check=True)
        print("Git commit and push successful.")
    except subprocess.CalledProcessError as e:
        print(f"Git command failed: {e}")
        sys.exit(1)

def main():
    read_and_forecast(CSV_FILE, JSON_FILE)
    git_commit_and_push(JSON_FILE, GIT_COMMIT_MSG)

if __name__ == '__main__':
    main()

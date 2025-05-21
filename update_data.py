import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import json
import subprocess
import sys
from datetime import datetime
from dateutil.relativedelta import relativedelta

# === Configuration ===
CSV_FILE = 'Price Dataset.csv'
JSON_FILE = 'data_combined.json'
GIT_COMMIT_MSG = f"Monthly data update: {datetime.now():%Y-%m-%d}"

def read_and_forecast(csv_path, json_path):
    try:
        df = pd.read_csv(csv_path)  # assumes comma-separated
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    if 'Date' not in df.columns:
        print("Error: 'Date' column not found in CSV.")
        sys.exit(1)

    df = df.dropna(subset=['Date'])
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    historical_dates = df['Date'].dt.strftime('%Y-%m-%d').tolist()
    df_prices = df.drop(columns=['Date'])

    results = {}

    for city in df_prices.columns:
        series = df_prices[city].astype(float).dropna()
        try:
            model = ExponentialSmoothing(series, trend='add', seasonal=None)
            fit = model.fit(optimized=True, use_brute=True)  # brute-force avoids convergence failure
            forecast = fit.forecast(60)
            full_series = pd.concat([series, forecast]).reset_index(drop=True)
            results[city] = full_series.tolist()
        except Exception as e:
            print(f"Error processing {city}: {e}")

    last_date = df['Date'].iloc[-1]
    future_dates = [
        (last_date + relativedelta(months=i)).strftime('%Y-%m-%d')
        for i in range(1, 61)
    ]

    combined = {
        'dates': historical_dates + future_dates,
        **results
    }

    try:
        with open(json_path, 'w') as f:
            json.dump(combined, f, indent=2)
        print(f"✅ Forecast data written to {json_path}")
    except Exception as e:
        print(f"Error writing JSON: {e}")
        sys.exit(1)

    try:
        subprocess.run(['git', 'add', '-A'], check=True)  # stage all modified files
        subprocess.run(['git', 'commit', '-m', GIT_COMMIT_MSG], check=True)
        subprocess.run(['git', 'push'], check=True)
        print("✅ Git commit and push complete.")
    except subprocess.CalledProcessError as e:
        print(f"Git error: {e}")
        sys.exit(1)

def main():
    read_and_forecast(CSV_FILE, JSON_FILE)

if __name__ == '__main__':
    main()
import pandas as pd
from statsmodels.tsa.holtwinters import Holt     # <— use Holt instead
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
    df = pd.read_csv(csv_path).dropna(subset=['Date'])
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    historical_dates = df['Date'].dt.strftime('%Y-%m-%d').tolist()
    df_prices = df.drop(columns=['Date'])

    results = {}
    for city in df_prices.columns:
        series = df_prices[city]

        # Holt’s linear trend with damping
        model = Holt(series, damped_trend=True)
        # Set damping_trend < 1 (e.g. 0.9 for gentle flattening; lower for more)
        fit = model.fit(damping_trend=0.9)
        forecast = fit.forecast(60)

        full = pd.concat([series, forecast])
        results[city] = full.tolist()

    last_date = df['Date'].iloc[-1]
    future_dates = [
        (last_date + relativedelta(months=i)).strftime('%Y-%m-%d')
        for i in range(1, 61)
    ]

    combined = {'dates': historical_dates + future_dates, **results}
    with open(json_path, 'w') as f:
        json.dump(combined, f, indent=2)
    print(f"Successfully wrote JSON to {json_path}")

    # commit & push as before
    subprocess.run(['git', 'add', json_path], check=True)
    subprocess.run(['git', 'commit', '-m', GIT_COMMIT_MSG], check=True)
    subprocess.run(['git', 'push'], check=True)

if __name__ == '__main__':
    read_and_forecast(CSV_FILE, JSON_FILE)

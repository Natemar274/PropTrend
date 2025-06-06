import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np
import json
import subprocess
import sys
from datetime import datetime
from dateutil.relativedelta import relativedelta

CASH_RATE_CSV = 'Cash Rate Target.csv'
CASH_RATE_JSON = 'cash_rate.json'

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
            # Apply log transform for more stable growth extrapolation
            log_series = np.log(series)

            # Fit damped trend exponential smoothing on log-transformed series
            model = ExponentialSmoothing(log_series, trend='add', damped_trend=True, seasonal=None)
            fit = model.fit(optimized=True, use_brute=True)
            forecast_log = fit.forecast(60)

            # Inverse transform
            forecast = np.exp(forecast_log)

            # Estimate historical CAGR
            n_years = len(series) / 12
            if n_years > 0 and series.iloc[0] > 0:
                cagr = (series.iloc[-1] / series.iloc[0]) ** (1 / n_years) - 1
            else:
                cagr = 0.0
            capped_growth = (1 + cagr) ** (np.arange(1, 61) / 12)
            max_vals = series.iloc[-1] * capped_growth
            forecast = np.minimum(forecast, max_vals)

            full_series = pd.concat([series, pd.Series(forecast)]).reset_index(drop=True).round(2)
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

    # === Cash Rate JSON Generation ===
    try:
        cash_df = pd.read_csv(CASH_RATE_CSV)
        cash_df.rename(columns={"Cash Rate Target (%)": "Cash Rate Target"}, inplace=True)
        cash_df["Date"] = pd.to_datetime(cash_df["Date"], dayfirst=True).dt.strftime("%Y-%m-%d")
        cash_data = cash_df.to_dict(orient="records")
        with open(CASH_RATE_JSON, "w") as f:
            json.dump(cash_data, f, indent=2)
        print("✅ Cash rate data written to cash_rate.json")
    except Exception as e:
        print(f"Error processing cash rate data: {e}")

    # === Additional Transformations ===
    df_pct = df.copy()
    df_pct.set_index('Date', inplace=True)

    # 12-month % change
    df_yoy = df_pct.pct_change(periods=12) * 100
    df_yoy = df_yoy.dropna()
    df_yoy.reset_index(inplace=True)
    df_yoy['Date'] = df_yoy['Date'].dt.strftime('%Y-%m-%d')
    try:
        with open('data_yoy.json', 'w') as f:
            json.dump(df_yoy.to_dict(orient='records'), f, indent=2)
        print("✅ YoY % change data written to data_yoy.json")
    except Exception as e:
        print(f"Error writing YoY JSON: {e}")

    # 1-month % change
    df_mom = df_pct.pct_change(periods=1) * 100
    df_mom = df_mom.dropna()
    df_mom.reset_index(inplace=True)
    df_mom['Date'] = df_mom['Date'].dt.strftime('%Y-%m-%d')
    try:
        with open('data_mom.json', 'w') as f:
            json.dump(df_mom.to_dict(orient='records'), f, indent=2)
        print("✅ MoM % change data written to data_mom.json")
    except Exception as e:
        print(f"Error writing MoM JSON: {e}")

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
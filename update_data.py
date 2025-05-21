import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import json
import subprocess
import sys
from datetime import datetime

# === Configuration ===
CSV_FILE = 'Price Dataset.csv'   # Your CSV file in repo root
JSON_FILE = 'data_combined.json' # Output JSON file in repo root
GIT_COMMIT_MSG = f'Monthly data update: {datetime.now().strftime("%Y-%m-%d")}'

def read_and_forecast(csv_path, json_path):
    try:
        # Load CSV with tab separator
        df = pd.read_csv(csv_path, sep='\t')
    except Exception as e:
        print(f'Error reading CSV: {e}')
        sys.exit(1)

    # Drop 'Date' column; keep only city price columns
    df_prices = df.drop(columns=['Date'])

    results = {}
    for city in df_prices.columns:
        prices = df_prices[city]
        # Fit exponential smoothing model
        model = ExponentialSmoothing(prices, trend='add', seasonal=None)
        fit = model.fit()
        forecast = fit.forecast(60)
        # Combine historic + forecast
        full_series = pd.concat([prices, forecast]).reset_index(drop=True)
        results[city] = full_series.tolist()

    # Write JSON to file
    try:
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'Successfully wrote JSON to {json_path}')
    except Exception as e:
        print(f'Error writing JSON: {e}')
        sys.exit(1)

def git_commit_and_push(file_path, commit_msg):
    try:
        subprocess.run(['git', 'add', file_path], check=True)
        subprocess.run(['git', 'commit', '-m', commit_msg], check=True)
        subprocess.run(['git', 'push'], check=True)
        print('Git commit and push successful.')
    except subprocess.CalledProcessError as e:
        print(f'Git command failed: {e}')
        sys.exit(1)

def main():
    read_and_forecast(CSV_FILE, JSON_FILE)
    git_commit_and_push(JSON_FILE, GIT_COMMIT_MSG)

if __name__ == '__main__':
    main()
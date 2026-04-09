import pandas as pd
import requests
from io import StringIO

def download_data(url:str):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        raise ValueError(f"Failed to fetch data. Status code: {response.status_code}")

    try:
        data = StringIO(response.text)
        df = pd.read_csv(data)
    except Exception as e:
        raise ValueError(f"Error reading CSV: {e}")

    return df

def load_data(path:str):
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise ValueError(f"Error loading file: {e}")

    df = df.dropna(how="all")

    if df.empty:
        raise ValueError("Loaded DataFrame is empty")

    return df
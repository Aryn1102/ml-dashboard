import pandas as pd

def load_data(path:str):
    df=pd.read_csv(path)
    df=df.dropna(how="all")
    return df
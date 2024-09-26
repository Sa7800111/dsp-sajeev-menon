import joblib
import numpy as np
import pandas as pd
from house_prices.preprocess import process_data
from house_prices.config import PATH


def make_predictions(df: pd.DataFrame, path: str = PATH) -> np.array:
    df = process_data(df).dropna()
    model_path = path + "lreg.joblib"
    model = joblib.load(model_path)
    predictions = model.predict(df)
    return predictions

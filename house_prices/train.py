import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import LinearRegression
from house_prices.preprocess import preprocess_data, fit_encoder, fit_scaler
from sklearn.model_selection import train_test_split
from house_prices.config import CATEGORICAL_COLUMNS, CONTINOUS_COLUMNS, MODEL_PATH

def build_model(data: pd.DataFrame) -> dict:
     X = data.drop("SalePrice", axis=1)
    y = data["SalePrice"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    encoder = fit_encoder(X_train[CATEGORICAL_COLUMNS])
    scaler = fit_scaler(X_train[CONTINOUS_COLUMNS])
    X_train_processed = preprocess_data(X_train, encoder, scaler)

    model = LinearRegression()
    model.fit(X_train_processed, y_train)

    X_test_processed = preprocess_data(X_test, encoder, scaler)
    y_pred = model.predict(X_test_processed)

    rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))

 
    save_artifacts(model, encoder, scaler)

    return {"rmsle": round(rmsle, 2)}

def save_artifacts(model: LinearRegression, encoder, scaler):
  
    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoder, 'C:/Users/CORE I5/dsp-sajeev-menon/model/encoder.joblib')
    joblib.dump(scaler, 'C:/Users/CORE I5/dsp-sajeev-menon/model/scaler.joblib')
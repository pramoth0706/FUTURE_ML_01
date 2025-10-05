# src/utils.py
import pandas as pd

def create_date_features(df):
    df["month"] = df["ds"].dt.month
    df["day"] = df["ds"].dt.day
    df["dayofweek"] = df["ds"].dt.dayofweek
    df["is_month_start"] = df["ds"].dt.is_month_start.astype(int)
    return df

def make_lags(df, lags=[1,7,14,28]):
    for L in lags:
        df[f"lag_{L}"] = df["y"].shift(L)
    return df

def make_rollings(df, windows=[7,14,28]):
    for w in windows:
        df[f"roll_mean_{w}"] = df["y"].shift(1).rolling(w).mean()
    return df

def prepare_ml_df(df, max_lag=28):
    df = df.copy()
    df = create_date_features(df)
    df = make_lags(df, lags=[1,7,14,28])
    df = make_rollings(df, windows=[7,14,28])
    df = df.dropna().reset_index(drop=True)
    return df
# src/models.py (Prophet)
from prophet import Prophet
import pandas as pd

def train_prophet(df):
    # df must have columns 'ds' (datetime) and 'y' (target)
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    m.fit(df)
    return m

def prophet_forecast(model, periods=30, freq="D"):
    future = model.make_future_dataframe(periods=periods, freq=freq)
    fcst = model.predict(future)
    return fcst[['ds','yhat','yhat_lower','yhat_upper']]
# src/models.py (LightGBM parts)
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

def train_lgb(df_features, target_col="y"):
    X = df_features.drop(columns=[target_col, "ds"])
    y = df_features[target_col]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)  # time-series split prefer non-shuffle
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    params = {"objective":"regression", "metric":"rmse", "num_leaves":31, "learning_rate":0.05}
    model = lgb.train(params, train_data, valid_sets=[val_data], early_stopping_rounds=50, num_boost_round=2000)
    preds = model.predict(X_val)
    print("VAL MAE:", mean_absolute_error(y_val, preds))
    return model
# src/utils.py (append)
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def save_model(model, path="models/lgb_model.joblib"):
    joblib.dump(model, path)

def load_model(path="models/lgb_model.joblib"):
    return joblib.load(path)

def mae(y_true, y_pred): return mean_absolute_error(y_true, y_pred)
def rmse(y_true, y_pred): return np.sqrt(mean_squared_error(y_true, y_pred))
def mape(y_true, y_pred): return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100

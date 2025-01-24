from flask import flash
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd

def train_model(final_data):
    # Feature ve target ayarları
    X = final_data[['month', 'unit', 'unit_price', 'inflation_rate', 'total_orders', 'unit_per_order']]
    y = final_data['total_sales']

    # Eğitim ve test verisi olarak bölelim
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modeli oluştur ve eğit
    model = XGBRegressor()
    model.fit(X_train, y_train)

    # Test verisi üzerinde tahminler yapalım
    y_pred = model.predict(X_test)

    # Performans metriklerini hesaplayalım
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    def calculate_mape(actual, predicted):
        return np.mean(np.abs((actual - predicted) / actual)) * 100

    def calculate_mae(actual, predicted):
        return np.mean(np.abs(actual - predicted))

    mape = calculate_mape(y_test, y_pred)
    mae = calculate_mae(y_test, y_pred)

    # Metrikleri bir sözlükte toplayalım
    metrics = {
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'mae': mae
    }

    # Model ve metrikleri döndürelim
    return model, metrics

def train_model_customer_seg(final_data):
    X = final_data[['Yas', 'Cinsiyet']]
    y = final_data['Adet']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    metrics = {
        'rmse': rmse,
        'mae': mae
    }

    return model, metrics

def train_model_return(final_data_predict):
    X = final_data_predict.drop(columns=['Month', 'Total Returns'])
    y = final_data_predict['Total Returns']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    metrics = {
        'rmse': rmse,
        'mae': mae
    }

    new_orders = pd.DataFrame({
    'Total Orders': [10000],
    'Financial Concerns': [0],
    'Fit/Size Issues': [0],
    'Logistics Issues': [0],
    'Personal Preferences': [0],
    'Product Issues': [0],
    'Special Cases': [0]
    })

    predicted_returns = model.predict(new_orders)[0]
    predicted_returns = round(predicted_returns)

    reason_columns = ['Financial Concerns', 'Fit/Size Issues', 'Logistics Issues', 
                    'Personal Preferences', 'Product Issues', 'Special Cases']

    reason_proportions = final_data_predict[reason_columns].sum() / final_data_predict['Total Returns'].sum()
    reason_counts = (reason_proportions * predicted_returns).round().astype(int)

    result_df = pd.DataFrame({
        'Reason': reason_columns,
        'Predicted Count': reason_counts.values
    })

    print("Final Data Predict Columns:", final_data_predict.columns)
    print("Final Data Predict Head:\n", final_data_predict.head())
    if final_data_predict.empty:
        flash("Error: Uploaded data is empty or incorrect format.")

    return model, metrics, result_df, predicted_returns




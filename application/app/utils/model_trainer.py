from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

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

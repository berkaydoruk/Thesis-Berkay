import pandas as pd

def predict_month_10(model, final_data):

    month_10_data = {
        'month': [10],
        'unit': [33888.81],
        'unit_price': [89.29],
        'inflation_rate': [74.395], 
        'total_orders': [15777.97],
        'unit_per_order': [2.11]
    }

    month_10_df = pd.DataFrame(month_10_data)

    month_10_pred = model.predict(month_10_df)

    return round(month_10_pred[0], 2)
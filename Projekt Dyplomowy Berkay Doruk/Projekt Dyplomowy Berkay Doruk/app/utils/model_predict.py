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

def predict_customer_seg(model, final_data, age, gender):

    next_sale_data = {
    'Yas': [age],  # Example: '31-40'
    'Cinsiyet': [gender]  # Example: 'Erkek'
    }

    next_sale_df = pd.DataFrame(next_sale_data)

    next_sale_pred = model.predict(next_sale_df)

    return round(next_sale_pred[0], 2)

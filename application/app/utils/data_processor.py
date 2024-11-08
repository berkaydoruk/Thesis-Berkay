import pandas as pd

def create_dataset(sales_file_path, inflation_file_path):
    sales_data = pd.read_excel(sales_file_path)
    total_sales_inflation_data = pd.read_excel(inflation_file_path)
    
    sales_data['order_date'] = pd.to_datetime(sales_data['order_date'])
    sales_data['year'] = sales_data['order_date'].dt.year
    sales_data['month'] = sales_data['order_date'].dt.month

    monthly_totals = sales_data.groupby(['year', 'month']).agg({
        'unit': 'sum',              
        'unit_price': 'mean',       
        'total_price': 'sum',       
        'order_date': 'count'       
    }).reset_index()

    monthly_totals.rename(columns={'order_date': 'total_orders'}, inplace=True)

    merged_data = pd.merge(monthly_totals, total_sales_inflation_data, on='month')

    final_data = merged_data[['month', 'unit', 'unit_price', 'total_price', 'inflation_rate', 'total_orders']].copy()
    final_data = final_data.rename(columns={'total_price': 'total_sales'})  
    final_data['unit_per_order'] = final_data['unit'] / final_data['total_orders']

    return final_data

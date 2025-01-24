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

def create_dataset_customerseg(customer_seg_file_path):
    sales_file_path = pd.read_excel(customer_seg_file_path)
    sales_file_path.dropna(inplace=True)

    sales_file_path['Cinsiyet'] = sales_file_path['Cinsiyet'].map({'Belirtilmemis': 0, 'Erkek': 1, 'Kadin': 2})

    age_mapping = {
    '0-20': 0,
    '21-30': 1,
    '31-40': 2,
    '41-50': 3,
    '51-60': 4,
    '61-70': 5,
    '71+': 6
    }
    sales_file_path['Yas'] = sales_file_path['Yas'].map(age_mapping)

    return sales_file_path

def create_dataset_return(return_predict_path, return_cluster_path):

    final_data_predict = pd.read_excel(return_predict_path)

    final_data_cluster = pd.read_excel(return_cluster_path)

    return final_data_predict, final_data_cluster

import numpy as np
import matplotlib.pyplot as plt

def plot_sales_predictions(monthly_results):
    plt.figure(figsize=(10, 6))
    plt.plot(monthly_results['month'], monthly_results['actual_total'], label='Actual Sales', marker='o', linestyle='--')
    plt.plot(monthly_results['month'], monthly_results['predicted_total'], label='Predicted Sales', marker='x', linestyle='--')

    z_actual = np.polyfit(monthly_results['month'], monthly_results['actual_total'], 1) 
    p_actual = np.poly1d(z_actual)
    plt.plot(monthly_results['month'], p_actual(monthly_results['month']), "r--", label='Actual Sales Trend')

    z_predicted = np.polyfit(monthly_results['month'], monthly_results['predicted_total'], 1)
    p_predicted = np.poly1d(z_predicted)
    plt.plot(monthly_results['month'], p_predicted(monthly_results['month']), "b--", label='Predicted Sales Trend')

    plt.title('Actual and Predicted Total Sales (Linear Trend)')
    plt.xlabel('Month')
    plt.ylabel('Total Sales')
    plt.legend()

    plt.grid(True)
    image_path = 'static/images/sales_prediction_plot.png'
    plt.savefig(image_path)
    plt.close()
    return image_path
from flask import Blueprint, render_template, request, redirect, url_for, flash
from utils.upload_handler_sales_predict import save_uploaded_files
from utils.data_processor import create_dataset
from utils.model_trainer import train_model
from utils.model_predict import predict_month_10
from utils.model_plot import plot_sales_predictions
import os

salesPredict_bp = Blueprint('salespredict', __name__)

@salesPredict_bp.route('/salespredict', methods=['GET', 'POST'])
def salesPredict():
    final_data = None
    metrics = None
    model = None  
    month_10_prediction = None  

    if request.method == 'POST':
        file1 = request.files['file1']
        file2 = request.files['file2']

        if not file1 or not file2:
            flash('Both files must be uploaded!')
            return redirect(request.url)

        try:
            filepath1, filepath2 = save_uploaded_files(file1, file2)

            final_data = create_dataset(filepath1, filepath2)

            model, metrics = train_model(final_data)

            # Predict for month 10
            month_10_prediction = predict_month_10(model, final_data)

            # Prepare data for plotting and display the plot
            monthly_results = final_data[['month', 'total_sales']].copy()
            monthly_results.rename(columns={'total_sales': 'actual_total'}, inplace=True)
            monthly_results['predicted_total'] = model.predict(final_data[['month', 'unit', 'unit_price', 'inflation_rate', 'total_orders', 'unit_per_order']])
            plot_image_path = plot_sales_predictions(monthly_results)

            return render_template('salespredict_result.html', 
                       final_data=final_data.to_html(classes='table table-striped'),
                       metrics=metrics,
                       month_10_prediction=month_10_prediction,
                       plot_image_path=plot_image_path) 

        except Exception as e:
            flash(f'An error occurred: {str(e)}')
            return redirect(request.url)

    return render_template('salespredict.html')


@salesPredict_bp.route('/train_model', methods=['POST'])
def train_and_display_model():
    try:
        final_data = request.form.get('final_data')

        metrics = train_model(final_data)

        return render_template('salespredict_result.html', metrics=metrics)

    except Exception as e:
        flash(f'Hata oluştu: {str(e)}')
        return redirect(url_for('salespredict.salesPredict'))

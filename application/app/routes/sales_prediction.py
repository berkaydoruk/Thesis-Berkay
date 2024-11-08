from flask import Blueprint, render_template, request, redirect, url_for, flash
from utils.upload_handler_sales_predict import save_uploaded_files
from utils.data_processor import create_dataset
from utils.model_trainer import train_model
import os

salesPredict_bp = Blueprint('salespredict', __name__)

@salesPredict_bp.route('/salespredict', methods=['GET', 'POST'])
def salesPredict():
    final_data = None
    metrics = None
    model = None  # Başlangıçta model None

    if request.method == 'POST':
        file1 = request.files['file1']
        file2 = request.files['file2']

        # Dosya kontrolü
        if not file1 or not file2:
            flash('Her iki dosya da yüklenmeli!')
            return redirect(request.url)

        try:
            filepath1, filepath2 = save_uploaded_files(file1, file2)

            # Dataset oluşturma
            final_data = create_dataset(filepath1, filepath2)

            # Modeli eğit ve metrikleri al
            model, metrics = train_model(final_data)

            # Sonuçları HTML'de göstermek için gönder
            return render_template('salespredict_result.html', 
                                   final_data=final_data.to_html(classes='table table-striped'),
                                   metrics=metrics)

        except Exception as e:
            flash(f'Hata oluştu: {str(e)}')
            return redirect(request.url)

    return render_template('salespredict.html')


# Modeli eğitmek için yeni bir rota
@salesPredict_bp.route('/train_model', methods=['POST'])
def train_and_display_model():
    try:
        # Final veri mevcutsa model eğit
        final_data = request.form.get('final_data')  # Burada veri gönderilecek

        # Modeli eğit
        metrics = train_model(final_data)

        # Sonuçları göster
        return render_template('salespredict_result.html', metrics=metrics)

    except Exception as e:
        flash(f'Hata oluştu: {str(e)}')
        return redirect(url_for('salespredict.salesPredict'))

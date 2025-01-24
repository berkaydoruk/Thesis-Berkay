import matplotlib
matplotlib.use('Agg')
from flask import Blueprint, render_template, request, redirect, url_for, flash
from utils.upload_handler_sales_predict import save_uploaded_files
from utils.data_processor import create_dataset_return
from utils.model_trainer import train_model_return
import os
from flask import session
import matplotlib.pyplot as plt
import io
import base64
import pandas as pd

returns_bp = Blueprint('returns_bp', __name__)

# Global değişkenler
final_data_cluster = None
final_data_predict = None
metrics_return = None
model_return = None
result_df_return = None

@returns_bp.route('/returns', methods=['GET', 'POST'])
def returnsPredict():
    global final_data_predict, final_data_cluster, metrics_return, model_return, result_df_return, predicted_returns

    if request.method == 'POST':
        file1 = request.files.get('file1')
        file2 = request.files.get('file2')

        if not file1 or not file2:
            flash('Both files must be uploaded!')
            return redirect(request.url)

        try:
            # Dosyaları kaydet
            filepath1, filepath2 = save_uploaded_files(file1, file2)

            # Veri setini oluştur
            final_data_predict, final_data_cluster = create_dataset_return(filepath1, filepath2)

            # Modeli eğit ve tahmin yap
            model_return, metrics_return, result_df_return, predicted_returns = train_model_return(final_data_predict)

            # Grafiği oluştur
            chart_url = generate_return_reason_chart(final_data_cluster)

            return render_template('returns_result.html', 
                                   final_data=final_data_predict.to_html(classes='table table-striped'),
                                   metrics=metrics_return,
                                   result_df=result_df_return.to_html(classes='table table-striped'),
                                   predicted_returns=predicted_returns,
                                   chart_url=chart_url)

        except Exception as e:
            flash(f'An error occurred: {str(e)}')
            return redirect(request.url)

    return render_template('returns_upload.html')


@returns_bp.route('/train_model', methods=['POST'])
def train_and_display_returns_model():
    try:
        final_data_predict = request.form.get('final_data')

        model_return, metrics_return, result_df_return = train_model_return(final_data_predict)

        return render_template('returns_result.html', 
                               metrics=metrics_return, 
                               result_df=result_df_return.to_html(classes='table table-striped'))

    except Exception as e:
        flash(f'An error occurred: {str(e)}')
        return redirect(url_for('returns_bp.returnsPredict'))
    
def generate_return_reason_chart(df):
    try:
        # Kategori eşleşmeleri
        category_mapping = {
            'Teslim edilemeyen gonderi': 'Logistics Issues',
            'Kargo Teslimati Gecikmesi': 'Logistics Issues',
            'Yanlis urun gonderildi': 'Product Issues',
            'Kusurlu urun gonderildi': 'Product Issues',
            'Yanlis siparis verdim': 'Product Issues',
            'Eksik Urun': 'Product Issues',
            'Urunumun parcasi/aksesuari eksik gonderildi': 'Product Issues',
            'Bedeni/Ebati Buyuk Geldi': 'Fit/Size Issues',
            'Bedeni/Ebati Kucuk Geldi': 'Fit/Size Issues',
            'Tazmin': 'Financial Concerns',
            'Daha iyi bir fiyat mevcut': 'Financial Concerns',
            'Vazgectim': 'Personal Preferences',
            'Begenmedim': 'Personal Preferences'
        }

        # 'iade Sebebi' sütununa göre kategorileri eşleştir
        df['Category'] = df['iade Sebebi'].map(category_mapping)

        # Kategori bazında sayı sayımı
        category_counts = df['Category'].value_counts()

        # Grafik çizimi
        plt.figure(figsize=(10, 6))
        category_counts.plot(kind='bar', color=['blue', 'green', 'red', 'purple', 'orange', 'brown'])
        plt.title('Return Reasons by Category')
        plt.xlabel('Return Categories')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Grafiği belleğe kaydet
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()

        # Base64 kodlaması yaparak HTML'ye gömülebilir hale getir
        chart_url = base64.b64encode(img.getvalue()).decode()
        return f"data:image/png;base64,{chart_url}"

    except Exception as e:
        print(f"Error generating chart: {e}")
        return None
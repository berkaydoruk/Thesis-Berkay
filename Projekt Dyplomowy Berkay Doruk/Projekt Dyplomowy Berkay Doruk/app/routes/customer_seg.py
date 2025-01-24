import matplotlib
matplotlib.use('Agg')
from flask import Blueprint, render_template, request, redirect, url_for, flash
from utils.upload_handler_sales_predict import save_uploaded_files_customer_seg
from utils.data_processor import create_dataset_customerseg
from utils.model_trainer import train_model_customer_seg
from utils.model_predict import predict_customer_seg
import os
from flask import session
import matplotlib.pyplot as plt
import io
import base64

customerSeg_bp = Blueprint('customersegmentation', __name__)

final_data_seg = None
metrics_seg = None
model_seg = None
chart_url_seg = None

@customerSeg_bp.route('/customersegmentation', methods=['GET', 'POST'])
def customerSeg():
    global model_seg, final_data_seg, metrics_seg

    if request.method == 'POST':
        file1 = request.files['file1']

        if not file1:
            flash('Files must be uploaded!')
            return redirect(request.url)

        try:
            filepath1 = save_uploaded_files_customer_seg(file1)
            final_data_seg = create_dataset_customerseg(filepath1)

            model_seg, metrics_seg = train_model_customer_seg(final_data_seg)

            chart_url = create_sales_chart(final_data_seg)

            return render_template(
                'customerseg_result.html',
                final_data=final_data_seg.to_html(classes='table table-striped'),
                metrics=metrics_seg,
                chart_url=chart_url
            )

        except Exception as e:
            flash(f'An error occurred: {str(e)}')
            return redirect(request.url)

    return render_template('customerseg.html')

def create_sales_chart(df):
    global chart_url_seg  # Global değişkeni kullan

    # Belirlenen yaş ve cinsiyet gruplarını filtrele
    filtered_df = df[
        ((df['Yas'] == 1) & (df['Cinsiyet'] == 2)) |  # 21-30 kadın
        ((df['Yas'] == 2) & (df['Cinsiyet'] == 2)) |  # 31-40 kadın
        ((df['Yas'] == 2) & (df['Cinsiyet'] == 1)) |  # 31-40 erkek
        ((df['Yas'] == 6) & (df['Cinsiyet'] == 2))    # 71+ kadın
    ]

    if filtered_df.empty:
        chart_url_seg = None
        return None

    # Grup bazında satış sayısını hesapla
    sales_count = filtered_df.groupby(['Yas', 'Cinsiyet']).size().reset_index(name='Sales')

    # Etiketleri tanımla
    age_labels = {0: '0-20', 1: '21-30', 2: '31-40', 3: '41-50', 4: '51-60', 5: '61-70', 6: '71+'}
    gender_labels = {0: 'Not Specified', 1: 'Male', 2: 'Female'}
    sales_count['Yas'] = sales_count['Yas'].map(age_labels)
    sales_count['Cinsiyet'] = sales_count['Cinsiyet'].map(gender_labels)

    # Grafik oluştur
    plt.figure(figsize=(8, 6))
    plt.bar(sales_count['Yas'] + " " + sales_count['Cinsiyet'], sales_count['Sales'])
    plt.xlabel('Age Group and Gender')
    plt.ylabel('Total Sales Count')
    plt.title('Sales Count by Different Age and Gender Groups')
    plt.xticks(rotation=45)
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    chart_url_seg = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return f"data:image/png;base64,{chart_url_seg}"


@customerSeg_bp.route('/predict_customerseg', methods=['POST'])
def predict_customerseg():
    global model_seg, final_data_seg, metrics_seg, chart_url_seg

    try:
        if final_data_seg is None or metrics_seg is None:
            flash("Data or model not found. Please train the model first.")
            return redirect(url_for('customersegmentation.customerSeg'))

        # Kullanıcıdan yaş ve cinsiyet bilgilerini al
        age = int(request.form['age'])
        gender = int(request.form['gender'])

        if model_seg is None:
            flash("Model has not been trained yet. Please upload and process the data first.")
            return redirect(url_for('customersegmentation.customerSeg'))

        # Tahmin işlemini yap
        prediction = predict_customer_seg(model_seg, None, age, gender)

        return render_template(
            'customerseg_result.html',
            result=f"Predicted quantity: {prediction}",
            metrics=metrics_seg,
            final_data=final_data_seg.to_html(classes='table table-striped'),
            chart_url=f"data:image/png;base64,{chart_url_seg}" if chart_url_seg else None
        )

    except Exception as e:
        flash(f"An error occurred during prediction: {str(e)}")
        return redirect(url_for('customersegmentation.customerSeg'))
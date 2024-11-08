# Base image olarak Python'un bir versiyonunu kullan
FROM python:3.9

# Çalışma dizinini ayarla
WORKDIR /app

# Gereksinim dosyasını container'a kopyala
COPY requirements.txt .

# Gereksinimleri yükle
RUN pip install --no-cache-dir -r requirements.txt

# Tüm proje dosyalarını container'a kopyala
COPY . .

# Uygulamayı başlat
CMD ["python", "app.py"]

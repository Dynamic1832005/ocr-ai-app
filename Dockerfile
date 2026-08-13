FROM python:3.10-slim

# System တွင် Tesseract OCR နှင့် OpenCV လိုအပ်ချက်များ တပ်ဆင်ခြင်း
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-mya \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt-lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 10000

CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--timeout", "120", "app:app"]

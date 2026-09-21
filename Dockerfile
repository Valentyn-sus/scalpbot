FROM python:3.12-slim

# Установка системных зависимостей (libgl1 вместо libgl1-mesa-glx)
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    gnupg \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    tesseract-ocr-ukr \
    tesseract-ocr-rus \
    && rm -rf /var/lib/apt-get/lists/*

WORKDIR /app

# Копирование и установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Установка браузеров Playwright и зависимости SeleniumBase
RUN playwright install --with-deps chromium
RUN python -m seleniumbase install chromedriver

# Копируем исходный код
COPY . .

# Открываем порт FastAPI
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
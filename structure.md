ai-crawler-pipeline/
├── app/
│   ├── __init__.py
│   ├── main.py                   # Точка входу FastAPI (ендпоінти, CORS, Lifespan)
│   ├── config.py                 # Зчитування змінних оточення (Pydantic Settings)
│   │
│   ├── api/                      # Маршрути та HTTP-контролери
│   │   ├── __init__.py
│   │   └── v1/
│   │       ├── __init__.py
│   │       ├── endpoints.py      # POST /api/v1/crawler/jobs, GET /healthz
│   │       └── schemas.py        # Pydantic-моделі (JobRequest, JobResponse, WebhookPayload)
│   │
│   ├── db/                       # Робота з PostgreSQL
│   │   ├── __init__.py
│   │   ├── connection.py         # Пул підключень asyncpg
│   │   ├── repository.py         # SQL-запити (CRUD для crawler_tasks, history, bot_credentials)
│   │   └── migrations/           # SQL-скрипти ініціалізації schemas/tables
│   │       └── init.sql
│   │
│   ├── services/                 # Сервісний шар та оркестрація
│   │   ├── __init__.py
│   │   ├── orchestrator.py       # Керування фоновими задачами, таймаутами (TTL) та Webhook
│   │   └── logger.py             # Структуроване Zero-PII JSON логування
│   │
│   └── crawler/                  # Модуль веб-розвідки (Твій SeleniumBase + Playwright)
│       ├── __init__.py
│       ├── runner.py             # Адаптер для виклику test_runner у фоновому процесі
│       ├── test_runner.py        # Твій майстер-файл (test.py), оформлений у функцію
│       ├── ocr.py                # Модуль розпізнавання картинок (Tesseract OCR fallback)
│       └── utils.py              # Семантичний пошук локаторів, генератор Faker-даних
│
├── .env.example                  # Приклад змінних оточення
├── Dockerfile                    # Базовий образ playwright/python + Tesseract
├── docker-compose.yml            # Локальний запуск (App + PostgreSQL)
├── requirements.txt              # Залежності проєкту
└── README.md                     # Документація з запуску
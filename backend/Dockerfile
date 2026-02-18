FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x entrypoint.sh

CMD ["/bin/bash", "-c", "python manage.py migrate && python manage.py import_BAC && python manage.py import_CHA && python manage.py import_CYB && python manage.py import_CZA && python manage.py import_HJC && python manage.py import_JOA && python manage.py import_KAB && python manage.py import_KKA && python manage.py import_KYI && python manage.py import_KZA && python manage.py import_OT && python manage.py import_OU && python manage.py import_OV && python manage.py import_OW && python manage.py import_OZ && python manage.py import_SED && python manage.py import_SKB && python manage.py import_SRB && python manage.py import_TYB && python manage.py import_UKC && python manage.py runserver 0.0.0.0:8000"]

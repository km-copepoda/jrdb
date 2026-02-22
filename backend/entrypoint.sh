#!/bin/bash

# 1. PostgreSQL 接続を待機
echo "Waiting for PostgreSQL..."
while ! python -c "import psycopg2; psycopg2.connect(dbname='${DB_NAME:-jrdb}', user='${DB_USER:-admin}', password='${DB_PASS:-admin}', host='${DB_HOST:-db}', port='${DB_PORT:-5432}')" 2>/dev/null; do
    echo "PostgreSQL is unavailable - sleeping"
    sleep 2
done
echo "PostgreSQL is ready!"

# 2. マイグレーション実行
echo "Running migrations..."
python manage.py migrate

# 3. JRDBデータインポート
echo "Importing data..."
BEGIN="${IMPORT_START:-20180101}"
END="${IMPORT_END:-20241231}"

# 基本マスタ + 前日基本情報
python manage.py group_import_PreviousDayBaseInformation -begin "$BEGIN" -end "$END" || echo "Warning: PreviousDayBaseInformation import failed"

# 前日詳細情報 (オッズ, 調教, 拡張, 詳細)
python manage.py group_import_PreviousDayInformation -begin "$BEGIN" -end "$END" || echo "Warning: PreviousDayInformation import failed"

# 成績情報
python manage.py group_import_GradeInformation -begin "$BEGIN" -end "$END" || echo "Warning: GradeInformation import failed"

# 直前情報
python manage.py group_import_ThatDayInformation -begin "$BEGIN" -end "$END" || echo "Warning: ThatDayInformation import failed"

# 4. サーバー起動
echo "Starting server..."
python manage.py runserver 0.0.0.0:8000

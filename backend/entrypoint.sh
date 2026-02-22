#!/bin/bash

# Wait for PostgreSQL to be ready
echo "Waiting for PostgreSQL..."
while ! python -c "
import psycopg2, os
psycopg2.connect(
    host=os.environ.get('DB_HOST', 'db'),
    port=5432,
    dbname=os.environ.get('DB_NAME', 'jrdb'),
    user=os.environ.get('DB_USER', 'admin'),
    password=os.environ.get('DB_PASS', 'admin'),
)
" 2>/dev/null; do
    echo "  PostgreSQL is not ready yet. Retrying in 2s..."
    sleep 2
done
echo "PostgreSQL is ready!"

# Run migrations
echo "Running migrations..."
python manage.py migrate

# Import data using group_import commands
# 環境変数 IMPORT_BEGIN / IMPORT_END で日付範囲を指定（YYYYMMDD）
# 未指定の場合は group_import 側のデフォルト（当日）が使われる
BEGIN_OPT=""
END_OPT=""
if [ -n "$IMPORT_BEGIN" ]; then
    BEGIN_OPT="-begin $IMPORT_BEGIN"
fi
if [ -n "$IMPORT_END" ]; then
    END_OPT="-end $IMPORT_END"
fi

echo "Importing data... (begin=${IMPORT_BEGIN:-today}, end=${IMPORT_END:-today})"

echo "=== group_import_PreviousDayBaseInformation ==="
python manage.py group_import_PreviousDayBaseInformation $BEGIN_OPT $END_OPT || echo "Warning: PreviousDayBaseInformation failed"

echo "=== group_import_PreviousDayInformation ==="
python manage.py group_import_PreviousDayInformation $BEGIN_OPT $END_OPT || echo "Warning: PreviousDayInformation failed"

echo "=== group_import_GradeInformation ==="
python manage.py group_import_GradeInformation $BEGIN_OPT $END_OPT || echo "Warning: GradeInformation failed"

echo "=== group_import_ThatDayInformation ==="
python manage.py group_import_ThatDayInformation $BEGIN_OPT $END_OPT || echo "Warning: ThatDayInformation failed"

echo "Starting server..."
python manage.py runserver 0.0.0.0:8000


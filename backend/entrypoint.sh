#!/bin/bash

# WAIT for PostgreSQL to be ready
echo "Waiting for POSTGRESQL..."
while ! python -c "
import psycopg2, os
psycopg2.connect(
    host=os.getenv('DB_HOST', "db"),
    port=5432,
    dbname=os.getenv('DB_NAME', "jrdb"),
    user=os.getenv('DB_USER', "admin"),
    password=os.getenv('DB_PASS', "admin"),
)
" 2>/dev/null; do
    echo " PostgreSQL is not ready yet. Retrying in 2s..."
    sleep 2
done
echo "PostgreSQL in ready!"

# Run migrations
echo "Running migrations..."
python manage.py migrate

# IMport data using group_import commands
# 環境変数 IMPORT_BEGIN と IMPORT_END を使用して、インポートする期間を指定
# 未指定の場合は group_import側のデフォルト値が使用される
BEGIN_OPT=""
END_OPT=""
if [ -n "$IMPORT_BEGIN" ]; then
    BEGIN_OPT="--begin $IMPORT_BEGIN"
fi
if [ -n "$IMPORT_END" ]; then
    END_OPT="--end $IMPORT_END"
fi

echo "Importing data from ${IMPORT_BEGIN:-today} to ${IMPORT_END:-today}..."

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

docker compose exec backend python manage.py group_import_PreviousDayBaseInformation -begin 20180101 -end 20251231
docker compose exec backend python manage.py group_import_PreviousDayInformation-begin 20180101 -end 20251231
docker compose exec backend python manage.py group_import_GradeInformation -begin 20180101 -end 20251231
docker compose exec backend python manage.py group_import_ThatDayInformation -begin 20180101 -end 20251231
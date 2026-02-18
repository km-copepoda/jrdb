import os
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from database.management.common import valid_year_month_day, date_range, getModel, getColumnsDict
from datetime import datetime as dt
from concurrent.futures import ThreadPoolExecutor
from django.db.models.fields.related import ManyToOneRel
from django.db.utils import OperationalError
import time


class Command(BaseCommand):
    help = "This command can not use!"

    def add_arguments(self, parser):
        parser.add_argument(
            "-begin",
            "--find_begin_date",
            type=valid_year_month_day,
            help="YYMMDD or YYYYMMDD",
        )
        parser.add_argument(
            "-end",
            "--find_end_date",
            type=valid_year_month_day,
            help="YYMMDD or YYYYMMDD",
        )

    def handle(self, *args, **options):
        begin_date = (
            options["find_begin_date"]
            if options["find_begin_date"]
            else dt.now().date()
        )
        end_date = (
            options["find_end_date"] if options["find_end_date"] else dt.now().date()
        )
        start_time = time.time()
        for file_date in date_range(begin_date, end_date):
            filename = self.file_format.format(date=file_date.strftime("%y%m%d"))
            if not os.path.isfile(filename):
                continue
            read_and_write(filename, self.__model, self.__colspecs)
        end_time = time.time() - start_time
        print(end_time)



def read_and_write(filename, my_model, colspecs):
    print(f"import file is START -> {filename}")
    create_models, update_models = read_file(filename, my_model, colspecs)
    write_db(create_models, update_models, my_model)
    print(f"import file is END -> {filename}")


def read_file(filename, my_model, colspecs):
    with open(filename, "r", encoding="cp932") as f:
        models = list(map(lambda line: getModel(my_model, line, colspecs), f))
    create_models = []
    update_models = []
    for m in models:
        if my_model.objects.filter(pk=m.pk).exists():
            update_models.append(m)
        else:
            create_models.append(m)
    return create_models, update_models


def write_db(create_models, update_models, my_model):
    with transaction.atomic():
        try:
            my_model.objects.bulk_create(create_models)
            for m in update_models:
                m.save()
        except Exception as e:
            print(e)

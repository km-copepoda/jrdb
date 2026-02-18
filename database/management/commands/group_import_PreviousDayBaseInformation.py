from django.core.management.base import BaseCommand
from django.core import management
from database.management.common import valid_year_month_day
from datetime import datetime as dt

class Command(BaseCommand):
    help = 'This command is group import of UKC, CZA, KZA, KAB, BAC, KYI'

    def add_arguments(self, parser):
        parser.add_argument('-begin', '--find_begin_date', type=valid_year_month_day, help='YYMMDD or YYYYMMDD')
        parser.add_argument('-end', '--find_end_date', type=valid_year_month_day, help='YYMMDD or YYYYMMDD')

    def handle(self, *args, **options):
        begin_date = options['find_begin_date'] if options['find_begin_date'] else dt.now().date()
        end_date = options['find_end_date'] if options['find_end_date'] else dt.now().date()

        management.call_command('import_UKC', find_begin_date=begin_date, find_end_date=end_date)
        management.call_command('import_CZA', find_begin_date=begin_date, find_end_date=end_date)
        management.call_command('import_KZA', find_begin_date=begin_date, find_end_date=end_date)
        management.call_command('import_KAB', find_begin_date=begin_date, find_end_date=end_date)
        management.call_command('import_BAC', find_begin_date=begin_date, find_end_date=end_date)
        management.call_command('import_KYI', find_begin_date=begin_date, find_end_date=end_date)


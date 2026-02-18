from django.core.management.base import BaseCommand
from django.core import management
from database.management.common import valid_year_month_day
from datetime import datetime as dt

class Command(BaseCommand):
    help = 'This command is group import of HJC, SRB, SKB, SED'

    def add_arguments(self, parser):
        parser.add_argument('-begin', '--find_begin_date', type=valid_year_month_day, help='YYMMDD or YYYYMMDD')
        parser.add_argument('-end', '--find_end_date', type=valid_year_month_day, help='YYMMDD or YYYYMMDD')

    def handle(self, *args, **options):
        begin_date = options['find_begin_date'] if options['find_begin_date'] else dt.now().date()
        end_date = options['find_end_date'] if options['find_end_date'] else dt.now().date()

        management.call_command('import_HJC', find_begin_date=begin_date, find_end_date=end_date)
        management.call_command('import_SRB', find_begin_date=begin_date, find_end_date=end_date)
        management.call_command('import_SKB', find_begin_date=begin_date, find_end_date=end_date)
        management.call_command('import_SED', find_begin_date=begin_date, find_end_date=end_date)


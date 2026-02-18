from database.models.PreviousDayInformation import 馬基本情報
from database.management.import_BaseCommand import Command

class Command(Command):
    help = 'command to import UKC(馬基本情報) files'
    file_format = './database/temp/UKC{date}.txt'
    def __init__(self):
        self.__model = 馬基本情報
        max_lengths = [fields.max_length for fields in self.__model._meta.fields]
        colspecs_end = []
        offset = len(colspecs_end)
        colspecs_end.extend([sum(max_lengths[offset:i]) + max_lengths[i] for i in range(offset, len(max_lengths))])
        colspecs_begin = [0]
        colspecs_begin.extend(colspecs_end[offset:-1])
        self.__colspecs = [(begin, end) for begin, end in zip(colspecs_begin, colspecs_end)]


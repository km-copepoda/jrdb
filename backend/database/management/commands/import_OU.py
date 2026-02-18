from database.models.PreviousDayInformation import 前日_基準三連複情報
from database.management.import_BaseCommand import Command

class Command(Command):
    help = 'command to import OT(前日_基準三連複情報) files'
    file_format = './database/temp/OT{date}.txt'
    def __init__(self):
        self.__model = 前日_基準三連複情報
        max_lengths = [fields.max_length for fields in self.__model._meta.fields]
        colspecs_end = [6, 8]
        offset = len(colspecs_end)
        colspecs_end.extend([sum(max_lengths[offset:i]) + max_lengths[i] for i in range(offset, len(max_lengths))])
        colspecs_begin = [0, 0, 0]
        colspecs_begin.extend(colspecs_end[offset:-1])
        self.__colspecs = [(begin, end) for begin, end in zip(colspecs_begin, colspecs_end)]


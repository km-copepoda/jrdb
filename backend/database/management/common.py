from django.core.management.base import CommandError
from django.db.models.fields.related import ForeignKey, OneToOneField
from django.db.models.fields import TextField
from datetime import datetime as dt
from datetime import timedelta
DECOMP_DIR = './database/temp/'

# CP932 (Windows-31J) を使用
_ENCODING = 'cp932'
def replaceSJIS(record):
    return record\
        .replace('隆', '隆')\
        .replace('﨑', '崎')\
        .replace('橳', 'ﾇﾃ')\
        .replace('－', '‐')\
        .replace('～', 'ｶﾗ')\
        .replace('Ⅰ', '１')\
        .replace('Ⅱ', '２')\
        .replace('Ⅲ', '３')\
        .replace('＇', '’')\
        .replace('㎏', 'kg')\
        .replace('㌔', 'ｷﾛ')\
        .replace('①', '１')\
        .replace('②', '２')\
        .replace('③', '３')\
        .replace('④', '４')\
        .replace('⑤', '５')\
        .replace('栁', '柳')\
        .replace('塚', '塚')\
        .replace('ⅰ', '１')\
        .replace('ⅱ', '２')\
        .replace('ⅲ', '３')\
        .replace('ⅳ', '４')\
        .replace('ⅴ', '５')\
        .replace('ⅵ', '６')\
        .replace('ⅶ', '７')\
        .replace('ⅷ', '８')\
        .replace('ⅸ', '９')\
        .replace('ⅹ', '10')\
        .replace('ⅺ', '11')\
        .replace('ⅻ', '12')\
        .replace('㈲', '有')\
        .replace('㈱', '株')\
        .replace('　', '  ')

def valid_year_month_day(t):
    try:
        return dt.strptime(t, '%Y%m%d').date()
    except ValueError:
        raise CommandError('invalid is date {date}. ex -> YYYYMMDD'.format(date = t))

def date_range(_begin, _end):
    for n in range((_end - _begin).days + 1):
        yield _begin + timedelta(n)

def getModel(model, record, colspecs):
    column_dict = getColumnsDict(model, replaceSJIS(record), colspecs)
    return model(**column_dict)

def getColumnsDict(model, record, colspecs):
    recordEncode = record.encode(_ENCODING, errors="replace")

    columnDict = {}
    for index, field in enumerate(model._meta.fields):
        begin_index = colspecs[index][0]
        end_index = colspecs[index][1]
        field_name = field.name + ('_id' if type(field) == ForeignKey or type(field) == OneToOneField else '')
        value = recordEncode[begin_index:end_index].decode(_ENCODING, errors="replace")
        columnDict[field_name] = value if type(field) == TextField else value.strip()

    return columnDict
